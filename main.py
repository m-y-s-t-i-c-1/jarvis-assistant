import os
import time
import queue
import itertools
import threading
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

from src import tools  # noqa: F401
from src.core.agent import agent_loop
from src.core.jobs import porneste_thread_watcher
from src.core.context_manager import proceseaza as proceseaza_context
from src.core.database import db
from src.core.memory import memorie
from src.core.rag import rag
from src.core.consolidare import consolidare
from src.core.router import clasifica, este_raspuns_scurt_de_continuare
from src.core.monitor_ecran import porneste_monitorizare_ecran
from src.core.monitor_log import porneste_monitorizare_log
from src.tools.vedere import analizeaza_ecran_complet
from src.core.tts import spune
from src.core.stare_conversatie import conversatie_activa
from src.core.llm_provider import (
    intreaba_nvidia_conversatie,
    intreaba_nvidia_cod,
    ruleaza_cascada_externa,
    istoric_la_mesaje_openai,
)

load_dotenv()

# ---- Chei API Gemini ----
_gemini_chei = [
    os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
    for i in range(5)
]
_gemini_chei = [k for k in _gemini_chei if k]

if not _gemini_chei:
    raise ValueError("Nu am găsit nicio GEMINI_API_KEY în .env.")

_gemini_clienti = [genai.Client(api_key=k) for k in _gemini_chei]
_gemini_rotatie = itertools.cycle(_gemini_clienti)

groq_key    = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT_BAZA = """Tu ești Jarvis, un asistent AI personal extrem de inteligent, polivalent și loial.

Reguli de comportament:
- Te adresezi întotdeauna utilizatorului cu "Vasea".
- Tonul tău este calm, profesionist, dar cu un strop de umor sec, britanic.
- Răspunzi concis și la obiect, fără să divaghezi inutil.
- Ești proactiv: dacă observi o problemă sau o soluție mai bună, o menționezi.
- Nu ești servil sau exagerat de politicos — ești un consilier de încredere.
- Vorbești în limba română, cu un vocabular elevat dar natural.

Capacități:
- Ești un expert universal: programare, știință, matematică, istorie, filosofie,
  scriere creativă, eseuri, analiză, sfaturi de viață — orice domeniu.
- Poți controla sistemul de operare, aplicații, hardware și API-uri externe
  prin uneltele disponibile — folosește-le când e nevoie.
- Când utilizatorul întreabă ceva factual (oră, dată, vreme, calendar),
  folosești OBLIGATORIU uneltele disponibile, nu ghicești.
- Când utilizatorul cere ajutor intelectual (eseu, cod, analiză, explicație),
  răspunzi direct și complet, ca un expert în domeniu.

Mod vocal:
- Când ești în modul vocal, răspunsurile tale vor fi rostite cu voce tare.
- Evită liste cu puncte, simboluri speciale și Markdown în modul vocal.
- Formulează răspunsuri ca propoziții naturale, de parcă vorbești cu cineva.
"""

# ---- Coduri HTTP care declanșează trecerea la următoarea cheie ----
ERORI_FALLBACK = (503, 429, 500, 403)

# ---- Excepții de rețea/conexiune care NU au cod HTTP în mesaj, dar tot
# trebuie tratate ca eșec temporar al cheii curente, nu ca eroare fatală.
# httpx.RemoteProtocolError ("Server disconnected without sending a
# response") e cel mai frecvent exemplu — apare la conexiuni instabile
# sau când Google închide conexiunea brusc, independent de cod HTTP.
try:
    import httpx
    EXCEPTII_RETEA = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)
except ImportError:
    EXCEPTII_RETEA = ()


def _este_eroare_temporara(e: Exception) -> bool:
    """
    Decide dacă o excepție trebuie tratată ca eșec temporar al cheii curente
    (încercăm cheia următoare) sau ca eroare fatală (o propagăm).

    Verifică două lucruri:
    1. Dacă mesajul erorii conține unul din codurile HTTP din ERORI_FALLBACK.
    2. Dacă excepția e de tip rețea/conexiune (EXCEPTII_RETEA) — acestea nu
       au neapărat un cod HTTP în mesaj (ex: conexiune întreruptă brusc).
    """
    mesaj_eroare = str(e)
    are_cod_cunoscut = any(str(cod) in mesaj_eroare for cod in ERORI_FALLBACK)
    e_eroare_retea = isinstance(e, EXCEPTII_RETEA)
    return are_cod_cunoscut or e_eroare_retea


def groq_fallback(istoric: list, system_prompt: str) -> str:
    if not groq_client:
        return "Gemini este momentan indisponibil și nu am o cheie Groq configurată ca backup."

    mesaje = [{"role": "system", "content": system_prompt}]
    for continut in istoric:
        if not hasattr(continut, "parts"):
            continue
        text = " ".join(
            parte.text for parte in continut.parts
            if hasattr(parte, "text") and parte.text
        )
        if not text:
            continue
        rol = "user" if continut.role == "user" else "assistant"
        mesaje.append({"role": rol, "content": text})

    raspuns = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=mesaje,
        max_tokens=1024,
    )
    return raspuns.choices[0].message.content


def _ruleaza_gemini_apoi_restul(istoric: list, system_prompt: str) -> str:
    """
    Calea "unelte": Gemini (toate cheile) -> Groq -> cascada finală
    (OpenRouter -> NVIDIA generic -> Bytez). Folosită pentru orice cerere
    clasificată drept "unelte" de router.py, și ca ultimă plasă și pentru
    celelalte categorii, dacă NVIDIA specializat + Groq eșuează.
    """
    for _ in range(len(_gemini_clienti)):
        client_curent = next(_gemini_rotatie)
        try:
            return agent_loop(client_curent, GEMINI_MODEL, system_prompt, istoric)
        except Exception as e:
            if _este_eroare_temporara(e):
                print(f"[Cheie Gemini indisponibilă ({str(e)[:80]}...) — încerc următoarea]")
                time.sleep(0.5)
                continue
            raise

    print("[Toate cheile Gemini sunt indisponibile — comut pe Groq]")
    try:
        return groq_fallback(istoric, system_prompt)
    except Exception as e:
        print(f"[Groq indisponibil ({str(e)[:80]}) — încerc cascada externă]")

    rezultat_extern = ruleaza_cascada_externa(istoric, system_prompt)
    if rezultat_extern:
        return rezultat_extern

    return (
        "Vasea, toți providerii disponibili (Gemini, Groq, OpenRouter, "
        "NVIDIA NIM, Bytez) sunt indisponibili momentan. Încearcă din nou "
        "peste câteva minute."
    )


# Ține minte categoria ultimei cereri, pentru "routare sticky" — un
# răspuns scurt de continuare ("da", "nu", "ok") nu se reclasifică de la
# zero, moștenește categoria turei precedente. Fără asta, o confirmare
# pentru o acțiune Gemini (ex: git commit) putea "sări" pe NVIDIA, care
# nu are unelte reale și halucinează că a executat acțiunea. (Task 6.7)
_ultima_categorie: str = "unelte"  # implicit "unelte" — cea mai sigură presupunere inițială

# System prompt separat pentru NVIDIA (fallback text, FĂRĂ unelte reale).
# NU folosim SYSTEM_PROMPT_BAZA direct — acela spune explicit "poți
# controla sistemul... prin uneltele disponibile", ceea ce l-a făcut pe
# NVIDIA să halucineze că a executat un git commit care nu s-a întâmplat
# niciodată. Aici îi spunem clar opusul.
AVERTISMENT_NVIDIA_FARA_UNELTE = """IMPORTANT — citește cu atenție înainte să răspunzi:
În acest mod NU ai acces la nicio unealtă reală. NU poți rula comenzi, NU poți
face commit-uri Git, NU poți controla sistemul, ecranul sau orice altceva.
Ești strict un model de conversație/cunoștințe generale, fără capacitate de
acțiune.

NU pretinde NICIODATĂ, sub nicio formă, că ai executat o acțiune reală
(fișiere, comenzi de sistem, Git, aplicații etc.) — asta ar fi o informație
falsă transmisă lui Vasea.

Dacă cererea lui pare să necesite o acțiune reală, spune-i CLAR și DIRECT că
cererea asta trebuie procesată de asistentul principal (cu unelte), nu te
preface că ai făcut-o tu."""


def _ultimul_mesaj_utilizator(istoric: list) -> str:
    """Extrage textul ultimului mesaj de tip user din istoric, pentru clasificare."""
    if not istoric:
        return ""
    ultimul = istoric[-1]
    if not hasattr(ultimul, "parts"):
        return ""
    return " ".join(
        p.text for p in ultimul.parts if hasattr(p, "text") and p.text
    )


def ruleaza_cu_fallback(istoric: list, system_prompt: str) -> str:
    """
    Punctul de intrare principal. Clasifică cererea (router.py) și decide
    ce provider încearcă întâi:

        "unelte"      -> direct Gemini (singurul cu tool-calling funcțional)
        "cod"         -> NVIDIA (cheia dedicată de coding) întâi
        "conversatie" -> NVIDIA (cheile dedicate de conversație) întâi

    Pentru "cod"/"conversatie", dacă NVIDIA specializat eșuează, cade pe
    calea completă Gemini -> Groq -> cascada finală (_ruleaza_gemini_apoi_restul).

    Routare sticky (Task 6.7): un răspuns scurt de continuare ("da", "nu",
    "ok" etc.) moștenește categoria turei precedente, în loc să fie
    reclasificat independent — previne ruperea unui flux de confirmare
    Gemini (ex: git commit) la jumătate, către un provider fără unelte.
    """
    global _ultima_categorie

    text_cerere = _ultimul_mesaj_utilizator(istoric)

    if este_raspuns_scurt_de_continuare(text_cerere):
        categorie = _ultima_categorie
        print(f"[Router] Răspuns scurt de continuare — păstrez categoria anterioară: '{categorie}'")
    else:
        categorie = clasifica(text_cerere)
        print(f"[Router] Cerere clasificată drept: '{categorie}'")

    _ultima_categorie = categorie

    if categorie == "unelte":
        return _ruleaza_gemini_apoi_restul(istoric, system_prompt)

    system_prompt_nvidia = AVERTISMENT_NVIDIA_FARA_UNELTE + "\n\n" + system_prompt
    mesaje_openai = istoric_la_mesaje_openai(istoric, system_prompt_nvidia)

    if categorie == "cod":
        rezultat = intreaba_nvidia_cod(mesaje_openai)
    else:
        rezultat = intreaba_nvidia_conversatie(mesaje_openai)

    if rezultat:
        return rezultat

    print(f"[NVIDIA indisponibil pentru categoria '{categorie}' — trec pe Gemini]")
    _ultima_categorie = "unelte"
    return _ruleaza_gemini_apoi_restul(istoric, system_prompt)


def bucla_text(istoric: list, sesiune_id: str, system_prompt: str):
    """Bucla de conversație prin terminal."""
    print("Scrie 'exit' pentru a încheia conversația.\n")

    while True:
        mesaj_utilizator = input("Tu: ")

        if mesaj_utilizator.lower() in ("exit", "quit", "stop"):
            print("Jarvis: La revedere, Vasea.")
            db.inchide_sesiune(sesiune_id)
            break

        if not mesaj_utilizator.strip():
            # Enter fără text (des posibil când wake word/monitoarele
            # scriu în terminal concurent) — reluăm fără să irosim un
            # apel API pe un mesaj gol.
            continue

        # Adăugăm în istoric
        istoric.append(
            types.Content(role="user", parts=[types.Part(text=mesaj_utilizator)])
        )

        # Sliding window
        proceseaza_context(istoric, sesiune_id, client=next(_gemini_rotatie))

        # Răspuns
        raspuns_text = ruleaza_cu_fallback(istoric, system_prompt)
        print("Jarvis:", raspuns_text)

        # Salvare în DB
        db.salveaza_mesaj(sesiune_id, "user", mesaj_utilizator)
        db.salveaza_mesaj(sesiune_id, "assistant", raspuns_text)

        # Indexare în RAG
        rag.indexeaza_mesaj("user", mesaj_utilizator, sesiune_id)
        rag.indexeaza_mesaj("assistant", raspuns_text, sesiune_id)

        # Extragere amintiri episodice (în fundal)
        try:
            client_mem = next(_gemini_rotatie)
            threading.Thread(
                target=memorie.extrage_si_salveaza,
                args=(mesaj_utilizator, raspuns_text, sesiune_id, client_mem, GEMINI_MODEL),
                daemon=True,
            ).start()
        except Exception as e:
            print(f"[Memorie] Extragere omisă: {e}")

        # Consolidare autonomă la fiecare 10 mesaje (în fundal)
        stats = db.statistici()
        if stats["mesaje_total"] % 10 == 0:
            threading.Thread(
                target=consolidare.ruleaza,
                kwargs={"client": next(_gemini_rotatie), "model": GEMINI_MODEL},
                daemon=True,
            ).start()
            print("[Memorie] Consolidare autonomă pornită în fundal.")


# ---- Pornire ----

# Răcire minimă separată pentru comentarii de personalitate (non-urgente) —
# mult mai mare decât cea de 30s de la monitor_ecran.py (care rămâne
# neschimbată pentru alertele urgente). Fără asta, Jarvis ar putea comenta
# la fiecare schimbare de ecran, devenind enervant în loc de carismatic.
RACIRE_COMENTARIU_SECUNDE = 600  # 10 minute
_ultimul_comentariu_timp = 0.0


_coada_alerte: queue.Queue[tuple[str, str]] = queue.Queue()

# Cât așteptăm, cel mult, ca o conversație activă să se termine înainte
# să vorbim o alertă oricum — plasă de siguranță ca să nu pierdem
# alerte pentru totdeauna dacă flag-ul rămâne blocat din vreun motiv
# neprevăzut (bug, thread mort etc.).
_ASTEPTARE_MAXIMA_ALERTA_SECUNDE = 45.0


def _bucla_alerte():
    """
    Thread dedicat, unic, care rostește alertele de ecran din
    _coada_alerte. Așteaptă ca stare_conversatie.conversatie_activa să
    fie clear() înainte să vorbească — altfel o alertă s-ar putea
    strecura ÎNTRE două propoziții ale aceluiași răspuns al lui Jarvis
    (spune() e protejat per-propoziție, nu pe durata întregii conversații
    — vezi stare_conversatie.py pentru detalii).
    """
    while True:
        mesaj, eticheta = _coada_alerte.get()

        astept_de = time.time()
        while conversatie_activa.is_set():
            if time.time() - astept_de > _ASTEPTARE_MAXIMA_ALERTA_SECUNDE:
                print(f"[Vedere] Aștept de peste {_ASTEPTARE_MAXIMA_ALERTA_SECUNDE:.0f}s conversația să se termine — vorbesc alerta oricum ({eticheta}).")
                break
            time.sleep(0.2)

        try:
            spune(mesaj)
        except RuntimeError as e:
            print(f"[Vedere] Nu am putut rosti alerta ({eticheta}): {e}")


threading.Thread(target=_bucla_alerte, daemon=True).start()


def _rosteste_alerta_pe_thread(mesaj: str, eticheta: str) -> None:
    """Pune alerta la coadă — vorbită de _bucla_alerte, doar când nu ești în mijlocul unei conversații."""
    _coada_alerte.put((mesaj, eticheta))


def _la_schimbare_ecran():
    """
    Callback apelat de monitor_ecran.py când detectează o schimbare
    vizuală locală confirmată. UN singur apel Gemini, care clasifică în
    "urgent" / "comentariu" / "nimic":

        "urgent"      -> te anunță imediat, în scris ȘI cu voce (răcirea
                         de 30s vine deja din monitor_ecran.py, înainte
                         să ajungă aici)
        "comentariu"  -> te anunță DOAR dacă au trecut cel puțin
                         RACIRE_COMENTARIU_SECUNDE de la ultimul
                         comentariu, în scris ȘI cu voce
        "nimic"       -> tace complet
    """
    global _ultimul_comentariu_timp

    rezultat = analizeaza_ecran_complet()
    tip, mesaj = rezultat["tip"], rezultat["mesaj"]

    if not mesaj:
        return

    if tip == "urgent":
        print(f"\n🔔 [Jarvis observă]: {mesaj}\nTu: ", end="", flush=True)
        _rosteste_alerta_pe_thread(mesaj, "urgent")

    elif tip == "comentariu":
        acum = time.time()
        if acum - _ultimul_comentariu_timp >= RACIRE_COMENTARIU_SECUNDE:
            _ultimul_comentariu_timp = acum
            print(f"\n💬 [Jarvis]: {mesaj}\nTu: ", end="", flush=True)
            _rosteste_alerta_pe_thread(mesaj, "comentariu")
        # altfel: comentariu detectat, dar suprimat — încă în perioada de răcire


def _la_eroare_log(sursa: str, linie: str):
    """
    Callback apelat de monitor_log.py când o linie din jurnalul de sistem
    sau kernel se potrivește unui pattern de eroare. Spre deosebire de
    _la_schimbare_ecran, NU mai trece prin Gemini — pattern-ul text e deja
    suficient de precis, deci alerta e instantă și gratuită.
    """
    print(f"\n🔔 [Jarvis observă — {sursa}]: {linie}\nTu: ", end="", flush=True)


def _porneste_ascultare_pasiva_daca_posibil(istoric_ref, rotatie_clienti, model_gemini):
    """
    Al treilea canal senzorial (alături de vedere și log-uri): ascultare
    pasivă ("Hey Jarvis"), pornită în fundal INDIFERENT de modul de
    interacțiune ales — nu doar dacă alegi explicit [2] sau [3].

    Eșuează elegant, cu mesaj clar, dacă dependențele audio (openwakeword,
    silero-vad, faster-whisper, sounddevice) lipsesc sau microfonul nu e
    configurat — restul lui Jarvis (text, vedere, log-uri) rămâne complet
    funcțional, nu blocăm pornirea pentru asta.
    """
    try:
        from src.core.wake_word import porneste_cu_wake_word
    except ImportError as e:
        print(f"[Senzorial] Ascultare pasivă indisponibilă (dependențe audio lipsă): {e}")
        return
    except Exception as e:
        print(f"[Senzorial] Ascultare pasivă indisponibilă: {e}")
        return

    def _bucla():
        try:
            porneste_cu_wake_word(
                istoric=istoric_ref,
                rotatie_clienti=rotatie_clienti,
                model_gemini=model_gemini,
            )
        except Exception as e:
            print(f"[Senzorial] Ascultare pasivă oprită din cauza unei erori: {e}")

    threading.Thread(target=_bucla, daemon=True).start()
    print("[Senzorial] Ascultare pasivă pornită în fundal — spune 'Hey Jarvis' oricând.")


# ---- Pornire ----
porneste_thread_watcher()
porneste_monitorizare_ecran(_la_schimbare_ecran)
porneste_monitorizare_log(_la_eroare_log)
istoric = []
sesiune_id = db.incepe_sesiune()

# Sincronizăm amintirile din SQLite în ChromaDB
rag.sincronizeaza_din_db()

# System prompt îmbogățit cu profil + amintiri
SYSTEM_PROMPT = memorie.construieste_system_prompt(SYSTEM_PROMPT_BAZA)

print(f"\nJarvis este activ. {len(_gemini_chei)} cheie(i) Gemini încărcată(e).")
print(f"[Memorie] System prompt: {len(SYSTEM_PROMPT)} caractere | "
      f"Amintiri: {db.statistici()['amintiri']}")
if not groq_key:
    print("[AVERTISMENT: GROQ_API_KEY lipsește din .env — fallback dezactivat]")

print("\nAlege modul de interacțiune:")
print("  [1] text  — conversație prin terminal")
print("  [2] voce  — wake word 'Hey Jarvis' + conversație vocală")
print("  [3] ambele — text în terminal + wake word în fundal\n")

alegere = input("Mod [1/2/3, default 1]: ").strip()

if alegere in ("2", "voce"):
    from src.core.wake_word import porneste_cu_wake_word
    porneste_cu_wake_word(
        istoric=istoric,
        rotatie_clienti=_gemini_rotatie,
        model_gemini=GEMINI_MODEL,
    )

elif alegere in ("3", "ambele"):
    from src.core.wake_word import porneste_cu_wake_word

    threading.Thread(
        target=porneste_cu_wake_word,
        kwargs={
            "istoric": istoric,
            "rotatie_clienti": _gemini_rotatie,
            "model_gemini": GEMINI_MODEL,
        },
        daemon=True,
    ).start()
    print("[Modul vocal pornit în fundal. Poți folosi și terminalul.]\n")
    bucla_text(istoric, sesiune_id, SYSTEM_PROMPT)

else:
    # Modul implicit [1]: text în terminal, DAR pornim și ascultarea
    # pasivă în fundal (Task 6.8) — canalul senzorial auditiv nu mai
    # depinde de alegerea explicită a modului [2]/[3]. Eșuează elegant
    # dacă lipsesc dependențele audio (vezi funcția de mai sus).
    _porneste_ascultare_pasiva_daca_posibil(istoric, _gemini_rotatie, GEMINI_MODEL)
    bucla_text(istoric, sesiune_id, SYSTEM_PROMPT)