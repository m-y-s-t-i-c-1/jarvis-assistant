import os
import re
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


class _RotatieSigura:
    """
    Iterator ciclic thread-safe peste clienții Gemini. Obiectul acesta e
    pasat și modulelor vocale (wake_word / audio_loop), care îl apelează
    cu next() din alte thread-uri — itertools.cycle nu e garantat sigur
    la acces concurent.
    """

    def __init__(self, elemente):
        self._ciclu = itertools.cycle(elemente)
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._ciclu)


_gemini_rotatie = _RotatieSigura(_gemini_clienti)


def urmatorul_client():
    """Returnează următorul client Gemini din rotație, thread-safe."""
    return next(_gemini_rotatie)


groq_key    = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None

# FIX: restul proiectului (vedere, memorie, consolidare, wake_word, server,
# context_manager, diagnostic) folosește "gemini-3.6-flash"; main.py era
# singurul pe "gemini-2.5-flash". Acum e o singură valoare, suprascriibilă
# din .env (GEMINI_MODEL=...).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
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
# FIX: adăugate 502/504 (gateway) — tot erori temporare, nu fatale.
ERORI_FALLBACK = (503, 429, 500, 502, 504, 403)

# Marcaje textuale folosite de Google în erorile de tip temporar, pentru
# cazul în care excepția nu expune un cod numeric.
_MARCAJE_TEMPORARE = ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "OVERLOADED", "DEADLINE_EXCEEDED")

# ---- Excepții de rețea/conexiune care NU au cod HTTP în mesaj, dar tot
# trebuie tratate ca eșec temporar al cheii curente, nu ca eroare fatală.
try:
    import httpx
    EXCEPTII_RETEA = (
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.TimeoutException,  # acoperă ReadTimeout, ConnectTimeout, WriteTimeout, PoolTimeout
    )
except ImportError:
    EXCEPTII_RETEA = ()


def _este_eroare_temporara(e: Exception) -> bool:
    """
    Decide dacă o excepție trebuie tratată ca eșec temporar al cheii curente
    (încercăm cheia următoare) sau ca eroare fatală (o propagăm).

    FIX: înainte se căuta substring-ul "500"/"503"/... în tot mesajul, deci
    orice eroare care conținea ex. "max_tokens=5000" sau un ID cu "429"
    era tratată greșit ca temporară (și invers, o eroare reală cu cod în
    atribut dar fără cod în text nu era recunoscută). Acum verificăm întâi
    atributul de cod al excepției (google.genai: .code / .status_code),
    apoi cuvinte întregi în mesaj.
    """
    if EXCEPTII_RETEA and isinstance(e, EXCEPTII_RETEA):
        return True

    for atribut in ("code", "status_code"):
        cod = getattr(e, atribut, None)
        if isinstance(cod, int) and cod in ERORI_FALLBACK:
            return True

    mesaj_eroare = str(e)
    if any(re.search(rf"(?<!\d){cod}(?!\d)", mesaj_eroare) for cod in ERORI_FALLBACK):
        return True

    mesaj_upper = mesaj_eroare.upper()
    return any(marcaj in mesaj_upper for marcaj in _MARCAJE_TEMPORARE)


def groq_fallback(istoric: list, system_prompt: str) -> str:
    if not groq_client:
        return "Gemini este momentan indisponibil și nu am o cheie Groq configurată ca backup."

    mesaje = [{"role": "system", "content": system_prompt}]
    for continut in istoric:
        if not hasattr(continut, "parts") or not continut.parts:
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
    # FIX: content poate fi None (ex. răspuns blocat) -> returnam sir gol
    # in loc de None, ca sa nu crape mai departe la print/DB/RAG.
    return raspuns.choices[0].message.content or ""


def _ruleaza_gemini_apoi_restul(istoric: list, system_prompt: str) -> str:
    """
    Calea "unelte": Gemini (toate cheile) -> Groq -> cascada finală
    (OpenRouter -> NVIDIA generic -> Bytez). Folosită pentru orice cerere
    clasificată drept "unelte" de router.py, și ca ultimă plasă și pentru
    celelalte categorii, dacă NVIDIA specializat + Groq eșuează.
    """
    for _ in range(len(_gemini_clienti)):
        client_curent = urmatorul_client()
        try:
            return agent_loop(client_curent, GEMINI_MODEL, system_prompt, istoric)
        except Exception as e:
            if _este_eroare_temporara(e):
                print(f"[Cheie Gemini indisponibilă ({str(e)[:80]}...) — încerc următoarea]")
                time.sleep(0.5)
                continue
            raise

    # FIX: fără cheie Groq, groq_fallback() întorcea un mesaj text "valid",
    # deci cascada externă (OpenRouter/NVIDIA/Bytez) nu mai ajungea
    # niciodată să fie încercată. Acum Groq e sărit dacă nu e configurat.
    if groq_client:
        print("[Toate cheile Gemini sunt indisponibile — comut pe Groq]")
        try:
            rezultat_groq = groq_fallback(istoric, system_prompt)
            if rezultat_groq:
                return rezultat_groq
        except Exception as e:
            print(f"[Groq indisponibil ({str(e)[:80]}) — încerc cascada externă]")
    else:
        print("[Toate cheile Gemini sunt indisponibile și Groq nu e configurat — încerc cascada externă]")

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
# zero, moștenește categoria turei precedente. (Task 6.7)
_ultima_categorie: str = "unelte"  # implicit "unelte" — cea mai sigură presupunere inițială

# System prompt separat pentru NVIDIA (fallback text, FĂRĂ unelte reale).
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
    # FIX: parts poate fi None (Content fără parts) -> iterarea ar crăpa.
    if not getattr(ultimul, "parts", None):
        return ""
    return " ".join(
        p.text for p in ultimul.parts if hasattr(p, "text") and p.text
    )


def _are_text(continut) -> bool:
    return any(getattr(p, "text", None) for p in (getattr(continut, "parts", None) or []))


def _are_function_response(continut) -> bool:
    return any(getattr(p, "function_response", None) for p in (getattr(continut, "parts", None) or []))


def _curata_inceput_istoric(istoric: list) -> None:
    """
    FIX: fereastra glisantă (context_manager.proceseaza) taie istoricul la o
    poziție fixă și poate lăsa la început un răspuns "model" sau un
    function_response orfan (fără function_call-ul dinaintea lui). Gemini
    respinge un astfel de istoric cu eroare 400. Eliminăm din față tot ce
    nu e un mesaj de tip user cu text. Ultimul element (mesajul curent al
    utilizatorului) e mereu text, deci bucla se oprește cel târziu la el.
    """
    while len(istoric) > 1:
        primul = istoric[0]
        if primul.role == "user" and _are_text(primul) and not _are_function_response(primul):
            break
        istoric.pop(0)


def _asigura_raspuns_in_istoric(istoric: list, text: str) -> None:
    """
    FIX: agent_loop adaugă singur răspunsul modelului în istoric, dar căile
    NVIDIA / Groq / cascada externă NU. Rezultat: istoricul rămânea cu
    mesaje "user" consecutive, iar Gemini pierdea răspunsurile date de
    ceilalți provideri. Dacă ultimul element nu e deja un răspuns "model",
    îl adăugăm.
    """
    if not text:
        return
    if istoric and getattr(istoric[-1], "role", None) == "model":
        return
    istoric.append(types.Content(role="model", parts=[types.Part(text=text)]))


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
    "ok" etc.) moștenește categoria turei precedente.
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
        text = _ruleaza_gemini_apoi_restul(istoric, system_prompt)
        _asigura_raspuns_in_istoric(istoric, text)
        return text

    system_prompt_nvidia = AVERTISMENT_NVIDIA_FARA_UNELTE + "\n\n" + system_prompt
    mesaje_openai = istoric_la_mesaje_openai(istoric, system_prompt_nvidia)

    if categorie == "cod":
        rezultat = intreaba_nvidia_cod(mesaje_openai)
    else:
        rezultat = intreaba_nvidia_conversatie(mesaje_openai)

    if rezultat:
        _asigura_raspuns_in_istoric(istoric, rezultat)
        return rezultat

    print(f"[NVIDIA indisponibil pentru categoria '{categorie}' — trec pe Gemini]")
    _ultima_categorie = "unelte"
    text = _ruleaza_gemini_apoi_restul(istoric, system_prompt)
    _asigura_raspuns_in_istoric(istoric, text)
    return text


# Previne pornirea a două consolidări simultane (ex. la mesaje rapide).
_lock_consolidare = threading.Lock()


def _consolidare_in_fundal(client):
    if not _lock_consolidare.acquire(blocking=False):
        return  # deja rulează una
    try:
        consolidare.ruleaza(client=client, model=GEMINI_MODEL)
    except Exception as e:
        print(f"[Memorie] Consolidare eșuată: {e}")
    finally:
        _lock_consolidare.release()


def bucla_text(istoric: list, sesiune_id: str, system_prompt: str):
    """Bucla de conversație prin terminal."""
    print("Scrie 'exit' pentru a încheia conversația.\n")

    try:
        while True:
            # FIX: Ctrl+C / Ctrl+D (EOF) închideau programul cu traceback și
            # sesiunea rămânea nemarcată ca închisă în DB.
            try:
                mesaj_utilizator = input("Tu: ")
            except (EOFError, KeyboardInterrupt):
                print("\nJarvis: La revedere, Vasea.")
                break

            if mesaj_utilizator.strip().lower() in ("exit", "quit", "stop"):
                print("Jarvis: La revedere, Vasea.")
                break

            if not mesaj_utilizator.strip():
                continue

            lungime_inainte = len(istoric)
            istoric.append(
                types.Content(role="user", parts=[types.Part(text=mesaj_utilizator)])
            )

            try:
                # Sliding window
                proceseaza_context(
                    istoric, sesiune_id,
                    client=urmatorul_client(), model=GEMINI_MODEL,
                )
                _curata_inceput_istoric(istoric)

                # Răspuns
                raspuns_text = ruleaza_cu_fallback(istoric, system_prompt)
            except KeyboardInterrupt:
                print("\n[Cererea a fost întreruptă.]")
                del istoric[lungime_inainte:]
                continue
            except Exception as e:
                # FIX: o eroare fatală (ex. 400/401) nu mai omoară bucla și
                # nu lasă mesajul utilizatorului "agățat" în istoric.
                print(f"Jarvis: Vasea, am întâmpinat o eroare: {str(e)[:200]}")
                del istoric[lungime_inainte:]
                continue

            if not raspuns_text:
                raspuns_text = "Vasea, nu am primit niciun răspuns de la provideri. Reîncearcă."

            print("Jarvis:", raspuns_text)

            # Salvare în DB + RAG — o eroare aici nu trebuie să oprească chatul
            try:
                db.salveaza_mesaj(sesiune_id, "user", mesaj_utilizator)
                db.salveaza_mesaj(sesiune_id, "assistant", raspuns_text)
                rag.indexeaza_mesaj("user", mesaj_utilizator, sesiune_id)
                rag.indexeaza_mesaj("assistant", raspuns_text, sesiune_id)
            except Exception as e:
                print(f"[Memorie] Salvare/indexare eșuată: {e}")

            # Extragere amintiri episodice (în fundal)
            try:
                threading.Thread(
                    target=memorie.extrage_si_salveaza,
                    args=(mesaj_utilizator, raspuns_text, sesiune_id,
                          urmatorul_client(), GEMINI_MODEL),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"[Memorie] Extragere omisă: {e}")

            # Consolidare autonomă la fiecare 10 mesaje (în fundal)
            try:
                stats = db.statistici()
                if stats["mesaje_total"] > 0 and stats["mesaje_total"] % 10 == 0:
                    threading.Thread(
                        target=_consolidare_in_fundal,
                        args=(urmatorul_client(),),
                        daemon=True,
                    ).start()
                    print("[Memorie] Consolidare autonomă pornită în fundal.")
            except Exception as e:
                print(f"[Memorie] Verificare consolidare eșuată: {e}")
    finally:
        # FIX: sesiunea se închide mereu, nu doar la 'exit' tastat.
        try:
            db.inchide_sesiune(sesiune_id)
        except Exception as e:
            print(f"[DB] Nu am putut închide sesiunea: {e}")


# ---- Alerte vocale / vizuale ----

# Răcire minimă separată pentru comentarii de personalitate (non-urgente).
RACIRE_COMENTARIU_SECUNDE = 600  # 10 minute
_ultimul_comentariu_timp = 0.0

_coada_alerte: queue.Queue = queue.Queue()

# Cât așteptăm, cel mult, ca o conversație activă să se termine înainte
# să vorbim o alertă oricum — plasă de siguranță.
_ASTEPTARE_MAXIMA_ALERTA_SECUNDE = 45.0


def _bucla_alerte():
    """
    Thread dedicat, unic, care rostește alertele de ecran din
    _coada_alerte. Așteaptă ca stare_conversatie.conversatie_activa să
    fie clear() înainte să vorbească.
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
        except Exception as e:
            # FIX: orice altă excepție din TTS omora thread-ul de alerte,
            # iar alertele ulterioare nu mai erau rostite niciodată.
            print(f"[Vedere] Eroare neașteptată la rostirea alertei ({eticheta}): {e}")


def _rosteste_alerta_pe_thread(mesaj: str, eticheta: str) -> None:
    """Pune alerta la coadă — vorbită de _bucla_alerte, doar când nu ești în mijlocul unei conversații."""
    _coada_alerte.put((mesaj, eticheta))


def _la_schimbare_ecran():
    """
    Callback apelat de monitor_ecran.py când detectează o schimbare
    vizuală locală confirmată. Clasifică în "urgent" / "comentariu" / "nimic".
    """
    global _ultimul_comentariu_timp

    try:
        rezultat = analizeaza_ecran_complet()
    except Exception as e:
        print(f"[Vedere] Analiza ecranului a eșuat: {e}")
        return

    # FIX: rezultatul poate fi None / fără chei -> KeyError în thread-ul monitorului.
    if not isinstance(rezultat, dict):
        return
    tip, mesaj = rezultat.get("tip"), rezultat.get("mesaj")

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
    sau kernel se potrivește unui pattern de eroare. Alertă instantă, fără Gemini.
    """
    print(f"\n🔔 [Jarvis observă — {sursa}]: {linie}\nTu: ", end="", flush=True)


def _porneste_ascultare_pasiva_daca_posibil(istoric_ref, rotatie_clienti, model_gemini):
    """
    Ascultare pasivă ("Hey Jarvis"), pornită în fundal INDIFERENT de modul de
    interacțiune ales. Eșuează elegant dacă dependențele audio lipsesc.
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

def main():
    # FIX: tot ce pornea la import (thread-uri, monitoare, input()) e acum
    # în main(), deci `import main` nu mai declanșează pornirea aplicației.
    threading.Thread(target=_bucla_alerte, daemon=True).start()

    porneste_thread_watcher()
    porneste_monitorizare_ecran(_la_schimbare_ecran)
    porneste_monitorizare_log(_la_eroare_log)
    istoric = []
    sesiune_id = db.incepe_sesiune()

    # Sincronizăm amintirile din SQLite în ChromaDB
    rag.sincronizeaza_din_db()

    # System prompt îmbogățit cu profil + amintiri
    system_prompt = memorie.construieste_system_prompt(SYSTEM_PROMPT_BAZA)

    print(f"\nJarvis este activ. {len(_gemini_chei)} cheie(i) Gemini încărcată(e).")
    print(f"[Memorie] System prompt: {len(system_prompt)} caractere | "
          f"Amintiri: {db.statistici()['amintiri']}")
    if not groq_key:
        print("[AVERTISMENT: GROQ_API_KEY lipsește din .env — fallback dezactivat]")

    print("\nAlege modul de interacțiune:")
    print("  [1] text  — conversație prin terminal")
    print("  [2] voce  — wake word 'Hey Jarvis' + conversație vocală")
    print("  [3] ambele — text în terminal + wake word în fundal\n")

    try:
        alegere = input("Mod [1/2/3, default 1]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        db.inchide_sesiune(sesiune_id)
        return

    if alegere in ("2", "voce"):
        from src.core.wake_word import porneste_cu_wake_word
        try:
            porneste_cu_wake_word(
                istoric=istoric,
                rotatie_clienti=_gemini_rotatie,
                model_gemini=GEMINI_MODEL,
            )
        except KeyboardInterrupt:
            print("\nJarvis: La revedere, Vasea.")
        finally:
            # FIX: în modul vocal sesiunea nu se închidea niciodată.
            db.inchide_sesiune(sesiune_id)

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
        bucla_text(istoric, sesiune_id, system_prompt)

    else:
        # Modul implicit [1]: text în terminal + ascultare pasivă în fundal (Task 6.8).
        _porneste_ascultare_pasiva_daca_posibil(istoric, _gemini_rotatie, GEMINI_MODEL)
        bucla_text(istoric, sesiune_id, system_prompt)


if __name__ == "__main__":
    main()