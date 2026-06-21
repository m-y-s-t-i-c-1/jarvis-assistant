import os
import time
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

ERORI_FALLBACK = (503, 429, 500, 403)


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


def ruleaza_cu_fallback(istoric: list, system_prompt: str) -> str:
    for _ in range(len(_gemini_clienti)):
        client_curent = next(_gemini_rotatie)
        try:
            return agent_loop(client_curent, GEMINI_MODEL, system_prompt, istoric)
        except Exception as e:
            mesaj_eroare = str(e)
            este_eroare_server = any(str(cod) in mesaj_eroare for cod in ERORI_FALLBACK)
            if este_eroare_server:
                print(f"[Cheie Gemini indisponibilă ({mesaj_eroare[:50]}...) — încerc următoarea]")
                time.sleep(0.5)
                continue
            raise

    print("[Toate cheile Gemini sunt indisponibile — comut pe Groq]")
    return groq_fallback(istoric, system_prompt)


def bucla_text(istoric: list, sesiune_id: str, system_prompt: str):
    """Bucla de conversație prin terminal."""
    print("Scrie 'exit' pentru a încheia conversația.\n")

    while True:
        mesaj_utilizator = input("Tu: ")

        if mesaj_utilizator.lower() in ("exit", "quit", "stop"):
            print("Jarvis: La revedere, Vasea.")
            db.inchide_sesiune(sesiune_id)
            break

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
porneste_thread_watcher()
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
    bucla_text(istoric, sesiune_id, SYSTEM_PROMPT)