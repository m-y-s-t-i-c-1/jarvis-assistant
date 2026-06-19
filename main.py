import os
import time
import itertools
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

from src import tools  # noqa: F401  -- declanșează înregistrarea uneltelor
from src.core.agent import agent_loop

load_dotenv()

# ---- Chei API Gemini (rotation între mai multe chei) ----
# Pune în .env: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 etc.
# Funcționează și cu o singură cheie, rotation e transparent.
_gemini_chei = [
    os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
    for i in range(5)  # caută GEMINI_API_KEY, _2, _3, _4, _5
]
_gemini_chei = [k for k in _gemini_chei if k]  # filtrăm None-urile

if not _gemini_chei:
    raise ValueError("Nu am găsit nicio GEMINI_API_KEY în .env. Verifică fișierul.")

# Construim un client Gemini pentru fiecare cheie
_gemini_clienti = [genai.Client(api_key=k) for k in _gemini_chei]

# Iteratorul circular — fiecare cerere ia următorul client din listă
_gemini_rotatie = itertools.cycle(_gemini_clienti)

# ---- Cheie Groq (fallback când toate cheile Gemini sunt indisponibile) ----
groq_key    = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Tu ești Jarvis, un asistent AI personal inteligent, eficient și loial.

Reguli de comportament:
- Te adresezi întotdeauna utilizatorului cu "Vasea".
- Tonul tău este calm, profesionist, dar cu un strop de umor sec, britanic.
- Răspunzi concis și la obiect, fără să divaghezi inutil.
- Ești proactiv: dacă observi o problemă sau o soluție mai bună, o menționezi.
- Nu ești servil sau exagerat de politicos - ești un consilier de încredere, nu un servitor.
- Vorbești în limba română, cu un vocabular elevat dar natural.
- Nu ai cunoștințe proprii despre ora, data sau alte fapte din lumea reală
  care se schimbă în timp. Când ai la dispoziție o unealtă (function call)
  care poate răspunde la o întrebare, FOLOSEȘTE-O întotdeauna, în loc să
  ghicești sau să estimezi un răspuns plauzibil. Un răspuns ghicit greșit
  este mai dăunător decât a admite că nu știi.
"""

# Codurile de eroare care justifică trecerea la cheia următoare / Groq
ERORI_FALLBACK = (503, 429, 500, 403)


def groq_fallback(istoric: list) -> str:
    """
    Fallback pe Groq când toate cheile Gemini sunt indisponibile.
    ATENȚIE: fără function calling în modul fallback, doar răspuns text.
    """
    if not groq_client:
        return "Gemini este momentan indisponibil și nu am o cheie Groq configurată ca backup."

    mesaje = [{"role": "system", "content": SYSTEM_PROMPT}]
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


def ruleaza_cu_fallback(istoric: list) -> str:
    """
    Încearcă fiecare cheie Gemini în ordine (round-robin).
    Dacă toate dau 503/429/500 → fallback pe Groq.
    Dacă e altă eroare (cheie invalidă, bug) → crapă normal.
    """
    # Încearcă toate cheile Gemini disponibile înainte să cadă pe Groq
    for _ in range(len(_gemini_clienti)):
        client_curent = next(_gemini_rotatie)
        try:
            return agent_loop(client_curent, GEMINI_MODEL, SYSTEM_PROMPT, istoric)

        except Exception as e:
            mesaj_eroare = str(e)
            este_eroare_server = any(str(cod) in mesaj_eroare for cod in ERORI_FALLBACK)

            if este_eroare_server:
                print(f"[Cheie Gemini indisponibilă ({mesaj_eroare[:50]}...) — încerc următoarea]")
                time.sleep(0.5)
                continue  # încearcă cu următoarea cheie

            raise  # eroare reală (cheie invalidă, bug de cod) — nu continuăm

    # Toate cheile Gemini au eșuat → Groq
    print("[Toate cheile Gemini sunt indisponibile — comut pe Groq]")
    return groq_fallback(istoric)


# ---- Bucla principală de conversație ----
istoric = []

print(f"Jarvis este activ. {len(_gemini_chei)} cheie(i) Gemini încărcată(e).")
if not groq_key:
    print("[AVERTISMENT: GROQ_API_KEY lipsește din .env — fallback dezactivat]")
print("Scrie 'exit' pentru a încheia conversația.\n")

while True:
    mesaj_utilizator = input("Tu: ")

    if mesaj_utilizator.lower() in ("exit", "quit", "stop"):
        print("Jarvis: La revedere, Vasea.")
        break

    istoric.append(
        types.Content(role="user", parts=[types.Part(text=mesaj_utilizator)])
    )

    raspuns_text = ruleaza_cu_fallback(istoric)
    print("Jarvis:", raspuns_text)