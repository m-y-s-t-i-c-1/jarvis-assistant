import os
from dotenv import load_dotenv
from google import genai

from src import tools  # noqa: F401  -- importul ăsta declanșează înregistrarea uneltelor
from src.core.agent import agent_loop
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Nu am găsit GEMINI_API_KEY în .env. Verifică fișierul .env.")

client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"  # gratuit (free tier), mai fiabil la tool-calling decât flash-lite

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

# ---- Bucla principală de conversație ----
istoric = []

print("Jarvis este activ. Scrie 'exit' pentru a încheia conversația.\n")

while True:
    mesaj_utilizator = input("Tu: ")

    if mesaj_utilizator.lower() in ("exit", "quit", "stop"):
        print("Jarvis: La revedere, Vasea.")
        break

    istoric.append(
        types.Content(role="user", parts=[types.Part(text=mesaj_utilizator)])
    )

    raspuns_text = agent_loop(client, MODEL, SYSTEM_PROMPT, istoric)
    print("Jarvis:", raspuns_text)