"""
Orchestratorul Audio (Task 3.5)

Leagă toate modulele din Faza 3 într-o buclă continuă:

    1. VAD     — ascultă microfonul, detectează când vorbești
    2. STT     — transcrie ce ai spus (Whisper)
    3. Agent   — trimite textul la Gemini, primește răspuns
    4. TTS     — redă răspunsul prin boxe (Piper)
    5. repeat  — revine la pasul 1

Comenzi vocale speciale (recunoscute după transcriere):
    "exit" / "stop" / "ieși" / "oprește-te"  — închide bucla vocală

Conectare cu main.py:
    Istoricul conversației e același obiect `istoric` folosit și de
    interfața text din main.py — poți comuta între voce și text oricând,
    conversația rămâne continuă.

Rulare standalone (mod voce pur, fără terminal):
    python -m src.core.audio_loop

Rulare din main.py (integrat):
    from src.core.audio_loop import porneste_bucla_audio
    porneste_bucla_audio(istoric)
"""

import os
import time
import itertools
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.core.vad import obtine_detector
from src.core.stt import transcrie
from src.core.tts import spune, opreste
from src.core.agent import agent_loop
from src import tools  # noqa: F401 — înregistrează toate uneltele

load_dotenv()

# ---- Comenzi care închid bucla vocală ----
COMENZI_OPRIRE = {"exit", "stop", "ieși", "oprește-te", "opreste-te", "quit"}

# ---- Fraze scurte afișate ca indicatori vizuali în terminal ----
INDICATOR = {
    "ascult":    "🎙  Ascult...",
    "transcriere": "📝 Transcriere...",
    "gandesc":   "🤔 Gândesc...",
    "vorbesc":   "🔊 Vorbesc...",
}


def _obtine_client_gemini():
    """
    Construiește clientul Gemini din cheile disponibile în .env.
    Suportă rotație (GEMINI_API_KEY, GEMINI_API_KEY_2 ... _5),
    identic cu logica din main.py.
    """
    chei = [
        os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
        for i in range(5)
    ]
    chei = [k for k in chei if k]

    if not chei:
        raise ValueError(
            "Nu am găsit nicio GEMINI_API_KEY în .env. "
            "Adaugă cel puțin o cheie pentru modul vocal."
        )

    clienti = [genai.Client(api_key=k) for k in chei]
    rotatie = itertools.cycle(clienti)
    return rotatie


SYSTEM_PROMPT = """Tu ești Jarvis, un asistent AI personal inteligent, eficient și loial.

Reguli de comportament:
- Te adresezi întotdeauna utilizatorului cu "Vasea".
- Tonul tău este calm, profesionist, dar cu un strop de umor sec, britanic.
- Răspunzi concis și la obiect, fără să divaghezi inutil.
- Ești proactiv: dacă observi o problemă sau o soluție mai bună, o menționezi.
- Nu ești servil sau exagerat de politicos — ești un consilier de încredere.
- Vorbești în limba română, cu un vocabular elevat dar natural.
- IMPORTANT pentru modul vocal: răspunsurile tale vor fi ROSTITE cu voce tare.
  Evită listele cu puncte, simbolurile speciale (*, #, →), URL-uri lungi și
  orice formatare Markdown — acestea sună ciudat când sunt citite cu voce tare.
  Formulează răspunsuri ca propoziții naturale, de parcă vorbești cu cineva.
- Nu ai cunoștințe proprii despre ora sau data curentă. Folosește uneltele disponibile.
"""


def porneste_bucla_audio(
    istoric: list | None = None,
    rotatie_clienti=None,
    model: str = "gemini-3.6-flash",
) -> None:
    """
    Pornește bucla audio continuă VAD → STT → Agent → TTS.

    Parametri:
        istoric:          lista de mesaje existentă (din main.py), sau None
                          pentru o conversație nouă independentă.
        rotatie_clienti:  iteratorul de clienți Gemini din main.py (opțional,
                          dacă None construim unul nou din .env).
        model:            modelul Gemini de folosit.
    """
    if istoric is None:
        istoric = []

    if rotatie_clienti is None:
        rotatie_clienti = _obtine_client_gemini()

    detector_vad = obtine_detector()

    print("\n" + "═" * 50)
    print("  Modul vocal activ. Vorbește cu Jarvis.")
    print("  Spune 'exit' sau 'stop' pentru a ieși.")
    print("═" * 50 + "\n")

    while True:
        # ── Pasul 1: VAD — așteptăm să vorbești ──────────────────────────
        print(INDICATOR["ascult"])
        audio = detector_vad.asculta_pana_la_pauza()

        if audio is None:
            # Timeout de siguranță din VAD — nu s-a detectat nicio vorbire
            continue

        # ── Pasul 2: STT — transcriem ce ai spus ─────────────────────────
        print(INDICATOR["transcriere"])
        text_utilizator = transcrie(audio)

        if not text_utilizator.strip():
            print("[Transcriere goală — poate zgomot de fundal, reiau ascultarea]")
            continue

        print(f"Tu (vocal): {text_utilizator}")

        # Verificăm comenzile de oprire (eliminăm punctuația, comparăm lowercase)
        text_curat = text_utilizator.strip().lower().rstrip("!?.,")
        if text_curat in COMENZI_OPRIRE or any(cmd in text_curat.split() for cmd in COMENZI_OPRIRE):
            raspuns_oprire = "La revedere, Vasea. Modul vocal dezactivat."
            print(f"Jarvis: {raspuns_oprire}")
            spune(raspuns_oprire)
            break

        # ── Pasul 3: Agent — trimitem la Gemini ──────────────────────────
        print(INDICATOR["gandesc"])
        istoric.append(
            types.Content(
                role="user",
                parts=[types.Part(text=text_utilizator)]
            )
        )

        try:
            client_curent = next(rotatie_clienti)
            raspuns_text = agent_loop(client_curent, model, SYSTEM_PROMPT, istoric)
        except Exception as e:
            raspuns_text = f"Îmi pare rău, Vasea, am întâmpinat o eroare: {str(e)[:100]}"
            print(f"[EROARE agent]: {e}")

        print(f"Jarvis: {raspuns_text}")

        # ── Pasul 4: TTS — rostim răspunsul ──────────────────────────────
        print(INDICATOR["vorbesc"])
        try:
            spune(raspuns_text)
        except RuntimeError as e:
            print(f"[EROARE TTS]: {e}")

        # Pauză scurtă între sfârșit redare și reluarea ascultării,
        # ca să nu captăm ecoul propriei voci (mai ales fără căști)
        time.sleep(0.3)


if __name__ == "__main__":
    # Mod standalone — conversație vocală pură, fără interfața text
    porneste_bucla_audio()