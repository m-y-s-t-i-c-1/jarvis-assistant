"""
Orchestratorul Audio (Task 3.5) + Streaming propoziție-cu-propoziție (Task 6.9)

Leagă toate modulele din Faza 3 într-o buclă continuă:

    1. VAD     — ascultă microfonul, detectează când vorbești
    2. STT     — transcrie ce ai spus (Whisper)
    3. Agent   — trimite textul la Gemini, STREAMEAZĂ răspunsul
    4. TTS     — vorbește FIECARE PROPOZIȚIE imediat ce e gata, nu așteaptă
                 tot răspunsul (Task 6.9 — "vorbește pe măsură ce gândește")
    5. repeat  — revine la pasul 1

Diferență față de varianta blocantă: în loc de
    raspuns_text = agent_loop(...); spune(raspuns_text)
folosim
    raspuns_text = agent_loop_streaming(..., la_propozitie_gata=...)
Jarvis începe să vorbească la prima propoziție completă generată, în loc
să aștepte tot răspunsul — latență percepută mult mai mică în conversație
vocală. Fiecare propoziție trece prin vorbeste_cu_intrerupere (Task 6.10)
ca să poți opri Jarvis vorbind peste el.

Comenzi vocale speciale (recunoscute după transcriere):
    "exit" / "stop" / "ieși" / "oprește-te"  — închide bucla vocală

Rulare standalone (mod voce pur, fără terminal):
    python -m src.core.audio_loop

Rulare din main.py (integrat):
    from src.core.audio_loop import porneste_bucla_audio
    porneste_bucla_audio(istoric)
"""

import os
import re
import time
import threading
import traceback
import itertools
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.core.vad import obtine_detector
from src.core.stt import transcrie
from src.core.tts import spune
from src.core.barge_in import vorbeste_cu_intrerupere
from src.core.agent import agent_loop_streaming, _PATTERN_SFARSIT_PROPOZITIE
from src.core.router import clasifica, este_raspuns_scurt_de_continuare
from src.core.stare_conversatie import conversatie_activa
from src.core.llm_provider import (
    intreaba_nvidia_cod,
    intreaba_nvidia_conversatie,
    istoric_la_mesaje_openai,
)
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


# System prompt separat pentru NVIDIA (conversație/cod, FĂRĂ unelte reale) —
# identic ca scop cu AVERTISMENT_NVIDIA_FARA_UNELTE din main.py, dar
# reformulat pentru contextul vocal (fără liste, fără markdown).
_AVERTISMENT_NVIDIA_VOCAL = """IMPORTANT — citește cu atenție înainte să răspunzi:
În acest mod NU ai acces la nicio unealtă reală. NU poți rula comenzi, NU poți
controla sistemul, ecranul sau orice altceva. Ești strict un model de
conversație/cunoștințe generale, fără capacitate de acțiune.

NU pretinde NICIODATĂ că ai executat o acțiune reală — asta ar fi o
informație falsă transmisă lui Vasea. Dacă cererea pare să necesite o
acțiune reală, spune-i clar că trebuie reformulată ca să ajungă la
asistentul principal, cu unelte."""

# Routare sticky (identic cu main.py, dar stare separată pentru modul
# vocal): un răspuns scurt de continuare moștenește categoria turei
# precedente, ca să nu rupem un flux de confirmare Gemini la jumătate.
_ultima_categorie_vocal: str = "unelte"


def _vorbeste_text_pe_propozitii(text: str, la_propozitie_gata) -> None:
    """
    Împarte un text complet (primit dintr-o singură bucată, de la NVIDIA —
    care nu suportă streaming propoziție-cu-propoziție ca Gemini) în
    propoziții, folosind același pattern ca agent_loop_streaming, și le
    trimite pe rând la callback-ul de vorbire. Rezultatul perceput e
    identic cu streaming-ul Gemini (Jarvis vorbește propoziție cu
    propoziție) — diferența e că textul a fost deja generat integral
    înainte să înceapă vorbirea, nu pe măsură ce vine.
    """
    pozitie = 0
    for potrivire in _PATTERN_SFARSIT_PROPOZITIE.finditer(text):
        capat = potrivire.end()
        propozitie = text[pozitie:capat].strip()
        if propozitie:
            la_propozitie_gata(propozitie)
        pozitie = capat

    rest = text[pozitie:].strip()
    if rest:
        la_propozitie_gata(rest)


SYSTEM_PROMPT = """Tu ești Jarvis, un asistent AI personal inteligent, eficient și loial.

Reguli de comportament:
- Te adresezi întotdeauna utilizatorului cu "Vasea".
- Tonul tău este calm, profesionist, dar cu un strop de umor sec, britanic.
- Răspunzi concis și la obiect, fără să divaghezi inutil.
- Ești proactiv: dacă observi o problemă sau o soluție mai bună, o menționezi.
- Nu ești servil sau exagerat de politicos — ești un consilier de încredere.
- Vorbești în limba română, cu un vocabular elevat dar natural.
- IMPORTANT pentru modul vocal: răspunsurile tale vor fi ROSTITE cu voce tare,
  PROPOZIȚIE CU PROPOZIȚIE, pe măsură ce le generezi. Evită listele cu puncte,
  simbolurile speciale (*, #, →), URL-uri lungi și orice formatare Markdown —
  acestea sună ciudat când sunt citite cu voce tare. Formulează propoziții
  scurte și complete — fiecare propoziție e rostită imediat ce o termini,
  deci evită propoziții foarte lungi cu multe virgule, care ar întârzia
  momentul în care începi să vorbești.
- Nu ai cunoștințe proprii despre ora sau data curentă. Folosește uneltele disponibile.
"""


def porneste_bucla_audio(
    istoric: list | None = None,
    rotatie_clienti=None,
    model: str = "gemini-2.5-flash",
) -> None:
    """
    Pornește bucla audio continuă VAD → STT → Agent (streaming) → TTS.

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
            try:
                spune(raspuns_oprire)
            except Exception as e:
                print(f"[EROARE TTS la mesajul de oprire]: {type(e).__name__}: {e}")
            break

        # ── Pasul 3+4: Router → Agent (Gemini sau NVIDIA) + TTS pe măsură ──
        print(INDICATOR["gandesc"])
        istoric.append(
            types.Content(
                role="user",
                parts=[types.Part(text=text_utilizator)]
            )
        )

        print(INDICATOR["vorbesc"])
        flag_intrerupere = threading.Event()

        def _la_propozitie(propozitie: str):
            """
            Vorbește o propoziție IMEDIAT prin vorbeste_cu_intrerupere
            (Task 6.10 — ascultă concurent pentru barge-in). Dacă Vasea
            vorbește peste Jarvis, setăm flag_intrerupere.
            """
            print(f"Jarvis: {propozitie}")

            a_fost_intrerupt = vorbeste_cu_intrerupere(propozitie)
            if a_fost_intrerupt:
                flag_intrerupere.set()

        # Clasificăm cererea — la fel ca în modul text (main.py), ca să nu
        # trimitem TOTUL pe Gemini și să epuizăm cota gratuită (429
        # RESOURCE_EXHAUSTED) pentru conversații banale care nu au nevoie
        # de nicio unealtă reală.
        global _ultima_categorie_vocal
        if este_raspuns_scurt_de_continuare(text_utilizator):
            categorie = _ultima_categorie_vocal
            print(f"[Router vocal] Răspuns scurt de continuare — păstrez categoria: '{categorie}'")
        else:
            categorie = clasifica(text_utilizator)
            print(f"[Router vocal] Cerere clasificată drept: '{categorie}'")
        _ultima_categorie_vocal = categorie

        eroare_de_afisat = None

        # IMPORTANT: conversatie_activa rămâne setat pe durata ÎNTREGII
        # ture (toate propozițiile răspunsului, nu doar una), ca alertele
        # de ecran (main.py, _la_schimbare_ecran) să nu se strecoare între
        # două propoziții ale aceluiași răspuns — vezi stare_conversatie.py.
        conversatie_activa.set()
        try:
            if categorie == "unelte":
                # Singura cale cu tool-calling real — Gemini, streaming nativ.
                try:
                    client_curent = next(rotatie_clienti)
                    agent_loop_streaming(
                        client_curent, model, SYSTEM_PROMPT, istoric,
                        la_propozitie_gata=_la_propozitie,
                        flag_intrerupere=flag_intrerupere,
                    )
                except Exception as e:
                    eroare_de_afisat = e
            else:
                # "cod" / "conversatie" — NVIDIA întâi, FĂRĂ unelte reale.
                mesaje_openai = istoric_la_mesaje_openai(
                    istoric, _AVERTISMENT_NVIDIA_VOCAL + "\n\n" + SYSTEM_PROMPT
                )
                rezultat = (
                    intreaba_nvidia_cod(mesaje_openai)
                    if categorie == "cod"
                    else intreaba_nvidia_conversatie(mesaje_openai)
                )

                if rezultat:
                    istoric.append(
                        types.Content(role="model", parts=[types.Part(text=rezultat)])
                    )
                    _vorbeste_text_pe_propozitii(rezultat, _la_propozitie)
                else:
                    # NVIDIA indisponibil — cădem pe Gemini, ca plasă de siguranță.
                    print(f"[NVIDIA indisponibil pentru '{categorie}' — trec pe Gemini]")
                    _ultima_categorie_vocal = "unelte"
                    try:
                        client_curent = next(rotatie_clienti)
                        agent_loop_streaming(
                            client_curent, model, SYSTEM_PROMPT, istoric,
                            la_propozitie_gata=_la_propozitie,
                            flag_intrerupere=flag_intrerupere,
                        )
                    except Exception as e:
                        eroare_de_afisat = e

            if eroare_de_afisat is not None:
                e = eroare_de_afisat
                raspuns_text = f"Îmi pare rău, Vasea, am întâmpinat o eroare: {str(e)[:100]}"
                print(f"[EROARE agent]: {e}")
                traceback.print_exc()
                try:
                    spune(raspuns_text)
                except Exception as e_tts:
                    print(f"[EROARE TTS]: {type(e_tts).__name__}: {e_tts}")
        finally:
            # Garantăm eliberarea flag-ului indiferent de rezultat — altfel
            # o eroare neprevăzută ar lăsa alertele de ecran blocate la
            # coadă pentru totdeauna, crezând că o conversație e încă activă.
            conversatie_activa.clear()

        if flag_intrerupere.is_set():
            print("[Barge-in] Revin imediat la ascultare — spune ce ai de spus.")
            # Fără pauză aici — vrem să captăm cât mai repede ce spune Vasea,
            # exact motivul pentru care a întrerupt.
            continue

        # Pauză scurtă între sfârșit redare și reluarea ascultării,
        # ca să nu captăm ecoul propriei voci (mai ales fără căști)
        time.sleep(0.3)


if __name__ == "__main__":
    # Mod standalone — conversație vocală pură, fără interfața text
    porneste_bucla_audio()