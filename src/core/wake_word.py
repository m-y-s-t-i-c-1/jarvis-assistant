"""
Wake Word — Ascultare Pasivă (Task 3.6)

Ascultă continuu microfonul la consum minim de resurse, detectează
cuvântul de activare "Hey Jarvis", apoi predă controlul buclei audio
complete (VAD → STT → Agent → TTS) din audio_loop.py.

Flux complet:
    wake_word loop → detectează "Hey Jarvis"
        → "Ascult, Vasea."
        → porneste_bucla_audio() (conversație completă)
        → revine la ascultarea pasivă

Rulare:
    python -m src.core.wake_word           # mod normal
    python -m src.core.wake_word --test    # doar testează detecția
    python -m src.core.wake_word --test --prag 0.4
"""

import sys
import os
import time
import argparse
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from src.core.audio_io import SAMPLE_RATE, DEVICE_IMPLICIT, safe_input_stream

# ── Configurare ───────────────────────────────────────────────────────────────

CALE_MODEL = (
    "/home/vaseoc/Downloads/jarvis-assistant-main/venv/lib/python3.14"
    "/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
)
MODEL_CHEIE  = "hey_jarvis_v0.1"
PRAG_DETECTIE = 0.35
MARIME_BLOC   = 1280       # 80ms la 16kHz — recomandat de OpenWakeWord

# Pauză după conversație, înainte de redeschiderea wake-word InputStream.
PAUZA_DUPA_ACTIVARE = 1.0


def _cale_model_wake() -> str:
    """Cale portabilă către modelul hey_jarvis (nu hardcodată pe un singur PC)."""
    env = os.getenv("WAKE_WORD_MODEL", "").strip()
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(CALE_MODEL):
        return CALE_MODEL
    try:
        import openwakeword
        baza = os.path.join(
            os.path.dirname(openwakeword.__file__),
            "resources",
            "models",
            "hey_jarvis_v0.1.onnx",
        )
        if os.path.isfile(baza):
            return baza
    except Exception:
        pass
    return CALE_MODEL

# ── Model (lazy, o singură dată) ─────────────────────────────────────────────

_model_oww: Model | None = None


def _obtine_model() -> Model:
    global _model_oww
    if _model_oww is None:
        cale = _cale_model_wake()
        print(f"[Wake Word] Încărcare model '{MODEL_CHEIE}'...")
        _model_oww = Model(wakeword_model_paths=[cale])
        print("[Wake Word] Model încărcat. Ascult după 'Hey Jarvis'...")
    return _model_oww


# ── Ascultare pasivă ─────────────────────────────────────────────────────────

def asteapta_wake_word() -> None:
    """Blochează până detectează 'Hey Jarvis', apoi returnează."""
    model = _obtine_model()

    with safe_input_stream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=MARIME_BLOC,
        device=DEVICE_IMPLICIT,
    ) as stream:
        while True:
            bloc, _ = stream.read(MARIME_BLOC)
            predictii = model.predict(bloc.flatten())
            scor = float(predictii.get(MODEL_CHEIE, 0.0))  # float() — sigur comparabil

            if scor >= PRAG_DETECTIE:
                print(f"[Wake Word] Detectat! (scor: {scor:.3f})")
                return


# ── Bucla principală ─────────────────────────────────────────────────────────

def porneste_cu_wake_word(
    istoric: list | None = None,
    rotatie_clienti=None,
    model_gemini: str = "gemini-3.6-flash",
) -> None:
    """
    Buclă permanentă: ascultă pasiv → detectează → conversație → repeat.

    Parametri:
        istoric:          lista de mesaje partajată cu main.py (opțional)
        rotatie_clienti:  iteratorul de clienți Gemini din main.py (opțional)
        model_gemini:     modelul Gemini de folosit
    """
    from src.core.audio_loop import porneste_bucla_audio
    from src.core.tts import spune
    from src.core.vad import obtine_detector

    if istoric is None:
        istoric = []

    # Preîncărcăm Silero-VAD ACUM, nu imediat după TTS. Încărcarea Torch
    # + deschiderea InputStream-ului VAD în aceeași clipă cu închiderea
    # OutputStream-ului TTS a produs pe acest sistem:
    #   free(): corrupted unsorted chunks  → abort (core dumped)
    # (corupție de heap în PortAudio/PipeWire + alocatori nativi).
    obtine_detector()

    print("\n" + "═" * 50)
    print("  Jarvis în așteptare. Spune 'Hey Jarvis'.")
    print("  Ctrl+C pentru a opri.")
    print("═" * 50 + "\n")

    while True:
        try:
            print("💤 Aștept wake word...")
            asteapta_wake_word()

            print("✅ Wake word detectat!")

            # TTS e acum pe player extern (proces separat) — nu mai deschide
            # PortAudio OutputStream, deci nu mai e nevoie de pauze lungi.
            try:
                _obtine_model().reset()
            except Exception:
                pass

            try:
                spune("Ascult, Vasea.")
            except RuntimeError as e:
                print(f"[TTS eroare]: {e}")

            porneste_bucla_audio(
                istoric=istoric,
                rotatie_clienti=rotatie_clienti,
                model=model_gemini,
            )

            print("\n[Revin la ascultarea pasivă...]\n")
            time.sleep(PAUZA_DUPA_ACTIVARE)

        except KeyboardInterrupt:
            print("\n[Wake Word] Oprit.")
            break
        except Exception as e:
            print(f"[Wake Word] Eroare: {e}. Reiau în 2s...")
            time.sleep(2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Wake Word")
    parser.add_argument("--test", action="store_true",
                        help="Mod test: afișează scoruri, nu pornește agentul")
    parser.add_argument("--prag", type=float, default=PRAG_DETECTIE,
                        help=f"Pragul de detecție (default: {PRAG_DETECTIE})")
    args = parser.parse_args()

    if args.test:
        print(f"[Test] Model: {MODEL_CHEIE}, prag: {args.prag}")
        print("[Test] Spune 'Hey Jarvis'. Ctrl+C pentru stop.\n")
        model = _obtine_model()

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=MARIME_BLOC,
                device=DEVICE_IMPLICIT,
            ) as stream:
                while True:
                    bloc, _ = stream.read(MARIME_BLOC)
                    predictii = model.predict(bloc.flatten())
                    scor = float(predictii.get(MODEL_CHEIE, 0.0))
                    if scor > 0.05:
                        detectat = "✅ DETECTAT!" if scor >= args.prag else ""
                        print(f"Scor: {scor:.4f}  {detectat}")
        except KeyboardInterrupt:
            print("\n[Test oprit]")
        sys.exit(0)

    porneste_cu_wake_word()