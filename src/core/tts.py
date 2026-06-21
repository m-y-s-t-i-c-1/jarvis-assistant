"""
Sintetizarea și Redarea Vocii — Text-to-Speech (Task 3.4)

Folosește Piper TTS (local, offline) pentru a transforma textul răspunsurilor
lui Jarvis în audio redat prin boxe. Piper scrie WAV într-un fișier temporar
(are nevoie de seek() pentru a scrie headerul), pe care îl citim și redăm
prin sounddevice, apoi îl ștergem.

Arhitectură:
    spune(text) — funcția principală, apelată din orchestratorul audio (Task 3.5)
    _reda_wav_bytes(bytes) — parsează headerul WAV și redă audio-ul prin sounddevice
    opreste() — întrerupe redarea curentă (pentru barge-in, Task 3.7)

Configurare în .env (opțional):
    PIPER_BINARY   — calea către executabilul piper (default: piper)
    PIPER_MODEL    — calea către fișierul .onnx (default: voci_piper/ro_RO-mihai-medium.onnx)
"""

import io
import os
import wave
import tempfile
import threading
import subprocess

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from src.core.audio_io import DEVICE_IMPLICIT

load_dotenv()

# ---- Configurare ----
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL  = os.getenv("PIPER_MODEL", "voci_piper/ro_RO-mihai-medium.onnx")

# Eveniment pentru oprirea redării în curs (barge-in, Task 3.7)
_stop_event = threading.Event()

# Lock ca să nu pornească două redări simultan
_redare_lock = threading.Lock()


def _reda_wav_bytes(wav_bytes: bytes) -> None:
    """
    Parsează un blob WAV din memorie și îl redă prin sounddevice.
    Blochează până se termină redarea SAU până _stop_event e setat.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate  = wf.getframerate()
        n_canale     = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames     = wf.getnframes()
        raw          = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width, np.int16)
    audio = np.frombuffer(raw, dtype=dtype)

    if n_canale > 1:
        audio = audio.reshape(-1, n_canale)

    audio_f32 = audio.astype(np.float32) / np.iinfo(dtype).max

    BLOC = 4096
    idx = 0

    with sd.OutputStream(
        samplerate=sample_rate,
        channels=n_canale,
        dtype="float32",
        device=DEVICE_IMPLICIT,
    ) as stream:
        while idx < len(audio_f32) and not _stop_event.is_set():
            bloc = audio_f32[idx : idx + BLOC]
            if bloc.ndim == 1:
                bloc = bloc.reshape(-1, 1)
            stream.write(bloc)
            idx += BLOC


def spune(text: str) -> None:
    """
    Transformă textul în voce și îl redă imediat prin boxe.

    Piper are nevoie de seek() pentru a scrie headerul WAV după sinteză,
    deci folosim un fișier temporar în loc de stdout.
    """
    if not text or not text.strip():
        return

    _stop_event.clear()

    with _redare_lock:
        # Creăm fișier temporar — Piper face seek() pe el pentru header WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            cale_tmp = tmp.name

        try:
            rezultat = subprocess.run(
                [
                    PIPER_BINARY,
                    "--model",       PIPER_MODEL,
                    "--output-file", cale_tmp,
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            os.unlink(cale_tmp)
            raise RuntimeError(
                f"Binarul Piper nu a fost găsit: '{PIPER_BINARY}'. "
                f"Pune calea completă în PIPER_BINARY din .env sau adaugă piper în PATH."
            )
        except subprocess.TimeoutExpired:
            os.unlink(cale_tmp)
            raise RuntimeError("Piper a depășit limita de timp. Textul e prea lung?")

        if rezultat.returncode != 0:
            os.unlink(cale_tmp)
            eroare = rezultat.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Piper a returnat eroare (cod {rezultat.returncode}): {eroare}")

        try:
            with open(cale_tmp, "rb") as f:
                wav_bytes = f.read()
        finally:
            os.unlink(cale_tmp)

        if not wav_bytes:
            raise RuntimeError(
                "Piper nu a generat niciun audio. "
                "Verifică modelul: PIPER_MODEL din .env sau calea hardcodată."
            )

        _reda_wav_bytes(wav_bytes)


def opreste() -> None:
    """
    Oprește redarea vocii în curs.
    Apelat din Task 3.7 (barge-in) când utilizatorul începe să vorbească.
    """
    _stop_event.set()


if __name__ == "__main__":
    import sys

    text_test = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Bună ziua, Vasea. Sistemul audio funcționează corect."
    )

    print(f"[Sintetizare: \"{text_test}\"]")
    print(f"[Model: {PIPER_MODEL}]")
    print(f"[Binar: {PIPER_BINARY}]")

    try:
        spune(text_test)
        print("[Redare completă]")
    except RuntimeError as e:
        print(f"[EROARE]: {e}")
        sys.exit(1)