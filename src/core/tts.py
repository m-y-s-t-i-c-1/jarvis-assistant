"""
Sintetizarea și Redarea Vocii — Text-to-Speech (Task 3.4)

Folosește Piper TTS (local, offline) pentru a transforma textul răspunsurilor
lui Jarvis în audio redat prin boxe. Piper scrie WAV într-un fișier temporar
(are nevoie de seek() pentru a scrie headerul), pe care îl citim și redăm
prin sounddevice, apoi îl ștergem.

Task 6.10 — redare prin echo-cancel-sink: dacă modulul PipeWire de anulare
ecou e configurat (vezi audio_io.py), redarea trece prin DEVICE_REDARE_DIFUZOARE
("echo-cancel-sink") în loc de device-ul implicit — necesar ca modulul să
aibă semnalul de referință pentru anularea ecoului în timpul barge-in.

Arhitectură:
    spune(text) — funcția principală, apelată din orchestratorul audio (Task 3.5)
    _reda_wav_bytes(bytes) — parsează headerul WAV și redă audio-ul prin sounddevice
    opreste() — întrerupe redarea curentă (pentru barge-in, Task 3.7 / 6.10)

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
import shutil
import time
from src.core import audio_player

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from dotenv import load_dotenv

from src.core.audio_io import DEVICE_REDARE_DIFUZOARE, _OPEN_CLOSE_LOCK

load_dotenv()

# ---- Configurare ----
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL  = os.getenv("PIPER_MODEL", "voci_piper/ro_RO-mihai-medium.onnx")

# Eveniment pentru oprirea redării în curs (barge-in, Task 3.7/6.10)
_stop_event = threading.Event()

# Lock ca să nu pornească două redări simultan
_redare_lock = threading.Lock()


def _reesantioneaza(audio: np.ndarray, rata_sursa: int, rata_tinta: int) -> np.ndarray:
    """
    Reeșantionare cu filtrare anti-aliasing folosind `resample_poly`.

    Suportă audio mono (1D) sau multi-canal (2D, shape (n, canale)).
    """
    if rata_sursa == rata_tinta or len(audio) == 0:
        return audio.astype(np.float32)

    up = rata_tinta
    down = rata_sursa

    # `resample_poly` lucrează pe axis=0; suportă array 1D sau 2D direct
    try:
        res = resample_poly(audio, up, down, axis=0).astype(np.float32)
    except Exception:
        # Fallback la interpolare liniară dacă ceva e în neregulă
        if audio.ndim == 1:
            n_esantioane_noi = max(1, int(len(audio) * rata_tinta / rata_sursa))
            indici_noi = np.linspace(0, len(audio) - 1, n_esantioane_noi)
            indici_vechi = np.arange(len(audio))
            res = np.interp(indici_noi, indici_vechi, audio).astype(np.float32)
        else:
            n_esantioane_noi = max(1, int(audio.shape[0] * rata_tinta / rata_sursa))
            indici_noi = np.linspace(0, audio.shape[0] - 1, n_esantioane_noi)
            indici_vechi = np.arange(audio.shape[0])
            canale_resample = [
                np.interp(indici_noi, indici_vechi, audio[:, c])
                for c in range(audio.shape[1])
            ]
            res = np.stack(canale_resample, axis=1).astype(np.float32)

    return res


def _reda_wav_bytes(wav_bytes: bytes) -> None:
    """
    Parsează un blob WAV din memorie și îl redă prin sounddevice, pe
    DEVICE_REDARE_DIFUZOARE (echo-cancel-sink dacă e disponibil).
    Blochează până se termină redarea SAU până _stop_event e setat.

    IMPORTANT: device-urile virtuale PipeWire (ex: echo-cancel-sink) au
    adesea o rată de eșantionare FIXĂ (setată la crearea filtrului),
    spre deosebire de device-ul generic "pipewire", care face resampling
    automat la orice rată. Piper generează frecvent la 22050 Hz — dacă
    rata WAV-ului nu se potrivește cu rata nativă a device-ului, PortAudio
    aruncă "Invalid sample rate" direct la deschiderea stream-ului. Ca să
    evităm asta, reeșantionăm manual înainte de redare, la rata nativă a
    device-ului curent.
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
    # Creăm un fișier temporar WAV și încercăm să-l redăm cu un player extern
    # (paplay sau aplay) înainte de a folosi PortAudio, pentru a evita crash-urile
    # cunoscute în anumite combinații libportaudio/driver.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_bytes)
            tmp_path = tf.name

        # Încercăm `paplay` (PulseAudio/PipeWire friendly), apoi `aplay` ca fallback
        player = shutil.which("paplay") or shutil.which("aplay")
        if player:
            try:
                proc = subprocess.Popen([player, tmp_path])
                # Așteptăm terminarea player-ului sau eventul de stop
                while proc.poll() is None and not _stop_event.is_set():
                    time.sleep(0.05)

                if _stop_event.is_set() and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except Exception:
                        proc.kill()
                return
            except FileNotFoundError:
                # Nu găsim player-ul; cădem în fallback la sounddevice
                pass

        # Dacă nu avem player extern sau elșuie, folosim player-ul persistent
        # care păstrează un OutputStream deschis (reduce riscul deschiderilor/închiderilor repetate)
        try:
            audio_player.play_blocking(audio_f32, sample_rate, _stop_event)
            return
        except Exception as e:
            print(f"[TTS] Player persistent a eșuat: {e} — încerc fallback sounddevice/paplay")

        # Verificăm rata nativă a device-ului de redare și reeșantionăm dacă diferă
        try:
            info_device = sd.query_devices(DEVICE_REDARE_DIFUZOARE)
            rata_device = int(info_device["default_samplerate"])
            if rata_device != sample_rate:
                audio_f32 = _reesantioneaza(audio_f32, sample_rate, rata_device)
                sample_rate = rata_device
        except Exception as e:
            print(f"[TTS] Nu am putut verifica rata device-ului, redau la rata originală: {e}")

        # Fallback final: try sd.play (serialized open/close)
        try:
            with _OPEN_CLOSE_LOCK:
                sd.play(audio_f32, samplerate=sample_rate, device=DEVICE_REDARE_DIFUZOARE)
                sd.wait()
        except Exception as e2:
            print(f"[TTS] sd.play a eșuat: {e2}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
    Apelat din Task 3.7/6.10 (barge-in) când utilizatorul începe să vorbească.
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