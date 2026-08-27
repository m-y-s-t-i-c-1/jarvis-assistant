"""
Sintetizarea și Redarea Vocii — Text-to-Speech (Task 3.4 + fix Task 6.10)

Folosește Piper TTS (local, offline) pentru a transforma textul răspunsurilor
lui Jarvis în audio redat prin boxe. Piper scrie WAV într-un fișier temporar
(are nevoie de seek() pentru a scrie headerul), pe care îl citim și redăm
prin sounddevice, apoi îl ștergem.

FIX (bug "Jarvis mă aude dar nu răspunde cu voce"):
    DEVICE_REDARE_DIFUZOARE (echo-cancel-sink, Task 6.10) e un nod PipeWire
    brut, NU device-ul generic "pipewire" care face resampling automat.
    Nodurile de filtru create de module-echo-cancel acceptă de regulă DOAR
    rata lor nativă configurată (de obicei 48000 Hz) — orice altă rată
    trimisă direct de aplicație (16000 Hz de la Whisper, sau rata WAV-ului
    generat de Piper, adesea 22050 Hz) e respinsă de PortAudio cu:

        PortAudioError: Error opening OutputStream: Invalid sample rate
        [PaErrorCode -9997]

    Asta cădea silențios din perspectiva utilizatorului: STT + agent
    funcționau (textul apărea în terminal), dar TTS-ul eșua mereu chiar
    la deschiderea stream-ului, înainte să scoată vreun sunet.

    Soluție: interogăm rata nativă a device-ului de redare cu
    sd.query_devices() și reeșantionăm audio-ul cu scipy.signal.resample_poly
    (aceeași tehnică deja folosită în barge_in.py pentru captură) înainte
    de a deschide stream-ul, la rata corectă. Dacă redarea tot eșuează
    (device ocupat, echo-cancel dezactivat între timp etc.), cădem elegant
    pe DEVICE_IMPLICIT (device-ul generic "pipewire", cu resampling automat)
    — Jarvis tot vorbește, doar fără beneficiul anulării de ecou pentru
    acel enunț.

Configurare în .env (opțional):
    PIPER_BINARY   — calea către executabilul piper (default: piper)
    PIPER_MODEL    — calea către fișierul .onnx
"""

import io
import os
import wave
import math
import tempfile
import threading
import subprocess

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from dotenv import load_dotenv

from src.core.audio_io import (
    BLOCKSIZE,
    DEVICE_IMPLICIT,
    DEVICE_REDARE_DIFUZOARE,
    safe_output_stream,
)

load_dotenv()

# ---- Configurare ----
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL  = os.getenv("PIPER_MODEL", "voci_piper/ro_RO-mihai-medium.onnx")

# Eveniment pentru oprirea redării în curs (barge-in, Task 3.7 / 6.10)
_stop_event = threading.Event()

# Lock ca să nu pornească două redări simultan
_redare_lock = threading.Lock()


def _rata_nativa(device) -> int | None:
    """Interoghează rata de eșantionare implicită a unui device sounddevice."""
    try:
        info = sd.query_devices(device)
        return int(info["default_samplerate"])
    except Exception:
        return None


def _reesantioneaza(audio_f32: np.ndarray, rata_sursa: int, rata_tinta: int) -> np.ndarray:
    """
    Reeșantionează audio float32 (1D mono sau 2D multi-canal) de la
    rata_sursa la rata_tinta, cu resample_poly + GCD (ca în barge_in.py).
    """
    if rata_sursa == rata_tinta or len(audio_f32) == 0:
        return audio_f32

    divizor = math.gcd(rata_sursa, rata_tinta)
    up = rata_tinta // divizor
    down = rata_sursa // divizor
    reesantionat = resample_poly(audio_f32, up, down, axis=0).astype(np.float32)

    # IMPORTANT: resample_poly poate produce mici depășiri peste ±1.0
    # (ringing/Gibbs), care la redare float32 sună ca trosnituri/distorsiune
    # ("difuzor stricat"). Reducem ușor nivelul înainte de clip ca să nu
    # tăiem vârfurile reale ale vocii Piper (clip hard = distorsiune dură).
    reesantionat *= 0.92
    return np.clip(reesantionat, -1.0, 1.0)


def _incearca_redare(audio_f32: np.ndarray, n_canale: int, rata: int, device) -> None:
    """
    Deschide un OutputStream pe device-ul dat, la rata dată, și redă blocul de audio.

    IMPORTANT: folosim safe_output_stream (nu sd.OutputStream direct) —
    serializes deschiderea/închiderea prin _OPEN_CLOSE_LOCK din audio_io.py,
    exact ca să nu intre în conflict cu InputStream-ul deschis concurent de
    barge_in.py (safe_input_stream) în timp ce Jarvis vorbește. Fără acest
    lock, deschiderea simultană a două stream-uri PortAudio pe noduri
    PipeWire înrudite (echo-cancel-source + echo-cancel-sink) poate produce
    glitch-uri audio și poate rupe detectarea întreruperii (barge-in).
    """
    # Blocuri scurte: latență mică la barge-in (opreste() oprește la următorul
    # write) + mai puține underrun-uri „robotice” decât un buffer uriaș.
    bloc_size = max(256, BLOCKSIZE)
    idx = 0

    with safe_output_stream(
        samplerate=rata,
        channels=n_canale,
        dtype="float32",
        device=device,
        blocksize=bloc_size,
    ) as stream:
        while idx < len(audio_f32) and not _stop_event.is_set():
            bloc = audio_f32[idx: idx + bloc_size]
            if bloc.ndim == 1:
                bloc = bloc.reshape(-1, 1)
            stream.write(bloc)
            idx += bloc_size

        # NU apelăm stream.abort() aici: pe PipeWire/PortAudio, abort() +
        # close() din __exit__ a dus la "free(): corrupted unsorted chunks".
        # Oprirea la următorul write (via _stop_event) e suficientă pentru
        # barge-in; eventualele ~50–100ms rămase în buffer sunt acceptabile.


def _reda_wav_bytes(wav_bytes: bytes) -> None:
    """
    Parsează un blob WAV din memorie și îl redă prin sounddevice.
    Blochează până se termină redarea SAU până _stop_event e setat.

    Încearcă întâi DEVICE_REDARE_DIFUZOARE (echo-cancel-sink, dacă e
    disponibil), reeșantionând la rata lui nativă. Dacă asta eșuează
    (ex: Invalid sample rate, device ocupat), cade pe DEVICE_IMPLICIT,
    care face resampling automat prin PipeWire — mai puțin optim pentru
    barge-in, dar garantează că Jarvis tot vorbește.
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

    erori = []
    device_preferat = DEVICE_REDARE_DIFUZOARE
    device_fallback = DEVICE_IMPLICIT
    # DEVICE_* cade pe "pipewire" (string), niciodată pe None — deci încercăm
    # mereu path-ul preferat, apoi fallback doar dacă e un device diferit.
    acelasi_device = device_preferat == device_fallback

    # ── Încercarea 1: device-ul de anulare ecou, la rata lui nativă ──────
    try:
        rata_nativa = _rata_nativa(device_preferat)
        if rata_nativa is None:
            raise RuntimeError("Nu am putut determina rata nativă a device-ului.")

        audio_convertit = audio_f32
        if rata_nativa != sample_rate:
            audio_convertit = _reesantioneaza(audio_f32, sample_rate, rata_nativa)

        _incearca_redare(audio_convertit, n_canale, rata_nativa, device_preferat)
        return  # succes — gata
    except Exception as e:
        erori.append(
            f"DEVICE_REDARE_DIFUZOARE [{device_preferat}]: "
            f"{type(e).__name__}: {e}"
        )
        if not acelasi_device:
            print(
                f"[TTS] Redare pe echo-cancel-sink a eșuat ({type(e).__name__}: {e}). "
                f"Cad pe device-ul implicit, fără anulare de ecou pentru acest enunț."
            )

    # ── Încercarea 2 (fallback): device-ul implicit, cu resampling automat ──
    if not acelasi_device:
        try:
            _incearca_redare(audio_f32, n_canale, sample_rate, device_fallback)
            return
        except Exception as e:
            erori.append(f"DEVICE_IMPLICIT [{device_fallback}]: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Redarea TTS a eșuat pe toate device-urile încercate:\n  "
        + "\n  ".join(erori)
        + "\nRulează listeaza_device_uri() din audio_io.py pentru diagnostic."
    )


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
    Apelat din Task 3.7 / 6.10 (barge-in) când utilizatorul începe să vorbească.
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
    print(f"[Device redare (echo-cancel): {DEVICE_REDARE_DIFUZOARE}]")
    print(f"[Device implicit (fallback): {DEVICE_IMPLICIT}]")

    try:
        spune(text_test)
        print("[Redare completă]")
    except RuntimeError as e:
        print(f"[EROARE]: {e}")
        sys.exit(1)