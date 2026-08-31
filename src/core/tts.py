"""
Sintetizarea și Redarea Vocii — Text-to-Speech (Task 3.4 + 6.10)

Piper genereză WAV; redarea trece PRIORITAR printr-un player extern
(subprocess: paplay / pw-play / ffplay / afplay / aplay), NU prin
PortAudio OutputStream în același proces Python.

De ce: pe multe sisteme (mai ales PipeWire + openwakeword/onnxruntime +
torch/Silero), deschiderea InputStream (wake/VAD/barge-in) și OutputStream
(TTS) în același proces PortAudio duce la crash nativ:
    free(): corrupted unsorted chunks / double free or corruption
Separarea redării într-un proces dedicat păstrează rezultatul (voce +
oprire barge-in via terminate) și e mai portabilă între mașini.

Echo-cancel (barge-in): dacă există sink-ul Pulse/PipeWire
`echo-cancel-sink` (sau unul cu „Jarvis” în descriere), paplay îl folosește
cu `-d`, ca modulul de anulare ecou să aibă semnal de referință.

Fallback final: sounddevice, doar dacă niciun player extern nu e disponibil.

Configurare în .env (opțional):
    PIPER_BINARY   — calea către executabilul piper (default: piper)
    PIPER_MODEL    — calea către fișierul .onnx
    TTS_SINK       — nume sink Pulse/PipeWire forțat (ex: echo-cancel-sink)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import subprocess
import time

from dotenv import load_dotenv

load_dotenv()

PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "voci_piper/ro_RO-mihai-medium.onnx")
TTS_SINK_ENV = os.getenv("TTS_SINK", "").strip() or None
ALLOW_SOUNDDEVICE_TTS = os.getenv("ALLOW_SOUNDDEVICE_TTS", "0").strip().lower() not in {"0", "false", "no", "off", ""}

_stop_event = threading.Event()
_redare_lock = threading.Lock()
_player_proc: subprocess.Popen | None = None
_player_lock = threading.Lock()

# Cache sink echo-cancel (None = necunoscut încă, False = lipsă, str = nume)
_sink_ecou_cache: str | bool | None = None


def _gaseste_sink_ecou() -> str | None:
    """
    Găsește sink-ul Pulse/PipeWire pentru anulare ecou, dacă există.
    Universal: folosește `pactl` când e disponibil; altfel None.
    """
    global _sink_ecou_cache
    if TTS_SINK_ENV:
        return TTS_SINK_ENV
    if _sink_ecou_cache is False:
        return None
    if isinstance(_sink_ecou_cache, str):
        return _sink_ecou_cache

    if not shutil.which("pactl"):
        _sink_ecou_cache = False
        return None

    try:
        out = subprocess.check_output(
            ["pactl", "list", "sinks"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        _sink_ecou_cache = False
        return None

    nume_curent = None
    candidat_jarvis = None
    for linie in out.splitlines():
        s = linie.strip()
        if s.startswith("Name:"):
            nume_curent = s.split(":", 1)[1].strip()
        elif s.startswith("Description:") and nume_curent:
            desc = s.split(":", 1)[1].strip().lower()
            if nume_curent == "echo-cancel-sink":
                _sink_ecou_cache = nume_curent
                return nume_curent
            if "jarvis" in desc and candidat_jarvis is None:
                candidat_jarvis = nume_curent

    if candidat_jarvis:
        _sink_ecou_cache = candidat_jarvis
        return candidat_jarvis

    _sink_ecou_cache = False
    return None


def _construieste_comenzi_player(cale_wav: str) -> list[list[str]]:
    """
    Listează comenzi de redare, în ordinea preferinței.
    Fiecare e o listă argv gata de Popen.
    """
    comenzi: list[list[str]] = []
    sink = _gaseste_sink_ecou()

    if shutil.which("paplay"):
        if sink:
            comenzi.append(["paplay", "-d", sink, cale_wav])
        comenzi.append(["paplay", cale_wav])

    if shutil.which("pw-play"):
        comenzi.append(["pw-play", cale_wav])

    if shutil.which("ffplay"):
        # -nodisp -autoexit: fără fereastră, iese la sfârșitul fișierului
        comenzi.append(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", cale_wav]
        )

    if shutil.which("afplay"):  # macOS
        comenzi.append(["afplay", cale_wav])

    if shutil.which("aplay"):
        comenzi.append(["aplay", "-q", cale_wav])

    return comenzi


def _seteaza_player(proc: subprocess.Popen | None) -> None:
    global _player_proc
    with _player_lock:
        _player_proc = proc


def _opreste_player_curent() -> None:
    """Oprește procesul de redare (barge-in / opreste())."""
    with _player_lock:
        proc = _player_proc
    if proc is None:
        return
    if proc.poll() is not None:
        _seteaza_player(None)
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _seteaza_player(None)


def _reda_cu_player_extern(cale_wav: str) -> bool:
    """
    Redă WAV printr-un player extern. Returnează True dacă a reușit să
    pornească și să aștepte (sau să fie oprit de barge-in).
    False dacă niciun player nu a putut porni.
    """
    comenzi = _construieste_comenzi_player(cale_wav)
    if not comenzi:
        return False

    erori: list[str] = []
    for cmd in comenzi:
        if _stop_event.is_set():
            return True
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            erori.append(f"{cmd[0]}: {e}")
            continue
        except Exception as e:
            erori.append(f"{cmd[0]}: {type(e).__name__}: {e}")
            continue

        _seteaza_player(proc)
        try:
            while proc.poll() is None:
                if _stop_event.is_set():
                    _opreste_player_curent()
                    return True
                time.sleep(0.05)
            # Cod != 0: încercăm următorul player (ex: sink inexistent)
            if proc.returncode not in (0, None) and not _stop_event.is_set():
                erori.append(f"{cmd[0]} exit {proc.returncode}")
                continue
            return True
        finally:
            _seteaza_player(None)

    if erori:
        print(f"[TTS] Playere externe eșuate: {'; '.join(erori[:3])}")
    return False


def _reda_cu_sounddevice(cale_wav: str) -> None:
    """
    Fallback ultim: PortAudio via sounddevice.
    Folosit doar pe sisteme fără paplay/pw-play/ffplay/aplay/afplay.
    """
    import wave
    import io
    import math
    import numpy as np
    import sounddevice as sd
    from scipy.signal import resample_poly
    from src.core.audio_io import (
        BLOCKSIZE,
        DEVICE_IMPLICIT,
        DEVICE_REDARE_DIFUZOARE,
        safe_output_stream,
    )

    with open(cale_wav, "rb") as f:
        wav_bytes = f.read()

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_canale = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width, np.int16)
    audio = np.frombuffer(raw, dtype=dtype)
    if n_canale > 1:
        audio = audio.reshape(-1, n_canale)
    audio_f32 = audio.astype(np.float32) / np.iinfo(dtype).max

    device = DEVICE_REDARE_DIFUZOARE
    try:
        info = sd.query_devices(device)
        rata = int(info["default_samplerate"])
    except Exception:
        device = DEVICE_IMPLICIT
        rata = sample_rate

    if rata != sample_rate:
        div = math.gcd(sample_rate, rata)
        audio_f32 = resample_poly(
            audio_f32, rata // div, sample_rate // div, axis=0
        ).astype(np.float32)
        audio_f32 = np.clip(audio_f32 * 0.92, -1.0, 1.0)

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


def _reda_fisier_wav(cale_wav: str) -> None:
    """Redă un fișier WAV: player extern întâi; fără el, evită sounddevice pe sisteme fragile."""
    if _reda_cu_player_extern(cale_wav):
        return

    if not ALLOW_SOUNDDEVICE_TTS:
        print(
            "[TTS] Niciun player extern disponibil/utilizabil "
            "(paplay/pw-play/ffplay/afplay/aplay). "
            "Sounddevice e dezactivat implicit pentru a evita crash-ul nativ "
            "PipeWire/PortAudio. Setează ALLOW_SOUNDDEVICE_TTS=1 doar dacă ai "
            "testat cu atenție sistemul audio."
        )
        return

    print(
        "[TTS] Niciun player extern disponibil/utilizabil "
        "(paplay/pw-play/ffplay/afplay/aplay) — fallback sounddevice activat "
        "după opt-in explicit."
    )
    _reda_cu_sounddevice(cale_wav)


def spune(text: str) -> None:
    """
    Transformă textul în voce și îl redă imediat prin boxe.

    Piper are nevoie de seek() pentru headerul WAV, deci fișier temporar.
    """
    if not text or not text.strip():
        return

    _stop_event.clear()

    with _redare_lock:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            cale_tmp = tmp.name

        try:
            try:
                rezultat = subprocess.run(
                    [
                        PIPER_BINARY,
                        "--model",
                        PIPER_MODEL,
                        "--output-file",
                        cale_tmp,
                    ],
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Binarul Piper nu a fost găsit: '{PIPER_BINARY}'. "
                    f"Pune calea completă în PIPER_BINARY din .env sau adaugă piper în PATH."
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("Piper a depășit limita de timp. Textul e prea lung?")

            if rezultat.returncode != 0:
                eroare = rezultat.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Piper a returnat eroare (cod {rezultat.returncode}): {eroare}"
                )

            if not os.path.getsize(cale_tmp):
                raise RuntimeError(
                    "Piper nu a generat niciun audio. "
                    "Verifică modelul: PIPER_MODEL din .env sau calea hardcodată."
                )

            _reda_fisier_wav(cale_tmp)
        finally:
            try:
                os.unlink(cale_tmp)
            except OSError:
                pass


def opreste() -> None:
    """
    Oprește redarea vocii în curs (barge-in).
    Omoară playerul extern dacă rulează; pentru fallback sounddevice
    semnalează _stop_event.
    """
    _stop_event.set()
    _opreste_player_curent()


if __name__ == "__main__":
    import sys

    text_test = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Bună ziua, Vasea. Sistemul audio funcționează corect."
    )

    print(f'[Sintetizare: "{text_test}"]')
    print(f"[Model: {PIPER_MODEL}]")
    print(f"[Binar: {PIPER_BINARY}]")
    print(f"[Sink echo-cancel: {_gaseste_sink_ecou()}]")
    print(f"[Playere: {[c[0] for c in _construieste_comenzi_player('x.wav')]}]")

    try:
        spune(text_test)
        print("[Redare completă]")
    except RuntimeError as e:
        print(f"[EROARE]: {e}")
        sys.exit(1)
