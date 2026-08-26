"""
Configurarea I/O Audio (Task 3.1) + Anulare Ecou pentru Barge-in (Task 6.10)

Strat de bază peste hardware-ul audio, folosind sounddevice.

Task 6.10 — device-uri dedicate pentru anulare ecou:
    Pentru barge-in fiabil (întreruperea lui Jarvis din vorbire), microfonul
    și difuzoarele trebuie să treacă prin modulul PipeWire de anulare ecou
    (`echo-cancel-source` / `echo-cancel-sink`), NU prin device-urile
    implicite ale sistemului — altfel Jarvis își aude propria voce prin
    difuzoare și se poate întrerupe singur (fals-pozitiv).

    DEVICE_CAPTURA_MIC / DEVICE_REDARE_DIFUZOARE caută automat aceste
    noduri PipeWire după nume, la pornire. Dacă nu le găsește (modulul nu
    e configurat, sau ai dezinstalat webrtc-audio-processing), cad elegant
    pe DEVICE_IMPLICIT ("pipewire" generic) — Jarvis funcționează în
    continuare, doar fără beneficiul anulării de ecou.

Format audio standard folosit în tot proiectul:
    - Sample rate: 16000 Hz (standardul pentru Whisper, evită resampling ulterior)
    - Canale: 1 (mono — Whisper nu are nevoie de stereo)
    - Tip: float32 (formatul nativ sounddevice, ușor convertibil)
"""

import sounddevice as sd
import numpy as np
import threading
from contextlib import contextmanager

# Configurare audio standard — folosită peste tot în Faza 3
SAMPLE_RATE = 16000
CANALE = 1

# IMPORTANT: device-ul implicit ales automat de sistem poate fi o conexiune
# ALSA directă la hardware (ex: "HDA Intel PCH"), care nu acceptă orice
# sample rate — doar cele native (44100/48000 Hz), nu 16000 Hz cerut de Whisper.
# Forțăm folosirea PipeWire, care face resampling automat la orice rată.
DEVICE_IMPLICIT = "pipewire"

sd.default.device = DEVICE_IMPLICIT
sd.default.latency = "high"  # prioritizează stabilitatea în fața latenței minime
BLOCKSIZE = 1024  # buffer explicit — evită underrun-uri care sună "robotic"

# Lock folosit STRICT în jurul operațiilor de deschidere/închidere a stream-urilor
# (open/close). Nu blochează I/O per-se, doar serializes open/close pentru a evita
# race-uri native între PortAudio și backend-urile ALSA/PipeWire.
_OPEN_CLOSE_LOCK = threading.Lock()


@contextmanager
def safe_input_stream(*args, **kwargs):
    """Context manager care serializes open/close pentru InputStream.

    Folosește `_OPEN_CLOSE_LOCK` doar în timpul apelurilor de intrare/ieșire
    (open/close), nu în timpul citirii efective.
    """
    with _OPEN_CLOSE_LOCK:
        stream = sd.InputStream(*args, **kwargs)
        stream.__enter__()

    try:
        yield stream
    finally:
        with _OPEN_CLOSE_LOCK:
            try:
                stream.__exit__(None, None, None)
            except Exception:
                pass


@contextmanager
def safe_output_stream(*args, **kwargs):
    """Context manager similar pentru OutputStream."""
    with _OPEN_CLOSE_LOCK:
        stream = sd.OutputStream(*args, **kwargs)
        stream.__enter__()

    try:
        yield stream
    finally:
        with _OPEN_CLOSE_LOCK:
            try:
                stream.__exit__(None, None, None)
            except Exception:
                pass


def _gaseste_device_dupa_nume(nume_partial: str):
    """
    Caută în lista de device-uri sounddevice unul al cărui nume conține
    `nume_partial` (case-insensitive). Returnează indexul (int) dacă
    găsește, altfel None.

    Folosit pentru a localiza automat nodurile PipeWire de anulare ecou
    (echo-cancel-source / echo-cancel-sink), create de modulul configurat
    în ~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf.
    """
    try:
        device_uri = sd.query_devices()
    except Exception:
        return None

    for i, dev in enumerate(device_uri):
        if nume_partial.lower() in dev["name"].lower():
            return i

    return None


# ── Device-uri dedicate anulării de ecou (Task 6.10), cu fallback elegant ──
# IMPORTANT: PortAudio/sounddevice arată eticheta descriptivă (node.description
# din config), NU numele intern PipeWire ("echo-cancel-source"/"-sink") —
# de-aia căutăm după textele exacte puse în 99-echo-cancel.conf, nu după
# numele nodului. Confirmat empiric cu sd.query_devices() pe sistemul lui Vasea.
_index_captura_ec = _gaseste_device_dupa_nume("Jarvis - Microfon")
_index_redare_ec = _gaseste_device_dupa_nume("Jarvis - Difuzoare")

DEVICE_CAPTURA_MIC = _index_captura_ec if _index_captura_ec is not None else DEVICE_IMPLICIT
DEVICE_REDARE_DIFUZOARE = _index_redare_ec if _index_redare_ec is not None else DEVICE_IMPLICIT

if _index_captura_ec is not None and _index_redare_ec is not None:
    print("[Audio] Anulare ecou PipeWire detectată — barge-in va folosi echo-cancel-source/-sink.")
else:
    print(
        "[Audio] Nodurile PipeWire de anulare ecou NU au fost găsite — "
        "folosesc device-ul implicit. Barge-in-ul generic (VAD) poate avea "
        "fals-pozitive din propriul ecou al lui Jarvis. Vezi configurarea "
        "din ~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf."
    )


def listeaza_device_uri():
    """
    Afișează toate device-urile audio detectate de sistem, cu indexul lor.
    Util pentru a identifica microfonul/boxele corecte dacă ai mai multe
    (ex: microfon laptop + cască USB), sau pentru a verifica manual dacă
    nodurile echo-cancel-source/-sink apar în listă.
    """
    device_uri = sd.query_devices()
    print("\n=== Device-uri audio disponibile ===")
    for i, dev in enumerate(device_uri):
        tip = []
        if dev["max_input_channels"] > 0:
            tip.append("INPUT (microfon)")
        if dev["max_output_channels"] > 0:
            tip.append("OUTPUT (boxe)")
        tip_str = " + ".join(tip) if tip else "necunoscut"
        print(f"  [{i}] {dev['name']} — {tip_str}")

    print(f"\nDevice implicit input:  {sd.default.device[0]}")
    print(f"Device implicit output: {sd.default.device[1]}")
    print(f"Device captură (folosit de VAD/wake-word): {DEVICE_CAPTURA_MIC}")
    print(f"Device redare (folosit de TTS): {DEVICE_REDARE_DIFUZOARE}")
    print("=====================================\n")

    return device_uri


def inregistreaza(durata_secunde: float = 3.0, device=None) -> np.ndarray:
    """
    Înregistrează audio de la microfon pentru durata specificată.

    Parametri:
        durata_secunde: cât timp să înregistreze
        device:         numele sau indexul device-ului de input
                        (default: DEVICE_CAPTURA_MIC — echo-cancel-source
                        dacă e disponibil, altfel DEVICE_IMPLICIT)

    Returnează:
        numpy array, shape (n_samples, 1), dtype float32
    """
    if device is None:
        device = DEVICE_CAPTURA_MIC

    print(f"[Înregistrare {durata_secunde}s — vorbește acum...]")

    with _OPEN_CLOSE_LOCK:
        audio = sd.rec(
            int(durata_secunde * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CANALE,
            dtype="float32",
            device=device,
            blocksize=BLOCKSIZE,
        )
        sd.wait()  # blochează până se termină înregistrarea

    print("[Înregistrare terminată]")
    return audio


def reda(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, device=None):
    """
    Redă un numpy array prin boxele sistemului. Blochează până se termină
    reproducerea (sd.wait()), ca să nu se suprapună cu alte sunete.

    Parametri:
        audio:       numpy array cu datele audio
        sample_rate: rata de eșantionare a array-ului (default: 16000)
        device:      indexul device-ului de output (default:
                     DEVICE_REDARE_DIFUZOARE — echo-cancel-sink dacă e
                     disponibil, altfel None = implicit din sistem)
    """
    if device is None:
        device = DEVICE_REDARE_DIFUZOARE

    with _OPEN_CLOSE_LOCK:
        sd.play(audio, samplerate=sample_rate, device=device, blocksize=BLOCKSIZE)
        sd.wait()


def test_complet(durata_secunde: float = 3.0):
    """
    Test rapid de capăt-la-capăt: înregistrează N secunde, apoi le redă
    imediat înapoi. Dacă auzi exact ce ai spus, microfonul ȘI boxele
    funcționează corect prin Python — gata pentru Task 3.2+.
    """
    listeaza_device_uri()

    audio = inregistreaza(durata_secunde)

    print("[Redare — ar trebui să auzi ce ai înregistrat...]")
    reda(audio)
    print("[Test complet. Dacă ai auzit clar vocea ta, totul funcționează.]")


def test_sample_rate_nativ(durata_secunde: float = 3.0):
    """
    Test de diagnostic: înregistrează și redă la sample rate-ul NATIV al
    device-ului (de obicei 44100 sau 48000 Hz), fără nicio conversie.

    Folosește asta dacă test_complet() sună distorsionat — izolează dacă
    problema vine din resampling-ul către 16000 Hz sau e altceva (driver,
    format, hardware).
    """
    info_device = sd.query_devices(sd.default.device[0])
    rata_nativa = int(info_device["default_samplerate"])

    print(f"\n[Test la sample rate nativ: {rata_nativa} Hz, fără resampling]")
    print(f"[Înregistrare {durata_secunde}s — vorbește acum...]")

    with _OPEN_CLOSE_LOCK:
        audio = sd.rec(
            int(durata_secunde * rata_nativa),
            samplerate=rata_nativa,
            channels=CANALE,
            dtype="float32",
            blocksize=BLOCKSIZE,
        )
        sd.wait()
    print("[Înregistrare terminată — redare...]")

    with _OPEN_CLOSE_LOCK:
        sd.play(audio, samplerate=rata_nativa, blocksize=BLOCKSIZE)
        sd.wait()
    print("[Test terminat. Compară claritatea cu test_complet().]")


if __name__ == "__main__":
    # Rulează direct cu: python -m src.core.audio_io
    test_complet()