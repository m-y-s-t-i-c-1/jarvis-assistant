"""
Configurarea I/O Audio (Task 3.1)

Strat de bază peste hardware-ul audio, folosind sounddevice.
Nu are nicio legătură cu Whisper/Piper încă — doar verifică că
microfonul și boxele răspund corect prin Python, înainte să construim
STT/TTS peste el (Task 3.3, 3.4).

Funcții cheie:
    listeaza_device_uri()        — afișează toate device-urile audio disponibile
    inregistreaza(durata)        — înregistrează N secunde de la microfon, returnează numpy array
    reda(audio_array)            — redă un numpy array prin boxe
    test_complet()               — înregistrează 3 secunde și le redă imediat (loopback test)

Format audio standard folosit în tot proiectul:
    - Sample rate: 16000 Hz (standardul pentru Whisper, evită resampling ulterior)
    - Canale: 1 (mono — Whisper nu are nevoie de stereo)
    - Tip: float32 (formatul nativ sounddevice, ușor convertibil)
"""

import sounddevice as sd
import numpy as np

# Configurare audio standard — folosită peste tot în Faza 3
SAMPLE_RATE = 16000
CANALE = 1

# IMPORTANT: device-ul implicit ales automat de sistem poate fi o conexiune
# ALSA directă la hardware (ex: "HDA Intel PCH"), care nu acceptă orice
# sample rate — doar cele native (44100/48000 Hz), nu 16000 Hz cerut de Whisper.
# Forțăm folosirea PipeWire, care face resampling automat la orice rată.
#
# Dacă la tine indexul "pipewire" diferă, rulează listeaza_device_uri()
# și actualizează valoarea de mai jos.
DEVICE_IMPLICIT = "pipewire"

sd.default.device = DEVICE_IMPLICIT
sd.default.latency = "high"  # prioritizează stabilitatea în fața latenței minime
BLOCKSIZE = 1024  # buffer explicit — evită underrun-uri care sună "robotic"

# IMPORTANT: pe multe sisteme Linux, device-ul ALSA hardware direct
# (ex: "HDA Intel PCH") nu acceptă orice sample rate (ex: 16000 Hz),
# doar ratele native ale plăcii de sunet (de obicei 44100/48000 Hz).
# PipeWire/PulseAudio fac resampling automat, deci sunt alegerea sigură.
# Dacă 'pipewire' nu există pe sistemul tău, schimbă cu numele exact
# din lista afișată de listeaza_device_uri() (caută unul fără hw:X,Y).
DEVICE_IMPLICIT = "pipewire"


def listeaza_device_uri():
    """
    Afișează toate device-urile audio detectate de sistem, cu indexul lor.
    Util pentru a identifica microfonul/boxele corecte dacă ai mai multe
    (ex: microfon laptop + cască USB).
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
    print("=====================================\n")

    return device_uri


def inregistreaza(durata_secunde: float = 3.0, device=DEVICE_IMPLICIT) -> np.ndarray:
    """
    Înregistrează audio de la microfon pentru durata specificată.

    Parametri:
        durata_secunde: cât timp să înregistreze
        device:         numele sau indexul device-ului de input
                        (default: DEVICE_IMPLICIT = "pipewire")

    Returnează:
        numpy array, shape (n_samples, 1), dtype float32
    """
    print(f"[Înregistrare {durata_secunde}s — vorbește acum...]")

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


def reda(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, device=DEVICE_IMPLICIT):
    """
    Redă un numpy array prin boxele sistemului. Blochează până se termină
    reproducerea (sd.wait()), ca să nu se suprapună cu alte sunete.

    Parametri:
        audio:       numpy array cu datele audio
        sample_rate: rata de eșantionare a array-ului (default: 16000)
        device:      indexul device-ului de output (None = implicit din sistem)
    """
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

    audio = sd.rec(
        int(durata_secunde * rata_nativa),
        samplerate=rata_nativa,
        channels=CANALE,
        dtype="float32",
        blocksize=BLOCKSIZE,
    )
    sd.wait()
    print("[Înregistrare terminată — redare...]")

    sd.play(audio, samplerate=rata_nativa, blocksize=BLOCKSIZE)
    sd.wait()
    print("[Test terminat. Compară claritatea cu test_complet().]")


if __name__ == "__main__":
    # Rulează direct cu: python -m src.core.audio_io
    test_complet()