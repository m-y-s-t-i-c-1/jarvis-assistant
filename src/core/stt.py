"""
Transcrierea Audio — Speech-to-Text (Task 3.3)

Folosește faster-whisper (reimplementare CTranslate2, mult mai rapidă pe CPU
decât openai-whisper standard) pentru a transcrie audio-ul capturat de VAD
(Task 3.2) în text. Limba e fixată explicit la română — evită pasul de
auto-detectare a limbii, care ar adăuga latență inutilă pentru enunțuri scurte.

Modelul 'small' e încărcat o singură dată, lazy, la prima transcriere
necesară — descărcarea (prima rulare) durează ceva, rulările următoare
sunt instant din cache local (~/.cache/huggingface/).
"""

import numpy as np
from faster_whisper import WhisperModel

from src.core.audio_io import SAMPLE_RATE

# Configurare model — 'small' e echilibrul recomandat pentru CPU
DIMENSIUNE_MODEL = "small"

# CPU explicit, cu int8 — cuantizare care reduce mult timpul de inferență
# pe CPU, cu pierdere minimă de acuratețe (recomandat oficial pentru CPU-only)
TIP_DEVICE = "cpu"
TIP_COMPUTE = "int8"

# Limba fixată — evită auto-detectare, mai rapid pentru enunțuri scurte
LIMBA = "ro"

_model: WhisperModel | None = None


def _obtine_model() -> WhisperModel:
    """Încarcă modelul Whisper lazy, o singură dată, la prima cerere."""
    global _model
    if _model is None:
        print(f"[Încărcare model Whisper '{DIMENSIUNE_MODEL}' (CPU, int8)...]")
        _model = WhisperModel(
            DIMENSIUNE_MODEL,
            device=TIP_DEVICE,
            compute_type=TIP_COMPUTE,
        )
        print("[Model Whisper încărcat]")
    return _model


def transcrie(audio: np.ndarray) -> str:
    """
    Transcrie un array numpy de audio (din VAD, Task 3.2) în text românesc.

    Parametri:
        audio: numpy array, shape (n_samples, 1) sau (n_samples,), float32,
               la SAMPLE_RATE (16000 Hz) — exact formatul produs de
               core.audio_io și core.vad.

    Returnează:
        Textul transcris, ca string curat (fără spații extra la capete).
        String gol dacă nu s-a detectat nimic transcriptibil.
    """
    model = _obtine_model()

    # faster-whisper vrea audio 1D, nu (n_samples, 1)
    audio_flat = audio.flatten() if audio.ndim > 1 else audio

    segmente, info = model.transcribe(
        audio_flat,
        language=LIMBA,
        beam_size=3,                       # redus de la 8 — mult mai rapid pe CPU,
                                            # pierdere minimă de acuratețe pe enunțuri scurte
        temperature=0,                     # determinist, evită variații aleatoare pe enunțuri scurte
        condition_on_previous_text=False,  # nu propagă erori din context anterior
        vad_filter=False,  # VAD-ul nostru (Task 3.2) a filtrat deja tăcerea
    )

    # segmente e un generator — trebuie consumat ca să obținem textul
    text_complet = " ".join(segment.text.strip() for segment in segmente)

    return text_complet.strip()


if __name__ == "__main__":
    # Test integrat: VAD ascultă, Whisper transcrie, afișăm rezultatul text.
    from src.core.vad import obtine_detector

    detector = obtine_detector()
    audio = detector.asculta_pana_la_pauza()

    if audio is not None:
        print("[Transcriere...]")
        text = transcrie(audio)
        print(f"\n>>> Ai spus: \"{text}\"\n")
    else:
        print("[Nimic de transcris]")