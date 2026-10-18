"""
Barge-in — Întreruperea lui Jarvis din Vorbire (Task 6.10)

Rulează CONCURENT cu tts.spune(), ascultând microfonul (prin
echo-cancel-source, dacă modulul PipeWire de anulare ecou e configurat —
vezi audio_io.py) pentru detectarea vorbirii utilizatorului ÎN TIMP CE
Jarvis vorbește. Dacă detectează vorbire reală (nu ecoul propriei voci a
lui Jarvis), oprește TTS-ul instant prin tts.opreste().

De ce funcționează fiabil ACUM (spre deosebire de o variantă fără anulare
ecou): DEVICE_CAPTURA_MIC (echo-cancel-source) elimină din semnalul captat
partea corelată cu ce redă Jarvis prin DEVICE_REDARE_DIFUZOARE
(echo-cancel-sink) — microfonul "aude" mult mai puțin din vocea proprie a
lui Jarvis, deci pragul de detecție poate rămâne aproape de cel normal.

Chiar și așa, păstrăm un prag puțin mai strict și cerem câteva blocuri
consecutive de confirmare, ca plasă suplimentară de siguranță — anularea
de ecou reduce mult problema, dar nu o elimină 100% (calibrare, distanță
mic-difuzoare, volum etc. pot afecta cât de bine anulează).

Utilizare (din audio_loop.py):
    from src.core.barge_in import vorbeste_cu_intrerupere
    vorbeste_cu_intrerupere("Textul de rostit.", pe_intrerupere=callback_opțional)
"""

import threading
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from src.core.audio_io import SAMPLE_RATE, DEVICE_CAPTURA_MIC, safe_input_stream
from src.core.vad import obtine_detector
from src.core.tts import spune, opreste as opreste_tts

MARIME_BLOC = 512  # identic cu vad.py — 32ms la 16kHz, cerut de Silero

# Prag de probabilitate pentru barge-in — puțin mai strict decât VAD-ul
# normal de conversație (0.5), ca plasă suplimentară peste anularea de ecou.
PRAG_BARGE_IN = 0.65

# Câte blocuri consecutive de "vorbire" înainte să chiar întrerupem —
# evită întreruperi pe un singur zgomot scurt/reziduu de ecou.
BLOCURI_CONFIRMARE = 4  # ~128ms de vorbire continuă confirmată


def _asculta_pentru_intrerupere(stop_flag: threading.Event, a_intrerupt: threading.Event):
    """
    Rulează într-un thread separat, CONCURENT cu tts.spune(). Ascultă
    microfonul (DEVICE_CAPTURA_MIC) în blocuri scurte; dacă detectează
    BLOCURI_CONFIRMARE blocuri consecutive peste PRAG_BARGE_IN, oprește
    TTS-ul (tts.opreste()) și setează `a_intrerupt`, ca apelantul să știe
    că a fost o întrerupere reală, nu doar sfârșitul normal al redării.

    Se oprește singur când `stop_flag` e setat (redarea s-a terminat
    normal) — nu rămâne să asculte după ce Jarvis a tăcut oricum.
    """
    detector = obtine_detector()
    blocuri_consecutive = 0

    try:
        # Determinăm rata nativă ȘI numărul real de canale ale device-ului
        # de captură (echo-cancel-source e STEREO — 2 canale — pe sistemul
        # tău, confirmat cu `pactl list sources short`). Citim blocuri la
        # acea rată/canale, apoi re-eșantionăm/downmixăm noi manual către
        # ce cere Silero (MARIME_BLOC = 512 la 16kHz, mono).
        try:
            if isinstance(DEVICE_CAPTURA_MIC, int):
                dev_idx = DEVICE_CAPTURA_MIC
            else:
                dev_idx = sd.default.device[0]

            info = sd.query_devices(dev_idx)
            rata_nativa = int(info.get("default_samplerate", SAMPLE_RATE))
            canale_native = max(1, int(info.get("max_input_channels", 1) or 1))
        except Exception:
            # Dacă ceva e în neregulă, cădem înapoi la valorile implicite
            rata_nativa = SAMPLE_RATE
            canale_native = 1

        durata_bloc_sec = MARIME_BLOC / SAMPLE_RATE
        bloc_nativ = max(1, int(round(durata_bloc_sec * rata_nativa)))

        # IMPORTANT: cerem EXPLICIT canale_native (nu 1 fix). Forțarea
        # channels=1 pe un device stereo (echo-cancel-source, 2ch) obligă
        # PortAudio/PipeWire să facă un downmix intern care, pe acest
        # sistem, scrie dincolo de bufferul numpy alocat pentru citire —
        # exact sursa lui `malloc(): unsorted double linked list corrupted`.
        # Downmix-ul îl facem noi, manual, mai jos — sigur, în numpy pur.
        with safe_input_stream(
            samplerate=rata_nativa,
            channels=canale_native,
            dtype="float32",
            blocksize=bloc_nativ,
            device=DEVICE_CAPTURA_MIC,
        ) as stream:
            while not stop_flag.is_set():
                bloc, _ = stream.read(bloc_nativ)

                # Downmix manual la mono (medie pe canale) dacă e nevoie —
                # sigur, fără nicio implicare a PortAudio în conversie.
                if bloc.ndim == 2 and bloc.shape[1] > 1:
                    bloc_flat = bloc.mean(axis=1).astype(np.float32)
                else:
                    bloc_flat = bloc.flatten()

                # Dacă trebuie, re-eșantionăm către MARIME_BLOC (512 la 16kHz)
                if bloc_nativ != MARIME_BLOC or rata_nativa != SAMPLE_RATE:
                    try:
                        res = resample_poly(bloc_flat, MARIME_BLOC, bloc_nativ)
                    except Exception:
                        # Fallback la interpolare liniară foarte simplă dacă resample_poly eșuează
                        res = np.interp(
                            np.linspace(0, len(bloc_flat) - 1, MARIME_BLOC),
                            np.arange(len(bloc_flat)),
                            bloc_flat,
                        )

                    # Asigurăm lungimea exactă
                    if res.shape[0] > MARIME_BLOC:
                        bloc_proc = res[:MARIME_BLOC].astype(np.float32)
                    elif res.shape[0] < MARIME_BLOC:
                        bloc_proc = np.pad(res, (0, MARIME_BLOC - res.shape[0]), mode="constant").astype(np.float32)
                    else:
                        bloc_proc = res.astype(np.float32)
                else:
                    bloc_proc = bloc_flat

                probabilitate = detector.probabilitate_vorbire(bloc_proc)

                if probabilitate >= PRAG_BARGE_IN:
                    blocuri_consecutive += 1
                else:
                    blocuri_consecutive = 0

                if blocuri_consecutive >= BLOCURI_CONFIRMARE:
                    print("\n[Barge-in] Vorbire detectată — întrerup Jarvis.")
                    opreste_tts()
                    a_intrerupt.set()
                    return
    except Exception as e:
        # Nu blocăm TTS-ul normal dacă ascultarea de barge-in eșuează
        # (ex: device ocupat) — Jarvis tot vorbește, doar nu poate fi
        # întrerupt vocal în tura asta.
        print(f"[Barge-in] Ascultare indisponibilă: {str(e)[:150]}")


def vorbeste_cu_intrerupere(text: str, pe_intrerupere=None) -> bool:
    """
    Rostește `text` prin tts.spune(), în timp ce ascultă concurent pentru
    barge-in. Dacă utilizatorul vorbește peste Jarvis, redarea se oprește
    imediat.

    Parametri:
        text:           textul de rostit
        pe_intrerupere: callback() opțional, apelat dacă a avut loc o
                        întrerupere (util ca apelantul să știe să oprească
                        și generarea restului răspunsului, nu doar redarea)

    Returnează:
        True dacă a fost întreruptă redarea, False dacă a rulat complet.
    """
    stop_flag = threading.Event()
    a_intrerupt = threading.Event()

    thread_ascultare = threading.Thread(
        target=_asculta_pentru_intrerupere,
        args=(stop_flag, a_intrerupt),
        daemon=True,
    )
    thread_ascultare.start()

    try:
        spune(text)
    except RuntimeError as e:
        print(f"[EROARE TTS]: {e}")
    finally:
        stop_flag.set()
        # FĂRĂ timeout: trebuie să știm SIGUR că thread-ul a ieșit din
        # bucla lui înainte să continuăm la propoziția următoare — altfel
        # două thread-uri de barge-in pot rula concurent, apelând modelul
        # VAD partajat în paralel (vezi lock-ul nou din vad.py). Thread-ul
        # verifică stop_flag după fiecare bloc citit, deci iese rapid
        # (sub-secundă) — nu există risc real de blocare permanentă aici.
        thread_ascultare.join()

    if a_intrerupt.is_set():
        if pe_intrerupere:
            try:
                pe_intrerupere()
            except Exception as e:
                print(f"[Barge-in] Eroare în callback pe_intrerupere: {e}")
        return True

    return False