"""
Înregistrarea Inteligentă — VAD (Task 3.2) + suport Barge-in (Task 6.10)

Folosește Silero-VAD pentru a detecta automat când utilizatorul vorbește
și când tace, fără apăsare de taste (push-to-talk). Ascultă continuu pe
blocuri scurte de audio și acumulează vorbirea într-un buffer, până
detectează o pauză suficient de lungă — moment în care consideră că
utilizatorul a terminat și returnează audio-ul complet capturat.

Pe scurt, fluxul e:
    1. Ascultă în blocuri mici (32ms fiecare, recomandat de Silero)
    2. Pentru fiecare bloc, Silero spune: vorbire sau tăcere
    3. Cât timp e vorbire, acumulăm blocurile într-un buffer
    4. Când apare o pauză mai lungă decât PRAG_TACERE_SECUNDE, considerăm
       enunțul terminat și returnăm tot ce s-a acumulat

Acest modul NU face transcriere (asta e Task 3.3, Whisper) — doar decide
CÂND să trimită audio-ul către transcriere.

Task 6.10 — barge-in: expune și `probabilitate_vorbire()`, care returnează
scorul brut Silero (0-1), fără prag aplicat. Folosit de src/core/barge_in.py
ca să aplice un prag PROPRIU, mai strict decât cel de conversație normală
(PRAG_PROBABILITATE_VORBIRE), pentru a reduce fals-pozitivele din ecoul
acustic al lui Jarvis în timpul redării TTS.
"""

import numpy as np
import sounddevice as sd
import torch

from src.core.audio_io import SAMPLE_RATE, safe_input_stream

# Silero-VAD funcționează nativ pe blocuri de 512 samples la 16kHz (32ms)
MARIME_BLOC = 512

# Cât timp de tăcere continuă (în secunde) înseamnă "a terminat de vorbit"
PRAG_TACERE_SECUNDE = 0.8

# Praguri de probabilitate Silero — peste asta considerăm că e vorbire
PRAG_PROBABILITATE_VORBIRE = 0.5

# Durată maximă a unui enunț, ca plasă de siguranță (evită buffer infinit
# dacă cineva vorbește foarte mult fără pauză)
DURATA_MAXIMA_SECUNDE = 30


class DetectorActivitateVocala:
    """
    Wrapper peste modelul Silero-VAD, cu logică de acumulare a unui
    enunț complet (de la început de vorbire până la pauză semnificativă).
    """

    def __init__(self):
        # Încărcăm modelul Silero o singură dată, la inițializare —
        # e mic (~1MB) și rapid, dar nu vrem să-l reîncărcăm la fiecare apel
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.model.eval()

    def probabilitate_vorbire(self, bloc_audio: np.ndarray) -> float:
        """
        Returnează probabilitatea BRUTĂ de vorbire (0.0-1.0) pentru un bloc
        audio, FĂRĂ niciun prag aplicat — decizia "e vorbire sau nu" rămâne
        la apelant. Folosit de barge_in.py, care aplică un prag propriu,
        mai strict decât PRAG_PROBABILITATE_VORBIRE de mai jos.
        """
        tensor = torch.from_numpy(bloc_audio).float()
        with torch.no_grad():
            return self.model(tensor, SAMPLE_RATE).item()

    def _e_vorbire(self, bloc_audio: np.ndarray) -> bool:
        """Verifică dacă un singur bloc de audio conține vorbire (prag standard)."""
        return self.probabilitate_vorbire(bloc_audio) >= PRAG_PROBABILITATE_VORBIRE

    def asculta_pana_la_pauza(self) -> np.ndarray | None:
        """
        Ascultă continuu de la microfon până detectează un enunț complet:
        începe să acumuleze de la primul bloc cu vorbire, și se oprește
        după PRAG_TACERE_SECUNDE de tăcere continuă.

        Returnează:
            numpy array cu audio-ul enunțului complet, sau None dacă
            nu s-a detectat nicio vorbire (timeout de siguranță).
        """
        buffer_audio = []
        blocuri_tacere_consecutive = 0
        a_inceput_vorbirea = False

        # Câte blocuri de tăcere consecutive înseamnă "pauză suficientă"
        blocuri_pentru_pauza = int(
            (PRAG_TACERE_SECUNDE * SAMPLE_RATE) / MARIME_BLOC
        )
        blocuri_maxime = int(
            (DURATA_MAXIMA_SECUNDE * SAMPLE_RATE) / MARIME_BLOC
        )

        print("[Ascult... vorbește când ești gata]")

        # Folosim device-ul implicit al sistemului (pipewire/resampler),
        # pentru că VAD/Wake-word ascultă doar când Jarvis NU vorbește
        # și nu are nevoie de anulare de ecou dedicată.
        with safe_input_stream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=MARIME_BLOC,
            device=None,
        ) as stream:

            for _ in range(blocuri_maxime):
                bloc, _ = stream.read(MARIME_BLOC)
                bloc_flat = bloc.flatten()

                vorbire = self._e_vorbire(bloc_flat)

                if vorbire:
                    if not a_inceput_vorbirea:
                        print("[Vorbire detectată — ascult]")
                        a_inceput_vorbirea = True
                    buffer_audio.append(bloc_flat)
                    blocuri_tacere_consecutive = 0

                elif a_inceput_vorbirea:
                    # A început să vorbească, dar acum tace — numărăm tăcerea
                    buffer_audio.append(bloc_flat)  # păstrăm și tăcerea scurtă, sună natural
                    blocuri_tacere_consecutive += 1

                    if blocuri_tacere_consecutive >= blocuri_pentru_pauza:
                        print("[Pauză detectată — enunț complet]")
                        break

                # Dacă nu a început încă vorbirea și tot tăcere e, continuăm
                # să așteptăm, fără să acumulăm nimic (evită buffer gol lung)

        if not buffer_audio:
            print("[Nicio vorbire detectată]")
            return None

        return np.concatenate(buffer_audio).reshape(-1, 1)


# Instanță globală — modelul se încarcă o singură dată la primul import folosit
_detector: DetectorActivitateVocala | None = None


def obtine_detector() -> DetectorActivitateVocala:
    """Returnează instanța globală de VAD, inițializând-o lazy la prima cerere."""
    global _detector
    if _detector is None:
        print("[Încărcare model Silero-VAD...]")
        _detector = DetectorActivitateVocala()
        print("[Model VAD încărcat]")
    return _detector


if __name__ == "__main__":
    # Test rapid: ascultă un enunț și redă-l înapoi, ca să verifici
    # că taie corect la început/sfârșit de vorbire.
    from src.core.audio_io import reda

    detector = obtine_detector()
    audio = detector.asculta_pana_la_pauza()

    if audio is not None:
        print(f"[Enunț capturat: {len(audio) / SAMPLE_RATE:.1f} secunde — redare...]")
        reda(audio)
    else:
        print("[Nimic de redat]")