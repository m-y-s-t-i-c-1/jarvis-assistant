"""
Rulează bucla vocală cu microfon, STT și TTS simulate: verifică reflexul (fără LLM) și faptul că,
la barge-in, în istoric rămâne doar ce a fost rostit. Modulele grele (audio, torch, chromadb) sunt
înlocuite cu stub-uri doar pe durata testului.
"""
import importlib
import sys
import types as pytypes
from unittest.mock import MagicMock

import pytest

_STUBURI = ["sounddevice", "torch", "faster_whisper", "openwakeword", "openwakeword.model", "chromadb",
            "chromadb.utils", "chromadb.utils.embedding_functions", "sentence_transformers", "scipy",
            "scipy.signal", "silero_vad"]


@pytest.fixture
def audio_loop(monkeypatch):
    inainte = set(sys.modules)
    for nume in _STUBURI:
        if nume not in sys.modules:
            monkeypatch.setitem(sys.modules, nume, MagicMock())
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    try:
        modul = importlib.import_module("src.core.audio_loop")
    except Exception as e:  # dependență lipsă în mediul curent
        _curata(inainte)
        pytest.skip(f"audio_loop nu se poate importa aici: {e}")
    yield modul
    _curata(inainte)


# Modulele construite pe stub-uri (audio, torch, chromadb). Doar pe acestea le scoatem după test: restul
# (ex. src.tools) își înregistrează unelte la import, iar o a doua importare ar ridica „unealtă duplicată".
_PE_STUB = {
    "src.core.audio_loop", "src.core.audio_io", "src.core.audio_player", "src.core.vad", "src.core.stt",
    "src.core.tts", "src.core.barge_in", "src.core.rag", "src.core.context_rag", "src.core.consolidare",
    "src.core.wake_word",
}


def _curata(inainte):
    for nume in (set(sys.modules) - inainte) & _PE_STUB:
        sys.modules.pop(nume, None)


def _ruleaza(audio_loop, monkeypatch, enunturi, nvidia="", intrerupe_la=None):
    """Simulează ture vocale; întoarce (istoric, propoziții rostite, apeluri salveaza_tura, apeluri NVIDIA)."""
    rostite, salvari, apeluri_nvidia = [], [], []

    class _Detector:
        def __init__(self):
            self.i = -1

        def asculta_pana_la_pauza(self):
            self.i += 1
            return f"audio{self.i}"

    transcrieri = {f"audio{i}": t for i, t in enumerate(enunturi)}

    def vorbeste(propozitie):
        rostite.append(propozitie)
        return intrerupe_la is not None and len(rostite) >= intrerupe_la   # True = utilizatorul a întrerupt

    monkeypatch.setattr(audio_loop, "obtine_detector", lambda: _Detector())
    monkeypatch.setattr(audio_loop, "transcrie", lambda a: transcrieri.get(a, "stop"))
    monkeypatch.setattr(audio_loop, "vorbeste_cu_intrerupere", vorbeste)
    monkeypatch.setattr(audio_loop, "spune", lambda t: None)
    monkeypatch.setattr(audio_loop, "prompt_pentru_tura", lambda *a, **k: "prompt")
    monkeypatch.setattr(audio_loop, "salveaza_tura", lambda *a, **k: salvari.append(a))
    monkeypatch.setattr(audio_loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(audio_loop, "intreaba_nvidia_conversatie",
                        lambda m: apeluri_nvidia.append(m) or nvidia)

    def _nu_chema_gemini(*a, **k):
        raise AssertionError("Gemini chemat, deși tura trebuia rezolvată fără el")

    monkeypatch.setattr(audio_loop, "_agent_streaming_cu_fallback_chei", _nu_chema_gemini)
    # „conversație" -> NVIDIA; routerul real decide, deci folosim fraze fără cuvinte de unelte
    istoric = []
    audio_loop._porneste_bucla_audio_interna(istoric, iter([MagicMock()]), "m", "sesiune-test", "")
    return istoric, rostite, salvari, apeluri_nvidia


def test_tura_reflex_nu_chema_niciun_model(audio_loop, monkeypatch):
    istoric, rostite, salvari, nvidia = _ruleaza(audio_loop, monkeypatch, ["cât e ceasul?"])
    assert len(rostite) == 1 and rostite[0].startswith("Este ora ")
    assert nvidia == []
    assert [c.role for c in istoric] == ["user", "model"]
    # salvată în DB/RAG, dar FĂRĂ client => fără extragere de amintiri (nu consumă cotă)
    assert len(salvari) == 1 and salvari[0][3] is None


def test_barge_in_pastreaza_in_istoric_doar_ce_s_a_rostit(audio_loop, monkeypatch):
    raspuns = "Prima propoziție. A doua propoziție. A treia propoziție."
    istoric, rostite, salvari, _ = _ruleaza(
        audio_loop, monkeypatch, ["salut, ce mai faci"], nvidia=raspuns, intrerupe_la=2)
    assert rostite == ["Prima propoziție.", "A doua propoziție."]     # a treia nu se mai rostește
    assert istoric[-1].role == "model"
    assert istoric[-1].parts[0].text == "Prima propoziție. A doua propoziție."
    assert salvari and salvari[0][2] == "Prima propoziție. A doua propoziție."


def test_fara_barge_in_tot_raspunsul_ramane(audio_loop, monkeypatch):
    raspuns = "Prima propoziție. A doua propoziție."
    istoric, rostite, _, _ = _ruleaza(audio_loop, monkeypatch, ["salut, ce mai faci"], nvidia=raspuns)
    assert rostite == ["Prima propoziție.", "A doua propoziție."]
    assert istoric[-1].parts[0].text == raspuns
