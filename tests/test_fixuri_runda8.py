"""Teste pentru runda 8: monitor de ecran oprit implicit, contor de ture, stderr în monitor_log, player TTS."""
import importlib
import subprocess
import sys
import threading
from unittest.mock import MagicMock

import pytest

from src.core import config, monitor_log, persistenta_tura


# ── monitorul de ecran: oprit implicit (confidențialitate) ───────────────────
@pytest.mark.parametrize("env,asteptat", [(None, False), ("0", False), ("false", False), ("", False),
                                          ("1", True), ("true", True), ("DA", True), ("on", True)])
def test_monitor_ecran_implicit_oprit(monkeypatch, env, asteptat):
    monkeypatch.setattr(config, "_DATE", {})
    if env is None:
        monkeypatch.delenv("JARVIS_MONITOR_ECRAN", raising=False)
    else:
        monkeypatch.setenv("JARVIS_MONITOR_ECRAN", env)
    assert config.monitor_ecran_activ() is asteptat


def test_monitor_ecran_din_fisierul_de_config(monkeypatch):
    monkeypatch.delenv("JARVIS_MONITOR_ECRAN", raising=False)
    monkeypatch.setattr(config, "_DATE", {"monitor": {"ecran": True}})
    assert config.monitor_ecran_activ() is True
    monkeypatch.setenv("JARVIS_MONITOR_ECRAN", "0")          # env-ul are prioritate
    assert config.monitor_ecran_activ() is False


# ── contorul de ture: nicio incrementare pierdută între thread-uri ───────────
def test_consolidarea_porneste_exact_la_fiecare_n_ture(monkeypatch):
    from src.core import database
    monkeypatch.setattr(database.db, "salveaza_mesaj", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "src.core.rag", MagicMock())
    monkeypatch.setitem(sys.modules, "src.core.memory", MagicMock())
    pornite = []
    monkeypatch.setattr(persistenta_tura, "_consolidare_in_fundal", lambda c, m: pornite.append(1))
    monkeypatch.setattr(persistenta_tura, "CONSOLIDARE_LA_N_TURE", 5)
    monkeypatch.setattr(persistenta_tura, "_ture", 0)
    vechi = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                      # forțează întrețeserea thread-urilor
    try:
        def lucrator():
            for _ in range(50):
                persistenta_tura.salveaza_tura("s", "întrebare", "răspuns", client=object())
        thread_uri = [threading.Thread(target=lucrator) for _ in range(8)]
        [t.start() for t in thread_uri]
        [t.join() for t in thread_uri]
    finally:
        sys.setswitchinterval(vechi)
    import time
    time.sleep(0.2)                                  # thread-urile de consolidare pornite în fundal
    assert persistenta_tura._ture == 400
    assert len(pornite) == 400 // 5


# ── monitor_log: stderr voluminos nu mai blochează monitorul ─────────────────
def test_stderr_voluminos_nu_blocheaza_monitorul(capfd):
    # 300 KB pe stderr (> bufferul unui pipe), apoi o linie pe stdout și ieșire cu eroare.
    cmd = ["sh", "-c", "head -c 300000 /dev/zero | tr '\\0' x >&2; echo 'kernel error: test'; exit 3"]
    primite = []
    t = threading.Thread(target=monitor_log._urmareste_sursa, args=(cmd, "test", lambda s, l: primite.append(l)),
                         daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "monitorul s-a blocat pe stderr"
    assert primite == ["kernel error: test"]
    assert "cod 3" in capfd.readouterr().out         # mesajul despre oprirea neașteptată a rămas


# ── TTS: un player oprit la jumătate nu se reia cu altul ─────────────────────
_STUBURI = ["sounddevice", "torch", "faster_whisper", "scipy", "scipy.signal"]


@pytest.fixture
def tts(monkeypatch):
    inainte = set(sys.modules)
    for nume in _STUBURI:
        if nume not in sys.modules:
            monkeypatch.setitem(sys.modules, nume, MagicMock())
    try:
        modul = importlib.import_module("src.core.tts")
    except Exception as e:
        for nume in (set(sys.modules) - inainte) & {"src.core.tts", "src.core.audio_io"}:
            sys.modules.pop(nume, None)
        pytest.skip(f"tts nu se poate importa aici: {e}")
    yield modul
    for nume in (set(sys.modules) - inainte) & {"src.core.tts", "src.core.audio_io"}:
        sys.modules.pop(nume, None)


class _Proc:
    def __init__(self, cod):
        self.returncode, self._poll = cod, 1

    def poll(self):
        if self._poll:
            self._poll -= 1
            return None
        return self.returncode

    def terminate(self): pass
    def wait(self, timeout=None): return self.returncode


def _scenariu(tts, monkeypatch, durata_player_1):
    lansate = []
    ceasuri = iter([0.0, durata_player_1, 100.0, 100.1])           # pornit_la, la eșec, apoi playerul 2
    monkeypatch.setattr(tts, "_construieste_comenzi_player", lambda c: [["p1"], ["p2"]])
    monkeypatch.setattr(tts.subprocess, "Popen", lambda cmd, **k: lansate.append(cmd[0]) or _Proc(1 if cmd[0] == "p1" else 0))
    monkeypatch.setattr(tts.time, "monotonic", lambda: next(ceasuri))
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)
    tts._stop_event.clear()
    return tts._reda_cu_player_extern("x.wav"), lansate


def test_player_oprit_la_jumatate_nu_e_inlocuit(tts, monkeypatch):
    rezultat, lansate = _scenariu(tts, monkeypatch, durata_player_1=2.0)   # a rulat 2 s: a redat ceva
    assert rezultat is True and lansate == ["p1"]


def test_player_care_esueaza_imediat_e_inlocuit(tts, monkeypatch):
    rezultat, lansate = _scenariu(tts, monkeypatch, durata_player_1=0.05)  # n-a putut porni (ex. sink lipsă)
    assert rezultat is True and lansate == ["p1", "p2"]
