"""Teste pentru runda 6: monitorul de log nu lasă procese orfane."""
import subprocess
import threading
import time

from src.core import monitor_log


def _asteapta(conditie, timp=3.0):
    t0 = time.time()
    while time.time() - t0 < timp:
        if conditie():
            return True
        time.sleep(0.02)
    return False


def test_opreste_monitorizarea_termina_procesele_chiar_fara_linii_noi():
    # `sleep` nu scrie nimic pe stdout: înainte, thread-ul rămânea blocat și procesul orfan.
    proc_inainte = len(monitor_log._procese)
    t = threading.Thread(target=monitor_log._urmareste_sursa,
                         args=(["sleep", "30"], "test", lambda *a: None), daemon=True)
    t.start()
    assert _asteapta(lambda: len(monitor_log._procese) > proc_inainte)
    proces = monitor_log._procese[-1]
    assert proces.poll() is None

    monitor_log.opreste_monitorizarea_log()
    assert _asteapta(lambda: proces.poll() is not None), "procesul n-a fost oprit"
    assert _asteapta(lambda: not t.is_alive()), "thread-ul a rămas blocat"
    monitor_log._opreste_monitorizarea.clear()


def test_pornirea_dubla_nu_dubleaza_urmaririle(monkeypatch):
    porniri = []

    class _FalsProces:
        stdout = iter(())
        stderr = None
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fals_popen(comanda, **k):
        porniri.append(comanda[0])
        time.sleep(0.3)           # sursa „rulează" cât timp facem al doilea apel
        return _FalsProces()

    monkeypatch.setattr(subprocess, "Popen", fals_popen)
    monkeypatch.setattr(monitor_log, "_thread_uri", [])
    monitor_log.porneste_monitorizare_log(lambda *a: None)
    monitor_log.porneste_monitorizare_log(lambda *a: None)   # al doilea apel: ignorat
    assert _asteapta(lambda: len(porniri) >= 2)
    time.sleep(0.5)
    assert sorted(porniri) == ["dmesg", "journalctl"]
