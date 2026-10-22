"""Teste pentru fixurile din runda 3 (blacklist, router, chei invalide, checkpoint, joburi)."""
import re
import threading
import pytest

from src.core import security
from src.core.router import clasifica


@pytest.mark.parametrize("nume,args", [
    ("git_add", {"fisiere": "halt.py"}),
    ("git_checkout", {"branch": "fix/shutdown-bug"}),
    ("git_push", {"remote": "origin", "branch": "poweroff-ui"}),
    ("porneste_server_http", {"folder": "/home/v/halt-site"}),
])
def test_blacklist_nu_blocheaza_nume_legitime(nume, args):
    assert security.verifica_blacklist(nume, args)[0] is True


@pytest.mark.parametrize("comanda", [
    "mkfs /dev/sdb1", "dd of=/dev/sda if=/dev/zero", "node -e 'x'",
    "python3 -c print(1)", "perl -e 'system(1)'",
])
def test_blacklist_prinde_ocolirile(comanda):
    ok, _ = security.verifica_blacklist("porneste_job_fundal", {"comanda": comanda, "descriere": "x"})
    assert ok is False


def test_blacklist_nu_blocheaza_python_script_cu_flag_c():
    ok, _ = security.verifica_blacklist("porneste_job_fundal", {"comanda": "python build.py -c cfg", "descriere": "x"})
    assert ok is True


@pytest.mark.parametrize("text", [
    "ce vreme va fi mâine în Iași", "caută pe Google ultimele știri",
    "dă mai tare", "cât de plină e memoria",
])
def test_router_trimite_pe_unelte(text):
    assert clasifica(text) == "unelte"


def test_router_nu_mai_prinde_procesare():
    assert clasifica("cum procesează creierul informația") != "unelte"


def test_checkpoint_foloseste_director_pentru_joburi(monkeypatch):
    apeluri = []
    monkeypatch.setattr(security, "cere_confirmare", lambda *a: True)
    monkeypatch.setattr(security, "creeaza_checkpoint", lambda motiv, cale_repo: apeluri.append(cale_repo))
    security.trece_prin_securitate("porneste_job_fundal", {"comanda": "ls", "descriere": "x", "director": "/tmp/proj"}, True)
    assert apeluri == ["/tmp/proj"]


@pytest.mark.parametrize("msg", [
    "400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.",
    "API_KEY_INVALID", "API key expired. Please renew the API key.",
])
def test_cheie_invalida_e_tratata_ca_temporara(msg):
    from src.core.subagenti import _eroare_temporara
    assert _eroare_temporara(Exception(msg)) is True


def test_eroare_400_obisnuita_ramane_fatala():
    from src.core.subagenti import _eroare_temporara
    assert _eroare_temporara(Exception("400 INVALID_ARGUMENT: bad request body")) is False


def test_job_se_termina_si_devine_neanuntat():
    from src.core import jobs
    m = jobs.JobManager()
    jid = m.porneste_job(["true"], "t")
    import time
    for _ in range(50):
        if m.joburi_neanuntate():
            break
        time.sleep(0.05)
    else:
        pytest.fail("jobul nu s-a terminat")
