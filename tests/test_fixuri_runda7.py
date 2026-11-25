"""Teste pentru runda 7: lansare aplicații, joburi fără shell, mesaje de eroare git."""
import pytest

from src.tools import developer, joburi, sistem


# ── deschide_aplicatie ────────────────────────────────────────────────────────
@pytest.fixture
def fara_config(monkeypatch):
    for alias in sistem.ALIAS_APLICATII:
        monkeypatch.delenv(f"JARVIS_APP_{alias.upper()}", raising=False)
    monkeypatch.setattr(sistem, "valoare", lambda s, c, i="", env=None: __import__("os").getenv(env or "", i))


def test_alege_prima_aplicatie_instalata(monkeypatch, fara_config):
    instalate = {"firefox", "kitty"}
    monkeypatch.setattr(sistem.shutil, "which", lambda c: f"/usr/bin/{c}" if c in instalate else None)
    assert sistem._alege_aplicatie("browser")[0] == ["firefox"]     # vivaldi lipsește -> următorul instalat
    assert sistem._alege_aplicatie("terminal")[0] == ["kitty"]
    assert sistem._alege_aplicatie("editor")[0] is None


def test_configurarea_explicita_are_prioritate(monkeypatch, fara_config):
    monkeypatch.setattr(sistem.shutil, "which", lambda c: f"/usr/bin/{c}")
    monkeypatch.setenv("JARVIS_APP_BROWSER", "flatpak run org.mozilla.firefox")
    assert sistem._alege_aplicatie("browser")[0] == ["flatpak", "run", "org.mozilla.firefox"]
    monkeypatch.setattr(sistem.shutil, "which", lambda c: None)
    assert sistem._alege_aplicatie("browser")[0] is None            # configurat, dar neinstalat


def test_deschide_aplicatie_lanseaza_binarul_gasit(monkeypatch, fara_config):
    lansate = []
    monkeypatch.setattr(sistem.shutil, "which", lambda c: f"/usr/bin/{c}" if c == "firefox" else None)
    monkeypatch.setattr(sistem.subprocess, "Popen", lambda args, **k: lansate.append(args))
    assert sistem.deschide_aplicatie("Browser") == "Am lansat 'firefox'."
    assert lansate == [["firefox"]]


def test_deschide_aplicatie_fara_nicio_aplicatie_explica_cum_se_seteaza(monkeypatch, fara_config):
    monkeypatch.setattr(sistem.shutil, "which", lambda c: None)
    monkeypatch.setattr(sistem.subprocess, "Popen", lambda *a, **k: pytest.fail("nu trebuie lansat nimic"))
    raspuns = sistem.deschide_aplicatie("terminal")
    assert "JARVIS_APP_TERMINAL" in raspuns and "alacritty" in raspuns


@pytest.mark.parametrize("functie", ["deschide_aplicatie", "ruleaza_comanda_info"])
def test_alias_none_nu_arunca(functie):
    assert "necunoscut" in getattr(sistem, functie)(None).lower()


# ── joburi: operatorii de shell nu se pierd în tăcere ────────────────────────
@pytest.fixture
def manager_interzis(monkeypatch):
    pornite = []
    monkeypatch.setattr(joburi.manager_joburi, "porneste_job",
                        lambda parti, desc, cwd=".": pornite.append(parti) or "id1")
    return pornite


@pytest.mark.parametrize("comanda", [
    "npm install && npm run build", "make || echo eșuat", "cd src; make", "cat log | grep eroare",
    "echo salut > fisier.txt", "sleep 5 &", "a&&b",
])
def test_job_cu_operatori_de_shell_e_respins(manager_interzis, comanda):
    raspuns = joburi.porneste_job_fundal(comanda, "test")
    assert "operatori de shell" in raspuns
    assert manager_interzis == []          # nimic nu a pornit


@pytest.mark.parametrize("comanda", [
    "npm run build", 'git commit -m "a && b"', 'curl "http://x.ro/?a=1&b=2"', "python script.py --opt=x",
])
def test_job_normal_porneste(manager_interzis, comanda):
    assert "Job pornit" in joburi.porneste_job_fundal(comanda, "test")
    assert len(manager_interzis) == 1


# ── git: directorul inexistent nu mai pare „git lipsă" ───────────────────────
def test_git_director_inexistent():
    raspuns = developer.git_status("/tmp/nu_exista_deloc_123")
    assert "nu există" in raspuns and "instalat" not in raspuns
