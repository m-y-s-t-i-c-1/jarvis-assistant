"""
Teste pentru a doua rundă de corecturi (router, blacklist, git safety, NVIDIA, calendar, UI confirm).
Rulează fără microfon, rețea sau chei API:  python -m pytest tests -q
"""
import asyncio
import subprocess
import threading

import pytest

from src.core import security
from src.core.router import clasifica


# ── router: întrebări cu unealtă nu mai cad pe „conversatie" ──────────────────
@pytest.mark.parametrize("text", [
    "ce fișiere am în director", "care e adresa mea IP", "ce versiune de kernel rulez",
    "cine sunt pe sistem", "ce procese consumă cel mai mult", "ce face jobul abc123",
])
def test_router_trimite_pe_unelte(text):
    assert clasifica(text) == "unelte"


@pytest.mark.parametrize("text", ["povestește-mi despre Ștefan cel Mare", "ce e un program", "explică-mi fotosinteza"])
def test_router_nu_exagereaza(text):
    assert clasifica(text) == "conversatie"


# ── blacklist: text legitim nu mai e blocat ───────────────────────────────────
def test_scrie_text_cu_reboot_in_propozitie_trece():
    ok, _ = security.verifica_blacklist("scrie_text", {"text": "trebuie să dau reboot la router"})
    assert ok is True


@pytest.mark.parametrize("text", ["reboot", "sudo reboot", "ls; shutdown -h now", "systemctl poweroff"])
def test_scrie_text_cu_comanda_de_oprire_ramane_blocat(text):
    ok, _ = security.verifica_blacklist("scrie_text", {"text": text})
    assert ok is False


def test_scrie_text_pastreaza_restul_blacklist():
    ok, _ = security.verifica_blacklist("scrie_text", {"text": "rm -rf /"})
    assert ok is False


def test_deschide_vscode_halt_py():
    ok, _ = security.verifica_blacklist("deschide_vscode", {"cale": "halt.py"})
    assert ok is True


def test_oprirea_sistemului_ramane_blocata_la_comenzi():
    ok, _ = security.verifica_blacklist("ruleaza_comanda", {"comanda": "reboot"})
    assert ok is False


def test_python_c_al_scriptului_nu_e_blocat_dar_python_c_da():
    ok, _ = security.verifica_blacklist("porneste_job_fundal", {"comanda": "python build.py -c 'cfg'"})
    assert ok is True
    ok, _ = security.verifica_blacklist("porneste_job_fundal", {"comanda": "python -c 'import os'"})
    assert ok is False
    ok, _ = security.verifica_blacklist("porneste_job_fundal", {"comanda": "python3 -u -c \"print(1)\""})
    assert ok is False


# ── git safety net ────────────────────────────────────────────────────────────
def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_checkpoint_include_fisiere_untracked(repo, monkeypatch, tmp_path):
    from src.core import git_safety
    monkeypatch.setattr(git_safety, "LOG_CHECKPOINT", tmp_path / "log.txt")
    (repo / "nou.txt").write_text("nou\n")
    h = git_safety.creeaza_checkpoint("test", str(repo))
    assert h
    assert "nou.txt" in _git(repo, "ls-tree", "-r", "--name-only", h)
    # nu atinge working tree-ul / indexul real
    assert "?? nou.txt" in _git(repo, "status", "--short")


def test_checkpoint_gol_cand_arborele_e_curat(repo, monkeypatch, tmp_path):
    from src.core import git_safety
    monkeypatch.setattr(git_safety, "LOG_CHECKPOINT", tmp_path / "log.txt")
    assert git_safety.creeaza_checkpoint("test", str(repo)) == ""


def test_checkpoint_foloseste_cale_repo_din_argumente(repo, monkeypatch, tmp_path):
    from src.core import git_safety
    apelat = {}
    monkeypatch.setattr(security, "creeaza_checkpoint", lambda motiv, cale_repo=".": apelat.update(cale=cale_repo))
    monkeypatch.setattr(security, "cere_confirmare", lambda *a, **k: True)
    ok, _ = security.trece_prin_securitate("git_commit", {"mesaj": "x", "cale_repo": str(repo)}, True)
    assert ok and apelat["cale"] == str(repo)


# ── NVIDIA: răspuns tăiat ─────────────────────────────────────────────────────
def test_nvidia_detecteaza_finish_reason_length():
    from src.core.llm_provider import _a_fost_taiat, _proceseaza_stream, _MESAJ_TAIAT
    assert _a_fost_taiat({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
    assert not _a_fost_taiat({"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]})
    linii = [
        b'data: {"choices":[{"delta":{"content":"salut"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
        b"data: [DONE]",
    ]
    assert _proceseaza_stream(iter(linii)) == "salut" + _MESAJ_TAIAT


# ── voce: fallback între chei ─────────────────────────────────────────────────
def test_voce_incearca_urmatoarea_cheie(monkeypatch):
    try:
        from src.core import audio_loop
    except Exception as e:  # sounddevice/PortAudio/torch lipsă (ex. CI fără audio)
        pytest.skip(f"audio_loop nu se poate importa aici: {e}")

    class Eroare429(Exception):
        code = 429

    apeluri = []

    def fals(client, model, prompt, istoric, **kw):
        apeluri.append(client)
        if client == "k1":
            raise Eroare429("RESOURCE_EXHAUSTED")
        return "ok"

    monkeypatch.setattr(audio_loop, "agent_loop_streaming", fals)
    monkeypatch.setattr(audio_loop.time, "sleep", lambda s: None)
    rot = audio_loop._RotatieClienti(["k1", "k2"])
    istoric = ["u"]
    r = audio_loop._agent_streaming_cu_fallback_chei(rot, "m", "p", istoric, 0, lambda p: None, None)
    assert r == "ok" and apeluri == ["k1", "k2"]


# ── server: confirmări prin WebSocket ─────────────────────────────────────────
def test_server_confirmare_prin_ws_aproba_si_refuza():
    pytest.importorskip("fastapi")
    try:
        from src.ui import server
    except Exception as e:  # dependențe grele lipsă (chromadb etc.)
        pytest.skip(f"server.py nu se poate importa aici: {e}")

    mesaje = []

    async def fals_broadcast(m):
        mesaje.append(m)
        id_conf = m["id"]
        server._confirmari[id_conf]["aprobat"] = m["functie"] == "git_commit"
        server._confirmari[id_conf]["event"].set()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    server._loop_principal = loop
    server.conexiuni_active.append(object())
    original = server.broadcast
    server.broadcast = fals_broadcast
    try:
        assert server._confirma_prin_ws("git_commit", {"mesaj": "x"}) is True
        assert server._confirma_prin_ws("git_push", {}) is False
        assert not server._confirmari
    finally:
        server.broadcast = original
        server.conexiuni_active.clear()
        server._loop_principal = None
        loop.call_soon_threadsafe(loop.stop)


def test_server_refuza_fara_client_conectat():
    try:
        from src.ui import server
    except Exception as e:
        pytest.skip(f"server.py nu se poate importa aici: {e}")
    server.conexiuni_active.clear()
    assert server._confirma_prin_ws("git_push", {}) is False
