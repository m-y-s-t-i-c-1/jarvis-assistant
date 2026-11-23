"""Serverul UI refuză cereri HTTP cu Host necunoscut (protecție împotriva DNS rebinding)."""
import importlib
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

_STUBURI = ["sounddevice", "torch", "faster_whisper", "chromadb", "chromadb.utils",
            "chromadb.utils.embedding_functions", "sentence_transformers"]
_PE_STUB = {"src.ui.server", "src.core.rag", "src.core.context_rag", "src.core.consolidare"}


@pytest.fixture
def server(monkeypatch):
    inainte = set(sys.modules)
    for nume in _STUBURI:
        if nume not in sys.modules:
            monkeypatch.setitem(sys.modules, nume, MagicMock())
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("JARVIS_PORT", "8080")
    monkeypatch.setenv("JARVIS_ALLOWED_ORIGINS", "http://192.168.1.5:8080")
    from src.core.database import db
    monkeypatch.setattr(db, "incepe_sesiune", lambda: "sesiune-test")   # nu poluăm baza de date reală
    try:
        modul = importlib.import_module("src.ui.server")
    except Exception as e:
        for nume in (set(sys.modules) - inainte) & _PE_STUB:
            sys.modules.pop(nume, None)
        pytest.skip(f"server nu se poate importa aici: {e}")
    yield modul
    for nume in (set(sys.modules) - inainte) & _PE_STUB:
        sys.modules.pop(nume, None)


@pytest.mark.parametrize("host", ["localhost:8080", "127.0.0.1:8080", "[::1]:8080", "LOCALHOST:8080",
                                  "192.168.1.5:8080"])   # ultimul vine din JARVIS_ALLOWED_ORIGINS
def test_host_permis(server, host):
    from fastapi.testclient import TestClient
    assert TestClient(server.app).get("/api/history", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example.com:8080", "evil.example.com", "localhost.evil.com:8080",
                                  "localhost:9999", "localhost", "", "10.0.0.7:8080"])
def test_host_nepermis_respins(server, host):
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    for ruta in ("/api/history", "/api/stats", "/"):
        raspuns = client.get(ruta, headers={"host": host})
        assert raspuns.status_code == 403, (ruta, host)
        assert "mesaje" not in raspuns.text


def test_websocket_ramane_functional_cu_origin_corect(server):
    from fastapi.testclient import TestClient
    with TestClient(server.app).websocket_connect("/ws", headers={"origin": "http://localhost:8080"}) as ws:
        assert ws.receive_json()["tip"] == "status"
