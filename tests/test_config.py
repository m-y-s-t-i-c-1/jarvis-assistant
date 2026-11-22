"""Teste pentru faza 0: configurația centrală și eliminarea numelui hardcodat din cod."""
import ast
import importlib
from pathlib import Path

import pytest

from src.core import config

RADACINA = Path(__file__).resolve().parents[1]


# ── valoare(): prioritatea env > fișier > implicit ───────────────────────────
def test_valoare_prioritate(monkeypatch):
    monkeypatch.setattr(config, "_DATE", {"locatie": {"oras": "Iași"}})
    monkeypatch.delenv("JARVIS_ORAS", raising=False)
    assert config.valoare("locatie", "oras", "Chișinău", env="JARVIS_ORAS") == "Iași"      # fișier > implicit
    monkeypatch.setenv("JARVIS_ORAS", "București")
    assert config.valoare("locatie", "oras", "Chișinău", env="JARVIS_ORAS") == "București"  # env > fișier
    monkeypatch.setattr(config, "_DATE", {})
    monkeypatch.delenv("JARVIS_ORAS")
    assert config.valoare("locatie", "oras", "Chișinău", env="JARVIS_ORAS") == "Chișinău"   # implicit


def test_valoare_ignora_valori_goale(monkeypatch):
    monkeypatch.setattr(config, "_DATE", {"persona": {"utilizator": "   "}})
    monkeypatch.setenv("JARVIS_USER", "  ")
    assert config.valoare("persona", "utilizator", "Vasea", env="JARVIS_USER") == "Vasea"


def test_fisier_lipsa_sau_stricat_nu_opreste_pornirea(tmp_path):
    assert config._citeste(tmp_path / "nu_exista.toml") == {}
    stricat = tmp_path / "x.toml"
    stricat.write_text("[persona\nutilizator = ", encoding="utf-8")
    assert config._citeste(stricat) == {}
    bun = tmp_path / "ok.toml"
    bun.write_text('[persona]\nutilizator = "Ion"\n', encoding="utf-8")
    assert config._citeste(bun) == {"persona": {"utilizator": "Ion"}}


def test_personalizeaza_inlocuieste_jetonul(monkeypatch):
    monkeypatch.setattr(config, "UTILIZATOR", "Ion")
    assert config.personalizeaza('Te adresezi cu "{utilizator}".') == 'Te adresezi cu "Ion".'


# ── utilizatorul din env ajunge în prompturi ─────────────────────────────────
def test_jarvis_user_ajunge_in_prompturi(monkeypatch):
    from src.core import memory
    monkeypatch.setenv("JARVIS_USER", "Ion")
    importlib.reload(config)
    try:
        importlib.reload(memory)
        assert "numit Ion" in memory.PROMPT_EXTRACTIE and "Vasea" not in memory.PROMPT_EXTRACTIE
        # promptul rămâne formatabil (acoladele JSON sunt dublate)
        assert "Ion: salut" in memory.PROMPT_EXTRACTIE.format(mesaj_user="salut", raspuns_jarvis="x")
    finally:
        monkeypatch.delenv("JARVIS_USER")
        importlib.reload(config)
        importlib.reload(memory)


def test_prompturile_formatabile_pastreaza_acoladele():
    from src.core import context_manager, memory
    assert "{utilizator}" not in memory.PROMPT_EXTRACTIE
    assert "{utilizator}" not in context_manager.PROMPT_REZUMARE
    assert "x" in context_manager.PROMPT_REZUMARE.format(conversatie="x")


# ── vremea: orașul implicit vine din config ──────────────────────────────────
def test_vremea_foloseste_orasul_din_config(monkeypatch):
    import requests
    from src.tools import external
    vazut = {}

    def fals_get(url, params=None, timeout=None):
        vazut.update(params or {})
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(external.requests, "get", fals_get)
    monkeypatch.setenv("JARVIS_ORAS", "Iași")
    external.vremea("")
    assert vazut["name"] == "Iași"


# ── regresie: niciun „Vasea" hardcodat în șirurile executabile ───────────────
def _siruri_executabile(cale: Path):
    """Constantele șir din cod, fără docstring-uri și fără blocul `if __name__ == "__main__"` (demo)."""
    arbore = ast.parse(cale.read_text(encoding="utf-8"))
    docstringuri, de_sarit = set(), []
    for nod in ast.walk(arbore):
        if isinstance(nod, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if nod.body and isinstance(nod.body[0], ast.Expr) and isinstance(nod.body[0].value, ast.Constant):
                docstringuri.add(id(nod.body[0].value))
        if (isinstance(nod, ast.If) and isinstance(nod.test, ast.Compare)
                and isinstance(nod.test.left, ast.Name) and nod.test.left.id == "__name__"):
            de_sarit.append((nod.lineno, nod.end_lineno))
    for nod in ast.walk(arbore):
        if isinstance(nod, ast.Constant) and isinstance(nod.value, str) and id(nod) not in docstringuri:
            if any(a <= nod.lineno <= b for a, b in de_sarit):
                continue
            yield nod.lineno, nod.value


def test_niciun_nume_hardcodat_in_siruri_executabile():
    fisiere = [RADACINA / "main.py", *sorted((RADACINA / "src").rglob("*.py"))]
    gasite = [
        f"{f.relative_to(RADACINA)}:{linie}"
        for f in fisiere
        if f.name != "config.py"
        for linie, text in _siruri_executabile(f)
        if "Vasea" in text
    ]
    assert gasite == [], f"„Vasea\" hardcodat; folosește UTILIZATOR / {{utilizator}}: {gasite}"
