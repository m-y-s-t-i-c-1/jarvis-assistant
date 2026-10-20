"""
Teste de regresie pentru bug-urile corectate în rundele de revizuire.
Rulează fără microfon, rețea sau chei API:  python -m pytest tests -q
"""
import threading

import pytest

from src.core import security
from src.core.confirmare_vocala import interpreteaza_raspuns
from src.core.router import clasifica


# ── security: blacklist rm (bypass cu -vrf / -v -rf) ─────────────────────────
@pytest.mark.parametrize("comanda", [
    "rm -rf /", "rm -vrf x", "rm -v -rf x", "rm -iR d", "rm -f a", "/bin/rm -rf x",
])
def test_blacklist_blocheaza_rm_periculos(comanda):
    ok, _ = security.verifica_blacklist("ruleaza_comanda", {"comanda": comanda})
    assert ok is False


@pytest.mark.parametrize("comanda", ["rm file.txt", "rm -v file", "confirm -f", "perform -rf"])
def test_blacklist_lasa_comenzi_inofensive(comanda):
    ok, _ = security.verifica_blacklist("ruleaza_comanda", {"comanda": comanda})
    assert ok is True


# ── confirmare vocală ────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,asteptat", [
    ("Da.", True), ("da, te rog", True), ("Fă-o!", True), ("Desigur, execută", True),
    ("Nu", False), ("nu, stop", False), ("anulează", False),
    ("Da, dar nu acum", False),            # negația bate afirmația
    ("", None), ("poate", None),
    ("da da da da da da da da da da", None),  # prea lung = neclar
])
def test_interpretare_raspuns_vocal(text, asteptat):
    assert interpreteaza_raspuns(text) is asteptat


def test_canal_confirmare_este_per_thread():
    rezultate = {}

    def in_thread():
        with security.canal_confirmare(lambda n, a: True):
            rezultate["voce"] = security.cere_confirmare("f", {})

    t = threading.Thread(target=in_thread)
    t.start(); t.join()
    # thread-ul principal nu are canal și (în pytest) nici terminal: fail closed
    rezultate["text"] = security.cere_confirmare("f", {})
    assert rezultate == {"voce": True, "text": False}


def test_canal_confirmare_cu_eroare_refuza():
    def stricat(n, a):
        raise RuntimeError("microfon")

    with security.canal_confirmare(stricat):
        assert security.cere_confirmare("f", {}) is False


# ── registry: INTEGER primit ca float ────────────────────────────────────────
def test_registry_converteste_float_intreg():
    from src.core.registry import unealta, ruleaza_functie

    @unealta(description="test", parameters={"n": {"type": "INTEGER", "description": "n"}})
    def _tip_n(n):
        return type(n).__name__

    assert ruleaza_functie("_tip_n", {"n": 5.0})["rezultat"] == "int"
    assert ruleaza_functie("_tip_n", {"n": 5.5})["rezultat"] == "float"  # nu strică valorile reale


# ── database: același fapt nu se mai dublează ────────────────────────────────
def test_memorie_upsert_nu_dubleaza(tmp_path):
    from src.core.database import JarvisDB

    db = JarvisDB(tmp_path / "t.db")
    a = db.salveaza_memorie("fapt", "Vasea lucrează la Jarvis", "s1", 0.6)
    b = db.salveaza_memorie("fapt", "  vasea lucrează la jarvis ", "s2", 1.0)
    c = db.salveaza_memorie("preferinta", "Vasea lucrează la Jarvis", "s2", 1.0)
    assert a == b and a != c
    fapte = db.cauta_memorie(tip="fapt")
    assert len(fapte) == 1 and fapte[0]["relevanta"] == 1.0


# ── router ───────────────────────────────────────────────────────────────────
def test_router_clasifica_categorii_de_baza():
    assert clasifica("deschide firefox") == "unelte"
    assert clasifica("salut, ce mai faci?") == "conversatie"
