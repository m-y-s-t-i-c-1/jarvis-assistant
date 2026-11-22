"""Teste pentru reflexul determinist (faza 3): cereri banale fără LLM."""
import pytest

from src.core import reflex

ORA = {"rezultat": "19:03, duminică, 22 noiembrie 2026 (22.11.2026)"}


@pytest.fixture
def unelte_false(monkeypatch):
    """Registru fals: înregistrează apelurile și întoarce rezultate fixe."""
    apeluri = []
    rezultate = {"get_ora_curenta": ORA}

    def fals(nume, argumente):
        apeluri.append((nume, argumente))
        return rezultate.get(nume, {"rezultat": f"ok:{nume}"})

    monkeypatch.setattr(reflex, "ruleaza_functie", fals)
    monkeypatch.setattr(reflex, "REGISTRU_FUNCTII", {n: None for n in (
        "get_ora_curenta", "seteaza_volum", "status_volum", "comuta_mute", "seteaza_luminozitate",
        "status_luminozitate", "status_cpu", "status_ram", "status_disc", "status_temperatura",
        "status_sistem_general")})
    monkeypatch.setattr(reflex, "CONFIRMARE_FUNCTII", {})
    monkeypatch.setattr(reflex, "trece_prin_securitate", lambda *a, **k: (True, ""))
    return apeluri, rezultate


@pytest.mark.parametrize("text,unealta,argumente", [
    ("Cât e ceasul?", "get_ora_curenta", {}),
    ("Jarvis, ce oră e, te rog?", "get_ora_curenta", {}),
    ("ce zi e azi", "get_ora_curenta", {}),
    ("în ce an suntem?", "get_ora_curenta", {}),
    ("pune volumul la 50%", "seteaza_volum", {"procent": 50}),
    ("volum 30", "seteaza_volum", {"procent": 30}),
    ("setează volumul la 20 la sută", "seteaza_volum", {"procent": 20}),
    ("volum 130", "seteaza_volum", {"procent": 100}),                       # plafonat la 100
    ("cât de tare e volumul", "status_volum", {}),
    ("dă mute", "comuta_mute", {"activeaza_mute": "da"}),
    ("oprește sunetul", "comuta_mute", {"activeaza_mute": "da"}),
    ("scoate mute", "comuta_mute", {"activeaza_mute": "nu"}),
    ("pornește sunetul", "comuta_mute", {"activeaza_mute": "nu"}),
    ("pune luminozitatea la 40", "seteaza_luminozitate", {"procent": 40}),
    ("care e luminozitatea", "status_luminozitate", {}),
    ("cât de încărcat e procesorul", "status_cpu", {}),
    ("câtă memorie RAM folosesc?", "status_ram", {}),
    ("cât spațiu am pe disc", "status_disc", {}),
    ("cât de cald e procesorul", "status_temperatura", {}),
    ("cum stă sistemul?", "status_sistem_general", {}),
])
def test_reflex_recunoaste_si_apeleaza_unealta(unelte_false, text, unealta, argumente):
    apeluri, _ = unelte_false
    assert reflex.incearca(text) is not None
    assert apeluri == [(unealta, argumente)]


@pytest.mark.parametrize("text", [
    "", "salut", "deschide firefox", "cât e ceasul în Tokyo", "ce părere ai despre ora de vară",
    "explică-mi ce e un procesor", "pune volumul", "pune volumul la", "pune muzică",
    "cât spațiu am în cameră", "fă un commit", "ce oră e și ce vreme e",
])
def test_reflex_nu_se_potriveste_pe_fraze_ambigue(unelte_false, text):
    apeluri, _ = unelte_false
    assert reflex.incearca(text) is None
    assert apeluri == []          # nicio unealtă nu rulează când nu e potrivire exactă


def test_sabloane_ora_data_an(unelte_false):
    assert reflex.incearca("cât e ceasul") == "Este ora 19:03."
    assert reflex.incearca("ce zi e azi") == "Azi este duminică, 22 noiembrie 2026."
    assert reflex.incearca("în ce an suntem") == "Suntem în anul 2026."


def test_format_necunoscut_cade_pe_model(unelte_false):
    _, rezultate = unelte_false
    rezultate["get_ora_curenta"] = {"rezultat": "format neașteptat"}
    assert reflex.incearca("cât e ceasul") is None


def test_eroarea_unelte_e_spusa_utilizatorului(unelte_false):
    _, rezultate = unelte_false
    rezultate["status_cpu"] = {"eroare": "psutil lipsește"}
    assert reflex.incearca("cât de încărcat e procesorul") == "psutil lipsește"


def test_unealta_cu_confirmare_nu_ruleaza_niciodata(unelte_false, monkeypatch):
    apeluri, _ = unelte_false
    monkeypatch.setattr(reflex, "CONFIRMARE_FUNCTII", {"seteaza_volum": True})
    assert reflex.incearca("pune volumul la 50") is None
    assert apeluri == []


def test_blacklistul_se_aplica_si_pe_calea_reflex(unelte_false, monkeypatch):
    apeluri, _ = unelte_false
    monkeypatch.setattr(reflex, "trece_prin_securitate", lambda *a, **k: (False, "blocat"))
    assert reflex.incearca("pune volumul la 50") is None
    assert apeluri == []


def test_unealta_neinregistrata_cade_pe_model(unelte_false, monkeypatch):
    monkeypatch.setattr(reflex, "REGISTRU_FUNCTII", {})
    assert reflex.incearca("cât e ceasul") is None


def test_statistici_numara_cererile_si_reflexele(unelte_false):
    inainte = dict(reflex.STATISTICI)
    reflex.incearca("cât e ceasul")
    reflex.incearca("salut")
    assert reflex.STATISTICI["cereri"] == inainte["cereri"] + 2
    assert reflex.STATISTICI["reflex"] == inainte["reflex"] + 1


def test_adauga_in_istoric_pastreaza_alternanta():
    istoric = []
    reflex.adauga_in_istoric(istoric, "cât e ceasul", "Este ora 19:03.")
    assert [c.role for c in istoric] == ["user", "model"]
    assert istoric[1].parts[0].text == "Este ora 19:03."


def test_cu_unelte_reale_ora_se_rezolva_fara_llm():
    import src.tools  # noqa: F401  (înregistrează uneltele)
    raspuns = reflex.incearca("cât e ceasul?")
    assert raspuns is not None and raspuns.startswith("Este ora ")
