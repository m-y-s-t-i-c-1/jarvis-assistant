"""Plasa anti-halucinație pe calea streaming: afirmațiile fără unealtă nu se rostesc, se cere retry forțat."""
import pytest
from google.genai import types

from src.core import agent


def _text(t):
    return [types.Part(text=t)]


def _apel(nume="get_ora_curenta"):
    return [types.Part(function_call=types.FunctionCall(name=nume, args={}))]


class _ClStream:
    """Gemini fals: fiecare create() consumă următorul pas din `pasi` (listă de liste de Part)."""

    def __init__(self, pasi):
        self.chats = self
        self.pasi = list(pasi)
        self.configuri = []

    def create(self, model, history, config):
        self.configuri.append(config)
        self._parti = self.pasi.pop(0)
        return self

    def send_message_stream(self, _):
        c = types.Content(role="model", parts=self._parti)
        yield types.GenerateContentResponse(candidates=[types.Candidate(content=c)])


@pytest.fixture(autouse=True)
def unelte_inregistrate():
    import src.tools  # noqa: F401


def _ruleaza(pasi):
    rostite = []
    cl = _ClStream(pasi)
    ist = [types.Content(role="user", parts=[types.Part(text="fă ceva")])]
    rez = agent.agent_loop_streaming(cl, "m", "p", ist, la_propozitie_gata=rostite.append)
    return rez, rostite, cl, ist


def _mod_fortat(config):
    tc = getattr(config, "tool_config", None)
    return bool(tc and tc.function_calling_config and tc.function_calling_config.mode == "ANY")


def test_afirmatie_fara_unealta_nu_se_rostește_si_se_cere_retry_fortat():
    rez, rostite, cl, ist = _ruleaza([
        _text("Am deschis Firefox. Totul e gata. "),     # halucinație: nicio unealtă apelată
        _apel(),                                         # retry forțat: modelul apelează o unealtă
        _text("Este ora 10:00. "),                       # răspuns după rezultatul real al uneltei
    ])
    assert "Am deschis Firefox." not in rostite and "Totul e gata." not in rostite
    assert rostite == ["Este ora 10:00."]
    assert [_mod_fortat(c) for c in cl.configuri] == [False, True, False]
    assert [c.role for c in ist] == ["user", "model", "user", "model"]   # nicio urmă a răspunsului respins


def test_propozitiile_normale_se_rostesc_imediat_chiar_daca_urmeaza_una_suspecta():
    _, rostite, cl, _ = _ruleaza([
        _text("Verific acum. Am lansat aplicația. "), _apel(), _text("Gata. "),
    ])
    assert rostite[0] == "Verific acum."             # înainte de afirmația suspectă: fără întârziere
    assert "Am lansat aplicația." not in rostite


def test_afirmatia_insotita_de_apel_real_se_rosteste():
    _, rostite, _, _ = _ruleaza([
        _text("Am deschis Firefox. ") + _apel(),     # afirmație + apel real în același pas
        _text("Gata. "),
    ])
    assert "Am deschis Firefox." in rostite and "Gata." in rostite


def test_dupa_o_unealta_folosita_nu_se_mai_retine_nimic():
    _, rostite, cl, _ = _ruleaza([_apel(), _text("Este ora 10:00. Am rulat comanda. ")])
    assert rostite == ["Este ora 10:00.", "Am rulat comanda."]
    assert not any(_mod_fortat(c) for c in cl.configuri)


def test_negatia_nu_declanseaza_retinerea():
    _, rostite, cl, _ = _ruleaza([_text("Nu am deschis nimic. ")])
    assert rostite == ["Nu am deschis nimic."] and len(cl.configuri) == 1


def test_retry_o_singura_data_apoi_se_accepta_raspunsul():
    rez, rostite, cl, _ = _ruleaza([
        _text("Am pornit serverul. "),               # suspect -> reținut, retry forțat
        _text("Am pornit serverul. "),               # tot suspect, dar retry-ul s-a făcut deja -> se rostește
    ])
    assert rostite == ["Am pornit serverul."]
    assert len(cl.configuri) == 2


def test_fara_unelte_disponibile_nu_se_retine(monkeypatch):
    monkeypatch.setattr(agent, "get_unelte_pentru_gemini", lambda doar_functiile=None: None)
    _, rostite, cl, _ = _ruleaza([_text("Am deschis Firefox. ")])
    assert rostite == ["Am deschis Firefox."] and len(cl.configuri) == 1
