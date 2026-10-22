"""Teste pentru runda 4: context, anti-halucinație, TTS, MAX_PASI."""
import pytest
from google.genai import types

from src.core import agent, context_manager


@pytest.mark.parametrize("text,suspect", [
    ("Am deschis browserul.", True),
    ("Nu am deschis nimic. Este ora exactă 14:00.", True),   # „nu" din altă propoziție nu mai scutește
    ("Nu am rulat comanda, ai refuzat.", False),
    ("Este un răspuns normal.", False),
])
def test_detectie_halucinatie_pe_propozitie(text, suspect):
    assert agent._raspunsul_pare_suspect(text) is suspect


def _taie(text):
    buf, out = text, []
    while True:
        m = agent._PATTERN_SFARSIT_PROPOZITIE.search(buf)
        if not m:
            break
        out.append(buf[:m.end()].strip())
        buf = buf[m.end():]
    return out, buf


def test_nu_taie_dupa_abrevieri_si_liste():
    out, rest = _taie("Avem mere, pere etc. și struguri. A doua propoziție. ")
    assert out == ["Avem mere, pere etc. și struguri.", "A doua propoziție."]
    out, _ = _taie("Pașii: 1. Deschide fișierul. Apoi salvează. ")
    assert out == ["Pașii: 1. Deschide fișierul.", "Apoi salvează."]


def test_context_rezuma_si_mesajele_taiate_la_granita(monkeypatch):
    vazute = {}
    monkeypatch.setattr(context_manager, "_genereaza_rezumat",
                        lambda c, m, mesaje: vazute.setdefault("n", len(mesaje)) and "rezumat")
    monkeypatch.setattr(context_manager.db, "salveaza_memorie", lambda **k: None)
    monkeypatch.setattr(context_manager.db, "salveaza_rezumat_sesiune", lambda *a: None)
    istoric = []
    for i in range(32):
        istoric.append(types.Content(role="user" if i % 2 == 0 else "model",
                                     parts=[types.Part(text=f"m{i}")]))
    # mesajul de la poziția 10 devine function_call -> va fi tăiat de graniță
    istoric[10] = types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name="x", args={}))])
    context_manager.proceseaza(istoric, "s", client=object())
    assert vazute["n"] >= 11   # 10 din bloc + cel(e) tăiat(e)


class _Cl:
    """Client fals: cere unealtă la nesfârșit cât are tools; fără tools răspunde text."""
    def __init__(self): self.chats = self; self.fara_unelte = 0
    def create(self, model, history, config):
        self.cfg = config
        return self
    def send_message(self, _):
        class R: pass
        r = R()
        if self.cfg.tools:
            fc = types.FunctionCall(name="get_ora_curenta", args={})
            c = types.Content(role="model", parts=[types.Part(function_call=fc)])
            r.candidates = [types.Candidate(content=c)]
            r.text = ""
        else:
            self.fara_unelte += 1
            c = types.Content(role="model", parts=[types.Part(text="Gata.")])
            r.candidates = [types.Candidate(content=c)]
            r.text = "Gata."
        return r


def test_ultimul_pas_e_fara_unelte_si_da_raspuns_final():
    import src.tools  # noqa: F401  (înregistrează unelte)
    cl = _Cl()
    ist = [types.Content(role="user", parts=[types.Part(text="salut")])]
    rasp = agent.agent_loop(cl, "m", "p", ist)
    assert rasp == "Gata." and cl.fara_unelte == 1
