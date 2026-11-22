"""
Reflexul determinist (faza 3): cererile banale se rezolvă fără LLM.

„Cât e ceasul?", „pune volumul la 40", „cât RAM folosesc?" nu au nevoie să treacă prin router,
prin Gemini sau prin NVIDIA. Aici o cerere e recunoscută printr-un tipar strict (potrivire pe
ÎNTREAGA frază, după normalizare), se apelează direct unealta potrivită și răspunsul se formează
dintr-un șablon. Rezultat: răspuns în milisecunde, zero tokeni, zero cotă consumată.

Reguli de siguranță:
  - doar unelte FĂRĂ confirmare (citire de stare, volum, luminozitate); o unealtă marcată
    `necesita_confirmare` nu rulează niciodată pe calea reflex;
  - apelul trece prin același blacklist ca restul uneltelor;
  - o frază care nu se potrivește EXACT cade pe calea obișnuită (router + LLM), deci tiparele
    pot fi conservatoare fără risc: ce nu recunoaște reflexul rezolvă modelul.

Utilizare (în fiecare buclă de conversație, înaintea routerului):
    raspuns = reflex.incearca(text)
    if raspuns is not None: ...  # gata, nu mai chemi niciun model
"""

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

from google.genai import types

from src.core.registry import CONFIRMARE_FUNCTII, REGISTRU_FUNCTII, ruleaza_functie
from src.core.security import trece_prin_securitate

# Cuvinte de politețe/adresare ignorate la potrivire („Jarvis, cât e ceasul, te rog?").
_UMPLUTURA = {"jarvis", "hei", "hey", "te", "rog", "va", "domnule"}

STATISTICI = {"cereri": 0, "reflex": 0}


def _normalizeaza(text: str) -> str:
    """Litere mici, fără diacritice, fără punctuație (inclusiv %), fără cuvinte de umplutură."""
    fara_diacritice = "".join(
        c for c in unicodedata.normalize("NFD", (text or "").lower())
        if unicodedata.category(c) != "Mn"
    )
    cuvinte = re.findall(r"[a-z0-9]+", fara_diacritice)
    return " ".join(c for c in cuvinte if c not in _UMPLUTURA)


# ── Șabloane de răspuns ───────────────────────────────────────────────────────
_TIPAR_ORA = re.compile(r"^(\d{1,2}:\d{2}), ([^,]+), (.+?) \(")


def _ca_text(rezultat: str, _m=None) -> Optional[str]:
    return rezultat


def _fraza_ora(rezultat: str, _m=None) -> Optional[str]:
    p = _TIPAR_ORA.match(rezultat)
    return f"Este ora {p.group(1)}." if p else None


def _fraza_data(rezultat: str, _m=None) -> Optional[str]:
    p = _TIPAR_ORA.match(rezultat)
    return f"Azi este {p.group(2)}, {p.group(3)}." if p else None


def _fraza_an(rezultat: str, _m=None) -> Optional[str]:
    p = _TIPAR_ORA.match(rezultat)
    return f"Suntem în anul {p.group(3).split()[-1]}." if p else None


@dataclass(frozen=True)
class Regula:
    tipar: re.Pattern
    unealta: str
    argumente: Callable[[re.Match], dict]
    formateaza: Callable[..., Optional[str]]


def _r(tipar: str, unealta: str, formateaza=_ca_text, argumente=lambda m: {}) -> Regula:
    return Regula(re.compile(tipar), unealta, argumente, formateaza)


def _procent(m: re.Match) -> dict:
    return {"procent": min(100, int(m.group("n")))}


_PROCENT = r"(?P<n>\d{1,3})(?: la suta| procente)?"
_E = r"(?:e|este)"
_CPU = r"(?:procesorul|cpu(?: ul)?)"

REGULI: list[Regula] = [
    # timp
    _r(rf"cat {_E} (?:ceasul|ora)|(?:ce|care) ora {_E}(?: acum)?|spune(?: mi)? ora|ora exacta|ora(?: acum)?",
       "get_ora_curenta", _fraza_ora),
    _r(rf"ce zi {_E}(?: azi| astazi)?|in ce zi suntem|ce data {_E}(?: azi| astazi)?"
       rf"|care {_E} data(?: de azi| de astazi)?|ce zi a saptamanii {_E}(?: azi| astazi)?",
       "get_ora_curenta", _fraza_data),
    _r(rf"in ce an suntem|ce an {_E}(?: acum)?", "get_ora_curenta", _fraza_an),
    # volum și sunet
    _r(rf"(?:(?:pune|seteaza|fa|da)(?: mi)? )?(?:volumul|sunetul|volum)(?: la| pe)? {_PROCENT}",
       "seteaza_volum", argumente=_procent),
    _r(rf"cat de tare {_E} volumul|ce volum (?:am|{_E})|care {_E} volumul|status volum", "status_volum"),
    _r(r"(?:(?:da|pune|fa)(?: pe)? mute|(?:opreste|inchide) sunetul|pune pe silent|mute)",
       "comuta_mute", argumente=lambda m: {"activeaza_mute": "da"}),
    _r(r"(?:(?:scoate|dezactiveaza|opreste)(?: de pe)? mute(?:ul)?|(?:porneste|deschide|da) sunetul"
       r"|da sunet|scoate de pe silent)",
       "comuta_mute", argumente=lambda m: {"activeaza_mute": "nu"}),
    # luminozitate
    _r(rf"(?:(?:pune|seteaza|fa|da)(?: mi)? )?luminozitatea?(?: la| pe)? {_PROCENT}",
       "seteaza_luminozitate", argumente=_procent),
    _r(rf"ce luminozitate am|care {_E} luminozitatea|status luminozitate", "status_luminozitate"),
    # resurse
    _r(rf"cat {_E} {_CPU} incarcat|cat de incarcat {_E} {_CPU}|cat (?:cpu|procesor) folosesc"
       rf"|cat cpu {_E} folosit|status cpu", "status_cpu"),
    _r(rf"cata memorie(?: ram)? (?:folosesc|am ocupata|{_E} ocupata|{_E} folosita)|cat ram folosesc"
       rf"|cat ram {_E} folosit|cat de plina {_E} memoria(?: ram)?|status ram", "status_ram"),
    _r(rf"cat spatiu (?:liber )?(?:mai )?(?:am|{_E})(?: liber)? pe disc|cat de plin {_E} discul|status disc",
       "status_disc"),
    _r(rf"cat de cald {_E} {_CPU}|ce temperatura (?:are|au) (?:procesorul|calculatorul|cpu(?: ul)?)"
       r"|status temperatura|temperatura procesorului", "status_temperatura"),
    _r(rf"cum sta sistemul|status sistem(?: general)?|cum stau resursele|cum merge calculatorul"
       rf"|cum {_E} sistemul|status calculator", "status_sistem_general"),
]


def _potrivire(text: str):
    normalizat = _normalizeaza(text)
    if not normalizat:
        return None
    for regula in REGULI:
        m = regula.tipar.fullmatch(normalizat)
        if m:
            return regula, m
    return None


def incearca(text: str) -> Optional[str]:
    """Răspunsul gata format dacă `text` e o cerere reflex, altfel None (cade pe calea obișnuită)."""
    STATISTICI["cereri"] += 1
    gasit = _potrivire(text)
    if gasit is None:
        return None
    regula, m = gasit

    nume = regula.unealta
    if nume not in REGISTRU_FUNCTII or CONFIRMARE_FUNCTII.get(nume, False):
        return None
    argumente = regula.argumente(m)
    aprobat, _motiv = trece_prin_securitate(nume, argumente, False)
    if not aprobat:
        return None

    start = time.perf_counter()
    rezultat = ruleaza_functie(nume, argumente)
    if "eroare" in rezultat:
        raspuns = str(rezultat["eroare"])
    else:
        raspuns = regula.formateaza(str(rezultat.get("rezultat", "")), m)
        if not raspuns:
            return None  # șablonul nu a recunoscut formatul -> las modelul să răspundă
    STATISTICI["reflex"] += 1
    print(f"[Reflex] '{text.strip()}' -> {nume} ({(time.perf_counter() - start) * 1000:.0f} ms, fără LLM)")
    return raspuns


def adauga_in_istoric(istoric: list, text_utilizator: str, raspuns: str) -> None:
    """Păstrează tura reflex în istoricul modelului, ca „și asta?" de după să aibă context."""
    istoric.append(types.Content(role="user", parts=[types.Part(text=text_utilizator)]))
    istoric.append(types.Content(role="model", parts=[types.Part(text=raspuns)]))
