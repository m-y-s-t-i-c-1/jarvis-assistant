"""
Configurație centrală (faza 0).

Înainte, numele utilizatorului („Vasea"), orașul implicit și fusul orar erau împrăștiate prin cod.
Acum există o singură sursă, cu această prioritate (cea mai mare câștigă):

    1. variabila de mediu (ex. JARVIS_USER, JARVIS_ORAS) — din shell sau din .env
    2. fișierul config/jarvis.toml (necomis în git; copiază config/jarvis.example.toml)
    3. valoarea implicită din cod

Folosește doar biblioteca standard (tomllib, Python >= 3.11): fără dependențe noi.
Un fișier lipsit sau stricat nu oprește pornirea — se folosesc valorile implicite.

Utilizare:
    from src.core.config import UTILIZATOR, personalizeaza, valoare
    personalizeaza("Te adresezi lui {utilizator}.")        # înlocuiește jetonul {utilizator}
    valoare("locatie", "oras", "Chișinău", env="JARVIS_ORAS")
"""

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

# .env trebuie citit ÎNAINTE de orice valoare: modulele se importă înaintea load_dotenv() din main.py.
load_dotenv()

RADACINA = Path(__file__).resolve().parents[2]
CALE_CONFIG_IMPLICITA = RADACINA / "config" / "jarvis.toml"

_JETON_UTILIZATOR = "{utilizator}"


def _citeste(cale: Path) -> dict:
    """Citește un fișier TOML; orice problemă => dicționar gol (se folosesc valorile implicite)."""
    try:
        with open(cale, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (tomllib.TOMLDecodeError, OSError) as e:
        print(f"[Config] {cale} nu poate fi citit ({e}) — folosesc valorile implicite.")
        return {}


def _incarca() -> dict:
    cale = Path(os.getenv("JARVIS_CONFIG", "").strip() or CALE_CONFIG_IMPLICITA).expanduser()
    return _citeste(cale)


_DATE: dict = _incarca()


def valoare(sectiune: str, cheie: str, implicit: str = "", env: str | None = None) -> str:
    """Env (dacă e dat și nenul) > config/jarvis.toml > `implicit`. Întoarce mereu un șir."""
    if env:
        din_env = os.getenv(env, "").strip()
        if din_env:
            return din_env
    brut = _DATE.get(sectiune, {})
    din_fisier = brut.get(cheie) if isinstance(brut, dict) else None
    if din_fisier is not None and str(din_fisier).strip():
        return str(din_fisier).strip()
    return implicit


UTILIZATOR: str = valoare("persona", "utilizator", "Vasea", env="JARVIS_USER")


def personalizeaza(text: str) -> str:
    """Înlocuiește jetonul {utilizator} din prompturi cu numele configurat."""
    return text.replace(_JETON_UTILIZATOR, UTILIZATOR)
