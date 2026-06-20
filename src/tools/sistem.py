"""
Control sistem de operare (Task 2.2).

Două categorii de unelte:
1. deschide_aplicatie  -> lansează un program GUI printr-un alias generic
2. ruleaza_comanda_info -> rulează o comandă read-only dintr-un set FIX,
   nu execuție liberă de shell. Gemini alege o cheie din ALIAS_COMENZI,
   nu poate trimite text arbitrar către subprocess.

IMPORTANT pentru Vasea: ALIAS_APLICATII de mai jos e generic. Modifică
valorile ca să corespundă cu ce ai instalat tu pe Arch (ex: terminalul
tău exact - alacritty/kitty/konsole, file manager-ul tău, etc).
"""

import subprocess
from src.core.registry import unealta

# ---- Alias -> comanda reală de pe sistemul tău ----
# Modifică valorile astea ca să corespundă cu binarele instalate la tine.
ALIAS_APLICATII = {
    "browser": "firefox",
    "editor": "code",
    "terminal": "alacritty",       # schimbă cu kitty/konsole/etc dacă ai altceva
    "file_manager": "nautilus",    # schimbă cu dolphin/thunar/etc dacă ai altceva
}

# ---- Comenzi read-only permise. Gemini trimite DOAR cheia, nu text liber. ----
ALIAS_COMENZI = {
    "lista_fisiere": ["ls", "-la"],
    "spatiu_disc": ["df", "-h"],
    "memorie_ram": ["free", "-h"],
    "procese_active": ["ps", "aux"],
    "timp_functionare": ["uptime"],
    "utilizator_curent": ["whoami"],
    "director_curent": ["pwd"],
    "info_sistem": ["uname", "-a"],
    "info_cpu": ["lscpu"],
    "info_retea": ["ip", "a"],
}


@unealta(
    description=(
        "Deschide o aplicație grafică pe sistemul utilizatorului. Folosește "
        "asta când utilizatorul cere să deschizi browserul, editorul de cod, "
        "terminalul sau managerul de fișiere. Alias-urile disponibile sunt: "
        + ", ".join(ALIAS_APLICATII.keys())
    ),
    parameters={
        "alias": {
            "type": "STRING",
            "description": (
                "Numele generic al aplicației de deschis. Trebuie să fie unul "
                "dintre: " + ", ".join(ALIAS_APLICATII.keys())
            ),
        }
    },
)
def deschide_aplicatie(alias: str):
    """Lansează o aplicație GUI prin alias, fără să blocheze Jarvis."""
    alias = alias.lower().strip()

    if alias not in ALIAS_APLICATII:
        return (
            f"Alias necunoscut: '{alias}'. Aplicații disponibile: "
            f"{', '.join(ALIAS_APLICATII.keys())}."
        )

    comanda = ALIAS_APLICATII[alias]

    try:
        # Popen, nu run -- nu vrem să blocăm Jarvis până se închide aplicația
        subprocess.Popen(
            [comanda],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Am lansat '{comanda}'."
    except FileNotFoundError:
        return (
            f"Comanda '{comanda}' nu a fost găsită pe sistem. "
            f"Verifică dacă e instalată sau dacă alias-ul din ALIAS_APLICATII "
            f"corespunde binarului real."
        )
    except Exception as e:
        return f"Eroare la lansarea aplicației: {str(e)}"


@unealta(
    description=(
        "Rulează o comandă de sistem read-only (doar informativă, nu "
        "modifică nimic) și returnează rezultatul. Folosește asta pentru "
        "întrebări despre spațiu pe disc, memorie RAM, procese active, "
        "informații despre sistem sau rețea. Comenzile disponibile sunt: "
        + ", ".join(ALIAS_COMENZI.keys())
    ),
    parameters={
        "alias": {
            "type": "STRING",
            "description": (
                "Cheia comenzii de rulat. Trebuie să fie una dintre: "
                + ", ".join(ALIAS_COMENZI.keys())
            ),
        }
    },
    max_linii=35,  # ps aux poate fi 150+ linii; 35 = top procese + header, suficient pentru analiză
)
def ruleaza_comanda_info(alias: str):
    """Rulează o comandă read-only dintr-un set fix și returnează output-ul."""
    alias = alias.lower().strip()

    if alias not in ALIAS_COMENZI:
        return (
            f"Comandă necunoscută: '{alias}'. Comenzi disponibile: "
            f"{', '.join(ALIAS_COMENZI.keys())}."
        )

    comanda = ALIAS_COMENZI[alias]

    try:
        rezultat = subprocess.run(
            comanda,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if rezultat.returncode != 0:
            return f"Comanda a returnat eroare: {rezultat.stderr.strip()}"

        return rezultat.stdout.strip()

    except subprocess.TimeoutExpired:
        return "Comanda a durat prea mult și a fost întreruptă."
    except FileNotFoundError:
        return f"Binarul pentru '{alias}' nu a fost găsit pe sistem."
    except Exception as e:
        return f"Eroare la rularea comenzii: {str(e)}"