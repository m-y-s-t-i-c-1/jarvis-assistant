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

import shlex
import shutil
import subprocess
from src.core.config import valoare
from src.core.registry import unealta

# ---- Alias -> comanda reală de pe sistemul tău ----
# Modifică valorile astea ca să corespundă cu binarele instalate la tine.
# FIX: aplicațiile erau hardcodate (vivaldi, alacritty, nautilus, code): pe un sistem fără ele,
# „deschide browserul" eșua mereu. Acum se alege prima instalată din listă; poți fixa una anume în
# config/jarvis.toml ([aplicatii]) sau cu JARVIS_APP_BROWSER / _EDITOR / _TERMINAL / _FILE_MANAGER.
ALIAS_APLICATII = {
    "browser": ["vivaldi", "firefox", "chromium", "google-chrome-stable", "brave", "opera"],
    "editor": ["code", "codium", "code-oss", "zed", "kate", "gedit"],
    "terminal": ["alacritty", "kitty", "konsole", "gnome-terminal", "xfce4-terminal", "foot", "wezterm", "xterm"],
    "file_manager": ["nautilus", "dolphin", "thunar", "pcmanfm", "nemo"],
}


def _alege_aplicatie(alias: str):
    """(argumente de lansare sau None, lista de binare încercate). Configurarea explicită are prioritate."""
    configurat = valoare("aplicatii", alias, "", env=f"JARVIS_APP_{alias.upper()}")
    if configurat:
        argumente = shlex.split(configurat)
        return (argumente if argumente and shutil.which(argumente[0]) else None), [configurat]
    candidati = ALIAS_APLICATII[alias]
    for c in candidati:
        if shutil.which(c):
            return [c], candidati
    return None, candidati

# ---- Comenzi read-only permise. Gemini trimite DOAR cheia, nu text liber. ----
ALIAS_COMENZI = {
    "lista_fisiere": ["ls", "-la"],
    "spatiu_disc": ["df", "-h"],
    "memorie_ram": ["free", "-h"],
    # FIX: `ps aux` e sortat după PID; cu max_linii=35 se vedeau doar procesele vechi (systemd,
    # kthreadd), nu cele care consumă CPU. Sortăm descrescător după CPU.
    "procese_active": ["ps", "aux", "--sort=-%cpu"],
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
    alias = str(alias or "").lower().strip()  # FIX: None făcea .lower() să arunce

    if alias not in ALIAS_APLICATII:
        return (
            f"Alias necunoscut: '{alias}'. Aplicații disponibile: "
            f"{', '.join(ALIAS_APLICATII.keys())}."
        )

    argumente, incercate = _alege_aplicatie(alias)
    if argumente is None:
        return (
            f"Nu am găsit nicio aplicație instalată pentru '{alias}' (am căutat: {', '.join(incercate)}). "
            f"Setează binarul în config/jarvis.toml, secțiunea [aplicatii], "
            f"sau cu variabila JARVIS_APP_{alias.upper()}."
        )

    comanda = " ".join(argumente)
    try:
        # Popen, nu run -- nu vrem să blocăm Jarvis până se închide aplicația
        subprocess.Popen(
            argumente,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Am lansat '{comanda}'."
    except FileNotFoundError:
        return f"Comanda '{comanda}' nu a fost găsită pe sistem."
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
    alias = str(alias or "").lower().strip()

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