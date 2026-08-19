"""
Controlul Ecranului — Mouse & Tastatură (Task 6.2)

Permite lui Jarvis să interacționeze fizic cu ecranul: mișcare mouse,
click, scroll, tastare text, apăsare taste/combinații. Se folosește
împreună cu vedere.py — de obicei fluxul e:

    1. gaseste_pe_ecran("butonul X")  -> Jarvis obține coordonate (x, y)
    2. click_la_pozitie(x, y)          -> Jarvis acționează pe acele coordonate

Folosește pyautogui, care pe X11 funcționează nativ (prin python3-xlib).

IMPORTANT — Wayland: pyautogui NU funcționează pe Wayland din motive de
securitate ale protocolului (nu poate injecta evenimente de input global).
Dacă vreodată treci de pe X11 pe Wayland, acest modul trebuie înlocuit cu
ydotool (necesită daemon + permisiuni uinput) sau wtype/wlrctl.

Dependențe:
    pip install pyautogui --break-system-packages

IMPORTANT — Securitate: TOATE acțiunile care modifică efectiv starea
sistemului (click, scriere text, apăsare taste) sunt marcate cu
necesita_confirmare=True, pentru că pot avea impact real (click greșit
pe un buton important, text scris în locul nepotrivit etc.). Poți relaxa
asta funcție cu funcție dacă devine incomod în workflow-ul zilnic, dar ai
grijă mai ales cu apasa_tasta (poate include Ctrl+W, Alt+F4 etc.).

Failsafe: dacă vrei să oprești instant orice acțiune automată în curs,
mută mouse-ul manual în colțul stânga-sus al ecranului (0, 0) — pyautogui
aruncă o excepție și oprește execuția (pyautogui.FAILSAFE = True).
"""

import pyautogui
from src.core.registry import unealta

# Pauză de siguranță după fiecare acțiune pyautogui, ca interfața să aibă
# timp să reacționeze înainte de comanda următoare.
pyautogui.PAUSE = 0.2

# Failsafe activ — mișcă mouse-ul în (0,0) ca să oprești orice automatizare.
pyautogui.FAILSAFE = True


@unealta(
    description=(
        "Mută cursorul mouse-ului la o poziție exactă pe ecran, fără click. "
        "Folosește coordonate obținute de la 'gaseste_pe_ecran' — nu ghici "
        "niciodată coordonate arbitrare."
    ),
    parameters={
        "x": {"type": "INTEGER", "description": "Coordonata X în pixeli."},
        "y": {"type": "INTEGER", "description": "Coordonata Y în pixeli."},
        "durata": {
            "type": "NUMBER",
            "description": "Durata mișcării în secunde (default 0.3, mișcare lină, nu instant).",
            "optional": True,
        },
    },
)
def muta_mouse(x: int, y: int, durata: float = 0.3):
    """Mută mouse-ul la coordonate exacte, fără nicio acțiune de click."""
    try:
        pyautogui.moveTo(int(x), int(y), duration=float(durata))
        return f"Mouse mutat la ({x}, {y})."
    except Exception as e:
        return f"Eroare la mutarea mouse-ului: {str(e)}"


@unealta(
    description=(
        "Dă click la o poziție exactă pe ecran. Folosește ÎNTOTDEAUNA "
        "coordonate obținute anterior de la 'gaseste_pe_ecran' — nu ghici "
        "niciodată coordonate. Suportă click stânga (default), dreapta "
        "sau dublu-click."
    ),
    parameters={
        "x": {"type": "INTEGER", "description": "Coordonata X în pixeli."},
        "y": {"type": "INTEGER", "description": "Coordonata Y în pixeli."},
        "tip": {
            "type": "STRING",
            "description": "Tipul de click: 'stanga' (default), 'dreapta' sau 'dublu'.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def click_la_pozitie(x: int, y: int, tip: str = "stanga"):
    """Mută mouse-ul la (x, y) și execută click-ul cerut."""
    try:
        pyautogui.moveTo(int(x), int(y), duration=0.2)
        tip = tip.lower().strip()
        if tip == "dreapta":
            pyautogui.rightClick()
        elif tip == "dublu":
            pyautogui.doubleClick()
        else:
            pyautogui.click()
        return f"Click ({tip}) executat la ({x}, {y})."
    except Exception as e:
        return f"Eroare la click: {str(e)}"


@unealta(
    description=(
        "Scrie text la poziția curentă a cursorului/focus-ului (de obicei "
        "un câmp de input selectat anterior cu 'click_la_pozitie'). "
        "Simulează tastare reală, caracter cu caracter."
    ),
    parameters={
        "text": {"type": "STRING", "description": "Textul de scris."},
        "interval": {
            "type": "NUMBER",
            "description": "Pauza între caractere, în secunde (default 0.02, ca tastare naturală).",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def scrie_text(text: str, interval: float = 0.02):
    """Tastează un text la poziția curentă de focus."""
    try:
        pyautogui.write(text, interval=float(interval))
        return f"Text scris ({len(text)} caractere)."
    except Exception as e:
        return f"Eroare la scriere: {str(e)}"


@unealta(
    description=(
        "Apasă o tastă sau o combinație de taste. Pentru combinații, "
        "separă tastele prin '+' (ex: 'ctrl+c', 'ctrl+v', 'alt+tab', "
        "'ctrl+shift+t'). Pentru o tastă simplă, trimite doar numele "
        "ei (ex: 'enter', 'esc', 'tab', 'backspace')."
    ),
    parameters={
        "tasta": {
            "type": "STRING",
            "description": "Tasta sau combinația de apăsat, format pyautogui (litere mici).",
        }
    },
    necesita_confirmare=True,
)
def apasa_tasta(tasta: str):
    """Apasă o tastă sau o combinație de taste."""
    try:
        parti = [p.strip() for p in tasta.lower().split("+") if p.strip()]
        if not parti:
            return "Nicio tastă specificată."
        if len(parti) > 1:
            pyautogui.hotkey(*parti)
        else:
            pyautogui.press(parti[0])
        return f"Tastă apăsată: {tasta}."
    except Exception as e:
        return f"Eroare la apăsarea tastei: {str(e)}"


@unealta(
    description=(
        "Derulează (scroll) la poziția curentă a mouse-ului. Valoare "
        "pozitivă derulează în sus, negativă derulează în jos."
    ),
    parameters={
        "cantitate": {
            "type": "INTEGER",
            "description": "Cât de mult să derulezi. Ex: 10 (sus) sau -10 (jos).",
        }
    },
)
def deruleaza(cantitate: int):
    """Derulează la poziția curentă a mouse-ului."""
    try:
        pyautogui.scroll(int(cantitate))
        return f"Derulat {cantitate} unități."
    except Exception as e:
        return f"Eroare la derulare: {str(e)}"


@unealta(
    description=(
        "Returnează rezoluția ecranului curent și poziția curentă a "
        "cursorului mouse-ului. Util pentru orientare rapidă, fără "
        "screenshot complet."
    ),
)
def status_ecran():
    """Informații rapide despre ecran și poziția mouse-ului."""
    try:
        latime, inaltime = pyautogui.size()
        x, y = pyautogui.position()
        return f"Rezoluție ecran: {latime}x{inaltime}. Cursor la: ({x}, {y})."
    except Exception as e:
        return f"Eroare: {str(e)}"