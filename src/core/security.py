"""
Barierele de Securitate și Validarea (Task 2.3) + Git Safety Net (Task 6.7)

Trei straturi de protecție:
1. BLACKLIST: pattern-uri de argumente care sunt blocate complet, indiferent
   de context. Niciodată nu ajung la subprocess.
2. CONFIRMARE: unelte marcate cu necesita_confirmare=True în @unealta cer
   aprobare explicită înainte de execuție.
3. GIT SAFETY NET (nou): chiar înainte de a aproba o acțiune care necesită
   confirmare, creăm automat un checkpoint Git neintruziv (git_safety.py).
   Dacă acțiunea aprobată strică ceva, ai un punct de recuperare, fără să
   fi cerut tu explicit asta.

Confirmarea are două moduri, comutabile din CONFIRMATION_MODE:
   - TEXT  (activ acum): întreabă în terminal, aștepți 'da'/'nu'
   - VOICE (Faza 3): va apela modulul audio când e implementat

Când implementezi Faza 3, caută comentariul "FAZA 3:" din această clasă
și înlocuiește blocul TEXT cu apelul la modulul tău audio.
"""

import re
from enum import Enum
from src.core.git_safety import creeaza_checkpoint

# ---- Modul de confirmare activ ----
# Schimbă asta în VOICE când implementezi Faza 3 audio
class ConfirmationMode(Enum):
    TEXT  = "text"
    VOICE = "voice"

CONFIRMATION_MODE = ConfirmationMode.TEXT


# ---- Pattern-uri blocate complet (blacklist) ----
# Dacă vreun argument al unui apel de funcție conține unul din aceste
# pattern-uri, execuția e oprită imediat, fără confirmare, fără excepții.
# Sunt regex-uri, ca să prindem variante (rm -rf, rm -Rf, rm -r -f etc.)
BLACKLIST_PATTERNS = [
    r"rm\s+-[rRfF]+",          # rm -rf și variante
    r"rm\s+--recursive",       # rm --recursive
    r":\s*\(\s*\)\s*\{.*\}",   # fork bomb
    r"mkfs\.",                 # formatare disc
    r"dd\s+if=",               # dd (suprascrierea discului)
    r">\s*/dev/(sd|nvme|vd)",  # scriere directă pe disc
    r"chmod\s+-R\s+777",       # permisiuni globale nesigure
    r"chown\s+-R.*root",       # schimbare owner la root recursiv
    r"sudo\s+rm",              # rm cu sudo
    r"shutdown",                # oprire sistem
    r"reboot",                 # restart sistem
    r"halt",                   # oprire hard
    r"poweroff",               # oprire sistem
    r"passwd",                 # schimbare parole
    r"useradd|userdel|usermod",# modificare utilizatori sistem
    r"visudo|sudoers",         # modificare privilegii sudo
    r"crontab\s+-[re]",        # modificare cron jobs
    r"iptables|nftables",      # modificare firewall
    r"curl.+\|\s*(bash|sh)",   # pipe curl direct în shell
    r"wget.+\|\s*(bash|sh)",   # pipe wget direct în shell
    r"eval\s+",                # eval arbitrar
    r"exec\s+",                # exec arbitrar
    r"base64\s+--decode.*\|",  # decode base64 și pipe (ofuscare)
    r"python.*-c\s+['\"]",     # python -c "cod arbitrar"
    r"os\.system|subprocess",  # injecție de cod Python
]

# Pre-compilăm regex-urile o dată la import, nu la fiecare apel
_BLACKLIST_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]

# ---- Unelte pentru care Git Safety Net-ul NU are sens ----
# Checkpoint-ul Git protejează fișierele din repo — n-are rost să facem
# un `git stash create` înainte de acțiuni care nu ating deloc filesystem-ul
# repo-ului (ex: click pe ecran, ștergere eveniment din Google Calendar).
# Îl păstrăm activ pentru orice altceva marcat necesita_confirmare=True,
# ca plasă implicită — mai bine un checkpoint în plus, nederanjant, decât
# unul lipsă exact când ai avea nevoie de el.
_FUNCTII_FARA_SAFETY_NET = {
    "click_la_pozitie", "scrie_text", "apasa_tasta",
    "sterge_eveniment", "creeaza_eveniment",
}


def verifica_blacklist(nume_functie: str, argumente: dict) -> tuple[bool, str]:
    """
    Verifică dacă argumentele unui apel de funcție conțin pattern-uri periculoase.

    Returnează:
        (True, "") dacă apelul e curat
        (False, motiv) dacă e blocat
    """
    text_de_verificat = " ".join(str(v) for v in argumente.values())

    for pattern in _BLACKLIST_COMPILED:
        if pattern.search(text_de_verificat):
            return False, (
                f"Apelul funcției '{nume_functie}' cu argumentele "
                f"{argumente} a fost blocat de securitate: "
                f"pattern periculos detectat ({pattern.pattern!r})."
            )

    return True, ""


def cere_confirmare(nume_functie: str, argumente: dict) -> bool:
    """
    Cere confirmare explicită înainte de a executa o acțiune marcată
    ca periculoasă. Returnează True dacă utilizatorul aprobă, False altfel.

    Modul activ: CONFIRMATION_MODE (TEXT sau VOICE).
    """
    mesaj = (
        f"\n⚠️  Jarvis dorește să execute: {nume_functie}({argumente})\n"
        f"    Această acțiune necesită confirmare. Aprobi? "
    )

    if CONFIRMATION_MODE == ConfirmationMode.TEXT:
        return _confirmare_text(mesaj)

    elif CONFIRMATION_MODE == ConfirmationMode.VOICE:
        # FAZA 3: înlocuiește această linie cu apelul la modulul tău audio
        # from src.audio.output import spune_text
        # from src.audio.input import asculta_confirmare
        # spune_text(mesaj)
        # return asculta_confirmare()  # returnează True dacă userul zice "da"
        print("[VOICE mode nu e implementat încă, folosesc TEXT ca fallback]")
        return _confirmare_text(mesaj)


def _confirmare_text(mesaj: str) -> bool:
    """Confirmare simplă prin input în terminal."""
    while True:
        raspuns = input(mesaj + "[da/nu]: ").strip().lower()
        if raspuns in ("da", "d", "yes", "y"):
            return True
        if raspuns in ("nu", "n", "no"):
            print("Acțiune anulată de utilizator.")
            return False
        print("Răspuns invalid. Scrie 'da' sau 'nu'.")


def trece_prin_securitate(
    nume_functie: str,
    argumente: dict,
    necesita_confirmare: bool = False,
) -> tuple[bool, str]:
    """
    Punctul unic de intrare pentru toate verificările de securitate.
    Apelat din agent.py înainte de orice ruleaza_functie().

    Ordinea pașilor:
        1. Blacklist — blocare imediată, fără confirmare
        2. Confirmare explicită (dacă unealta e marcată)
        3. Git Safety Net — checkpoint automat, DUPĂ ce utilizatorul a
           aprobat, chiar înainte de execuția efectivă. Neintruziv,
           silențios — nu cere el însuși nicio confirmare.

    Returnează:
        (True, "")       -> execuție aprobată, continuă normal
        (False, motiv)   -> execuție blocată, motiv e trimis înapoi la model
    """
    # Pasul 1: blacklist — blocare imediată, fără confirmare
    curat, motiv = verifica_blacklist(nume_functie, argumente)
    if not curat:
        return False, motiv

    # Pasul 2: confirmare — doar dacă unealta e marcată ca periculoasă
    if necesita_confirmare:
        aprobat = cere_confirmare(nume_functie, argumente)
        if not aprobat:
            return False, f"Utilizatorul a refuzat execuția funcției '{nume_functie}'."

        # Pasul 3: Git Safety Net — checkpoint automat, silențios, doar
        # pentru acțiuni unde chiar are sens (nu ating repo-ul altfel)
        if nume_functie not in _FUNCTII_FARA_SAFETY_NET:
            creeaza_checkpoint(motiv=f"înainte de {nume_functie}({argumente})")

    return True, ""