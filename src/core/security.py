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
import sys
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
    # FIX: pattern-urile nu aveau limite de cuvânt, deci "halt" bloca
    # "asphalt", "rm -f" bloca "confirm -f"/"perform -rf", "exec " bloca
    # "execute"-uri etc. Acum (?<![\w-]) / \b ancorează la cuvânt întreg.
    r"(?<![\w-])rm\s+-[rRfF]+",     # rm -rf și variante
    r"(?<![\w-])rm\s+--recursive",  # rm --recursive
    r":\s*\(\s*\)\s*\{.*\}",         # fork bomb
    r"\bmkfs\.",                    # formatare disc
    r"\bdd\s+if=",                  # dd (suprascrierea discului)
    r">\s*/dev/(sd|nvme|vd)",       # scriere directă pe disc
    r"\bchmod\s+-R\s+777",          # permisiuni globale nesigure
    r"\bchown\s+-R.*root",          # schimbare owner la root recursiv
    r"\bsudo\s+rm\b",              # rm cu sudo
    r"\b(shutdown|reboot|halt|poweroff)\b",  # oprire/restart sistem
    r"(?:^|[;&|]\s*)(?:sudo\s+)?passwd\b",  # comanda passwd (nu orice text cu "passwd")
    r"\b(useradd|userdel|usermod)\b",  # modificare utilizatori sistem
    r"\b(visudo|sudoers)\b",        # modificare privilegii sudo
    r"\bcrontab\s+-[re]",           # modificare cron jobs
    r"\b(iptables|nftables)\b",     # modificare firewall
    r"\bcurl\b.+\|\s*(bash|sh)\b",  # pipe curl direct în shell
    r"\bwget\b.+\|\s*(bash|sh)\b",  # pipe wget direct în shell
    r"(?:^|[;&|]\s*)(?:eval|exec)\s+",   # eval/exec ca și comandă shell
    r"\bbase64\s+(--decode|-d).*\|",  # decode base64 și pipe (ofuscare)
    r"\bpython[\d.]*\s.*-c\s+['\"]", # python -c "cod arbitrar"
    # FIX: înainte `\bsubprocess\b` bloca orice text cu cuvântul "subprocess";
    # acum doar apelurile efective.
    r"\bos\.system\s*\(|\bsubprocess\.(run|call|Popen|check_output|check_call)\b",
    # Adăugate: operații distructive neacoperite înainte
    r"(?<![\w-])--force(?:-with-lease)?\b",   # git push --force
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-\w*f",
    r"\bshred\b",
    r"\bfind\b.*\s-delete\b",
    r"\bfind\b.*-exec\s+rm\b",   # FIX: ocolea regula de mai sus
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


# FIX: blacklist-ul scana TOATE argumentele, deci text liber legitim era blocat
# (git commit -m \"fix shutdown handler\", eveniment \"Reboot router\", căutarea
# \"cum fac shutdown\", sarcina dată unui sub-agent). Parametrii de mai jos sunt
# text liber, nu comenzi: nu se scanează. Sub-agenții își trec oricum uneltele
# prin securitate la execuție.
_PARAMETRI_TEXT_LIBER = {
    "git_commit": {"mesaj"}, "git_stash": {"mesaj"},
    "creeaza_eveniment": {"titlu", "descriere"},
    "cauta_web": {"interogare"},
    "porneste_job_fundal": {"descriere"},
    "deleaga_agent_software": {"sarcina"},
    "deleaga_agent_devops": {"sarcina"},
    "deleaga_agent_cercetare": {"sarcina"},
    "gaseste_pe_ecran": {"element"}, "vezi_ecranul": {"intrebare"},
}


def verifica_blacklist(nume_functie: str, argumente: dict) -> tuple[bool, str]:
    """
    Verifică dacă argumentele unui apel de funcție conțin pattern-uri periculoase.

    Returnează:
        (True, "") dacă apelul e curat
        (False, motiv) dacă e blocat
    """
    exceptate = _PARAMETRI_TEXT_LIBER.get(nume_functie, set())
    text_de_verificat = " ".join(
        str(v) for k, v in argumente.items() if k not in exceptate
    )

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
    # FIX: în server/widget (fără terminal) input() bloca thread-ul sau
    # arunca EOFError. Fără canal de confirmare, refuzăm (fail closed).
    if not sys.stdin or not sys.stdin.isatty():
        print("[Securitate] Nu există terminal pentru confirmare — acțiune refuzată.")
        return False
    while True:
        try:
            raspuns = input(mesaj + "[da/nu]: ").strip().lower()
        except EOFError:
            return False
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