"""
Git Safety Net (Task 6.7)

Plasă de siguranță automată — înainte de orice acțiune care necesită
confirmare (git commit/push, joburi în fundal, ștergere evenimente
calendar, click/tastare pe ecran etc.), creăm un checkpoint local Git,
NEINTRUZIV, care nu modifică branch-ul activ, nu murdărește `git log`
și nu deranjează working tree-ul vizibil.

Mecanism: `git stash create` — construiește un obiect de commit care
conține exact starea curentă (staged + unstaged), FĂRĂ să atingă
indexul sau working tree-ul (spre deosebire de `git stash push`, care
le-ar goli). Obiectul rezultat e "orfan" (nu apare în `git log` normal),
dar poate fi recuperat oricând cu hash-ul lui.

Recuperare, dacă ceva merge prost:
    git stash apply <hash>          # aplică checkpoint-ul peste starea curentă
    git reset --hard <hash>         # revino EXACT la starea din checkpoint (distructiv)

Checkpoint-urile sunt loghate local (fișier text simplu), ca să le
găsești ușor după fapt, fără să cauți prin reflog.

IMPORTANT: dacă working tree-ul e curat (nimic de salvat), `git stash
create` nu returnează niciun hash — tratăm asta ca "nimic de făcut",
nu ca eroare.
"""

import subprocess
from datetime import datetime
from pathlib import Path

LOG_CHECKPOINT = Path.home() / ".jarvis" / "git_safety_log.txt"


def creeaza_checkpoint(motiv: str, cale_repo: str = ".") -> str:
    """
    Creează un checkpoint Git neintruziv (git stash create) înainte de o
    acțiune riscantă. Nu modifică branch-ul activ, nu cere confirmare.

    Parametri:
        motiv:     descriere scurtă a acțiunii care urmează (pentru log)
        cale_repo: calea repo-ului. Default: directorul curent.

    Returnează:
        - hash-ul checkpoint-ului (string), dacă a fost creat cu succes
        - "" (string gol) dacă nu era nimic de salvat (working tree curat)
          sau dacă directorul nu e un repo Git — NU e tratat ca eroare,
          doar înseamnă că nu era nevoie de checkpoint.

    Eșecurile sunt înghițite silențios (returnează "") — un checkpoint
    ratat nu trebuie NICIODATĂ să blocheze acțiunea reală a utilizatorului.
    """
    try:
        rezultat = subprocess.run(
            ["git", "stash", "create", f"[Jarvis Safety Net] {motiv}"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cale_repo,
        )

        hash_checkpoint = rezultat.stdout.strip()

        if not hash_checkpoint:
            # Working tree curat — nimic de salvat, nu e eroare
            return ""

        # Salvăm referința explicit — obiectele din `stash create` NU sunt
        # ținute de niciun ref, deci ar fi eligibile pentru garbage collection
        # dacă nu le "ancorăm" cu `stash store`.
        subprocess.run(
            ["git", "stash", "store", "-m", f"[Jarvis Safety Net] {motiv}", hash_checkpoint],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cale_repo,
        )

        _logheaza_checkpoint(hash_checkpoint, motiv)
        print(f"[Git Safety Net] Checkpoint creat: {hash_checkpoint[:10]} ({motiv})")
        return hash_checkpoint

    except FileNotFoundError:
        return ""  # git nu e instalat — nimic de făcut
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""  # orice altă eroare — nu blocăm acțiunea reală pentru asta


def _logheaza_checkpoint(hash_checkpoint: str, motiv: str):
    """Adaugă o linie în log-ul local de checkpoint-uri, pentru recuperare ușoară ulterioară."""
    try:
        LOG_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_CHECKPOINT, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} | {hash_checkpoint} | {motiv}\n")
    except Exception:
        pass  # logarea e best-effort, nu blocăm nimic pentru ea


def listeaza_checkpoint_uri(limita: int = 20) -> str:
    """Returnează ultimele N checkpoint-uri din log, cel mai recent primul."""
    if not LOG_CHECKPOINT.exists():
        return "Niciun checkpoint înregistrat încă."

    with open(LOG_CHECKPOINT, "r", encoding="utf-8") as f:
        linii = f.readlines()

    if not linii:
        return "Niciun checkpoint înregistrat încă."

    ultimele = linii[-limita:]
    ultimele.reverse()
    return "".join(ultimele)