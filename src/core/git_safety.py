"""
Git Safety Net (Task 6.7)

Plasă de siguranță automată — înainte de orice acțiune care necesită
confirmare (git commit/push, joburi în fundal, ștergere evenimente
calendar, click/tastare pe ecran etc.), creăm un checkpoint local Git,
NEINTRUZIV, care nu modifică branch-ul activ, nu murdărește `git log`
și nu deranjează working tree-ul vizibil.

Mecanism: se construiește un commit dintr-un index TEMPORAR care conține starea
curentă completă (tracked + untracked, respectând .gitignore), FĂRĂ să atingă
indexul real sau working tree-ul. Commit-ul e ancorat sub refs/jarvis-safety/*
(nu apare în `git log` normal, nu murdărește stash-ul).

Recuperare, dacă ceva merge prost:
    git restore --source=<hash> --worktree -- .   # readuce fișierele din checkpoint

Checkpoint-urile sunt loghate local (fișier text simplu), ca să le
găsești ușor după fapt, fără să cauți prin reflog.

IMPORTANT: dacă working tree-ul e curat (nimic de salvat), nu se creează niciun
checkpoint — tratăm asta ca "nimic de făcut", nu ca eroare.
"""

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

LOG_CHECKPOINT = Path.home() / ".jarvis" / "git_safety_log.txt"
_REF_PREFIX = "refs/jarvis-safety"
_ARBORE_GOL = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # hash-ul arborelui Git gol


def _git(args: list[str], cwd: str, env: dict | None = None, timeout: int = 15) -> str:
    rezultat = subprocess.run(
        ["git", *args], capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env,
    )
    if rezultat.returncode != 0:
        raise RuntimeError(rezultat.stderr.strip() or f"git {args[0]} a eșuat")
    return rezultat.stdout.strip()


def creeaza_checkpoint(motiv: str, cale_repo: str = ".") -> str:
    """
    Creează un checkpoint Git neintruziv înainte de o acțiune riscantă. Nu modifică
    branch-ul activ, indexul real, working tree-ul sau stash-ul și nu cere confirmare.

    FIX: `git stash create` ignora fișierele UNTRACKED, deci fișierele noi, necomise,
    nu erau protejate. Acum construim checkpoint-ul cu un index TEMPORAR
    (`git add -A` pe acel index, respectând .gitignore -> `write-tree` -> `commit-tree`)
    și îl ancorăm cu un ref propriu (refs/jarvis-safety/...), ca să nu fie șters de gc.

    Parametri:
        motiv:     descriere scurtă a acțiunii care urmează (pentru log)
        cale_repo: calea repo-ului. Default: directorul curent.

    Returnează:
        - hash-ul commit-ului de checkpoint, dacă a fost creat
        - "" dacă nu era nimic de salvat (working tree curat), dacă directorul nu e
          repo Git sau dacă a apărut orice eroare — un checkpoint ratat nu trebuie
          NICIODATĂ să blocheze acțiunea reală a utilizatorului.

    Recuperare:
        git restore --source=<hash> --worktree -- .   # readuce fișierele (inclusiv cele noi)
        git show <hash> --stat                        # vezi ce conține
    """
    cale_repo = os.path.abspath(cale_repo or ".")
    index_tmp = None
    try:
        radacina = _git(["rev-parse", "--show-toplevel"], cale_repo)
        try:
            cap = _git(["rev-parse", "--verify", "-q", "HEAD"], radacina)
        except RuntimeError:
            cap = ""  # repo fără niciun commit

        fd, index_tmp = tempfile.mkstemp(prefix="jarvis-idx-")
        os.close(fd)
        os.unlink(index_tmp)  # git vrea să creeze el fișierul de index
        env = {**os.environ, "GIT_INDEX_FILE": index_tmp}

        if cap:
            _git(["read-tree", cap], radacina, env)
        _git(["add", "-A"], radacina, env, timeout=30)
        arbore = _git(["write-tree"], radacina, env)

        arbore_cap = _git(["rev-parse", f"{cap}^{{tree}}"], radacina) if cap else _ARBORE_GOL
        if arbore == arbore_cap:
            return ""  # working tree curat — nimic de salvat

        mesaj = f"[Jarvis Safety Net] {motiv}"
        comanda = ["commit-tree", arbore, "-m", mesaj] + (["-p", cap] if cap else [])
        env_commit = {
            **env,
            "GIT_AUTHOR_NAME": "Jarvis", "GIT_AUTHOR_EMAIL": "jarvis@localhost",
            "GIT_COMMITTER_NAME": "Jarvis", "GIT_COMMITTER_EMAIL": "jarvis@localhost",
        }
        hash_checkpoint = _git(comanda, radacina, env_commit)
        _git(["update-ref", f"{_REF_PREFIX}/{datetime.now():%Y%m%d-%H%M%S}-{hash_checkpoint[:7]}",
              hash_checkpoint], radacina)

        _logheaza_checkpoint(hash_checkpoint, motiv)
        print(f"[Git Safety Net] Checkpoint creat: {hash_checkpoint[:10]} ({motiv[:80]})")
        return hash_checkpoint

    except FileNotFoundError:
        return ""  # git nu e instalat — nimic de făcut
    except Exception:
        return ""  # nu repo / timeout / orice altă eroare — nu blocăm acțiunea reală
    finally:
        if index_tmp:
            try:
                os.unlink(index_tmp)
            except OSError:
                pass


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