"""
Developer Workspace (Task 2.5)

Macro-uri pentru fluxul de lucru zilnic:
1. VS Code  — deschide fișiere/foldere, proiecte recente
2. Server HTTP local — pornește/oprește python -m http.server
3. Git — status, log, branch management, commit, push, stash

Comenzile distructive (commit, push, merge, stash drop) au
necesita_confirmare=True și trec prin mecanismul din Task 2.3.
"""

import subprocess
import os
import signal
from src.core.registry import unealta

# ---- PID-ul serverului HTTP activ (dacă e pornit) ----
# Îl ținem în memorie ca să-l putem opri la cerere.
_server_pid: int | None = None


# ==============================================================
# VS CODE
# ==============================================================

@unealta(
    description=(
        "Deschide VS Code. Poate deschide: directorul curent (fără argument), "
        "un fișier specific, sau un folder/proiect specific. "
        "Folosește când utilizatorul zice 'deschide editorul', "
        "'deschide VS Code', 'deschide fișierul X în editor' etc."
    ),
    parameters={
        "cale": {
            "type": "STRING",
            "description": (
                "Calea fișierului sau folderului de deschis. "
                "Lasă gol sau trimite '.' pentru directorul curent."
            ),
            "optional": True,
        }
    },
)
def deschide_vscode(cale: str = "."):
    """Deschide VS Code la calea specificată."""
    cale = cale.strip() or "."

    try:
        subprocess.Popen(
            ["code", cale],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"VS Code deschis la '{cale}'."
    except FileNotFoundError:
        return (
            "Comanda 'code' nu a fost găsită. Verifică dacă VS Code e instalat "
            "și dacă 'code' e în PATH (în VS Code: Ctrl+Shift+P → "
            "'Shell Command: Install code command in PATH')."
        )
    except Exception as e:
        return f"Eroare la deschiderea VS Code: {str(e)}"


# ==============================================================
# SERVER HTTP LOCAL
# ==============================================================

@unealta(
    description=(
        "Pornește un server HTTP local folosind Python. Util pentru a servi "
        "fișiere HTML/CSS/JS local în browser. Folosește când utilizatorul zice "
        "'pornește serverul', 'vreau să văd site-ul local', 'servește folderul X' etc."
    ),
    parameters={
        "port": {
            "type": "INTEGER",
            "description": "Portul pe care să ruleze serverul. Default: 8000.",
            "optional": True,
        },
        "folder": {
            "type": "STRING",
            "description": (
                "Folderul de servit. Default: directorul curent. "
                "Trimite calea absolută sau relativă."
            ),
            "optional": True,
        },
    },
)
def porneste_server_http(port: int = 8000, folder: str = "."):
    """Pornește python -m http.server în fundal."""
    global _server_pid

    if _server_pid is not None:
        return (
            f"Un server HTTP e deja activ pe PID {_server_pid}. "
            f"Oprește-l mai întâi cu 'opreste_server_http' înainte să pornești altul."
        )

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return f"Folderul '{folder}' nu există."

    try:
        proces = subprocess.Popen(
            ["python", "-m", "http.server", str(port), "--directory", folder],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _server_pid = proces.pid
        return (
            f"Server HTTP pornit pe http://localhost:{port} "
            f"(servește '{folder}', PID {_server_pid})."
        )
    except Exception as e:
        return f"Eroare la pornirea serverului: {str(e)}"


@unealta(
    description=(
        "Oprește serverul HTTP local dacă rulează. "
        "Folosește când utilizatorul zice 'oprește serverul', 'închide serverul HTTP' etc."
    ),
    necesita_confirmare=False,  # oprirea serverului propriu nu e distructivă
)
def opreste_server_http():
    """Oprește serverul HTTP local pornit de Jarvis."""
    global _server_pid

    if _server_pid is None:
        return "Nu există niciun server HTTP activ pornit de Jarvis."

    try:
        os.kill(_server_pid, signal.SIGTERM)
        pid_oprit = _server_pid
        _server_pid = None
        return f"Server HTTP (PID {pid_oprit}) oprit."
    except ProcessLookupError:
        _server_pid = None
        return "Serverul nu mai era activ (probabil s-a închis singur)."
    except Exception as e:
        return f"Eroare la oprirea serverului: {str(e)}"


@unealta(
    description="Returnează statusul serverului HTTP local — dacă rulează și pe ce port.",
)
def status_server_http():
    """Verifică dacă serverul HTTP local e activ."""
    if _server_pid is None:
        return "Niciun server HTTP activ."

    # Verificăm dacă procesul mai există efectiv
    try:
        os.kill(_server_pid, 0)  # signal 0 = verificare existență, fără efect
        return f"Server HTTP activ pe PID {_server_pid}."
    except ProcessLookupError:
        _server_pid_local = _server_pid
        return f"Serverul (PID {_server_pid_local}) nu mai rulează (s-a închis extern)."


# ==============================================================
# GIT
# ==============================================================

def _ruleaza_git(args: list[str], cale_repo: str = ".") -> str:
    """Helper intern — rulează o comandă git și returnează outputul."""
    try:
        rezultat = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.abspath(cale_repo),
        )
        output = rezultat.stdout.strip()
        eroare = rezultat.stderr.strip()

        if rezultat.returncode != 0:
            return f"Eroare git: {eroare or output}"

        return output or "Comandă executată (fără output)."
    except FileNotFoundError:
        return "Git nu e instalat sau nu e în PATH."
    except subprocess.TimeoutExpired:
        return "Comanda git a durat prea mult și a fost întreruptă."
    except Exception as e:
        return f"Eroare la rularea git: {str(e)}"


@unealta(
    description=(
        "Returnează statusul curent al repo-ului git: fișiere modificate, "
        "staged, untracked. Folosește pentru 'ce am modificat?', 'git status' etc."
    ),
    parameters={
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        }
    },
    max_linii=50,
)
def git_status(cale_repo: str = "."):
    return _ruleaza_git(["status"], cale_repo)


@unealta(
    description=(
        "Afișează istoricul de commit-uri git. Returnează ultimele N commit-uri "
        "cu hash, autor, dată și mesaj. Folosește pentru 'ce am făcut recent?', "
        "'arată-mi commit-urile', 'git log' etc."
    ),
    parameters={
        "numar": {
            "type": "INTEGER",
            "description": "Câte commit-uri să afișeze. Default: 10.",
            "optional": True,
        },
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        },
    },
    max_linii=40,
)
def git_log(numar: int = 10, cale_repo: str = "."):
    return _ruleaza_git(
        ["log", f"--max-count={numar}", "--oneline", "--decorate"],
        cale_repo
    )


@unealta(
    description=(
        "Listează branch-urile git disponibile (locale și remote). "
        "Marchează branch-ul curent. Folosește pentru 'pe ce branch sunt?', "
        "'ce branch-uri am?', 'arată branch-urile' etc."
    ),
    parameters={
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        }
    },
    max_linii=30,
)
def git_branches(cale_repo: str = "."):
    return _ruleaza_git(["branch", "-a"], cale_repo)


@unealta(
    description=(
        "Schimbă branch-ul activ sau creează unul nou. "
        "Folosește pentru 'treci pe branch-ul X', 'creează branch Y', "
        "'checkout la main' etc."
    ),
    parameters={
        "branch": {
            "type": "STRING",
            "description": "Numele branch-ului de activat.",
        },
        "creeaza_nou": {
            "type": "STRING",
            "description": "Dacă 'da', creează branch-ul dacă nu există (-b).",
            "optional": True,
        },
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def git_checkout(branch: str, creeaza_nou: str = "nu", cale_repo: str = "."):
    args = ["checkout"]
    if creeaza_nou.lower() in ("da", "yes", "true"):
        args.append("-b")
    args.append(branch)
    return _ruleaza_git(args, cale_repo)


@unealta(
    description=(
        "Adaugă fișiere la staging (git add). Folosește înainte de commit. "
        "Trimite '.' pentru a adăuga toate fișierele modificate."
    ),
    parameters={
        "fisiere": {
            "type": "STRING",
            "description": "Fișierele de adăugat. '.' pentru toate.",
        },
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def git_add(fisiere: str = ".", cale_repo: str = "."):
    return _ruleaza_git(["add", fisiere], cale_repo)


@unealta(
    description=(
        "Creează un commit git cu mesajul specificat. "
        "Asigură-te că ai rulat git_add înainte. "
        "Folosește pentru 'commit cu mesajul X', 'salvează modificările' etc."
    ),
    parameters={
        "mesaj": {
            "type": "STRING",
            "description": "Mesajul commit-ului.",
        },
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def git_commit(mesaj: str, cale_repo: str = "."):
    return _ruleaza_git(["commit", "-m", mesaj], cale_repo)


@unealta(
    description=(
        "Trimite commit-urile locale pe remote (git push). "
        "Folosește pentru 'push', 'trimite pe GitHub', 'urcă modificările' etc."
    ),
    parameters={
        "remote": {
            "type": "STRING",
            "description": "Remote-ul destinație. Default: origin.",
            "optional": True,
        },
        "branch": {
            "type": "STRING",
            "description": "Branch-ul de push. Default: branch-ul curent.",
            "optional": True,
        },
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def git_push(remote: str = "origin", branch: str = "", cale_repo: str = "."):
    args = ["push", remote]
    if branch:
        args.append(branch)
    return _ruleaza_git(args, cale_repo)


@unealta(
    description=(
        "Salvează temporar modificările nestagiate în stash (git stash). "
        "Util când vrei să schimbi branch-ul fără să pierzi munca în curs."
    ),
    parameters={
        "mesaj": {
            "type": "STRING",
            "description": "Mesaj descriptiv pentru stash (opțional).",
            "optional": True,
        },
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def git_stash(mesaj: str = "", cale_repo: str = "."):
    args = ["stash", "push"]
    if mesaj:
        args += ["-m", mesaj]
    return _ruleaza_git(args, cale_repo)


@unealta(
    description=(
        "Aplică ultimul stash salvat (git stash pop). "
        "Restaurează modificările salvate anterior cu git_stash."
    ),
    parameters={
        "cale_repo": {
            "type": "STRING",
            "description": "Calea repo-ului. Default: directorul curent.",
            "optional": True,
        }
    },
    necesita_confirmare=True,
)
def git_stash_pop(cale_repo: str = "."):
    return _ruleaza_git(["stash", "pop"], cale_repo)