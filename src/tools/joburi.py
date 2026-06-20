"""
Unelte pentru joburi în fundal (Task 2.8).

Expune JobManager din core/jobs.py către Gemini. Util pentru comenzi care
durează mult — build-uri, download-uri mari, compilări, sincronizări etc.
"""

from src.core.registry import unealta
from src.core.jobs import manager_joburi


@unealta(
    description=(
        "Pornește o comandă lungă în fundal, fără să blocheze conversația. "
        "Folosește pentru comenzi care pot dura mult (build, compilare, "
        "download, sincronizare etc.) — orice de unde NU ai nevoie de "
        "rezultat imediat. Jarvis va anunța automat când jobul se termină. "
        "NU folosi pentru comenzi rapide — pentru alea, folosește direct "
        "uneltele existente (ruleaza_comanda_info, comenzile git, etc.)."
    ),
    parameters={
        "comanda": {
            "type": "STRING",
            "description": (
                "Comanda de rulat, ca text simplu, cu argumente separate prin "
                "spațiu (ex: 'npm run build', 'python script.py'). Va fi "
                "împărțită automat în argumente."
            ),
        },
        "descriere": {
            "type": "STRING",
            "description": "O descriere scurtă a ce face jobul, pentru identificare ulterioară.",
        },
        "director": {
            "type": "STRING",
            "description": "Directorul în care să ruleze comanda. Default: directorul curent.",
            "optional": True,
        },
    },
    necesita_confirmare=True,  # poate rula orice comandă — trece prin blacklist + confirmare
)
def porneste_job_fundal(comanda: str, descriere: str, director: str = "."):
    """Pornește o comandă lungă într-un thread separat și returnează ID-ul jobului."""
    parti_comanda = comanda.split()

    if not parti_comanda:
        return "Comanda e goală, nu am ce rula."

    job_id = manager_joburi.porneste_job(parti_comanda, descriere, cwd=director)
    return (
        f"Job pornit în fundal: '{descriere}' (ID: {job_id}). "
        f"Te anunț automat când se termină, sau poți întreba 'cum merge jobul {job_id}'."
    )


@unealta(
    description=(
        "Verifică statusul unui job specific pornit anterior în fundal. "
        "Folosește când utilizatorul întreabă 'cum merge jobul X?', "
        "'s-a terminat build-ul?' etc."
    ),
    parameters={
        "job_id": {
            "type": "STRING",
            "description": "ID-ul jobului de verificat (oferit la pornirea jobului).",
        }
    },
)
def status_job_fundal(job_id: str):
    return manager_joburi.status_job(job_id)


@unealta(
    description=(
        "Listează toate joburile pornite în fundal, active sau terminate. "
        "Folosește pentru 'ce joburi rulează?', 'arată-mi toate joburile' etc."
    ),
    max_linii=20,
)
def listeaza_joburi_fundal():
    return manager_joburi.toate_joburile()