"""
Execuția Asincronă și Gestionarea Joburilor în Fundal (Task 2.8)

Permite rularea unor comenzi lungi (subprocess) într-un thread separat,
fără să blocheze conversația. Jarvis poate continua să răspundă la
întrebări cât timp un job rulează, și anunță proactiv când se termină.

Arhitectură:
    - JobManager ține evidența tuturor joburilor (id, status, rezultat).
    - Fiecare job rulează într-un thread separat (subprocess.run e blocant
      DOAR în acel thread, nu blochează main thread-ul unde rulează agent_loop).
    - Un thread "watcher" separat verifică periodic joburile terminate
      și le afișează proactiv în terminal, chiar dacă utilizatorul e la input().

Notă despre UX în terminal: anunțul proactiv poate întrerupe vizual linia
de input curentă a utilizatorului. E un compromis acceptat — în Faza 5
(UI web) asta se rezolvă elegant cu WebSockets; în terminal e inerent.
"""

import threading
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class StatusJob(Enum):
    IN_DESFASURARE = "în desfășurare"
    TERMINAT = "terminat"
    EROARE = "eroare"


@dataclass
class Job:
    id: str
    descriere: str
    status: StatusJob = StatusJob.IN_DESFASURARE
    rezultat: str = ""
    anuntat: bool = False  # True după ce watcher-ul l-a afișat proactiv


class JobManager:
    """
    Ține evidența tuturor joburilor pornite. Thread-safe prin lock,
    pentru că main thread-ul (agent_loop) și watcher thread-ul îl
    accesează concurent.
    """

    def __init__(self):
        self._joburi: dict[str, Job] = {}
        self._lock = threading.Lock()

    def porneste_job(self, comanda: list[str], descriere: str, cwd: str = ".") -> str:
        """
        Pornește o comandă într-un thread separat. Returnează imediat ID-ul
        jobului, fără să aștepte finalizarea.
        """
        job_id = str(uuid.uuid4())[:8]  # ID scurt, suficient pentru uz uman
        job = Job(id=job_id, descriere=descriere)

        with self._lock:
            self._joburi[job_id] = job

        thread = threading.Thread(
            target=self._ruleaza_in_fundal,
            args=(job_id, comanda, cwd),
            daemon=True,  # nu blochează închiderea programului
        )
        thread.start()

        return job_id

    def _ruleaza_in_fundal(self, job_id: str, comanda: list[str], cwd: str):
        """Execută efectiv comanda. Rulează în thread-ul dedicat jobului."""
        try:
            rezultat = subprocess.run(
                comanda,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=600,  # 10 minute — plasă de siguranță pentru joburi blocate
            )

            with self._lock:
                job = self._joburi[job_id]
                if rezultat.returncode == 0:
                    job.status = StatusJob.TERMINAT
                    job.rezultat = rezultat.stdout.strip() or "Comandă executată cu succes."
                else:
                    job.status = StatusJob.EROARE
                    job.rezultat = rezultat.stderr.strip() or "Comandă eșuată."

        except subprocess.TimeoutExpired:
            with self._lock:
                job = self._joburi[job_id]
                job.status = StatusJob.EROARE
                job.rezultat = "Job întrerupt: a depășit limita de 10 minute."

        except Exception as e:
            with self._lock:
                job = self._joburi[job_id]
                job.status = StatusJob.EROARE
                job.rezultat = f"Eroare neașteptată: {str(e)}"

    def status_job(self, job_id: str) -> str:
        """Returnează statusul curent al unui job specific."""
        with self._lock:
            job = self._joburi.get(job_id)

        if job is None:
            return f"Nu există niciun job cu ID-ul '{job_id}'."

        return (
            f"Job {job.id} ({job.descriere}): {job.status.value}. "
            f"{job.rezultat if job.status != StatusJob.IN_DESFASURARE else ''}"
        )

    def toate_joburile(self) -> str:
        """Returnează un sumar al tuturor joburilor (active și terminate)."""
        with self._lock:
            joburi = list(self._joburi.values())

        if not joburi:
            return "Niciun job pornit până acum."

        linii = []
        for job in joburi:
            linii.append(f"[{job.id}] {job.descriere} — {job.status.value}")

        return "\n".join(linii)

    def joburi_neanuntate(self) -> list[Job]:
        """
        Returnează joburile terminate (succes sau eroare) care încă nu au
        fost anunțate proactiv. Le marchează ca anunțate înainte de a le
        returna, ca să nu fie anunțate de două ori.
        """
        with self._lock:
            neanuntate = [
                job for job in self._joburi.values()
                if job.status != StatusJob.IN_DESFASURARE and not job.anuntat
            ]
            for job in neanuntate:
                job.anuntat = True

        return neanuntate


# Instanță globală unică, folosită de unelte și de main.py
manager_joburi = JobManager()


def porneste_thread_watcher(interval_secunde: float = 2.0):
    """
    Pornește thread-ul care verifică periodic joburile terminate și le
    afișează proactiv în terminal. Apelat o singură dată din main.py,
    la pornirea programului.
    """
    def bucla_watcher():
        while True:
            time.sleep(interval_secunde)
            joburi_noi = manager_joburi.joburi_neanuntate()

            for job in joburi_noi:
                simbol = "✅" if job.status == StatusJob.TERMINAT else "❌"
                print(
                    f"\n{simbol} [Job în fundal terminat] {job.descriere} "
                    f"({job.status.value}): {job.rezultat}\nTu: ",
                    end="",
                    flush=True,
                )

    thread = threading.Thread(target=bucla_watcher, daemon=True)
    thread.start()