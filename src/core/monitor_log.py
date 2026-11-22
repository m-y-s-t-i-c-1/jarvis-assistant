"""
Monitorizare Telemetrie & Log-uri de Sistem (Task 6.8)

A doua piesă (alături de monitor_ecran.py) din "Fluxul Senzorial Bazat pe
Evenimente". Urmărește în fundal jurnalul de sistem (journalctl) și
mesajele kernel (dmesg), căutând local pattern-uri de eroare — FĂRĂ NICIUN
apel API pentru detectare.

De ce nu trece prin Gemini ca monitor_ecran.py: aici sursa e deja text
structurat, nu o imagine care are nevoie de "interpretare". Un pattern
regex ("Traceback", "segfault", "kernel panic", "Failed to start") e deja
un semnal de încredere suficient de precis. Asta face acest monitor mai
ieftin (zero cost) și mai rapid (instant) decât cel vizual.

Surse urmărite (ambele în "follow mode", ca `tail -f`):
    - journalctl --user -f -p err..emerg   (jurnalul sesiunii utilizatorului,
                                              doar priorități eroare și mai grave)
    - dmesg -w                              (mesaje kernel: segfault, OOM
                                              killer, erori hardware)

Permisiuni: pe unele sisteme, `dmesg` sau `journalctl` (jurnalul complet
de sistem, nu doar --user) pot necesita apartenența la grupul
`systemd-journal` sau rulare cu privilegii. Dacă o sursă nu poate fi
citită, monitorul aferent se oprește silențios cu un mesaj explicativ —
NU blochează pornirea restului lui Jarvis.
"""

import atexit
import re
import subprocess
import threading
import time

# Pattern-uri care indică o problemă reală, nu zgomot normal de log
PATTERN_EROARE = re.compile(
    r"\b(error|fatal|critical|segfault|panic|failed|exception|traceback|"
    r"out of memory|oom[-_]?killer|core dumped|denied)\b",
    re.IGNORECASE,
)

# Răcire minimă între alerte, separat per sursă — identic ca filozofie cu
# monitor_ecran.py, ca să nu spamăm dacă o sursă produce erori repetitive
RACIRE_MINIMA = 30

_ultima_alerta = {"jurnal sistem": 0.0, "kernel": 0.0}
_opreste_monitorizarea = threading.Event()

# FIX: procesele `journalctl -f` / `dmesg -W` nu erau oprite niciodată: bucla citește stdout și verifică
# semnalul de oprire doar când sosește o linie nouă, deci pe un sistem liniștit procesele rămâneau orfane
# după ieșirea din Jarvis (și la fiecare repornire se adunau încă două).
_procese: list = []
_thread_uri: list = []
_lock_procese = threading.Lock()


def _termina_procesele() -> None:
    with _lock_procese:
        de_oprit = list(_procese)
        _procese.clear()
    for proces in de_oprit:
        try:
            if proces.poll() is None:
                proces.terminate()
        except Exception:
            pass


atexit.register(_termina_procesele)


def _urmareste_sursa(comanda: list[str], nume_sursa: str, callback):
    """Pornește un subproces în follow mode și verifică fiecare linie nouă."""
    try:
        proces = subprocess.Popen(
            comanda,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print(f"[Monitor Log] '{comanda[0]}' nu e instalat — monitorizare '{nume_sursa}' dezactivată.")
        return
    except Exception as e:
        print(f"[Monitor Log] Eroare la pornirea monitorizării '{nume_sursa}': {e}")
        return

    with _lock_procese:
        _procese.append(proces)
    print(f"[Monitor Log] Urmărire activă: {nume_sursa}")

    for linie in proces.stdout:
        if _opreste_monitorizarea.is_set():
            proces.terminate()
            return

        if not PATTERN_EROARE.search(linie):
            continue

        acum = time.time()
        if acum - _ultima_alerta.get(nume_sursa, 0.0) < RACIRE_MINIMA:
            continue
        _ultima_alerta[nume_sursa] = acum

        try:
            callback(nume_sursa, linie.strip())
        except Exception as e:
            print(f"[Monitor Log] Eroare în callback: {e}")

    # Dacă stdout s-a închis (procesul a murit), verificăm dacă a fost
    # din motive de permisiuni, ca să dăm un mesaj util în loc de tăcere.
    proces.wait(timeout=2) if proces.poll() is None else None
    if proces.returncode not in (0, None, -15):  # -15 = terminat de noi (SIGTERM)
        eroare = proces.stderr.read() if proces.stderr else ""
        print(f"[Monitor Log] '{nume_sursa}' s-a oprit neașteptat (cod {proces.returncode}). {eroare[:200]}")


def porneste_monitorizare_log(callback):
    """
    Pornește urmărirea jurnalului de sistem și a mesajelor kernel, în
    thread-uri daemon separate.

    Parametri:
        callback: funcție callback(sursa: str, linie: str), apelată doar
                  când o linie nouă se potrivește PATTERN_EROARE, cu
                  răcire de RACIRE_MINIMA secunde per sursă.
    """
    # FIX: un al doilea apel dubla urmăririle (și alertele) cât timp primele rulau încă.
    _thread_uri[:] = [t for t in _thread_uri if t.is_alive()]
    if _thread_uri:
        return
    _opreste_monitorizarea.clear()

    surse = (
        (["journalctl", "--user", "-f", "-n", "0", "-p", "err..emerg", "--no-pager"], "jurnal sistem"),
        (["dmesg", "-W"], "kernel"),
    )
    for comanda, nume in surse:
        t = threading.Thread(target=_urmareste_sursa, args=(comanda, nume, callback), daemon=True)
        _thread_uri.append(t)
        t.start()


def opreste_monitorizarea_log():
    """Oprește ambele thread-uri de monitorizare log, dacă rulează."""
    _opreste_monitorizarea.set()
    _termina_procesele()  # altfel thread-urile rămân blocate pe stdout până la următoarea linie de log
    print("[Monitor Log] Oprit.")


if __name__ == "__main__":
    def _test_callback(sursa: str, linie: str):
        print(f">>> EROARE DETECTATĂ [{sursa}]: {linie}")

    porneste_monitorizare_log(_test_callback)
    print("Monitorizare log activă. Ctrl+C pentru oprire.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        opreste_monitorizarea_log()