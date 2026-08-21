"""
Monitorizare Ecran — Trigger Vizual Bazat pe Evenimente (Task 6.5)

Prima piesă din "Fluxul Senzorial Bazat pe Evenimente" (arhitectura Vasea,
secțiunea 1). Înlocuiește polling-ul de tip "trimite fiecare screenshot
la LLM" cu un trigger LOCAL, gratuit, instant:

    1. La fiecare INTERVAL_VERIFICARE secunde, facem un screenshot redus
       la 8x8 pixeli grayscale și calculăm un "hash perceptual" (average
       hash / aHash) — o amprentă de 64 biți a conținutului vizual general.
    2. Comparăm hash-ul curent cu ultimul salvat (distanță Hamming =
       câți biți diferă din 64).
    3. Dacă distanța depășește PRAG_SCHIMBARE, N verificări la rând
       (CONFIRMARI_NECESARE), ȘI a trecut cel puțin RACIRE_MINIMA secunde
       de la ultima declanșare -> DOAR ATUNCI apelăm callback-ul.

De ce aHash și nu diff pixel-cu-pixel: aHash e insensibil la mișcări
minore de cursor, la un ceas care ticăie în system tray, la animații
mici — dar prinde schimbări reale (fereastră nouă, tab schimbat, dialog
apărut). Diff pixel-cu-pixel ar declanșa aproape constant.

De ce RACIRE_MINIMA: fără ea, dacă lucrezi activ (tastezi, dai scroll),
ecranul se schimbă vizual la fiecare verificare — callback-ul (care de
obicei costă un apel API, ex: vezi_ecranul) s-ar declanșa la fiecare
3 secunde și ai epuiza cota gratuită în minute. Cu răcire de 30s, chiar
dacă ecranul se schimbă constant, callback-ul nu rulează mai des de o
dată la 30 de secunde.

NU face niciun apel API în sine — doar screenshot local (pyautogui,
deja dependență din vedere.py) + hash. Cost efectiv zero.
"""

import time
import threading
from PIL import Image
import pyautogui

# ── Configurare ──────────────────────────────────────────────────────────
INTERVAL_VERIFICARE = 3        # secunde între verificări locale (ieftine)
PRAG_SCHIMBARE = 6             # distanță Hamming minimă (din 64 biți) ca să conteze
DIMENSIUNE_HASH = 8            # 8x8 = 64 biți
CONFIRMARI_NECESARE = 1        # câte verificări la rând confirmă schimbarea
RACIRE_MINIMA = 30             # secunde minime între două apeluri de callback


def _hash_perceptual(imagine: Image.Image) -> int:
    """Average hash (aHash): 8x8 grayscale, comparat cu media, întreg pe 64 biți."""
    mica = imagine.convert("L").resize((DIMENSIUNE_HASH, DIMENSIUNE_HASH), Image.LANCZOS)
    pixeli = list(mica.getdata())
    medie = sum(pixeli) / len(pixeli)

    hash_val = 0
    for i, p in enumerate(pixeli):
        if p > medie:
            hash_val |= (1 << i)
    return hash_val


def _distanta_hamming(a: int, b: int) -> int:
    """Câți biți diferă între două hash-uri — 0 = identice, 64 = complet diferite."""
    return bin(a ^ b).count("1")


class MonitorEcran:
    """Ține evidența ultimului hash de ecran, detectează schimbări, fără apeluri API."""

    def __init__(self):
        self._ultimul_hash: int | None = None
        self._schimbari_consecutive = 0
        self._ultima_declansare: float = 0.0

    def a_aparut_schimbare(self) -> bool:
        """
        Screenshot -> hash -> comparație. Actualizează starea internă.
        Returnează True doar dacă schimbarea e confirmată ȘI perioada de
        răcire de la ultima declanșare a trecut.
        """
        try:
            imagine = pyautogui.screenshot()
        except Exception as e:
            print(f"[Monitor Ecran] Eroare la screenshot: {e}")
            return False

        hash_curent = _hash_perceptual(imagine)

        if self._ultimul_hash is None:
            self._ultimul_hash = hash_curent
            return False

        distanta = _distanta_hamming(hash_curent, self._ultimul_hash)
        self._ultimul_hash = hash_curent

        if distanta < PRAG_SCHIMBARE:
            self._schimbari_consecutive = 0
            return False

        self._schimbari_consecutive += 1
        if self._schimbari_consecutive < CONFIRMARI_NECESARE:
            return False

        acum = time.time()
        if acum - self._ultima_declansare < RACIRE_MINIMA:
            return False  # schimbare reală, dar suntem încă în perioada de răcire

        self._ultima_declansare = acum
        self._schimbari_consecutive = 0
        return True


_monitor_global = MonitorEcran()
_thread_monitorizare: threading.Thread | None = None
_opreste_monitorizarea = threading.Event()


def porneste_monitorizare_ecran(callback, interval: float = INTERVAL_VERIFICARE):
    """
    Pornește un thread daemon care verifică local ecranul și apelează
    `callback()` DOAR când detectează o schimbare vizuală semnificativă,
    confirmată, și în afara perioadei de răcire.

    Parametri:
        callback:  funcție fără argumente, apelată la fiecare schimbare
                   confirmată. Aici se conectează, de exemplu, vezi_ecranul
                   + salvare în memorie, sau o notificare către Vasea.
        interval:  secunde între verificările locale (default 3, ieftin).
    """
    global _thread_monitorizare
    _opreste_monitorizarea.clear()

    def bucla():
        print(f"[Monitor Ecran] Pornit — verificare locală la fiecare {interval}s, "
              f"răcire minimă {RACIRE_MINIMA}s între declanșări.")
        while not _opreste_monitorizarea.is_set():
            if _monitor_global.a_aparut_schimbare():
                print("[Monitor Ecran] Schimbare vizuală confirmată — declanșez callback-ul.")
                try:
                    callback()
                except Exception as e:
                    print(f"[Monitor Ecran] Eroare în callback: {e}")
            time.sleep(interval)

    _thread_monitorizare = threading.Thread(target=bucla, daemon=True)
    _thread_monitorizare.start()


def opreste_monitorizarea():
    """Oprește thread-ul de monitorizare, dacă rulează."""
    _opreste_monitorizarea.set()
    print("[Monitor Ecran] Oprit.")


if __name__ == "__main__":
    def _test_callback():
        print(">>> SCHIMBARE DETECTATĂ! (aici ai conecta vezi_ecranul)")

    porneste_monitorizare_ecran(_test_callback, interval=2)
    print("Monitorizare activă. Schimbă ceva pe ecran... Ctrl+C pentru oprire.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        opreste_monitorizarea()