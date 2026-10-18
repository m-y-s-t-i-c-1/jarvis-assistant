"""
Monitorizarea Hardware și Controlul Perifericelor (Task 2.6)

Două categorii:
1. Monitorizare (read-only) — CPU, RAM, disc, temperatură dacă e disponibilă.
   Folosește psutil, nu parsare de text din comenzi shell.
2. Control periferice — volum (pactl, merge peste PipeWire/PulseAudio) și
   luminozitate (brightnessctl). Validăm intervalul [0, 100] ca să nu
   trimitem valori absurde către sistem.

Dependențe externe necesare pe sistem:
    pip install psutil
    sudo pacman -S brightnessctl   (pentru luminozitate)
    pactl ar trebui să existe deja cu PipeWire (pipewire-pulse)
"""

import subprocess
from src.core.registry import unealta

try:
    import psutil
    _PSUTIL_DISPONIBIL = True
except ImportError:
    _PSUTIL_DISPONIBIL = False


# ==============================================================
# MONITORIZARE HARDWARE (read-only)
# ==============================================================

@unealta(
    description=(
        "Returnează utilizarea curentă a procesorului (CPU), în procente. "
        "Folosește pentru 'cât e încărcat procesorul?', 'cum stă CPU-ul?' etc."
    ),
)
def status_cpu():
    """Procentul de utilizare CPU, măsurat pe o fereastră scurtă de timp."""
    if not _PSUTIL_DISPONIBIL:
        return "Librăria psutil nu e instalată. Rulează: pip install psutil"

    # interval=1 = măsoară pe parcursul a 1 secundă, pentru o citire reală
    # (fără interval, prima citire e adesea 0.0% sau nereprezentativă)
    procent = psutil.cpu_percent(interval=1)
    nuclee = psutil.cpu_count(logical=True)
    nuclee_fizice = psutil.cpu_count(logical=False)

    return (
        f"CPU: {procent}% utilizare curentă "
        f"({nuclee_fizice} nuclee fizice, {nuclee} thread-uri logice)."
    )


@unealta(
    description=(
        "Returnează utilizarea curentă a memoriei RAM: total, folosit, liber, "
        "în procente și GB. Folosește pentru 'câtă memorie am liberă?', "
        "'cum stă RAM-ul?' etc."
    ),
)
def status_ram():
    """Detalii despre utilizarea memoriei RAM."""
    if not _PSUTIL_DISPONIBIL:
        return "Librăria psutil nu e instalată. Rulează: pip install psutil"

    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    folosit_gb = mem.used / (1024 ** 3)
    liber_gb = mem.available / (1024 ** 3)

    return (
        f"RAM: {mem.percent}% utilizat — "
        f"{folosit_gb:.1f} GB folosiți din {total_gb:.1f} GB total "
        f"({liber_gb:.1f} GB disponibili)."
    )


@unealta(
    description=(
        "Returnează spațiul pe disc: total, folosit, liber, în procente și GB. "
        "Folosește pentru 'cât spațiu am pe disc?', 'mai am loc?' etc."
    ),
    parameters={
        "cale": {
            "type": "STRING",
            "description": "Punctul de montare de verificat. Default: '/' (rădăcina).",
            "optional": True,
        }
    },
)
def status_disc(cale: str = "/"):
    """Detalii despre spațiul pe disc pentru un anumit punct de montare."""
    if not _PSUTIL_DISPONIBIL:
        return "Librăria psutil nu e instalată. Rulează: pip install psutil"

    try:
        disc = psutil.disk_usage(cale)
    except FileNotFoundError:
        return f"Calea '{cale}' nu există."

    total_gb = disc.total / (1024 ** 3)
    folosit_gb = disc.used / (1024 ** 3)
    liber_gb = disc.free / (1024 ** 3)

    return (
        f"Disc ({cale}): {disc.percent}% utilizat — "
        f"{folosit_gb:.1f} GB folosiți din {total_gb:.1f} GB total "
        f"({liber_gb:.1f} GB liberi)."
    )


@unealta(
    description=(
        "Returnează temperatura senzorilor hardware disponibili (CPU, etc.), "
        "dacă sistemul expune aceste informații. Folosește pentru "
        "'cât de cald e procesorul?', 'ce temperatură are sistemul?' etc."
    ),
)
def status_temperatura():
    """Temperaturile senzorilor hardware, dacă sunt disponibile pe acest sistem."""
    if not _PSUTIL_DISPONIBIL:
        return "Librăria psutil nu e instalată. Rulează: pip install psutil"

    try:
        senzori = psutil.sensors_temperatures()
    except AttributeError:
        return "Citirea temperaturii nu e suportată pe acest sistem."

    if not senzori:
        return (
            "Nu am găsit senzori de temperatură accesibili. "
            "Pe unele laptopuri necesită module kernel suplimentare "
            "(ex: modprobe sau lm_sensors)."
        )

    rezultate = []
    for nume_senzor, citiri in senzori.items():
        for citire in citiri:
            eticheta = citire.label or nume_senzor
            rezultate.append(f"{eticheta}: {citire.current:.0f}°C")

    return " | ".join(rezultate)


@unealta(
    description=(
        "Returnează un rezumat general al stării sistemului: CPU, RAM și disc "
        "într-un singur răspuns. Folosește pentru 'cum stă sistemul?', "
        "'dă-mi un status general' etc."
    ),
)
def status_sistem_general():
    """Rezumat combinat CPU + RAM + disc, pentru o privire rapidă de ansamblu."""
    if not _PSUTIL_DISPONIBIL:
        return "Librăria psutil nu e instalată. Rulează: pip install psutil"

    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disc = psutil.disk_usage("/")

    return (
        f"CPU: {cpu}% | "
        f"RAM: {mem.percent}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB) | "
        f"Disc: {disc.percent}% ({disc.used / (1024**3):.1f}/{disc.total / (1024**3):.1f} GB)"
    )


# ==============================================================
# CONTROL PERIFERICE — VOLUM
# ==============================================================

@unealta(
    description=(
        "Setează volumul sistemului la un anumit procent (0-100). "
        "Folosește pentru 'pune volumul la X%', 'dă mai tare/încet' etc."
    ),
    parameters={
        "procent": {
            "type": "INTEGER",
            "description": "Nivelul de volum dorit, între 0 și 100.",
        }
    },
)
def seteaza_volum(procent: int):
    """Setează volumul sink-ului audio implicit, prin pactl."""
    procent = max(0, min(100, int(procent)))  # clamp în interval valid

    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{procent}%"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return f"Volum setat la {procent}%."
    except FileNotFoundError:
        return "Comanda 'pactl' nu a fost găsită. Verifică instalarea PipeWire/PulseAudio."
    except subprocess.CalledProcessError as e:
        return f"Eroare la setarea volumului: {e.stderr.strip()}"
    except Exception as e:
        return f"Eroare neașteptată: {str(e)}"


@unealta(
    description=(
        "Returnează nivelul curent al volumului sistemului, în procente. "
        "Folosește pentru 'cât e volumul?', 'ce volum am setat?' etc."
    ),
)
def status_volum():
    """Citește volumul curent al sink-ului audio implicit."""
    try:
        rezultat = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if rezultat.returncode != 0:
            return f"Eroare la citirea volumului: {rezultat.stderr.strip()}"
        return rezultat.stdout.strip()
    except FileNotFoundError:
        return "Comanda 'pactl' nu a fost găsită. Verifică instalarea PipeWire/PulseAudio."
    except Exception as e:
        return f"Eroare neașteptată: {str(e)}"


@unealta(
    description=(
        "Pornește sau oprește sunetul sistemului (mute/unmute). "
        "Folosește pentru 'oprește sunetul', 'pune pe silent', 'pornește sunetul' etc."
    ),
    parameters={
        "activeaza_mute": {
            "type": "STRING",
            "description": "Trimite 'da' pentru a opri sunetul, 'nu' pentru a-l reactiva.",
        }
    },
)
def comuta_mute(activeaza_mute: str):
    """Pornește sau oprește mute pe sink-ul audio implicit."""
    valoare = "1" if activeaza_mute.lower() in ("da", "yes", "true") else "0"

    try:
        subprocess.run(
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", valoare],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return "Sunet oprit (mute)." if valoare == "1" else "Sunet pornit."
    except FileNotFoundError:
        return "Comanda 'pactl' nu a fost găsită."
    except subprocess.CalledProcessError as e:
        return f"Eroare: {e.stderr.strip()}"


# ==============================================================
# CONTROL PERIFERICE — LUMINOZITATE
# ==============================================================

@unealta(
    description=(
        "Setează luminozitatea ecranului la un anumit procent (0-100). "
        "Folosește pentru 'pune luminozitatea la X%', 'fă ecranul mai luminos/întunecat' etc."
    ),
    parameters={
        "procent": {
            "type": "INTEGER",
            "description": "Nivelul de luminozitate dorit, între 0 și 100.",
        }
    },
)
def seteaza_luminozitate(procent: int):
    """Setează luminozitatea ecranului prin brightnessctl."""
    procent = max(0, min(100, int(procent)))  # clamp în interval valid

    try:
        subprocess.run(
            ["brightnessctl", "set", f"{procent}%"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return f"Luminozitate setată la {procent}%."
    except FileNotFoundError:
        return (
            "Comanda 'brightnessctl' nu a fost găsită. "
            "Instalează cu: sudo pacman -S brightnessctl "
            "(și adaugă-ți userul în grupul 'video' pentru acces fără sudo)."
        )
    except subprocess.CalledProcessError as e:
        return (
            f"Eroare la setarea luminozității: {e.stderr.strip()}. "
            f"Verifică dacă userul tău e în grupul 'video' "
            f"(sudo usermod -aG video $USER, apoi relogare)."
        )
    except Exception as e:
        return f"Eroare neașteptată: {str(e)}"


@unealta(
    description=(
        "Returnează nivelul curent al luminozității ecranului, în procente. "
        "Folosește pentru 'cât e luminozitatea?', 'ce luminozitate am?' etc."
    ),
)
def status_luminozitate():
    """Citește luminozitatea curentă prin brightnessctl."""
    try:
        rezultat = subprocess.run(
            ["brightnessctl", "g"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rezultat_max = subprocess.run(
            ["brightnessctl", "m"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if rezultat.returncode != 0:
            return f"Eroare la citirea luminozității: {rezultat.stderr.strip()}"

        curent = int(rezultat.stdout.strip())
        maxim = int(rezultat_max.stdout.strip())
        procent = round((curent / maxim) * 100)

        return f"Luminozitate: {procent}% ({curent}/{maxim})."
    except FileNotFoundError:
        return "Comanda 'brightnessctl' nu a fost găsită."
    except Exception as e:
        return f"Eroare neașteptată: {str(e)}"