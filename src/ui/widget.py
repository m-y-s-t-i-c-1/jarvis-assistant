"""
Container Desktop Nativ și Transparența (Task 5.7)

Transformă interfața web Jarvis într-un widget flotant transparent
pe desktop, folosind pywebview — o fereastră nativă fără borduri,
mereu deasupra celorlalte ferestre, cu fundal transparent.

Arhitectură:
    1. Pornește serverul FastAPI într-un thread separat
    2. Așteaptă ca serverul să fie ready (health check)
    3. Deschide fereastra pywebview cu proprietăți de widget:
       - fără borduri (frameless)
       - mereu deasupra (on_top)
       - fundal transparent
       - poziționată în colțul dorit al ecranului

Comenzi rapide (configurabile):
    Ctrl+Shift+J  — arată/ascunde widget-ul
    Ctrl+Shift+Q  — închide complet

Rulare:
    python -m src.ui.widget

Notă Linux/Wayland:
    pywebview pe Wayland poate necesita GTK backend.
    Dacă nu merge, încearcă: PYWEBVIEW_GUI=gtk python -m src.ui.widget
    Sau pe X11: DISPLAY=:0 python -m src.ui.widget
"""

import threading
import time
import sys
import os
import subprocess
import socket

# ── Configurare widget ────────────────────────────────────────────────────────

TITLU        = "Jarvis"
LATIME       = 420       # lățimea widget-ului în pixeli
INALTIME     = 680       # înălțimea widget-ului în pixeli
PORT_SERVER  = 8080
URL_UI       = f"http://localhost:{PORT_SERVER}"

# Poziția pe ecran: "dreapta-jos", "dreapta-sus", "stanga-jos", "stanga-sus", "centru"
POZITIE      = "dreapta-jos"
MARGINE      = 20        # distanța față de marginea ecranului

# ── Helpers ───────────────────────────────────────────────────────────────────

def _serverul_ruleaza(port: int, timeout: float = 0.5) -> bool:
    """Verifică dacă serverul FastAPI ascultă pe portul dat."""
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def _calculeaza_pozitie(latime: int, inaltime: int, pozitie: str, margine: int):
    """
    Calculează coordonatele x, y ale ferestrei pe ecran.
    Folosește xrandr pentru a detecta rezoluția reală a ecranului.
    """
    try:
        output = subprocess.check_output(
            ["xrandr", "--current"],
            text=True, stderr=subprocess.DEVNULL
        )
        # Căutăm "current WxH" în output xrandr
        for line in output.split("\n"):
            if "current" in line:
                parts = line.split("current")[1].strip().split(",")[0].strip()
                w_scr, h_scr = map(int, parts.split(" x "))
                break
        else:
            w_scr, h_scr = 1920, 1080  # fallback
    except Exception:
        w_scr, h_scr = 1920, 1080

    pozitii = {
        "dreapta-jos":  (w_scr - latime - margine,  h_scr - inaltime - margine),
        "dreapta-sus":  (w_scr - latime - margine,  margine),
        "stanga-jos":   (margine,                    h_scr - inaltime - margine),
        "stanga-sus":   (margine,                    margine),
        "centru":       ((w_scr - latime) // 2,     (h_scr - inaltime) // 2),
    }
    return pozitii.get(pozitie, pozitii["dreapta-jos"])


def _porneste_server_background():
    """Pornește serverul FastAPI într-un thread daemon."""
    def _run():
        # Importăm uvicorn și app-ul din server.py
        import uvicorn
        from src.ui.server import app
        uvicorn.run(app, host="0.0.0.0", port=PORT_SERVER, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _asteapta_server(timeout: float = 60.0) -> bool:
    """Blochează până serverul e ready sau timeout."""
    print(f"[Widget] Aștept serverul pe portul {PORT_SERVER}...")
    start = time.time()
    while time.time() - start < timeout:
        if _serverul_ruleaza(PORT_SERVER):
            print("[Widget] Server ready.")
            return True
        time.sleep(0.3)
    print("[Widget] Timeout — serverul nu a pornit.")
    return False


# ── Widget principal ──────────────────────────────────────────────────────────

def porneste_widget():
    """
    Pornește serverul FastAPI (dacă nu rulează deja) și deschide
    fereastra pywebview ca widget flotant transparent.
    """
    try:
        import webview
    except ImportError:
        print("[Widget] pywebview nu e instalat. Rulează: pip install pywebview")
        sys.exit(1)

    # Pornim serverul dacă nu e deja activ
    if not _serverul_ruleaza(PORT_SERVER):
        print("[Widget] Pornesc serverul FastAPI...")
        _porneste_server_background()
        if not _asteapta_server():
            print("[Widget] Nu pot porni serverul. Ieșire.")
            sys.exit(1)
    else:
        print(f"[Widget] Serverul deja rulează pe portul {PORT_SERVER}.")

    # Calculăm poziția pe ecran
    x, y = _calculeaza_pozitie(LATIME, INALTIME, POZITIE, MARGINE)
    print(f"[Widget] Poziție: {POZITIE} → x={x}, y={y}")

    # Creăm fereastra pywebview
    fereastra = webview.create_window(
        title=TITLU,
        url=URL_UI,
        width=LATIME,
        height=INALTIME,
        x=x,
        y=y,
        resizable=True,
        frameless=True,       # fără bara de titlu nativă
        on_top=True,          # mereu deasupra celorlalte ferestre
        transparent=True,     # fundal transparent (funcționează cu compositor)
        background_color="#000000",
        min_size=(300, 400),
    )

    print(f"[Widget] Jarvis UI pornit la {URL_UI}")
    print("[Widget] Apasă Ctrl+C în terminal pentru a închide.\n")

    # Pornim GUI-ul — blochează până utilizatorul închide fereastra
    webview.start(debug=False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jarvis Widget")
    parser.add_argument("--pozitie", default=POZITIE,
        choices=["dreapta-jos", "dreapta-sus", "stanga-jos", "stanga-sus", "centru"],
        help="Poziția widget-ului pe ecran")
    parser.add_argument("--latime",  type=int, default=LATIME,  help="Lățimea în pixeli")
    parser.add_argument("--inaltime",type=int, default=INALTIME,help="Înălțimea în pixeli")
    parser.add_argument("--port",    type=int, default=PORT_SERVER, help="Portul serverului")
    args = parser.parse_args()

    POZITIE     = args.pozitie
    LATIME      = args.latime
    INALTIME    = args.inaltime
    PORT_SERVER = args.port
    URL_UI = f"http://localhost:{PORT_SERVER}/desktop"
    porneste_widget()