"""
Serverul FastAPI (Task 5.1 + 5.2)

Routes:
    GET  /          — UI web principal (index.html)
    GET  /desktop   — UI widget desktop (index_desktop.html)
    WS   /ws        — WebSocket real-time
    GET  /api/stats — statistici hardware + memorie
    GET  /api/history — ultimele mesaje din sesiunea curentă
"""

import os
import json
import asyncio
import time
import itertools
import re
import threading
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

load_dotenv()

from src import tools  # noqa: F401
from src.core.agent import agent_loop
from src.core.database import db
from src.core.memory import memorie
from src.core.rag import rag
from src.core.context_manager import proceseaza as proceseaza_context
from google import genai
from google.genai import types

# ── Configurare ───────────────────────────────────────────────────────────────

UI_DIR = Path(__file__).parent / "static"
UI_DIR.mkdir(parents=True, exist_ok=True)

_gemini_chei = [
    os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
    for i in range(5)
]
_gemini_chei    = [k for k in _gemini_chei if k]
_gemini_clienti = [genai.Client(api_key=k) for k in _gemini_chei]
_gemini_rotatie = itertools.cycle(_gemini_clienti)

GEMINI_MODEL   = "gemini-3.6-flash"

HOST = os.getenv("JARVIS_HOST", "127.0.0.1")   # 0.0.0.0 doar explicit, via env
PORT = int(os.getenv("JARVIS_PORT", "8080"))
_ORIGINI_PERMISE = {
    f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}",
    *[o.strip() for o in os.getenv("JARVIS_ALLOWED_ORIGINS", "").split(",") if o.strip()],
}
ERORI_FALLBACK = (503, 429, 500, 403)

# FIX: `istoric` e global și e modificat din thread-uri de executor —
# două mesaje simultane se intercalau și corupeau istoricul.
_LOCK_AGENT = threading.Lock()

SYSTEM_PROMPT_BAZA = """Tu ești Jarvis, un asistent AI personal extrem de inteligent, polivalent și loial.
- Te adresezi întotdeauna utilizatorului cu "Vasea".
- Tonul tău este calm, profesionist, dar cu un strop de umor sec, britanic.
- Răspunzi concis și la obiect, fără să divaghezi inutil.
- Ești un expert universal: programare, știință, matematică, istorie, scriere creativă, eseuri — orice domeniu.
- Când utilizatorul întreabă ceva factual (oră, dată, vreme), folosești OBLIGATORIU uneltele disponibile.
- Răspunsurile tale pot conține Markdown — interfața le va reda corect.
"""

# ── State global ──────────────────────────────────────────────────────────────

istoric       = []
sesiune_id    = db.incepe_sesiune()
SYSTEM_PROMPT = memorie.construieste_system_prompt(SYSTEM_PROMPT_BAZA)

conexiuni_active: list[WebSocket] = []

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Jarvis UI")


async def broadcast(mesaj: dict):
    text = json.dumps(mesaj, ensure_ascii=False)
    deconectati = []
    for ws in conexiuni_active:
        try:
            await ws.send_text(text)
        except Exception:
            deconectati.append(ws)
    for ws in deconectati:
        if ws in conexiuni_active:
            conexiuni_active.remove(ws)


def _e_eroare_de_fallback(mesaj_err: str) -> bool:
    # FIX: înainte `str(c) in mesaj_err` potrivea "500" oriunde în text
    # (ex: "8500 tokens"). Acum doar cod de sine stătător.
    return any(re.search(rf"(?<!\d){c}(?!\d)", mesaj_err) for c in ERORI_FALLBACK)


def ruleaza_agent(mesaj_user: str) -> str:
    # FIX: fără chei, next(cycle([])) arunca StopIteration în executor
    # (apărea ca un TypeError confuz).
    if not _gemini_clienti:
        return "Nu există nicio GEMINI_API_KEY configurată în .env."
    with _LOCK_AGENT:
        return _ruleaza_agent_blocat(mesaj_user)


def _ruleaza_agent_blocat(mesaj_user: str) -> str:
    istoric.append(
        types.Content(role="user", parts=[types.Part(text=mesaj_user)])
    )
    proceseaza_context(istoric, sesiune_id, client=next(_gemini_rotatie))

    for _ in range(len(_gemini_clienti)):
        client = next(_gemini_rotatie)
        try:
            return agent_loop(client, GEMINI_MODEL, SYSTEM_PROMPT, istoric)
        except Exception as e:
            mesaj_err = str(e)
            if _e_eroare_de_fallback(mesaj_err):
                time.sleep(0.5)
                continue
            raise

    return "Toate cheile Gemini sunt indisponibile momentan."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """UI web principal — versiunea mare pentru browser."""
    html_path = UI_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>UI lipsă — pune index.html în src/ui/static/</h1>")


@app.get("/desktop", response_class=HTMLResponse)
async def desktop():
    """UI widget desktop — versiunea compactă frosted glass pentru pywebview."""
    html_path = UI_DIR / "index_desktop.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Desktop UI lipsă — pune index_desktop.html în src/ui/static/</h1>")


@app.get("/api/stats")
async def api_stats():
    """Statistici hardware și memorie pentru dashboard."""
    try:
        import psutil
        cpu  = psutil.cpu_percent(interval=0.3)
        mem  = psutil.virtual_memory()
        disc = psutil.disk_usage("/")
        hardware = {
            "cpu_pct":    cpu,
            "ram_pct":    mem.percent,
            "ram_gb":     round(mem.used / (1024**3), 1),
            "ram_total":  round(mem.total / (1024**3), 1),
            "disc_pct":   disc.percent,
            "disc_gb":    round(disc.used / (1024**3), 1),
            "disc_total": round(disc.total / (1024**3), 1),
        }
    except Exception:
        hardware = {}

    return {
        "hardware":   hardware,
        "db":         db.statistici(),
        "rag":        rag.statistici(),
        "sesiune_id": sesiune_id,
    }


@app.get("/api/history")
async def api_history():
    """Ultimele 20 mesaje din sesiunea curentă."""
    mesaje = db.incarca_sesiune(sesiune_id)
    return {"mesaje": mesaje[-20:]}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Blochează cross-site WebSocket hijacking: acceptă doar origini locale
    # (sau cele listate în JARVIS_ALLOWED_ORIGINS, separate prin virgulă).
    origin = ws.headers.get("origin", "")
    # FIX: înainte, un Origin lipsă trecea (clienți non-browser). Browserele
    # trimit mereu Origin la WebSocket, deci cerem să fie prezent și permis.
    if origin not in _ORIGINI_PERMISE:
        await ws.close(code=1008)
        return
    await ws.accept()
    conexiuni_active.append(ws)
    print(f"[WS] Client conectat. Total: {len(conexiuni_active)}")

    await ws.send_text(json.dumps({
        "tip":   "status",
        "stare": "idle",
        "mesaj": f"Jarvis activ. {len(_gemini_chei)} chei Gemini.",
    }))

    try:
        while True:
            data    = await ws.receive_text()
            payload = json.loads(data)
            tip     = payload.get("tip", "")

            if tip == "ping":
                await ws.send_text(json.dumps({"tip": "pong"}))
                continue

            if tip == "mesaj":
                text_user = payload.get("text", "").strip()
                if not text_user:
                    continue

                await broadcast({"tip": "status", "stare": "thinking"})

                loop = asyncio.get_event_loop()
                try:
                    raspuns = await loop.run_in_executor(None, ruleaza_agent, text_user)
                except Exception as e:
                    raspuns = f"Eroare: {str(e)[:200]}"
                    await broadcast({"tip": "eroare", "text": raspuns})
                    await broadcast({"tip": "status", "stare": "idle"})
                    continue

                db.salveaza_mesaj(sesiune_id, "user", text_user)
                db.salveaza_mesaj(sesiune_id, "assistant", raspuns)
                rag.indexeaza_mesaj("user", text_user, sesiune_id)
                rag.indexeaza_mesaj("assistant", raspuns, sesiune_id)

                await broadcast({
                    "tip":    "raspuns",
                    "text":   raspuns,
                    "user":   text_user,
                    "status": "done",
                })
                await broadcast({"tip": "status", "stare": "idle"})

    except WebSocketDisconnect:
        if ws in conexiuni_active:
            conexiuni_active.remove(ws)
        print(f"[WS] Client deconectat. Rămași: {len(conexiuni_active)}")
    except Exception as e:
        print(f"[WS] Eroare: {e}")
        if ws in conexiuni_active:
            conexiuni_active.remove(ws)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag.sincronizeaza_din_db()
    print(f"\n[Jarvis UI Web]     http://localhost:{PORT}  (bind {HOST})")
    print(f"[Jarvis UI Desktop] http://localhost:{PORT}/desktop")
    print(f"[Memorie] {db.statistici()['amintiri']} amintiri | "
          f"System prompt: {len(SYSTEM_PROMPT)} caractere\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")