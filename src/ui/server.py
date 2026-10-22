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
from src.core.security import canal_confirmare
from src.core.context_rag import prompt_pentru_tura
from src.core.jobs import porneste_thread_watcher
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
ERORI_FALLBACK = (503, 429, 500, 502, 504, 403)  # FIX: lipseau 502 și 504 (existente în main.py)

# FIX: uneltele cu necesita_confirmare (git_commit, git_push, click_la_pozitie, creeaza_eveniment...)
# nu puteau fi folosite din UI: nu exista canal de confirmare -> input() bloca thread-ul (cu lock-ul
# ținut, deci tot UI-ul îngheța) sau acțiunea era refuzată automat. Acum confirmarea se cere în
# browser/widget prin WebSocket (mesaj "confirmare" -> răspuns {"tip": "confirmare", "id", "aprobat"}).
TIMEOUT_CONFIRMARE = 60  # secunde; fără răspuns => refuz (fail closed)
_confirmari: dict[str, dict] = {}
_loop_principal: asyncio.AbstractEventLoop | None = None
_taskuri_mesaje: set = set()

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

# FIX: sesiunea creată la import nu se marca niciodată închisă (main.py o face).
import atexit
atexit.register(lambda: db.inchide_sesiune(sesiune_id))

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


def _e_eroare_de_fallback(eroare) -> bool:
    # FIX: înainte `str(c) in mesaj_err` potrivea "500" oriunde în text
    # (ex: "8500 tokens"). Acum doar cod de sine stătător.
    # FIX 2: acceptă și excepția în sine — erorile de rețea (httpx) și atributul .code nu
    # aveau cod în text, deci UI-ul nu încerca cheia următoare (spre deosebire de main.py).
    try:
        import httpx
        if isinstance(eroare, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError, httpx.TimeoutException)):
            return True
    except ImportError:
        pass
    for atribut in ("code", "status_code"):
        cod = getattr(eroare, atribut, None)
        if isinstance(cod, int) and cod in ERORI_FALLBACK:
            return True
    mesaj_err = str(eroare)
    # FIX: cheie invalidă/expirată (400) -> încercăm cheia următoare
    if re.search(r"API[ _]KEY[ _](NOT VALID|INVALID|EXPIRED)", mesaj_err.upper()):
        return True
    return any(re.search(rf"(?<!\d){c}(?!\d)", mesaj_err) for c in ERORI_FALLBACK)


def ruleaza_agent(mesaj_user: str) -> str:
    # FIX: fără chei, next(cycle([])) arunca StopIteration în executor
    # (apărea ca un TypeError confuz).
    if not _gemini_clienti:
        return "Nu există nicio GEMINI_API_KEY configurată în .env."
    with _LOCK_AGENT, canal_confirmare(_confirma_prin_ws):
        return _ruleaza_agent_blocat(mesaj_user)


def _confirma_prin_ws(nume_functie: str, argumente: dict) -> bool:
    """Cere confirmarea utilizatorului în UI (WebSocket). Rulează în thread-ul agentului."""
    import uuid
    if _loop_principal is None or not conexiuni_active:
        print("[Securitate] Niciun client UI conectat pentru confirmare — acțiune refuzată.")
        return False

    id_conf = uuid.uuid4().hex[:12]
    intrare = {"event": threading.Event(), "aprobat": False}
    _confirmari[id_conf] = intrare
    try:
        asyncio.run_coroutine_threadsafe(
            broadcast({
                "tip": "confirmare",
                "id": id_conf,
                "functie": nume_functie,
                "argumente": {str(k): str(v)[:300] for k, v in (argumente or {}).items()},
                "timeout": TIMEOUT_CONFIRMARE,
            }),
            _loop_principal,
        ).result(timeout=5)
        if not intrare["event"].wait(TIMEOUT_CONFIRMARE):
            print(f"[Securitate] Confirmare expirată pentru '{nume_functie}' — acțiune refuzată.")
            return False
        return bool(intrare["aprobat"])
    except Exception as e:
        print(f"[Securitate] Confirmarea din UI a eșuat ({e}) — acțiune refuzată.")
        return False
    finally:
        _confirmari.pop(id_conf, None)


def _anuleaza_tura(mesaj_curent) -> None:
    """
    FIX: dacă agent_loop pica (eroare fatală sau toate cheile indisponibile),
    mesajul utilizatorului rămânea în `istoric` global fără răspuns, iar
    următorul mesaj producea "user" consecutiv (+ eventuale function_call
    parțiale). Tăiem de la mesajul curent încolo, după identitate (nu după
    index: proceseaza_context poate comprima lista in-place).
    """
    for i in range(len(istoric) - 1, -1, -1):
        if istoric[i] is mesaj_curent:
            del istoric[i:]
            return


def _ruleaza_agent_blocat(mesaj_user: str) -> str:
    mesaj_curent = types.Content(role="user", parts=[types.Part(text=mesaj_user)])
    istoric.append(mesaj_curent)
    try:
        proceseaza_context(istoric, sesiune_id, client=next(_gemini_rotatie))

        for _ in range(len(_gemini_clienti)):
            client = next(_gemini_rotatie)
            try:
                # FIX: profil + amintiri proaspete + RAG pe tură (înainte: prompt static, calculat la import)
                prompt_tura = prompt_pentru_tura(SYSTEM_PROMPT_BAZA, SYSTEM_PROMPT, mesaj_user, sesiune_id)
                raspuns = agent_loop(client, GEMINI_MODEL, prompt_tura, istoric)
            except Exception as e:
                if _e_eroare_de_fallback(e):
                    time.sleep(0.5)
                    continue
                raise
            # agent_loop poate întoarce un mesaj explicativ (răspuns gol/blocat,
            # buclă prea lungă) fără să-l adauge în istoric -> păstrăm alternanța.
            if raspuns and getattr(istoric[-1], "role", None) != "model":
                istoric.append(types.Content(role="model", parts=[types.Part(text=raspuns)]))
            return raspuns
    except Exception:
        _anuleaza_tura(mesaj_curent)
        raise

    _anuleaza_tura(mesaj_curent)
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


_ture_ui = 0
_lock_consolidare_ui = threading.Lock()
CONSOLIDARE_LA_N_TURE_UI = 5


def _memorie_in_fundal(text_user: str, raspuns: str) -> None:
    """Extrage amintiri din tură și, la fiecare N ture, consolidează memoria. Nu aruncă niciodată."""
    global _ture_ui
    try:
        client = next(_gemini_rotatie)
        memorie.extrage_si_salveaza(text_user, raspuns, sesiune_id, client, GEMINI_MODEL)
        _ture_ui += 1
        if _ture_ui % CONSOLIDARE_LA_N_TURE_UI == 0 and _lock_consolidare_ui.acquire(blocking=False):
            try:
                from src.core.consolidare import consolidare
                consolidare.ruleaza(client=client, model=GEMINI_MODEL)
            finally:
                _lock_consolidare_ui.release()
    except Exception as e:
        print(f"[Memorie] Extragere/consolidare UI omisă: {e}")


async def _proceseaza_mesaj(text_user: str):
    await broadcast({"tip": "status", "stare": "thinking"})

    loop = asyncio.get_running_loop()  # FIX: get_event_loop() e deprecat în coroutine
    try:
        raspuns = await loop.run_in_executor(None, ruleaza_agent, text_user)
    except Exception as e:
        raspuns = f"Eroare: {str(e)[:200]}"
        await broadcast({"tip": "eroare", "text": raspuns})
        await broadcast({"tip": "status", "stare": "idle"})
        return

    # FIX: salvarea + indexarea RAG (embeddings) rulau SINCRON în event loop (UI-ul îngheța,
    # ping/pong întârziau) și o eroare aici omora conexiunea înainte să trimită răspunsul.
    def _persista():
        db.salveaza_mesaj(sesiune_id, "user", text_user)
        db.salveaza_mesaj(sesiune_id, "assistant", raspuns)
        rag.indexeaza_mesaj("user", text_user, sesiune_id)
        rag.indexeaza_mesaj("assistant", raspuns, sesiune_id)

    try:
        await loop.run_in_executor(None, _persista)
    except Exception as e:
        print(f"[Memorie] Salvare/indexare eșuată: {e}")

    # FIX: UI-ul nu extrăgea amintiri episodice și nu consolida memoria (main.py o face).
    if _gemini_clienti:
        loop.run_in_executor(None, _memorie_in_fundal, text_user, raspuns)

    await broadcast({
        "tip":    "raspuns",
        "text":   raspuns,
        "user":   text_user,
        "status": "done",
    })
    await broadcast({"tip": "status", "stare": "idle"})


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
    global _loop_principal
    _loop_principal = asyncio.get_running_loop()
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
            # FIX: un mesaj JSON invalid (sau non-obiect) arunca excepție -> conexiunea cădea.
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            tip     = payload.get("tip", "")

            if tip == "ping":
                await ws.send_text(json.dumps({"tip": "pong"}))
                continue

            if tip == "confirmare":
                intrare = _confirmari.get(str(payload.get("id", "")))
                if intrare is not None:
                    intrare["aprobat"] = payload.get("aprobat") is True
                    intrare["event"].set()
                continue

            if tip == "mesaj":
                text_user = payload.get("text", "").strip()
                if not text_user:
                    continue
                # FIX: tura rula în acest handler (await), deci bucla de recepție era blocată și
                # răspunsul de confirmare al utilizatorului nu putea fi citit. O rulăm ca task.
                task = asyncio.create_task(_proceseaza_mesaj(text_user))
                _taskuri_mesaje.add(task)
                task.add_done_callback(_taskuri_mesaje.discard)

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
    porneste_thread_watcher()  # FIX: joburile din fundal nu erau anunțate niciodată în UI
    print(f"\n[Jarvis UI Web]     http://localhost:{PORT}  (bind {HOST})")
    print(f"[Jarvis UI Desktop] http://localhost:{PORT}/desktop")
    print(f"[Memorie] {db.statistici()['amintiri']} amintiri | "
          f"System prompt: {len(SYSTEM_PROMPT)} caractere\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")