"""
Sub-Agenți Specializați cu Buclă Limitată (Task 6.6)

Implementarea pasului 3 din arhitectura Vasea — versiune "delegare
controlată", NU swarm complet cu dezbatere între agenți (decizie
confirmată explicit, ca să nu multiplicăm costul/complexitatea).

Trei sub-agenți, fiecare cu:
    - system prompt specializat pe domeniul lui
    - un SUBSET fix de unelte (nu tot registrul) — Software nu poate
      opri sistemul, Cercetare nu poate face git push etc.
    - aceeași "Regulă de Aur a Buclei" ca agentul principal: MAX_PASI=5
      din agent.py, moștenit automat (agent_loop e reutilizat, nu duplicat)

Siguranță împotriva recursiei infinite: niciun sub-agent nu are acces la
uneltele de delegare (deleaga_agent_*) — sunt excluse intenționat din
toate cele 3 liste de unelte permise de mai jos. Un sub-agent nu poate
delega mai departe la alt sub-agent.

Buget total per cerere a utilizatorului: agentul principal are MAX_PASI=5
pași; fiecare pas ÎN CARE deleagă consumă unul din cei 5, iar sub-agentul
apelat are propriul buget de 5 pași. Worst case teoretic: 5x5=25 apeluri
model într-o singură tură de conversație — generos, dar cu limită hard,
nu buclă infinită.

Fiecare sub-agent are propria rotație de chei Gemini + fallback (identic
cu vedere.py), independentă de rotația din main.py.
"""

import os
import itertools
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.core.agent import agent_loop

load_dotenv()

# ── Client Gemini propriu, cu fallback pe toate cheile ──────────────────────
_gemini_chei = [
    os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
    for i in range(5)
]
_gemini_chei = [k for k in _gemini_chei if k]
_clienti_subagenti = [genai.Client(api_key=k) for k in _gemini_chei] if _gemini_chei else []
_rotatie_subagenti = itertools.cycle(_clienti_subagenti) if _clienti_subagenti else None

MODEL_SUBAGENTI = "gemini-3.6-flash"

_ERORI_FALLBACK = (403, 429, 500, 503)
try:
    import httpx
    _EXCEPTII_RETEA = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)
except ImportError:
    _EXCEPTII_RETEA = ()


def _eroare_temporara(e: Exception) -> bool:
    return any(str(cod) in str(e) for cod in _ERORI_FALLBACK) or isinstance(e, _EXCEPTII_RETEA)


# ── Configurația celor 3 sub-agenți ──────────────────────────────────────────

AGENTI_SPECIALIZATI = {
    "software": {
        "nume_afisat": "Agentul Software",
        "system_prompt": (
            "Ești un sub-agent specializat în DEZVOLTARE SOFTWARE, parte din "
            "sistemul Jarvis al lui Vasea. Scopul tău e strict tehnic: scrii, "
            "editezi, verifici cod și gestionezi Git/VS Code/server local. "
            "Nu ai acces la controlul sistemului de operare general, hardware "
            "sau cercetare web — dacă sarcina cere așa ceva, spune clar că "
            "depășește scopul tău, nu încerca să ghicești. Răspunde concis, "
            "tehnic, la obiect."
        ),
        "unelte": [
            "deschide_vscode",
            "porneste_server_http", "opreste_server_http", "status_server_http",
            "git_status", "git_log", "git_branches", "git_checkout",
            "git_add", "git_commit", "git_push", "git_stash", "git_stash_pop",
        ],
    },
    "devops": {
        "nume_afisat": "Agentul DevOps",
        "system_prompt": (
            "Ești un sub-agent specializat în DEVOPS / CONTROLUL SISTEMULUI, "
            "parte din sistemul Jarvis al lui Vasea. Te ocupi de comenzi de "
            "sistem read-only, monitorizare hardware, control periferice și "
            "joburi în fundal. Nu scrii cod și nu faci cercetare web — dacă "
            "sarcina cere așa ceva, spune clar că depășește scopul tău. "
            "Fii precaut cu orice acțiune care modifică starea sistemului; "
            "preferă mereu verificarea (status) înainte de acțiune."
        ),
        "unelte": [
            "deschide_aplicatie", "ruleaza_comanda_info",
            "status_cpu", "status_ram", "status_disc", "status_temperatura",
            "status_sistem_general",
            "seteaza_volum", "status_volum", "comuta_mute",
            "seteaza_luminozitate", "status_luminozitate",
            "porneste_job_fundal", "status_job_fundal", "listeaza_joburi_fundal",
        ],
    },
    "cercetare": {
        "nume_afisat": "Agentul de Cercetare",
        "system_prompt": (
            "Ești un sub-agent specializat în CERCETARE ȘI INFORMAȚII "
            "EXTERNE, parte din sistemul Jarvis al lui Vasea. Cauți fapte, "
            "definiții, vreme și informații din surse externe. Nu controlezi "
            "sistemul și nu scrii cod — dacă sarcina cere așa ceva, spune "
            "clar că depășește scopul tău. Fii transparent când o căutare "
            "nu găsește un răspuns direct — nu inventa informații."
        ),
        "unelte": [
            "cauta_web", "vremea",
        ],
    },
}


def ruleaza_subagent(cheie_agent: str, sarcina: str) -> str:
    """
    Pornește un mini-agent_loop specializat, cu propriul system prompt și
    doar uneltele lui, pentru o singură sarcină. Independent de istoricul
    conversației principale — primește DOAR sarcina delegată, ca context nou.

    Parametri:
        cheie_agent: "software" | "devops" | "cercetare"
        sarcina:     descrierea sarcinii de rezolvat, ca text

    Returnează textul răspunsului sub-agentului, sau un mesaj de eroare
    clar dacă ceva eșuează (cheie necunoscută, toate cheile Gemini indisponibile).
    """
    if cheie_agent not in AGENTI_SPECIALIZATI:
        return (
            f"Eroare internă: sub-agent necunoscut '{cheie_agent}'. "
            f"Disponibili: {', '.join(AGENTI_SPECIALIZATI.keys())}."
        )

    if _rotatie_subagenti is None:
        return "Sub-agenții nu au nicio GEMINI_API_KEY disponibilă în .env."

    config = AGENTI_SPECIALIZATI[cheie_agent]
    istoric_subagent = [
        types.Content(role="user", parts=[types.Part(text=sarcina)])
    ]

    print(f"[Sub-agent] {config['nume_afisat']} primește sarcina: {sarcina[:80]}...")

    for _ in range(len(_clienti_subagenti)):
        client = next(_rotatie_subagenti)
        try:
            rezultat = agent_loop(
                client,
                MODEL_SUBAGENTI,
                config["system_prompt"],
                istoric_subagent,
                unelte_permise=config["unelte"],
            )
            print(f"[Sub-agent] {config['nume_afisat']} a terminat.")
            return rezultat
        except Exception as e:
            if _eroare_temporara(e):
                print(f"[Sub-agent] Cheie indisponibilă ({str(e)[:80]}) — încerc următoarea]")
                continue
            return f"Eroare în {config['nume_afisat']}: {str(e)}"

    return f"{config['nume_afisat']} — toate cheile Gemini disponibile au eșuat."