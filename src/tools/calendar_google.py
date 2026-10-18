"""
Google Calendar (Task 2.7 — partea OAuth)

Necesită:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
    credentials.json descărcat din Google Cloud Console, la rădăcina proiectului.

Flow de autentificare:
    Prima rulare: se deschide browser, ceri permisiune, se salvează token.json local.
    Rulările următoare: token.json se reîmprospătează automat, fără interacțiune,
    cât timp refresh_token-ul rămâne valid.

IMPORTANT: credentials.json și token.json conțin date private — sunt deja
în .gitignore, NU le urca niciodată pe GitHub.
"""

import os
import datetime
from src.core.registry import unealta

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    _GOOGLE_LIBS_DISPONIBILE = True
except ImportError:
    _GOOGLE_LIBS_DISPONIBILE = False

# Scope-ul cere acces complet de citire+scriere la calendar.
# Dacă vrei doar citire, schimbă în 'calendar.readonly' (mai sigur, dar
# atunci nu poți crea evenimente prin Jarvis).
_SCOPES = ["https://www.googleapis.com/auth/calendar"]

_CALE_CREDENTIALS = "credentials.json"
_CALE_TOKEN = "token.json"

# Serviciul Google Calendar e construit lazy, o singură dată, la prima utilizare
_service = None


def _obtine_service():
    """
    Returnează un obiect 'service' autentificat pentru Google Calendar API.
    Gestionează automat: token existent valid, refresh token expirat,
    sau autentificare nouă completă (deschide browser).
    """
    global _service

    if not _GOOGLE_LIBS_DISPONIBILE:
        return None

    if _service is not None:
        return _service

    creds = None

    if os.path.exists(_CALE_TOKEN):
        creds = Credentials.from_authorized_user_file(_CALE_TOKEN, _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(_CALE_CREDENTIALS):
                return None  # nu putem autentifica fără credentials.json

            flow = InstalledAppFlow.from_client_secrets_file(_CALE_CREDENTIALS, _SCOPES)
            creds = flow.run_local_server(port=0)

        # Salvăm token-ul (nou sau reîmprospătat) pentru rulările viitoare
        with open(_CALE_TOKEN, "w") as f:
            f.write(creds.to_json())

    _service = build("calendar", "v3", credentials=creds)
    return _service


def _mesaj_lipsa_config() -> str:
    if not _GOOGLE_LIBS_DISPONIBILE:
        return (
            "Librăriile Google Calendar nu sunt instalate. Rulează: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
    if not os.path.exists(_CALE_CREDENTIALS):
        return (
            "Lipsește fișierul credentials.json din rădăcina proiectului. "
            "Descarcă-l din Google Cloud Console (Google Auth platform → Clients)."
        )
    return "Eroare de configurare necunoscută la Google Calendar."


# ==============================================================
# UNELTE
# ==============================================================

@unealta(
    description=(
        "Returnează următoarele evenimente din calendarul Google al utilizatorului. "
        "Folosește pentru 'ce am în program?', 'ce evenimente am azi/mâine?', "
        "'arată-mi calendarul' etc."
    ),
    parameters={
        "numar": {
            "type": "INTEGER",
            "description": "Câte evenimente viitoare să returneze. Default: 10.",
            "optional": True,
        }
    },
    max_linii=20,
)
def evenimente_viitoare(numar: int = 10):
    """Listează următoarele evenimente din calendarul principal al utilizatorului."""
    service = _obtine_service()
    if service is None:
        return _mesaj_lipsa_config()

    try:
        acum = datetime.datetime.utcnow().isoformat() + "Z"
        rezultat = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=acum,
                maxResults=numar,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        evenimente = rezultat.get("items", [])

        if not evenimente:
            return "Nu ai niciun eveniment viitor în calendar."

        linii = []
        for ev in evenimente:
            inceput = ev["start"].get("dateTime", ev["start"].get("date"))
            titlu = ev.get("summary", "(fără titlu)")
            id_eveniment = ev.get("id", "")
            linii.append(f"{inceput} — {titlu} [id: {id_eveniment}]")

        return "\n".join(linii)

    except HttpError as e:
        return f"Eroare la accesarea calendarului: {str(e)}"


@unealta(
    description=(
        "Creează un eveniment nou în calendarul Google al utilizatorului. "
        "Folosește pentru 'adaugă o întâlnire X', 'programează-mă la X', "
        "'pune în calendar X' etc. Datele trebuie în format ISO 8601 "
        "(ex: '2026-06-25T14:00:00')."
    ),
    parameters={
        "titlu": {
            "type": "STRING",
            "description": "Titlul evenimentului.",
        },
        "inceput": {
            "type": "STRING",
            "description": "Data și ora de început, format ISO 8601 (ex: '2026-06-25T14:00:00').",
        },
        "sfarsit": {
            "type": "STRING",
            "description": "Data și ora de sfârșit, format ISO 8601 (ex: '2026-06-25T15:00:00').",
        },
        "descriere": {
            "type": "STRING",
            "description": "Descriere opțională a evenimentului.",
            "optional": True,
        },
    },
    necesita_confirmare=True,
)
def creeaza_eveniment(titlu: str, inceput: str, sfarsit: str, descriere: str = ""):
    """Creează un eveniment nou în calendarul principal."""
    service = _obtine_service()
    if service is None:
        return _mesaj_lipsa_config()

    eveniment = {
        "summary": titlu,
        "description": descriere,
        "start": {"dateTime": inceput, "timeZone": "Europe/Chisinau"},
        "end": {"dateTime": sfarsit, "timeZone": "Europe/Chisinau"},
    }

    try:
        rezultat = service.events().insert(calendarId="primary", body=eveniment).execute()
        link = rezultat.get("htmlLink", "")
        return f"Eveniment creat: '{titlu}'. Link: {link}"
    except HttpError as e:
        return f"Eroare la crearea evenimentului: {str(e)}"


@unealta(
    description=(
        "Șterge un eveniment din calendar, după ID-ul lui. "
        "ID-ul evenimentului se obține din 'evenimente_viitoare' sau e cunoscut deja."
    ),
    parameters={
        "eveniment_id": {
            "type": "STRING",
            "description": "ID-ul evenimentului de șters.",
        }
    },
    necesita_confirmare=True,
)
def sterge_eveniment(eveniment_id: str):
    """Șterge un eveniment din calendarul principal, după ID."""
    service = _obtine_service()
    if service is None:
        return _mesaj_lipsa_config()

    try:
        service.events().delete(calendarId="primary", eventId=eveniment_id).execute()
        return f"Eveniment {eveniment_id} șters."
    except HttpError as e:
        return f"Eroare la ștergerea evenimentului: {str(e)}"