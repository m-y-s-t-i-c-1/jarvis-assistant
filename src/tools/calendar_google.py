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
    from google.auth.exceptions import RefreshError
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

# FIX: fusul orar era hardcodat. Configurabil din .env (JARVIS_TIMEZONE), default Chișinău.
_FUS_ORAR = os.getenv("JARVIS_TIMEZONE", "Europe/Chisinau")

# Serviciul Google Calendar e construit lazy, o singură dată, la prima utilizare
_service = None


def _poate_autentifica_interactiv() -> bool:
    """True doar dacă avem terminal interactiv și (pe Linux) un display pentru browser."""
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        return False
    if sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        return False
    return True


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
        try:
            creds = Credentials.from_authorized_user_file(_CALE_TOKEN, _SCOPES)
        except Exception as e:
            # token.json corupt / de la alt scope -> îl tratăm ca lipsă și refacem autentificarea
            print(f"[Calendar] token.json invalid ({e}) — refac autentificarea.")
            creds = None

    if not creds or not creds.valid:
        reimprospatat = False
        if creds and creds.expired and creds.refresh_token:
            # FIX: un refresh eșuat (token revocat / expirat — la proiectele OAuth în
            # modul „Testing" refresh_token-ul expiră după 7 zile) strica unealta
            # permanent, până ștergeai token.json manual.
            try:
                creds.refresh(Request())
                reimprospatat = True
            except RefreshError as e:
                print(f"[Calendar] Refresh-ul token-ului a eșuat ({e}) — șterg token.json și refac autentificarea.")
                try:
                    os.remove(_CALE_TOKEN)
                except OSError:
                    pass
                creds = None
            except Exception as e:
                # eroare de rețea etc.: nu ștergem token-ul, doar renunțăm acum
                print(f"[Calendar] Nu am putut reîmprospăta token-ul: {e}")
                return None

        if not reimprospatat:
            if not os.path.exists(_CALE_CREDENTIALS):
                return None  # nu putem autentifica fără credentials.json

            # FIX: run_local_server() deschide un browser și așteaptă la nesfârșit. Din
            # server/widget sau fără display, ar bloca tura. Permitem autentificarea
            # interactivă doar din terminal cu display.
            if not _poate_autentifica_interactiv():
                print("[Calendar] Token invalid și nu pot deschide browser-ul de aici. "
                      "Rulează o dată `python -m src.tools.calendar_google` dintr-un terminal.")
                return None

            flow = InstalledAppFlow.from_client_secrets_file(_CALE_CREDENTIALS, _SCOPES)
            creds = flow.run_local_server(port=0, timeout_seconds=120)

        # Salvăm token-ul (nou sau reîmprospătat) pentru rulările viitoare
        with open(_CALE_TOKEN, "w") as f:
            f.write(creds.to_json())

    _service = build("calendar", "v3", credentials=creds)
    return _service


def _mesaj_lipsa_config() -> str:
    if _GOOGLE_LIBS_DISPONIBILE and os.path.exists(_CALE_CREDENTIALS):
        return (
            "Nu mă pot autentifica la Google Calendar (token expirat sau revocat). "
            "Rulează `python -m src.tools.calendar_google` dintr-un terminal ca să te reautentifici."
        )
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
        },
        "zi": {
            "type": "STRING",
            "description": (
                "Opțional: 'azi', 'maine' sau o dată ISO (ex: '2026-06-25'). "
                "Dacă e dat, returnează doar evenimentele din ziua respectivă."
            ),
            "optional": True,
        },
    },
    max_linii=20,
)
def evenimente_viitoare(numar: int = 10, zi: str = ""):
    """Listează următoarele evenimente din calendarul principal al utilizatorului."""
    service = _obtine_service()
    if service is None:
        return _mesaj_lipsa_config()

    try:
        # FIX: utcnow() e deprecat; numar poate veni ca float (5.0) -> API-ul respinge maxResults
        acum_dt = datetime.datetime.now(datetime.timezone.utc)
        acum = acum_dt.isoformat()
        try:
            numar = max(1, min(int(numar), 100))
        except (TypeError, ValueError):
            numar = 10

        parametri = dict(calendarId="primary", timeMin=acum, maxResults=numar,
                         singleEvents=True, orderBy="startTime")

        # FIX: nu se putea cere „azi"/„mâine" — se luau doar următoarele N evenimente.
        zi_n = (zi or "").strip().lower().replace("â", "a")
        if zi_n:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(_FUS_ORAR)
            except Exception:
                tz = datetime.timezone.utc
            azi = datetime.datetime.now(tz).date()
            if zi_n == "azi":
                data_zi = azi
            elif zi_n == "maine":
                data_zi = azi + datetime.timedelta(days=1)
            else:
                try:
                    data_zi = datetime.date.fromisoformat(zi_n[:10])
                except ValueError:
                    return f"Nu înțeleg ziua '{zi}'. Folosește 'azi', 'maine' sau o dată ISO (2026-06-25)."
            inceput_zi = datetime.datetime.combine(data_zi, datetime.time.min, tzinfo=tz)
            parametri["timeMin"] = max(inceput_zi, acum_dt.astimezone(tz)).isoformat() \
                if data_zi == azi else inceput_zi.isoformat()
            parametri["timeMax"] = (inceput_zi + datetime.timedelta(days=1)).isoformat()

        rezultat = service.events().list(**parametri).execute()
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
        "start": {"dateTime": inceput, "timeZone": _FUS_ORAR},
        "end": {"dateTime": sfarsit, "timeZone": _FUS_ORAR},
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


if __name__ == "__main__":
    # Reautentificare manuală (din terminal): `python -m src.tools.calendar_google`
    _service = None
    print("Calendar OK." if _obtine_service() else _mesaj_lipsa_config())
