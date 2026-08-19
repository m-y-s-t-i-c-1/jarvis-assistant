"""
Vederea Ecranului (Task 6.1)

Jarvis poate acum să "vadă" ecranul: face un screenshot și îl trimite la
Gemini (multimodal) pentru analiză vizuală. Două unelte:

1. vezi_ecranul(intrebare)   -> descriere generală / răspuns la o întrebare
                                despre ce se vede pe ecran
2. gaseste_pe_ecran(element) -> localizează un element specific (buton,
                                câmp, iconiță) și returnează coordonatele
                                lui exacte în pixeli, gata de folosit cu
                                uneltele din control_ecran.py

Cum funcționează localizarea:
    Gemini poate returna bounding box-uri normalizate (scală 0-1000, format
    [y_min, x_min, y_max, x_max]) pentru obiecte dintr-o imagine. Convertim
    coordonatele normalizate în pixeli reali, folosind rezoluția ecranului
    curent (dimensiunea screenshot-ului efectiv, nu una presupusă).

Dependențe:
    pip install pyautogui pillow --break-system-packages

    Pe Arch/X11, dacă pyautogui.screenshot() dă eroare, instalează scrot
    ca fallback (Pillow >= 10 are suport nativ, dar unele setup-uri tot
    au nevoie de el):
        sudo pacman -S scrot

IMPORTANT: acest modul funcționează doar pe X11. Pe Wayland, capturarea
ecranului prin pyautogui nu funcționează din motive de securitate ale
protocolului — ar trebui înlocuit cu grim/slurp sau alt tool nativ Wayland.
"""

import os
import json
import itertools
from io import BytesIO

import pyautogui
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.core.registry import unealta

load_dotenv()

GEMINI_MODEL_VEDERE = "gemini-3.6-flash"

# ---- Client Gemini propriu pentru acest modul ----
# Uneltele nu primesc clientul din agent_loop (Gemini le apelează direct,
# fără parametri Python în plus), deci construim propria rotație de chei,
# identic cu logica din main.py / audio_loop.py.
_gemini_chei = [
    os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
    for i in range(5)
]
_gemini_chei = [k for k in _gemini_chei if k]
_clienti_vedere = [genai.Client(api_key=k) for k in _gemini_chei] if _gemini_chei else []
_rotatie_vedere = itertools.cycle(_clienti_vedere) if _clienti_vedere else None

# Coduri HTTP care înseamnă "încearcă următoarea cheie", nu "eroare fatală"
_ERORI_FALLBACK = (403, 429, 500, 503)

try:
    import httpx
    _EXCEPTII_RETEA = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)
except ImportError:
    _EXCEPTII_RETEA = ()


def _eroare_temporara(e: Exception) -> bool:
    are_cod_cunoscut = any(str(cod) in str(e) for cod in _ERORI_FALLBACK)
    return are_cod_cunoscut or isinstance(e, _EXCEPTII_RETEA)


def _genereaza_cu_fallback(**kwargs):
    """
    Apelează generate_content încercând pe rând TOATE cheile Gemini
    disponibile, nu doar cea care iese din rotație în momentul curent.

    Fără asta, dacă cheia care iese la rând e invalidă/blocată (403 etc.),
    unealta de vedere eșuează direct — chiar dacă alte chei ar fi mers.
    """
    if _rotatie_vedere is None:
        raise RuntimeError(
            "Nicio GEMINI_API_KEY disponibilă pentru modulul de vedere (src/tools/vedere.py)."
        )

    ultima_eroare = None
    for _ in range(len(_clienti_vedere)):
        client = next(_rotatie_vedere)
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            ultima_eroare = e
            if _eroare_temporara(e):
                continue
            raise

    raise RuntimeError(f"Toate cheile Gemini au eșuat pentru modulul de vedere: {ultima_eroare}")


def _screenshot_bytes() -> tuple[bytes, int, int]:
    """Face un screenshot și returnează (bytes PNG, lățime, înălțime)."""
    imagine = pyautogui.screenshot()
    latime, inaltime = imagine.size
    buffer = BytesIO()
    imagine.save(buffer, format="PNG")
    return buffer.getvalue(), latime, inaltime


# ==============================================================
# VEDERE GENERALĂ
# ==============================================================

@unealta(
    description=(
        "Face un screenshot al ecranului curent și îl analizează vizual cu "
        "AI, returnând o descriere a ceea ce se vede (ferestre deschise, "
        "conținut, text vizibil, stare generală a ecranului). Folosește "
        "pentru 'ce am pe ecran?', 'ce fac acum?', 'descrie ecranul', "
        "'citește ce scrie acolo' etc."
    ),
    parameters={
        "intrebare": {
            "type": "STRING",
            "description": (
                "Ce anume să observe/răspundă despre ecran. "
                "Default: o descriere generală a ecranului."
            ),
            "optional": True,
        }
    },
)
def vezi_ecranul(intrebare: str = "Descrie pe scurt ce se vede pe ecran."):
    """Screenshot + analiză vizuală prin Gemini multimodal, cu fallback pe NVIDIA multimodal."""
    try:
        img_bytes, _, _ = _screenshot_bytes()
    except Exception as e:
        return f"Nu am putut face screenshot: {str(e)}"

    try:
        raspuns = _genereaza_cu_fallback(
            model=GEMINI_MODEL_VEDERE,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        types.Part(text=intrebare),
                    ],
                )
            ],
        )
        return (raspuns.text or "Nu am obținut niciun răspuns de la analiza vizuală.").strip()
    except Exception as e:
        print(f"[Vedere] Gemini indisponibil ({str(e)[:120]}) — încerc NVIDIA multimodal...")
        try:
            from src.core.llm_provider import intreaba_nvidia_multimodal
            rezultat = intreaba_nvidia_multimodal(img_bytes, intrebare)
            if rezultat:
                return rezultat
        except Exception as e2:
            print(f"[Vedere] NVIDIA multimodal a eșuat și el: {str(e2)[:120]}")

        return f"Eroare la analiza ecranului (Gemini și NVIDIA indisponibile): {str(e)}"


# ==============================================================
# LOCALIZARE ELEMENTE (pentru control ulterior)
# ==============================================================

_PROMPT_LOCALIZARE = """Analizează imaginea (un screenshot al ecranului) și găsește elementul descris mai jos.

Element căutat: {element}

Returnează DOAR un JSON valid (fără markdown, fără text explicativ), cu formatul exact:
{{"gasit": true, "box_2d": [y_min, x_min, y_max, x_max], "eticheta": "ce ai găsit"}}

box_2d este normalizat pe o scală 0-1000 (NU pixeli reali), format [y_min, x_min, y_max, x_max].
Dacă nu găsești elementul, returnează exact:
{{"gasit": false, "box_2d": null, "eticheta": ""}}
"""


@unealta(
    description=(
        "Localizează un element specific pe ecran (buton, câmp de text, "
        "iconiță, link, fereastră etc.) și returnează coordonatele lui "
        "exacte în pixeli. Folosește ÎNTOTDEAUNA această unealtă ÎNAINTE "
        "de orice click, mișcare de mouse sau interacțiune — nu ghici "
        "niciodată coordonate direct."
    ),
    parameters={
        "element": {
            "type": "STRING",
            "description": (
                "Descrierea elementului de găsit, cât mai specifică. "
                "Ex: 'butonul Trimite din formular', "
                "'câmpul de căutare din bara browserului', "
                "'iconița X din colțul din dreapta sus'."
            ),
        }
    },
)
def gaseste_pe_ecran(element: str):
    """Găsește un element pe ecran și returnează coordonatele lui centrale în pixeli."""
    try:
        img_bytes, latime, inaltime = _screenshot_bytes()
    except Exception as e:
        return f"Nu am putut face screenshot: {str(e)}"

    try:
        raspuns = _genereaza_cu_fallback(
            model=GEMINI_MODEL_VEDERE,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        types.Part(text=_PROMPT_LOCALIZARE.format(element=element)),
                    ],
                )
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = (raspuns.text or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        date = json.loads(text.strip())
    except Exception as e:
        return f"Eroare la localizare: {str(e)}"

    if not date.get("gasit") or not date.get("box_2d"):
        return f"Nu am găsit '{element}' pe ecran."

    y_min, x_min, y_max, x_max = date["box_2d"]

    # Conversie din scala normalizată 0-1000 în pixeli reali,
    # folosind rezoluția EFECTIVĂ a screenshot-ului (nu una presupusă).
    px_x_min = round(x_min / 1000 * latime)
    px_x_max = round(x_max / 1000 * latime)
    px_y_min = round(y_min / 1000 * inaltime)
    px_y_max = round(y_max / 1000 * inaltime)

    centru_x = (px_x_min + px_x_max) // 2
    centru_y = (px_y_min + px_y_max) // 2

    eticheta = date.get("eticheta", element)
    return f"Găsit: '{eticheta}' — centru la ({centru_x}, {centru_y}), colț stânga-sus ({px_x_min}, {px_y_min}), colț dreapta-jos ({px_x_max}, {px_y_max})."