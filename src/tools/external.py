"""
Integrarea API-urilor Externe (Task 2.7)

Două servicii, ambele gratuite, fără cheie API:
1. Meteo — Open-Meteo (geocoding + forecast, fără cont, fără limite practice)
2. Căutare web — DuckDuckGo Instant Answer API

IMPORTANT despre DuckDuckGo: NU e o căutare web completă cu linkuri clasate
ca Google. Returnează doar "instant answers" — definiții, fapte rapide,
subiecte cunoscute (oameni celebri, locuri, concepte). Pentru întrebări
foarte specifice sau recente, poate returna gol. Descrierea uneltei
reflectă explicit limitarea asta, ca Gemini să nu se aștepte la prea mult.

Calendarul Google (parte separată din Task 2.7) e într-un fișier diferit
(calendar_google.py), pentru că necesită OAuth și e mai complex.
"""

import requests
from src.core.registry import unealta


# ==============================================================
# METEO — Open-Meteo
# ==============================================================

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Coduri meteo Open-Meteo -> descriere în română
# (Open-Meteo folosește codurile standard WMO)
_COD_VREME = {
    0: "cer senin",
    1: "în general senin", 2: "parțial înnorat", 3: "înnorat",
    45: "ceață", 48: "ceață cu chiciură",
    51: "burniță ușoară", 53: "burniță moderată", 55: "burniță densă",
    61: "ploaie ușoară", 63: "ploaie moderată", 65: "ploaie puternică",
    71: "ninsoare ușoară", 73: "ninsoare moderată", 75: "ninsoare puternică",
    80: "averse ușoare", 81: "averse moderate", 82: "averse violente",
    95: "furtună", 96: "furtună cu grindină", 99: "furtună puternică cu grindină",
}


def _descrie_vreme(cod: int) -> str:
    return _COD_VREME.get(cod, f"cod meteo necunoscut ({cod})")


@unealta(
    description=(
        "Returnează vremea curentă și prognoza pentru un oraș sau localitate. "
        "Folosește pentru 'ce vreme e în X?', 'cum e afară?', 'plouă mâine?' etc. "
        "Dacă utilizatorul nu specifică un oraș, presupune Chișinău, Moldova."
    ),
    parameters={
        "oras": {
            "type": "STRING",
            "description": "Numele orașului sau localității. Ex: 'Chișinău', 'București', 'Iași'.",
        }
    },
)
def vremea(oras: str = "Chișinău"):
    """Vremea curentă pentru un oraș, via Open-Meteo (geocoding + forecast)."""
    try:
        # Pasul 1: geocoding — transformăm numele orașului în coordonate
        geo_raspuns = requests.get(
            _GEOCODING_URL,
            params={"name": oras, "count": 1, "language": "ro"},
            timeout=10,
        )
        geo_raspuns.raise_for_status()
        geo_date = geo_raspuns.json()

        rezultate = geo_date.get("results")
        if not rezultate:
            return f"Nu am găsit localitatea '{oras}'. Verifică numele și încearcă din nou."

        loc = rezultate[0]
        lat, lon = loc["latitude"], loc["longitude"]
        nume_gasit = loc.get("name", oras)
        tara = loc.get("country", "")

        # Pasul 2: forecast — cerem vremea curentă pe coordonatele găsite
        meteo_raspuns = requests.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "timezone": "auto",
                "forecast_days": 2,  # azi + mâine
            },
            timeout=10,
        )
        meteo_raspuns.raise_for_status()
        meteo_date = meteo_raspuns.json()

        curent = meteo_date["current"]
        zilnic = meteo_date["daily"]

        temp_curenta = curent["temperature_2m"]
        umiditate = curent["relative_humidity_2m"]
        vant = curent["wind_speed_10m"]
        vreme_curenta = _descrie_vreme(curent["weather_code"])

        temp_max_azi = zilnic["temperature_2m_max"][0]
        temp_min_azi = zilnic["temperature_2m_min"][0]
        temp_max_maine = zilnic["temperature_2m_max"][1]
        temp_min_maine = zilnic["temperature_2m_min"][1]
        vreme_maine = _descrie_vreme(zilnic["weather_code"][1])

        return (
            f"Vreme în {nume_gasit}, {tara}: {temp_curenta}°C, {vreme_curenta}, "
            f"umiditate {umiditate}%, vânt {vant} km/h. "
            f"Azi: {temp_min_azi}°C–{temp_max_azi}°C. "
            f"Mâine: {temp_min_maine}°C–{temp_max_maine}°C, {vreme_maine}."
        )

    except requests.exceptions.Timeout:
        return "Serviciul de meteo nu a răspuns la timp. Încearcă din nou."
    except requests.exceptions.RequestException as e:
        return f"Eroare la accesarea serviciului de meteo: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Răspuns neașteptat de la serviciul de meteo: {str(e)}"


# ==============================================================
# CĂUTARE WEB — DuckDuckGo Instant Answer API
# ==============================================================

_DDG_URL = "https://api.duckduckgo.com/"


@unealta(
    description=(
        "Caută un răspuns rapid pe internet pentru fapte, definiții sau subiecte "
        "cunoscute (persoane, locuri, concepte). ATENȚIE: NU este o căutare web "
        "completă — nu returnează liste de linkuri sau rezultate clasate, doar "
        "'instant answers' pentru subiecte bine documentate. Pentru știri recente "
        "sau întrebări foarte specifice, poate returna gol — în acest caz, "
        "informează utilizatorul că nu ai găsit un răspuns direct, nu inventa unul. "
        "Folosește pentru 'ce este X?', 'cine a fost X?', 'caută X' etc."
    ),
    parameters={
        "interogare": {
            "type": "STRING",
            "description": "Termenul sau întrebarea de căutat.",
        }
    },
    max_linii=15,
)
def cauta_web(interogare: str):
    """Caută un instant answer pe DuckDuckGo pentru interogarea dată."""
    try:
        raspuns = requests.get(
            _DDG_URL,
            params={
                "q": interogare,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        raspuns.raise_for_status()
        date = raspuns.json()

        # Verificăm sursele de răspuns în ordine de prioritate
        if date.get("AbstractText"):
            sursa = date.get("AbstractSource", "sursă necunoscută")
            url = date.get("AbstractURL", "")
            return f"{date['AbstractText']} (sursă: {sursa}, {url})"

        if date.get("Answer"):
            return date["Answer"]

        if date.get("Definition"):
            sursa = date.get("DefinitionSource", "")
            return f"{date['Definition']} (sursă: {sursa})"

        # Fallback: subiecte înrudite, dacă există
        subiecte_inrudite = date.get("RelatedTopics", [])
        rezultate_text = []
        for subiect in subiecte_inrudite[:5]:
            if "Text" in subiect:
                rezultate_text.append(subiect["Text"])

        if rezultate_text:
            return "Nu am găsit un răspuns direct, dar subiecte înrudite: " + " | ".join(rezultate_text)

        return (
            f"Nu am găsit niciun rezultat pentru '{interogare}'. "
            f"DuckDuckGo Instant Answer nu acoperă căutări web complete — "
            f"pentru rezultate detaliate, o căutare manuală în browser ar fi mai potrivită."
        )

    except requests.exceptions.Timeout:
        return "Serviciul de căutare nu a răspuns la timp. Încearcă din nou."
    except requests.exceptions.RequestException as e:
        return f"Eroare la accesarea serviciului de căutare: {str(e)}"