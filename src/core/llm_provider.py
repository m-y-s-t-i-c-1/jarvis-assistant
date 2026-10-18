"""
Provideri Externi — NVIDIA (dedicat pe categorie) + Cascadă Finală (Task 6.4)

Arhitectură (confirmată prin `python -m src.core.llm_provider` — lista
REALĂ de modele a contului, nu documentație/articole indirecte, care s-au
dovedit greșite de mai multe ori):
    - Gemini (3 chei): EXCLUSIV pentru tool-calling și localizare pe ecran.
    - NVIDIA "Conversație" (4 chei, cascadă de 3 modele distincte):
      nemotron-3.5-lightning -> kimi-k2.6 -> gpt-oss-120b -> nemotron-3.5-lightning.
    - NVIDIA "Cod" (1-3 chei, cascadă de 3 modele): deepseek-v4-pro-0813
      -> codestral-22b -> kimi-k3, pentru întrebări tehnice/de programare.
    - NVIDIA "Multimodal" (1-3 chei, cascadă de 3 modele): minimax-m3 ->
      llama-3.2-90b-vision -> neva-22b, pentru vedere ecran — folosit
      ÎNAINTEA lui Gemini (vezi src/tools/vedere.py).
    - Groq, OpenRouter, Bytez: cascada de siguranță.
"""

import os
import time
import base64
import itertools
import json
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()

TIMEOUT_PER_APEL = 18

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── NVIDIA — Modele Specifice din Snippet-uri (Conversație) ────────────────
_CONFIG_CONVERSATIE = [
    {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "params": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 4096}
        }
    },
    {
        # CORECTAT 2026-08-31: "z-ai/glm-4.7" NU EXISTĂ pe catalogul
        # contului tău (confirmat cu `python -m src.core.llm_provider` —
        # nu apare deloc în lista de 83 modele). Contul tău are acces la
        # un set de modele diferit de ce arată documentația/articolele
        # generale — kimi-k2.6 e confirmat REAL, prezent în listă.
        "model": "moonshotai/kimi-k2.6",
        "params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
        }
    },
    {
        # CORECTAT 2026-08-31: "mistralai/mistral-large-3-675b-instruct-2512"
        # NU EXISTĂ pe catalogul contului tău (confirmat, nu apare în
        # lista de 83). "openai/gpt-oss-120b" e confirmat REAL, prezent.
        "model": "openai/gpt-oss-120b",
        "params": {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024
        }
    },
    {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096
        }
    }
]

# Catalogul NVIDIA NIM se schimbă frecvent — modele pot fi retrase (410
# Gone) fără avertisment lung. Cache-uim aici orice model care ne-a
# răspuns vreodată cu 410 în sesiunea curentă, ca să-l sărim automat la
# următoarele apeluri, în loc să irosim un apel + 45s de timeout posibil
# pe un model deja mort. NU rezolvă modelul mort din config (tot trebuie
# actualizat manual, vezi comentariile de mai sus), dar reduce impactul
# până atunci. Dacă vezi frecvent modele sărite aici, verifică lista
# curentă la build.nvidia.com/models și actualizează _CONFIG_CONVERSATIE.
_modele_nvidia_retrase: set[str] = set()

_nvidia_chei_conversatie = [
    os.getenv(f"NVIDIA_API_KEY_CONVERSATION_{i}") for i in range(1, 5)
]
_nvidia_chei_conversatie = [k for k in _nvidia_chei_conversatie if k]

# Cuplăm fiecare cheie cu modelul ei specific
_pool_conversatie = []
for i, cheie in enumerate(_nvidia_chei_conversatie):
    config = _CONFIG_CONVERSATIE[i % len(_CONFIG_CONVERSATIE)]
    _pool_conversatie.append({
        "cheie": cheie,
        "model": config["model"],
        "params": config["params"]
    })

_nvidia_rotatie_conversatie = (
    itertools.cycle(_pool_conversatie) if _pool_conversatie else None
)
_NVIDIA_MODEL_CONVERSATIE_DEFAULT = os.getenv(
    "NVIDIA_MODEL_CONVERSATIE", "nvidia/llama-3.3-nemotron-super-49b-v1.5"
)

# ── NVIDIA — Cod (1-3 chei x 2 modele) ───────────────────────────────────────
# Citim NVIDIA_API_KEY_CODING (fără sufix) + NVIDIA_API_KEY_CODING_2,
# _3 ... — la fel ca la conversație. Fiecare cheie e o găleată SEPARATĂ
# de ~40 cereri/minut. Structura e identică cu cea de la multimodal:
# pentru fiecare cheie, încercăm pe rând toate modelele din
# _MODELE_CODING; pe 429 sărim direct la următoarea cheie (nu la
# următorul model pe aceeași cheie — vezi LimitaDeRataNvidia).
_nvidia_chei_coding = [
    os.getenv(f"NVIDIA_API_KEY_CODING{'' if i == 0 else f'_{i+1}'}")
    for i in range(3)
]
_nvidia_chei_coding = [k for k in _nvidia_chei_coding if k]

_MODELE_CODING = [
    os.getenv("NVIDIA_MODEL_CODING", "deepseek-ai/deepseek-v4-pro-0813"),
    # CORECTATE 2026-08-31: "qwen/..." și "z-ai/..." NU EXISTĂ deloc pe
    # catalogul contului tău (niciun model din familiile astea, confirmat
    # cu `python -m src.core.llm_provider`). Alese din lista reală:
    # mistralai/codestral-22b-instruct-v0.1 e un model DEDICAT de cod
    # (nu generalist adaptat), iar kimi-k3 e puternic pe coding agentic.
    "mistralai/codestral-22b-instruct-v0.1",
    "moonshotai/kimi-k3",
]
_MODELE_CODING = list(dict.fromkeys(_MODELE_CODING))

# ── NVIDIA — Multimodal / vedere (1-3 chei x 3 modele) ──────────────────────
# O cheie NVIDIA funcționează cu ORICE model din catalog (nu e legată de
# modelul pe care ai apăsat când ai generat-o) — deci putem combina mai
# multe chei CU mai multe modele. Structura: pentru fiecare cheie
# disponibilă, încercăm pe rând toate modelele din _MODELE_MULTIMODAL.
# Dacă o cheie dă 429 (limită de rată), oprim cascada de modele PE ACEA
# CHEIE (vezi LimitaDeRataNvidia mai jos — alt model pe aceeași cheie tot
# 429 ar da) și trecem la următoarea cheie, cu propria ei cascadă de
# modele de la capăt.
_nvidia_chei_multimodal = [
    os.getenv(f"NVIDIA_API_KEY_MULTIMODAL{'' if i == 0 else f'_{i+1}'}")
    for i in range(3)
]
_nvidia_chei_multimodal = [k for k in _nvidia_chei_multimodal if k]

_MODELE_MULTIMODAL = [
    os.getenv("NVIDIA_MODEL_MULTIMODAL", "minimaxai/minimax-m3"),
    # CORECTAT 2026-08-31: "nvidia/nemotron-nano-12b-v2-vl" NU EXISTĂ pe
    # catalogul contului tău (confirmat, nu apare în lista de 83).
    # meta/llama-3.2-90b-vision-instruct e un model dedicat de vedere,
    # confirmat real în listă; nvidia/neva-22b e modelul propriu NVIDIA
    # de vision-language, la fel confirmat real.
    "meta/llama-3.2-90b-vision-instruct",
    "nvidia/neva-22b",
]
# Eliminăm duplicate păstrând ordinea (dacă cineva a pus în .env exact
# unul din cele două modele adăugate implicit mai jos)
_MODELE_MULTIMODAL = list(dict.fromkeys(_MODELE_MULTIMODAL))



class LimitaDeRataNvidia(Exception):
    """
    Ridicată când o cheie NVIDIA a atins limita de RPM (429). Limita e
    GLOBALĂ pe cheie (~40 cereri/minut, partajată între TOATE modelele
    apelate cu acea cheie) — deci, spre deosebire de 410 (model retras)
    sau 503/timeout (problemă temporară a unui model anume), încercarea
    altui model CU ACEEAȘI CHEIE nu ajută deloc, tot 429 va da. Cascadele
    de mai jos prind explicit această excepție și opresc imediat
    încercarea altor modele pe aceeași cheie, trecând direct la fallback
    (altă cheie sau alt provider).
    """
    pass


def _apel_chat_openai(
    base_url: str,
    cheie: str,
    model: str,
    mesaje: list[dict],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int = 4096,
    stream: bool = False,
    seed: int | None = None,
    extra_body: dict | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> dict | Any | None:
    """Helper generic — POST {base_url}/chat/completions."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": mesaje,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if temperature is not None: payload["temperature"] = temperature
    if top_p is not None: payload["top_p"] = top_p
    if seed is not None: payload["seed"] = seed
    if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
    if extra_body: payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {cheie}",
        "Content-Type": "application/json",
    }

    _start = time.perf_counter()
    try:
        if stream:
            headers["Accept"] = "text/event-stream"
            raspuns = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT_PER_APEL, stream=True)
            raspuns.raise_for_status()
            print(f"[TIMP] NVIDIA '{model}' — până la primul byte (streaming): {time.perf_counter() - _start:.2f}s")
            return raspuns.iter_lines()
        else:
            raspuns = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT_PER_APEL)
            raspuns.raise_for_status()
            print(f"[TIMP] NVIDIA '{model}' — răspuns complet: {time.perf_counter() - _start:.2f}s")
            return raspuns.json()
    except requests.exceptions.Timeout:
        print(f"[NVIDIA] Timeout ({TIMEOUT_PER_APEL}s) pentru '{model}'")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[NVIDIA] HTTP {e.response.status_code} pentru '{model}': {e.response.text[:200]}")
        if e.response.status_code == 410:
            _modele_nvidia_retrase.add(model)
            print(f"[NVIDIA] Model '{model}' marcat ca RETRAS — va fi sărit automat de acum înainte.")
            return None
        if e.response.status_code == 429:
            # Limită de rată pe CHEIE, nu pe model — alt model cu aceeași
            # cheie va da tot 429. Semnalăm asta distinct cascadelor.
            raise LimitaDeRataNvidia(f"Cheia a atins limita de RPM (429) la modelul '{model}'")
        return None
    except Exception as e:
        print(f"[NVIDIA] Eroare la '{model}': {str(e)[:200]}")
        return None


def _extrage_continut(date: dict) -> str | None:
    try:
        if not isinstance(date, dict): return None
        choices = date.get("choices")
        if not choices or len(choices) == 0: return None
        message = choices[0].get("message", {})
        if message.get("refusal"):
            print(f"[NVIDIA] Model refuz: {message['refusal']}")
            return None
        return message.get("content")
    except Exception as e:
        print(f"[NVIDIA] Eroare extragere conținut: {e}")
        return None


def _proceseaza_stream(stream_generator) -> str:
    continut_final = []
    reasoning_final = []

    try:
        for line in stream_generator:
            if not line: continue
            line_decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_decoded.startswith("data: "): line_decoded = line_decoded[6:]
            if line_decoded.strip() == "[DONE]": break

            try:
                chunk = json.loads(line_decoded)
                if not chunk.get("choices"): continue
                delta = chunk["choices"][0].get("delta", {})

                reasoning = delta.get("reasoning_content")
                if reasoning: reasoning_final.append(reasoning)

                content = delta.get("content")
                if content is not None: continut_final.append(content)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[NVIDIA] Eroare în stream: {e}")

    if reasoning_final:
        print(f"[NVIDIA] Reasoning capturat ({len(''.join(reasoning_final))} caractere)")

    return "".join(continut_final)


def intreaba_nvidia_conversatie(
    mesaje: list[dict],
    *,
    stream: bool = False,
    **kwargs  # Ignorăm parametrii generici de la Jarvis, folosim optimizările modelelor
) -> str | None:
    if _nvidia_rotatie_conversatie is None:
        print("[NVIDIA] Nicio cheie de conversație configurată!")
        return None

    for _ in range(len(_pool_conversatie)):
        combo = next(_nvidia_rotatie_conversatie)
        cheie, model, params = combo["cheie"], combo["model"], combo["params"]

        if model in _modele_nvidia_retrase:
            print(f"[NVIDIA] Sar peste '{model}' (marcat anterior ca retras — 410 Gone)")
            continue

        print(f"[NVIDIA] Apelează Conversație: Model '{model}'")

        try:
            rezultat = _apel_chat_openai(
                _NVIDIA_BASE_URL,
                cheie,
                model,
                mesaje,
                stream=stream,
                **params
            )
        except LimitaDeRataNvidia as e:
            # Fiecare combo are o cheie DIFERITĂ (NVIDIA_API_KEY_CONVERSATION_1..4)
            # — un 429 pe una nu înseamnă că următoarea (altă cheie) va da
            # tot 429. Continuăm normal la următoarea combinație din rotație.
            print(f"[NVIDIA] {e} — încerc următoarea cheie/model")
            continue

        if rezultat is None:
            continue

        if stream:
            return _proceseaza_stream(rezultat)
        else:
            content = _extrage_continut(rezultat)
            if content: return content

    print(f"[NVIDIA] Toate cele {len(_pool_conversatie)} combinații au eșuat.")
    return None


def intreaba_nvidia_cod(
    mesaje: list[dict],
    *,
    stream: bool = False,
    **kwargs
) -> str | None:
    if not _nvidia_chei_coding:
        print("[NVIDIA] Nicio cheie de coding configurată!")
        return None

    # Optimizări extrase din snippet pentru z-ai/glm-5.2 — potrivite și
    # pentru celelalte modele de coding din cascadă
    params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "seed": 42
    }

    for index_cheie, cheie in enumerate(_nvidia_chei_coding):
        for model in _MODELE_CODING:
            if model in _modele_nvidia_retrase:
                print(f"[NVIDIA] Sar peste coding '{model}' (marcat anterior ca retras — 410 Gone)")
                continue

            print(f"[NVIDIA] Apelează Coding (cheie #{index_cheie + 1}): Model '{model}'")
            try:
                rezultat = _apel_chat_openai(
                    _NVIDIA_BASE_URL, cheie, model, mesaje, stream=stream, **params
                )
            except LimitaDeRataNvidia as e:
                # Toate modelele din bucla interioară folosesc ACEEAȘI
                # cheie — oprim doar bucla interioară și trecem la
                # cheia următoare, cu propria ei cascadă de la capăt.
                print(f"[NVIDIA] {e} — trec la următoarea cheie de coding")
                break

            if rezultat:
                continut = _proceseaza_stream(rezultat) if stream else _extrage_continut(rezultat)
                if continut:
                    return continut

    return None


def _detecteaza_mime_type(imagine_bytes: bytes) -> str:
    if imagine_bytes[:2] == b"\xff\xd8": return "image/jpeg"
    if imagine_bytes[:4] == b"\x89PNG": return "image/png"
    if imagine_bytes[:4] == b"GIF8": return "image/gif"
    if len(imagine_bytes) > 12 and imagine_bytes[:4] == b"RIFF" and imagine_bytes[8:12] == b"WEBP": return "image/webp"
    return "image/png"


def intreaba_nvidia_multimodal(
    imagine_bytes: bytes,
    intrebare: str,
    **kwargs
) -> str | None:
    if not _nvidia_chei_multimodal:
        print("[NVIDIA] Nicio cheie multimodal configurată!")
        return None

    # Optimizări generice, potrivite pentru toate modelele din cascadă
    params = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}

    mime_type = _detecteaza_mime_type(imagine_bytes)
    imagine_b64 = base64.b64encode(imagine_bytes).decode("utf-8")

    mesaje = [{"role": "user", "content": [
        {"type": "text", "text": intrebare},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{imagine_b64}"}}
    ]}]

    for index_cheie, cheie in enumerate(_nvidia_chei_multimodal):
        for model in _MODELE_MULTIMODAL:
            if model in _modele_nvidia_retrase:
                print(f"[NVIDIA] Sar peste multimodal '{model}' (marcat anterior ca retras — 410 Gone)")
                continue

            print(f"[NVIDIA] Apelează Multimodal (cheie #{index_cheie + 1}): Model '{model}'")
            try:
                rezultat = _apel_chat_openai(
                    _NVIDIA_BASE_URL, cheie, model, mesaje, stream=False, **params
                )
            except LimitaDeRataNvidia as e:
                # Toate modelele din bucla interioară folosesc ACEEAȘI
                # cheie — un 429 aici înseamnă limita de RPM a CHEII, nu a
                # modelului. Oprim doar bucla interioară (alt model pe
                # aceeași cheie tot 429 ar da) și trecem la cheia
                # următoare, cu propria ei cascadă de modele de la capăt.
                print(f"[NVIDIA] {e} — trec la următoarea cheie multimodal")
                break

            if rezultat:
                continut = _extrage_continut(rezultat)
                if continut:
                    return continut

    print(f"[NVIDIA] Cascada multimodal a eșuat pe toate cele {len(_nvidia_chei_multimodal)} chei.")
    return None


def _orice_cheie_nvidia() -> str | None:
    if _nvidia_chei_conversatie: return _nvidia_chei_conversatie[0]
    if _nvidia_chei_coding: return _nvidia_chei_coding[0]
    if _nvidia_chei_multimodal: return _nvidia_chei_multimodal[0]
    return None


_PROVIDERI_FINALI = [
    {
        "nume": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "cheie": lambda: os.getenv("OPENROUTER_API_KEY"),
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    },
    {
        "nume": "NVIDIA (generic)",
        "base_url": _NVIDIA_BASE_URL,
        "cheie": _orice_cheie_nvidia,
        "model": _NVIDIA_MODEL_CONVERSATIE_DEFAULT,
    },
    {
        "nume": "Bytez",
        "base_url": "https://api.bytez.com/models/v2/openai/v1",
        "cheie": lambda: os.getenv("BYTEZ_API_KEY"),
        "model": os.getenv("BYTEZ_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
    },
]


def _istoric_la_mesaje_openai(istoric: list, system_prompt: str) -> list[dict]:
    mesaje = [{"role": "system", "content": system_prompt}]
    for continut in istoric:
        if not hasattr(continut, "parts"): continue
        text = " ".join(p.text for p in continut.parts if hasattr(p, "text") and p.text)
        if not text: continue
        mesaje.append({"role": "user" if continut.role == "user" else "assistant", "content": text})
    return mesaje

istoric_la_mesaje_openai = _istoric_la_mesaje_openai


def ruleaza_cascada_externa(istoric: list, system_prompt: str, *, temperature: float = 0.7, max_tokens: int = 4096) -> str | None:
    mesaje = _istoric_la_mesaje_openai(istoric, system_prompt)
    for provider in _PROVIDERI_FINALI:
        cheie = provider["cheie"]()
        if not cheie: continue
        print(f"[Cascadă finală] Încerc {provider['nume']}...")
        try:
            rezultat = _apel_chat_openai(
                provider["base_url"], cheie, provider["model"], mesaje, temperature=temperature, max_tokens=max_tokens
            )
        except LimitaDeRataNvidia as e:
            # Fiecare provider din listă e un serviciu/cheie complet
            # separat — un 429 la unul nu spune nimic despre următorul.
            print(f"[Cascadă finală] {provider['nume']}: {e} — încerc următorul provider")
            continue
        if rezultat and isinstance(rezultat, dict):
            content = _extrage_continut(rezultat)
            if content:
                print(f"[Cascadă finală] {provider['nume']} a răspuns cu succes.")
                return content
    print("[Cascadă finală] Toți providerii au eșuat.")
    return None


def lista_modele_nvidia(cheie: str) -> list[str]:
    """
    Interoghează DIRECT endpoint-ul NVIDIA (/v1/models) cu o cheie dată și
    returnează lista EXACTĂ, reală, de ID-uri de modele disponibile acum.

    De ce există: modelele NVIDIA NIM se retrag/schimbă des (vezi
    comentariile din _CONFIG_CONVERSATIE/_MODELE_CODING/_MODELE_MULTIMODAL
    de mai sus — am nimerit greșit ID-uri de câteva ori, fie din surse
    indirecte (articole), fie din nume aparent logice care nu corespund
    exact slug-ului din API). Interogarea directă e singura sursă 100%
    de încredere.
    """
    try:
        raspuns = requests.get(
            f"{_NVIDIA_BASE_URL}/models",
            headers={"Authorization": f"Bearer {cheie}"},
            timeout=TIMEOUT_PER_APEL,
        )
        raspuns.raise_for_status()
        date = raspuns.json()
        return sorted(m["id"] for m in date.get("data", []))
    except Exception as e:
        print(f"[NVIDIA] Nu am putut obține lista de modele: {e}")
        return []


if __name__ == "__main__":
    # Rulează cu: python -m src.core.llm_provider
    # Afișează lista REALĂ de modele disponibile pentru fiecare cheie NVIDIA
    # configurată — folosește asta ca să confirmi ID-urile exacte înainte
    # să le pui în _CONFIG_CONVERSATIE / _MODELE_CODING / _MODELE_MULTIMODAL,
    # în loc să ghicești din articole sau documentație indirectă.
    toate_cheile = {
        "Conversație": _nvidia_chei_conversatie,
        "Cod": _nvidia_chei_coding,
        "Multimodal": _nvidia_chei_multimodal,
    }

    for categorie, chei in toate_cheile.items():
        if not chei:
            print(f"\n=== {categorie}: nicio cheie configurată ===")
            continue

        print(f"\n=== {categorie} (cheia #1) — modele disponibile ===")
        modele = lista_modele_nvidia(chei[0])
        if modele:
            for model_id in modele:
                print(f"  {model_id}")
            print(f"({len(modele)} modele găsite)")