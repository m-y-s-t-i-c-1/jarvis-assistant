"""
Provideri Externi — NVIDIA (dedicat pe categorie) + Cascadă Finală (Task 6.4)

Arhitectură (confirmată):
    - Gemini (3 chei): EXCLUSIV pentru tool-calling și localizare pe ecran.
    - NVIDIA "Conversație" (4 chei, rotative): asociate cu 4 modele distincte.
    - NVIDIA "Cod" (1 cheie): glm-5.2 pentru întrebări tehnice/de programare.
    - NVIDIA "Multimodal" (1 cheie): minimax-m3 pentru vedere generală.
    - Groq, OpenRouter, Bytez: cascada de siguranță.
"""

import os
import base64
import itertools
import json
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()

TIMEOUT_PER_APEL = 45

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
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
        }
    },
    {
        "model": "meta/llama-3.1-8b-instruct",
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

# ── NVIDIA — Cod (1 cheie) ───────────────────────────────────────────────────
_NVIDIA_CHEIE_CODING = os.getenv("NVIDIA_API_KEY_CODING")
_NVIDIA_MODEL_CODING = os.getenv("NVIDIA_MODEL_CODING", "z-ai/glm-5.2")

# ── NVIDIA — Multimodal / vedere (1 cheie) ──────────────────────────────────
_NVIDIA_CHEIE_MULTIMODAL = os.getenv("NVIDIA_API_KEY_MULTIMODAL")
_NVIDIA_MODEL_MULTIMODAL = os.getenv("NVIDIA_MODEL_MULTIMODAL", "minimaxai/minimax-m3")


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

    try:
        if stream:
            headers["Accept"] = "text/event-stream"
            raspuns = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT_PER_APEL, stream=True)
            raspuns.raise_for_status()
            return raspuns.iter_lines()
        else:
            raspuns = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT_PER_APEL)
            raspuns.raise_for_status()
            return raspuns.json()
    except requests.exceptions.Timeout:
        print(f"[NVIDIA] Timeout ({TIMEOUT_PER_APEL}s) pentru '{model}'")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[NVIDIA] HTTP {e.response.status_code} pentru '{model}': {e.response.text[:200]}")
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

        print(f"[NVIDIA] Apelează Conversație: Model '{model}'")

        rezultat = _apel_chat_openai(
            _NVIDIA_BASE_URL,
            cheie,
            model,
            mesaje,
            stream=stream,
            **params
        )

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
    if not _NVIDIA_CHEIE_CODING:
        print("[NVIDIA] Cheia de coding nu este configurată!")
        return None

    # Optimizări extrase din snippet pentru z-ai/glm-5.2
    params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "seed": 42
    }

    print(f"[NVIDIA] Apelează Coding: Model '{_NVIDIA_MODEL_CODING}'")
    rezultat = _apel_chat_openai(
        _NVIDIA_BASE_URL, _NVIDIA_CHEIE_CODING, _NVIDIA_MODEL_CODING, mesaje, stream=stream, **params
    )

    if rezultat:
        return _proceseaza_stream(rezultat) if stream else _extrage_continut(rezultat)
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
    if not _NVIDIA_CHEIE_MULTIMODAL:
        print("[NVIDIA] Cheia multimodal nu este configurată!")
        return None

    # Optimizări din snippet pentru minimax-m3
    params = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}

    mime_type = _detecteaza_mime_type(imagine_bytes)
    imagine_b64 = base64.b64encode(imagine_bytes).decode("utf-8")

    mesaje = [{"role": "user", "content": [
        {"type": "text", "text": intrebare},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{imagine_b64}"}}
    ]}]

    print(f"[NVIDIA] Apelează Multimodal: Model '{_NVIDIA_MODEL_MULTIMODAL}'")
    rezultat = _apel_chat_openai(
        _NVIDIA_BASE_URL, _NVIDIA_CHEIE_MULTIMODAL, _NVIDIA_MODEL_MULTIMODAL, mesaje, stream=False, **params
    )

    if rezultat: return _extrage_continut(rezultat)
    return None


def _orice_cheie_nvidia() -> str | None:
    if _nvidia_chei_conversatie: return _nvidia_chei_conversatie[0]
    return _NVIDIA_CHEIE_CODING or _NVIDIA_CHEIE_MULTIMODAL


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
        rezultat = _apel_chat_openai(
            provider["base_url"], cheie, provider["model"], mesaje, temperature=temperature, max_tokens=max_tokens
        )
        if rezultat and isinstance(rezultat, dict):
            content = _extrage_continut(rezultat)
            if content:
                print(f"[Cascadă finală] {provider['nume']} a răspuns cu succes.")
                return content
    print("[Cascadă finală] Toți providerii au eșuat.")
    return Nones