# Jarvis

Asistent personal în română, cu control asupra calculatorului (Linux), memorie între sesiuni și mod vocal.
Rulează local; textul trece prin API-uri cu nivel gratuit (Gemini în primul rând).

> Proiect personal, în dezvoltare. Controlul ecranului funcționează doar pe X11, nu pe Wayland.

## Ce poate face

- **Conversație** text în terminal, prin UI web sau prin voce („Hey Jarvis").
- **Unelte** (apelate de model): ora și data, vreme, căutare rapidă, Google Calendar, comenzi de sistem
  read-only, volum și luminozitate, aplicații, Git, server HTTP local, joburi în fundal, control mouse/tastatură,
  analiză de ecran.
- **Sub-agenți** specializați (software, devops, cercetare), fiecare cu propriul set limitat de unelte.
- **Memorie**: profil, amintiri extrase din conversații și căutare semantică (ChromaDB), în `~/.jarvis/`.
- **Securitate**: blacklist de comenzi periculoase, confirmare (text, vocală sau în UI) pentru acțiunile cu impact,
  checkpoint Git automat înainte de ele.

## Cerințe

- Linux (dezvoltat pe Arch), Python **3.11+** (config-ul folosește `tomllib`)
- O cheie Gemini ([Google AI Studio](https://aistudio.google.com/))
- Pentru voce: PipeWire sau PulseAudio, microfon, binarul [Piper](https://github.com/rhasspy/piper) și o voce
  românească (`ro_RO-mihai-medium.onnx`, din repo-ul de voci Piper) în `voci_piper/`
- Opțional: `xdotool` (taste cu diacritice), `brightnessctl` (luminozitate), `pactl` (volum)

## Instalare

```bash
git clone https://github.com/m-y-s-t-i-c-1/jarvis-assistant.git
cd jarvis-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # completează cel puțin GEMINI_API_KEY
cp config/jarvis.example.toml config/jarvis.toml   # opțional: nume, oraș, fus orar
```

`requirements.txt` listează dependențele grele (torch, chromadb, faster-whisper) fără versiuni fixate.
După ce ai un mediu care funcționează, poți să-l blochezi cu `pip freeze > requirements.lock`.

## Configurare

Prioritate: **variabile de mediu / `.env`** > **`config/jarvis.toml`** > valori implicite.

| Ce | Fișier de config | Variabilă de mediu |
|---|---|---|
| Cum te numește Jarvis | `persona.utilizator` | `JARVIS_USER` |
| Orașul pentru vreme | `locatie.oras` | `JARVIS_ORAS` |
| Fusul orar (Calendar) | `locatie.fus_orar` | `JARVIS_TIMEZONE` |

Toate variabilele de mediu sunt documentate în `.env.example`.

## Rulare

```bash
python main.py                 # alegi: [1] text, [2] voce, [3] text + wake word în fundal
python -m src.ui.server        # UI web: http://localhost:8080  (widget compact: /desktop)
python -m src.ui.widget        # fereastră flotantă (necesită pywebview)
```

Google Calendar: pune `credentials.json` în rădăcina proiectului și rulează o dată
`python -m src.tools.calendar_google` dintr-un terminal pentru autentificare.

## Confidențialitate

- Mesajele se trimit la providerii configurați (Gemini și, dacă ai chei, NVIDIA/Groq/OpenRouter/Bytez).
- **Monitorul de ecran** trimite periodic screenshot-uri către API-uri externe. Dezactivează-l cu
  `JARVIS_MONITOR_ECRAN=0` în `.env`.
- Datele locale: `~/.jarvis/jarvis.db` (conversații, memorie), `~/.jarvis/chroma/` (vectori),
  `~/.jarvis/git_safety_log.txt` (checkpoint-uri Git).
- UI-ul web ascultă implicit doar pe `127.0.0.1`.

## Siguranță

Acțiunile marcate `necesita_confirmare` (commit, push, click, taste, ștergere eveniment, joburi) cer aprobare.
Înainte de ele se creează un checkpoint Git sub `refs/jarvis-safety/*`; îl recuperezi cu
`git restore --source=<hash> --worktree -- .` (hash-ul e în `~/.jarvis/git_safety_log.txt`).
Blacklist-ul din `src/core/security.py` blochează comenzile periculoase fără confirmare.
E o plasă de siguranță, nu o garanție: nu rula Jarvis cu drepturi mai mari decât ai da unui script necunoscut.

## Teste

```bash
python -m pytest -q
```

Testele rulează fără microfon, rețea sau chei API. Cele care au nevoie de dependențe audio sau de server se
sar singure dacă acestea lipsesc. CI-ul (`.github/workflows/ci.yml`) rulează `ruff` (doar erori reale) și
`pytest` pe Python 3.12.

## Structură

```
main.py                 punctul de intrare (text / voce / ambele)
config/                 șablon de configurare
src/core/               agent, securitate, memorie, voce (VAD, STT, TTS, wake word), routing, sub-agenți
src/tools/              unelte expuse modelului (@unealta)
src/ui/                 server FastAPI, widget desktop, interfața web
tests/                  teste de regresie
voci_piper/             voce Piper (modelul .onnx se descarcă separat)
```

Adaugi o unealtă nouă scriind o funcție în `src/tools/` decorată cu `@unealta(...)`; apare automat în registru.

## Probleme frecvente

- **„Nu am găsit nicio GEMINI_API_KEY"**: lipsește `.env` sau cheia din el.
- **Fără sunet la voce**: verifică `PIPER_BINARY`, `PIPER_MODEL` și că ai `paplay` sau `pw-play`.
- **Barge-in nu merge**: ai nevoie de modulul PipeWire de anulare a ecoului; vezi comentariile din `src/core/barge_in.py`
  și `src/core/tts.py`.
- **Controlul ecranului nu pornește**: ai nevoie de X11 cu `DISPLAY` setat.
