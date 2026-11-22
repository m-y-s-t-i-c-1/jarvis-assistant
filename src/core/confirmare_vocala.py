"""
Confirmare vocală pentru acțiunile care cer aprobare (security.py).

Până acum confirmarea se făcea doar cu input() în terminal: în modul vocal Jarvis aștepta
în tăcere să tastezi „da", iar microfonul rămânea blocat. Aici Jarvis întreabă cu voce tare
și ascultă „da" / „nu".

Principii:
  - fail closed: orice neclaritate, eroare sau tăcere = NU se execută;
  - orice negație („nu", „stop", „anulează") bate orice afirmație („da, dar nu acum" = NU);
  - blacklist-ul din security.py rămâne necondiționat, fără confirmare.
Atenție: o confirmare vocală e mai slabă decât una tastată (oricine aude mesajul sau un
sunet din fundal poate spune „da"); de aceea cerem răspuns scurt și clar.
"""

import re
import unicodedata

MAX_INCERCARI = 3
MAX_CUVINTE_RASPUNS = 5   # un răspuns mai lung nu e o confirmare clară

_DA = {"da", "daa", "sigur", "confirm", "confirmat", "executa", "ok", "okay", "bine",
       "desigur", "aprob", "aprobat", "continua"}
_NU = {"nu", "nuu", "stop", "anuleaza", "anulez", "anulat", "renunt", "opreste", "niciodata"}
# Cuvinte de politețe/umplutură permise ÎN JURUL unei afirmații („da, te rog", „sigur, acum").
_UMPLUTURA = {"te", "rog", "va", "acum", "atunci", "hai", "o", "fa"}
_DA_FRAZE = {"fa o", "fao", "fa o te rog"}


def _normalizeaza(text: str) -> list[str]:
    fara_diacritice = "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.findall(r"[a-z]+", fara_diacritice)


def interpreteaza_raspuns(text: str):
    """True = confirmat, False = refuzat, None = neclar (se reîntreabă)."""
    cuvinte = _normalizeaza(text or "")
    if not cuvinte or len(cuvinte) > MAX_CUVINTE_RASPUNS:
        return None
    if any(c in _NU for c in cuvinte):
        return False
    text_curat = " ".join(cuvinte)
    if text_curat in _DA_FRAZE:
        return True
    # FIX: înainte, ORICE cuvânt din _DA într-un enunț de până la 8 cuvinte însemna „da" —
    # „ce execută?" sau „continuă cu ce?" aprobau o acțiune. Acum toate cuvintele trebuie să fie
    # afirmații sau umplutură, cu cel puțin o afirmație.
    if all(c in _DA or c in _UMPLUTURA for c in cuvinte) and any(c in _DA for c in cuvinte):
        return True
    return None


def _rezumat_actiune(nume_functie: str, argumente: dict) -> str:
    nume = nume_functie.replace("_", " ")
    valori = [str(v).strip() for v in (argumente or {}).values() if str(v).strip()]
    # FIX: se rosteau doar primele 2 valori, tăiate la 60 de caractere, fără nicio mențiune că
    # restul lipsește -> utilizatorul putea aproba o comandă pe care nu o auzise întreagă.
    detalii = "; ".join(v[:120] for v in valori[:3])
    if len(valori) > 3 or any(len(v) > 120 for v in valori[:3]):
        detalii += " și alte detalii"
    return f"{nume}: {detalii}" if detalii else nume


def cere_confirmare_vocala(nume_functie: str, argumente: dict) -> bool:
    # Importuri târzii: modulul audio e greu (torch) și nu trebuie încărcat în modul text.
    from src.core.tts import spune
    from src.core.vad import obtine_detector
    from src.core.stt import transcrie

    try:
        spune(f"Vasea, vreau să execut {_rezumat_actiune(nume_functie, argumente)}. "
              f"Confirmi? Spune da sau nu.")
        detector = obtine_detector()
        for _ in range(MAX_INCERCARI):
            audio = detector.asculta_pana_la_pauza()
            if audio is None:
                # FIX: tăcerea consuma o încercare fără niciun semn; utilizatorul nu știa că e ascultat.
                spune("Nu te-am auzit. Spune da sau nu.")
                continue
            text = transcrie(audio)
            print(f"[Confirmare vocală] Am auzit: '{text}'")
            decizie = interpreteaza_raspuns(text)
            if decizie is True:
                spune("Confirmat.")
                return True
            if decizie is False:
                spune("Anulat.")
                return False
            spune("Nu am înțeles. Spune doar da sau nu.")
        spune("Nu am primit o confirmare clară. Anulez acțiunea.")
    except Exception as e:
        print(f"[Confirmare vocală] Eroare: {type(e).__name__}: {e} — acțiune refuzată.")
    return False
