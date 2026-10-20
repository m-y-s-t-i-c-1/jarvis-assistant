"""
Prompt-ul pentru o tură: profil + amintiri proaspete + context RAG din conversații anterioare.

Înainte:
  - amintirile intrau în prompt o singură dată, la pornire (un fapt învățat azi apărea abia după
    repornire);
  - RAG-ul indexa fiecare mesaj, dar nu era citit nicăieri pentru răspunsuri.
"""

MIN_CARACTERE_INTEROGARE = 12   # „da", „salut" etc. nu merită o căutare semantică
PRAG_SCOR = 0.35
MAX_CARACTERE_FRAGMENT = 300


def prompt_pentru_tura(system_prompt_baza: str, system_prompt_rezerva: str,
                       text_user: str, sesiune_id: str | None) -> str:
    # 1) profil + amintiri citite acum din DB; dacă eșuează, folosim prompt-ul construit la pornire
    try:
        from src.core.memory import memorie
        prompt = memorie.construieste_system_prompt(system_prompt_baza)
    except Exception as e:
        print(f"[Prompt] Nu am putut reîmprospăta amintirile: {e}")
        prompt = system_prompt_rezerva

    # 2) context RAG din ALTE sesiuni (turele sesiunii curente sunt deja în istoric)
    if len((text_user or "").strip()) < MIN_CARACTERE_INTEROGARE:
        return prompt
    try:
        from src.core.rag import rag
        rezultate = rag.cauta(text_user, n=4)
    except Exception as e:
        print(f"[RAG] Căutare omisă: {e}")
        return prompt

    linii = []
    for r in rezultate:
        if r["scor"] < PRAG_SCOR:
            continue
        if r["sursa"] == "conversatii" and r["metadata"].get("sesiune_id") == sesiune_id:
            continue
        if r["text"] in prompt:  # amintirea e deja în prompt
            continue
        linii.append(f"- ({r['tip']}) {r['text'][:MAX_CARACTERE_FRAGMENT]}")

    if not linii:
        return prompt
    return (prompt + "\n\n=== Context din conversații/amintiri anterioare, posibil relevant pentru "
            "cererea curentă (folosește-l doar dacă ajută; nu îl cita) ===\n" + "\n".join(linii))
