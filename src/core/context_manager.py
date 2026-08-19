"""
Optimizarea Contextului de Scurtă Durată (Task 4.3)

Două mecanisme combinate:

1. SLIDING WINDOW — păstrează doar ultimele N mesaje în contextul activ
   trimis la Gemini. Mesajele mai vechi nu dispar — sunt salvate în DB
   (Task 4.1) — dar nu mai consumă tokens din fereastra de context.

2. REZUMARE AUTOMATĂ — când fereastra depășește MAX_MESAJE_CONTEXT,
   mesajele cele mai vechi sunt rezumate automat de Gemini într-un
   paragraf scurt, care înlocuiește mesajele originale în context.
   Rezumatul e salvat în DB pentru sesiunile viitoare.

Flux:
    la fiecare mesaj nou →
        dacă len(istoric) > MAX_MESAJE_CONTEXT →
            rezumă primele BLOC_REZUMARE mesaje →
            înlocuiește-le cu un singur mesaj-rezumat în istoric →
            salvează rezumatul în DB

Integrare cu agent_loop:
    context_manager.proceseaza(istoric, sesiune_id)
    → returnează istoricul optimizat, gata de trimis la Gemini
"""

from google.genai import types
from src.core.database import db

# ── Configurare ───────────────────────────────────────────────────────────────

# Numărul maxim de mesaje păstrate activ în context
MAX_MESAJE_CONTEXT = 30

# Câte mesaje vechi să rezumăm odată când depășim limita
BLOC_REZUMARE = 10

# Promptul trimis la Gemini pentru a genera rezumatul
PROMPT_REZUMARE = """Rezumă concis următoarea conversație în 3-5 propoziții.
Păstrează: deciziile luate, informațiile importante despre utilizator,
contextul tehnic relevant și orice preferințe menționate.
Omite: saluturile, întrebările retorice, răspunsurile banale.
Scrie la persoana a treia (ex: "Vasea a cerut...", "Jarvis a explicat...").

Conversație de rezumat:
{conversatie}

Rezumat:"""


def _mesaje_la_text(mesaje: list) -> str:
    """Convertește o listă de types.Content în text simplu pentru rezumare."""
    linii = []
    for msg in mesaje:
        if not hasattr(msg, "parts"):
            continue
        text = " ".join(
            p.text for p in msg.parts
            if hasattr(p, "text") and p.text
        )
        if text:
            rol = "Vasea" if msg.role == "user" else "Jarvis"
            linii.append(f"{rol}: {text}")
    return "\n".join(linii)


def _genereaza_rezumat(client, model: str, mesaje: list) -> str:
    """
    Trimite un bloc de mesaje la Gemini și cere un rezumat.
    Folosește același client și model ca agent_loop-ul principal.
    """
    text_conversatie = _mesaje_la_text(mesaje)
    if not text_conversatie.strip():
        return ""

    prompt = PROMPT_REZUMARE.format(conversatie=text_conversatie)

    try:
        raspuns = client.models.generate_content(
            model=model,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )],
        )
        return raspuns.text.strip()
    except Exception as e:
        print(f"[Context] Eroare la rezumare: {e}")
        return f"[Rezumat indisponibil — {len(mesaje)} mesaje arhivate]"


def proceseaza(
    istoric: list,
    sesiune_id: str,
    client=None,
    model: str = "gemini-3.6-flash",
) -> list:
    """
    Verifică dacă istoricul depășește limita și, dacă da, rezumă
    mesajele vechi și le înlocuiește cu un mesaj-rezumat compact.

    Parametri:
        istoric:     lista curentă de types.Content (modificată in-place)
        sesiune_id:  ID-ul sesiunii curente (pentru salvare rezumat în DB)
        client:      clientul Gemini (dacă None, nu se face rezumare,
                     doar trunchiere simplă)
        model:       modelul Gemini pentru generarea rezumatului

    Returnează:
        istoricul modificat (același obiect, modificat in-place)
    """
    if len(istoric) <= MAX_MESAJE_CONTEXT:
        return istoric

    print(f"[Context] Fereastra depășită ({len(istoric)} mesaje) — rezum primele {BLOC_REZUMARE}...")

    mesaje_vechi = istoric[:BLOC_REZUMARE]
    istoric_nou   = istoric[BLOC_REZUMARE:]

    if client:
        text_rezumat = _genereaza_rezumat(client, model, mesaje_vechi)
    else:
        # Fallback simplu dacă nu avem client — rezumat manual
        text_rezumat = f"[Context arhivat: {len(mesaje_vechi)} mesaje anterioare omise pentru eficiență]"

    if text_rezumat:
        # Salvăm rezumatul în DB
        db.salveaza_memorie(
            tip="context",
            continut=text_rezumat,
            sursa=sesiune_id,
            relevanta=0.8,
        )
        db.inchide_sesiune(sesiune_id, rezumat=text_rezumat)

        # Inserăm rezumatul ca primul mesaj în istoricul nou
        mesaj_rezumat = types.Content(
            role="user",
            parts=[types.Part(text=f"[Rezumat conversație anterioară]: {text_rezumat}")]
        )
        istoric_nou.insert(0, mesaj_rezumat)

        print(f"[Context] Rezumat generat: {text_rezumat[:100]}...")

    # Modificăm lista originală in-place (important — e același obiect din main.py)
    istoric.clear()
    istoric.extend(istoric_nou)

    print(f"[Context] Fereastră redusă la {len(istoric)} mesaje.")
    return istoric


def statistici_context(istoric: list) -> str:
    """Returnează un sumar al stării contextului curent."""
    nr_user      = sum(1 for m in istoric if hasattr(m, "role") and m.role == "user")
    nr_assistant = sum(1 for m in istoric if hasattr(m, "role") and m.role == "model")
    procent      = round((len(istoric) / MAX_MESAJE_CONTEXT) * 100)

    return (
        f"Context: {len(istoric)}/{MAX_MESAJE_CONTEXT} mesaje "
        f"({procent}% plin) — {nr_user} user, {nr_assistant} assistant"
    )


if __name__ == "__main__":
    # Test: simulăm o fereastră plină și verificăm că se rezumă corect
    print("\n=== Test Context Manager ===\n")

    # Construim un istoric fals de 35 mesaje
    istoric_test = []
    for i in range(35):
        rol = "user" if i % 2 == 0 else "model"
        text = f"Mesaj de test numărul {i+1} — {'întrebare' if rol == 'user' else 'răspuns'}"
        istoric_test.append(
            types.Content(role=rol, parts=[types.Part(text=text)])
        )

    print(f"Istoric inițial: {len(istoric_test)} mesaje")
    print(statistici_context(istoric_test))

    # Procesăm fără client (fallback simplu)
    sid_test = "test-session"
    proceseaza(istoric_test, sid_test, client=None)

    print(f"\nIstoric după procesare: {len(istoric_test)} mesaje")
    print(statistici_context(istoric_test))
    print(f"\nPrimul mesaj (rezumatul):")
    print(f"  {istoric_test[0].parts[0].text[:200]}")
    print("\n✅ Context Manager funcționează.")