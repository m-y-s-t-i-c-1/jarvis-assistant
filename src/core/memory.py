"""
Memoria Episodică (Task 4.4) + Logica de Injecție Dinamică (Task 4.6)

Două responsabilități combinate:

1. EXTRAGERE — după fiecare răspuns al lui Jarvis, analizează conversația
   și extrage automat fapte, preferințe și context relevant, salvându-le
   în tabelul `memorie` din DB.

2. INJECȚIE — la începutul fiecărei sesiuni, adună amintirile relevante
   din DB și le injectează în system prompt, ca Jarvis să "știe" ce s-a
   întâmplat în sesiunile anterioare.

Tipuri de memorie:
    fapt       — informații obiective despre utilizator sau proiect
    preferinta — cum preferă Vasea să lucreze sau să primească răspunsuri
    context    — starea curentă a unui proiect sau task
    corectie   — când Vasea a corectat o eroare a lui Jarvis

Utilizare în main.py:
    # La pornire — injectăm amintirile în system prompt
    from src.core.memory import memorie
    system_prompt_complet = memorie.construieste_system_prompt(SYSTEM_PROMPT)

    # După fiecare răspuns — extragem noi amintiri
    memorie.extrage_si_salveaza(mesaj_user, raspuns_jarvis, sesiune_id, client, model)
"""

from google.genai import types
from src.core.database import db

# ── Configurare ───────────────────────────────────────────────────────────────

# Câte amintiri injectăm în system prompt (cele mai relevante)
MAX_AMINTIRI_INJECTATE = 10

# Promptul pentru extragerea automată de fapte din conversație
PROMPT_EXTRACTIE = """Analizează acest schimb de conversație și extrage informații importante
despre utilizatorul numit Vasea sau despre contextul de lucru.

Returnează DOAR un JSON valid cu această structură (fără markdown, fără explicații):
{{
  "fapte": ["fapt obiectiv 1", "fapt obiectiv 2"],
  "preferinte": ["preferinta 1", "preferinta 2"],
  "context": ["context tehnic sau de proiect relevant"],
  "corectii": ["corecție dacă Vasea a corectat ceva"]
}}

Reguli stricte:
- Dacă nu există informații pentru o categorie, lasă lista goală [].
- Extrage DOAR informații noi, concrete și utile pentru conversații viitoare.
- Ignoră saluturile, întrebările banale și răspunsurile generice.
- Maximum 2-3 elemente per categorie.

Conversație:
Vasea: {mesaj_user}
Jarvis: {raspuns_jarvis}

JSON:"""


class ManagerMemorie:
    """Extrage, salvează și injectează amintiri episodice."""

    def extrage_si_salveaza(
        self,
        mesaj_user: str,
        raspuns_jarvis: str,
        sesiune_id: str,
        client,
        model: str = "gemini-2.5-flash",
    ) -> int:
        """
        Trimite conversația la Gemini, extrage fapte/preferințe/context
        și le salvează în DB. Apelat după fiecare răspuns al lui Jarvis.

        Returnează numărul de amintiri salvate.
        """
        prompt = PROMPT_EXTRACTIE.format(
            mesaj_user=mesaj_user,
            raspuns_jarvis=raspuns_jarvis,
        )

        try:
            raspuns = client.models.generate_content(
                model=model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )],
            )
            text = raspuns.text.strip()

            # Curățăm markdown dacă modelul a adăugat ```json
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            import json
            date = json.loads(text)

        except Exception as e:
            print(f"[Memorie] Eroare la extragere: {e}")
            return 0

        salvate = 0
        tip_map = {
            "fapte":      "fapt",
            "preferinte": "preferinta",
            "context":    "context",
            "corectii":   "corectie",
        }

        for cheie, tip in tip_map.items():
            for item in date.get(cheie, []):
                if item and len(item.strip()) > 10:  # ignorăm string-uri prea scurte
                    db.salveaza_memorie(
                        tip=tip,
                        continut=item.strip(),
                        sursa=sesiune_id,
                        relevanta=1.0,
                    )
                    salvate += 1

        if salvate:
            print(f"[Memorie] {salvate} amintiri noi salvate.")

        return salvate

    def incarca_amintiri_relevante(self, limita: int = MAX_AMINTIRI_INJECTATE) -> list[dict]:
        """Returnează cele mai relevante amintiri din DB."""
        return db.cauta_memorie(limita=limita)

    def bloc_amintiri(self) -> str:
        """
        Construiește un bloc text cu amintirile relevante,
        gata de injectat în system prompt.
        """
        amintiri = self.incarca_amintiri_relevante()

        if not amintiri:
            return ""

        linii = ["=== Ce știu despre tine din conversații anterioare ==="]

        pe_tip = {}
        for a in amintiri:
            tip = a["tip"]
            if tip not in pe_tip:
                pe_tip[tip] = []
            pe_tip[tip].append(a["continut"])

        etichete = {
            "fapt":       "Fapte",
            "preferinta": "Preferințe",
            "context":    "Context tehnic",
            "corectie":   "Corecții anterioare",
        }

        for tip, eticheta in etichete.items():
            if tip in pe_tip:
                linii.append(f"{eticheta}:")
                for item in pe_tip[tip]:
                    linii.append(f"  - {item}")

        return "\n".join(linii)

    def construieste_system_prompt(self, system_prompt_baza: str) -> str:
        """
        Adaugă profilul utilizatorului și amintirile episodice
        la system prompt-ul de bază.

        Apelat o singură dată la pornirea sesiunii.
        """
        from src.core.profile import profil

        sectiuni = [system_prompt_baza]

        # Profil utilizator
        bloc_profil = profil.bloc_context()
        if bloc_profil:
            sectiuni.append(bloc_profil)

        # Amintiri episodice
        bloc_mem = self.bloc_amintiri()
        if bloc_mem:
            sectiuni.append(bloc_mem)

        return "\n\n".join(sectiuni)


# ── Instanță globală ──────────────────────────────────────────────────────────

memorie = ManagerMemorie()


if __name__ == "__main__":
    print("\n=== Test ManagerMemorie ===\n")

    # Test bloc amintiri din DB (cele salvate în testele anterioare)
    bloc = memorie.bloc_amintiri()
    if bloc:
        print("Amintiri existente în DB:")
        print(bloc)
    else:
        print("Nicio amintire în DB încă.")

    # Test construire system prompt complet
    SYSTEM_PROMPT_TEST = "Tu ești Jarvis, asistentul lui Vasea."
    prompt_complet = memorie.construieste_system_prompt(SYSTEM_PROMPT_TEST)
    print(f"\nSystem prompt complet ({len(prompt_complet)} caractere):")
    print(prompt_complet[:500] + "..." if len(prompt_complet) > 500 else prompt_complet)
    print("\n✅ ManagerMemorie funcționează.")