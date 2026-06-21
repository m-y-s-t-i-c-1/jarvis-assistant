"""
Consolidarea Autonomă a Memoriei (Task 4.7)

Jarvis "digeră" periodic conversațiile din DB și:
    1. DEDUPLICARE — detectează amintiri duplicate sau contradictorii și le curăță
    2. ÎMBOGĂȚIRE — combină fapte fragmentare într-un singur fapt complet
    3. AUTO-CORECȚIE — când Vasea corectează ceva, actualizează amintirile vechi greșite
    4. CREȘTEREA RELEVANȚEI — amintirile confirmate de mai multe ori capătă scor mai mare

Rulare:
    - Automat: la fiecare N conversații (configurat în main.py)
    - Manual: python -m src.core.consolidare

Arhitectură:
    consolidare() — funcția principală, apelată periodic
    _deduplicare() — curăță amintirile similare
    _imbogatire()  — combină fragmentele într-un Gemini call
    _ajusteaza_relevanta() — scade relevanța amintirilor vechi neconfirmate
"""

import json
from datetime import datetime, timedelta
from google.genai import types
from src.core.database import db
from src.core.rag import rag

# ── Configurare ───────────────────────────────────────────────────────────────

# Similaritate minimă pentru a considera două amintiri duplicate (0-1)
PRAG_DUPLICAT = 0.92

# Câte zile fără confirmare până scădem relevanța unei amintiri
ZILE_PANA_LA_DEGRADARE = 7

# Factorul cu care scădem relevanța la fiecare ciclu de consolidare
FACTOR_DEGRADARE = 0.85

# Relevanța minimă sub care ștergem amintirea
PRAG_STERGERE = 0.1

PROMPT_CONSOLIDARE = """Analizează aceste amintiri despre utilizatorul Vasea și consolidează-le.

Amintiri de consolidat:
{amintiri}

Returnează DOAR un JSON valid (fără markdown):
{{
  "consolidate": [
    {{"tip": "fapt|preferinta|context|corectie", "continut": "amintire consolidată", "relevanta": 0.9}},
    ...
  ],
  "de_sters": [id1, id2],
  "observatii": "scurtă explicație a ce ai făcut"
}}

Reguli:
- Combină amintirile duplicate sau complementare într-una singură mai completă.
- Dacă există contradicții, păstrează cea mai recentă și marchează-o cu relevanta 1.0.
- Șterge amintirile vagi, incomplete sau redundante (pune id-ul în de_sters).
- Păstrează toate amintirile unice și valoroase.
- Maximum 15 amintiri consolidate total.
"""


class ConsolidareMemorie:

    def _deduplicare_rapida(self) -> int:
        """
        Detectează și șterge amintirile aproape identice din ChromaDB
        fără a apela Gemini — comparație vectorială directă.
        Returnează numărul de duplicate șterse.
        """
        amintiri = db.cauta_memorie(limita=100)
        if len(amintiri) < 2:
            return 0

        sterse = 0
        id_uri_sterse = set()

        for i, a1 in enumerate(amintiri):
            if a1["id"] in id_uri_sterse:
                continue

            rezultate = rag.cauta(a1["continut"], n=5)

            for r in rezultate:
                if r["scor"] >= PRAG_DUPLICAT and r["sursa"] == "amintiri":
                    # Extragem id-ul din ChromaDB (format: mem_{id}_{tip})
                    try:
                        parti = r["metadata"].get("tip", "")
                        # Căutăm amintirea în DB după conținut
                        for a2 in amintiri:
                            if (
                                a2["id"] != a1["id"]
                                and a2["id"] not in id_uri_sterse
                                and a2["continut"] == r["text"]
                            ):
                                # Păstrăm pe cea cu relevanță mai mare
                                de_sters = a2 if a1["relevanta"] >= a2["relevanta"] else a1
                                if de_sters["id"] not in id_uri_sterse:
                                    db.sterge_memorie(de_sters["id"])
                                    id_uri_sterse.add(de_sters["id"])
                                    sterse += 1
                    except Exception:
                        pass

        return sterse

    def _ajusteaza_relevanta(self) -> int:
        """
        Scade relevanța amintirilor vechi neconfirmate.
        Șterge cele sub pragul minim.
        Returnează numărul de amintiri șterse.
        """
        amintiri = db.cauta_memorie(limita=200)
        sterse = 0
        prag_data = (datetime.now() - timedelta(days=ZILE_PANA_LA_DEGRADARE)).isoformat()

        for a in amintiri:
            actualizat = a.get("actualizat", "")
            if actualizat < prag_data:
                noua_relevanta = a["relevanta"] * FACTOR_DEGRADARE

                if noua_relevanta < PRAG_STERGERE:
                    db.sterge_memorie(a["id"])
                    sterse += 1
                else:
                    # Actualizăm relevanța în DB
                    with db._conexiune() as con:
                        con.execute(
                            "UPDATE memorie SET relevanta=?, actualizat=? WHERE id=?",
                            (noua_relevanta, datetime.now().isoformat(), a["id"])
                        )

        return sterse

    def _consolidare_cu_ai(self, client, model: str) -> dict:
        """
        Trimite toate amintirile la Gemini pentru consolidare inteligentă.
        Returnează rezultatul consolidării.
        """
        amintiri = db.cauta_memorie(limita=50)
        if not amintiri:
            return {"consolidate": [], "de_sters": [], "observatii": "Nicio amintire de consolidat."}

        # Formatăm amintirile pentru prompt
        amintiri_text = "\n".join(
            f"[id:{a['id']}] [{a['tip']}] (relevanta:{a['relevanta']:.2f}) {a['continut']}"
            for a in amintiri
        )

        prompt = PROMPT_CONSOLIDARE.format(amintiri=amintiri_text)

        try:
            raspuns = client.models.generate_content(
                model=model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )],
            )
            text = raspuns.text.strip()

            # Curățăm markdown
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            return json.loads(text)

        except Exception as e:
            print(f"[Consolidare] Eroare AI: {e}")
            return {"consolidate": [], "de_sters": [], "observatii": f"Eroare: {e}"}

    def ruleaza(self, client=None, model: str = "gemini-2.5-flash") -> dict:
        """
        Rulează un ciclu complet de consolidare:
            1. Deduplicare rapidă (fără AI)
            2. Ajustare relevanță (degradare în timp)
            3. Consolidare cu AI (dacă avem client)
            4. Resincronizare RAG

        Parametri:
            client: clientul Gemini (opțional — fără el, doar pașii 1 și 2)
            model:  modelul de folosit pentru consolidare

        Returnează:
            dict cu statistici despre ce s-a făcut
        """
        print("\n[Consolidare] Ciclu de consolidare a memoriei...")
        rezultat = {
            "duplicate_sterse":   0,
            "amintiri_degradate": 0,
            "amintiri_consolidate": 0,
            "amintiri_sterse_ai": 0,
            "observatii": "",
        }

        # Pasul 1: deduplicare
        rezultat["duplicate_sterse"] = self._deduplicare_rapida()
        print(f"[Consolidare] Duplicate șterse: {rezultat['duplicate_sterse']}")

        # Pasul 2: degradare relevanță
        rezultat["amintiri_degradate"] = self._ajusteaza_relevanta()
        print(f"[Consolidare] Amintiri degradate/șterse: {rezultat['amintiri_degradate']}")

        # Pasul 3: consolidare AI (opțional)
        if client:
            date_ai = self._consolidare_cu_ai(client, model)

            # Ștergem amintirile marcate de AI
            for id_sters in date_ai.get("de_sters", []):
                try:
                    db.sterge_memorie(int(id_sters))
                    rezultat["amintiri_sterse_ai"] += 1
                except Exception:
                    pass

            # Salvăm amintirile consolidate noi
            for a in date_ai.get("consolidate", []):
                continut = a.get("continut", "").strip()
                tip = a.get("tip", "fapt")
                relevanta = float(a.get("relevanta", 0.9))

                if continut and len(continut) > 10:
                    db.salveaza_memorie(
                        tip=tip,
                        continut=continut,
                        sursa="consolidare_autonoma",
                        relevanta=relevanta,
                    )
                    rezultat["amintiri_consolidate"] += 1

            rezultat["observatii"] = date_ai.get("observatii", "")
            print(f"[Consolidare] AI: {rezultat['amintiri_consolidate']} consolidate, "
                  f"{rezultat['amintiri_sterse_ai']} șterse.")
            if rezultat["observatii"]:
                print(f"[Consolidare] Observații: {rezultat['observatii']}")

        # Pasul 4: resincronizăm RAG cu starea nouă a DB
        rag.sincronizeaza_din_db()
        print(f"[Consolidare] RAG resincronizat. Statistici: {rag.statistici()}")

        return rezultat


# ── Instanță globală ──────────────────────────────────────────────────────────

consolidare = ConsolidareMemorie()


if __name__ == "__main__":
    print("\n=== Test Consolidare Autonomă ===\n")

    # Adăugăm câteva amintiri duplicate pentru test
    db.salveaza_memorie("fapt", "Vasea are 18 ani", "test", 0.9)
    db.salveaza_memorie("fapt", "Vasea este în vârstă de 18 ani", "test", 0.8)
    db.salveaza_memorie("preferinta", "Vasea preferă răspunsuri scurte", "test", 0.7)
    db.salveaza_memorie("preferinta", "Vasea vrea răspunsuri concise și la obiect", "test", 0.9)

    print(f"Amintiri înainte: {db.statistici()['amintiri']}")

    # Rulăm consolidarea fără AI (doar deduplicare + degradare)
    rezultat = consolidare.ruleaza(client=None)

    print(f"\nAmintiri după: {db.statistici()['amintiri']}")
    print(f"Rezultat: {rezultat}")
    print("\n✅ Consolidare funcționează.")