"""
Memoria Semantică / RAG (Task 4.5)

Bază de date vectorială locală (ChromaDB) pentru căutare semantică în:
    - conversațiile anterioare salvate în DB
    - amintirile episodice (fapte, preferințe, context)
    - (opțional, viitor) codul sursă și documentațiile proiectului

Diferența față de memoria episodică (Task 4.4):
    - Task 4.4 = extragere structurată de fapte + injecție directă în prompt
    - Task 4.5 = căutare semantică prin similaritate vectorială
                 ("ce știu despre X?" găsește și lucruri care nu conțin exact cuvântul X)

Arhitectură:
    - ChromaDB persistent local în ~/.jarvis/chroma/
    - Embedding model: all-MiniLM-L6-v2 (mic, rapid, offline, ~80MB)
    - Două colecții: "conversatii" și "amintiri"

Utilizare:
    from src.core.rag import rag
    rag.indexeaza_mesaj("user", "Lucrez la un proiect Jarvis în Python", "ses-123")
    rezultate = rag.cauta("proiect Python", n=3)
"""

import os
from pathlib import Path
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

# ── Configurare ───────────────────────────────────────────────────────────────

CHROMA_DIR = Path.home() / ".jarvis" / "chroma"

# Modelul de embedding — rulează local, offline, fără API
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Numărul maxim de rezultate returnate la căutare
N_REZULTATE_DEFAULT = 5


class RAGManager:
    """
    Manager pentru baza de date vectorială ChromaDB.
    Indexează mesaje și amintiri, permite căutare semantică.
    """

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        # Client persistent — datele supraviețuiesc între sesiuni
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Funcția de embedding — all-MiniLM-L6-v2, descărcată automat prima dată
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Colecțiile
        self._col_conversatii = self._client.get_or_create_collection(
            name="conversatii",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._col_amintiri = self._client.get_or_create_collection(
            name="amintiri",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"[RAG] ChromaDB inițializat: {CHROMA_DIR}")
        print(f"[RAG] Conversații indexate: {self._col_conversatii.count()}")
        print(f"[RAG] Amintiri indexate: {self._col_amintiri.count()}")

    # ── Indexare ──────────────────────────────────────────────────────────────

    def indexeaza_mesaj(self, rol: str, continut: str, sesiune_id: str) -> str:
        """
        Indexează un mesaj din conversație în colecția vectorială.

        Parametri:
            rol:        'user' sau 'assistant'
            continut:   textul mesajului
            sesiune_id: ID-ul sesiunii

        Returnează:
            ID-ul documentului indexat
        """
        if not continut or len(continut.strip()) < 10:
            return ""

        doc_id = f"{sesiune_id}_{rol}_{datetime.now().timestamp()}"

        try:
            self._col_conversatii.add(
                documents=[continut],
                metadatas=[{
                    "rol": rol,
                    "sesiune_id": sesiune_id,
                    "timestamp": datetime.now().isoformat(),
                }],
                ids=[doc_id],
            )
            return doc_id
        except Exception as e:
            print(f"[RAG] Eroare indexare mesaj: {e}")
            return ""

    def indexeaza_amintire(
        self,
        tip: str,
        continut: str,
        sursa: str = "",
        amintire_id: int = 0,
    ) -> str:
        """
        Indexează o amintire episodică (din tabelul `memorie` din DB).

        Parametri:
            tip:         'fapt', 'preferinta', 'context', 'corectie'
            continut:    textul amântirii
            sursa:       sesiune_id de origine
            amintire_id: ID-ul din tabelul SQLite (pentru deduplicare)
        """
        if not continut or len(continut.strip()) < 5:
            return ""

        doc_id = f"mem_{amintire_id}_{tip}"

        try:
            # Verificăm dacă există deja (evităm duplicate)
            existente = self._col_amintiri.get(ids=[doc_id])
            if existente["ids"]:
                return doc_id  # deja indexat

            self._col_amintiri.add(
                documents=[continut],
                metadatas=[{
                    "tip": tip,
                    "sursa": sursa,
                    "timestamp": datetime.now().isoformat(),
                }],
                ids=[doc_id],
            )
            return doc_id
        except Exception as e:
            print(f"[RAG] Eroare indexare amintire: {e}")
            return ""

    def sincronizeaza_din_db(self):
        """
        Importă toate amintirile existente din SQLite în ChromaDB.
        Util la prima rulare sau după import manual de date.
        """
        from src.core.database import db

        amintiri = db.cauta_memorie(limita=500)
        indexate = 0

        for a in amintiri:
            doc_id = self.indexeaza_amintire(
                tip=a["tip"],
                continut=a["continut"],
                sursa=a.get("sursa", ""),
                amintire_id=a["id"],
            )
            if doc_id:
                indexate += 1

        print(f"[RAG] Sincronizat {indexate} amintiri din SQLite.")
        return indexate

    # ── Căutare ───────────────────────────────────────────────────────────────

    def cauta(self, interogare: str, n: int = N_REZULTATE_DEFAULT) -> list[dict]:
        """
        Caută semantic în ambele colecții (conversații + amintiri)
        și returnează cele mai relevante rezultate.

        Parametri:
            interogare: textul de căutat (semantic, nu exact)
            n:          numărul de rezultate per colecție

        Returnează:
            Listă de dicționare cu 'text', 'tip', 'scor', 'metadata'
        """
        rezultate = []

        # Căutare în amintiri
        try:
            if self._col_amintiri.count() > 0:
                res_amintiri = self._col_amintiri.query(
                    query_texts=[interogare],
                    n_results=min(n, self._col_amintiri.count()),
                )
                for doc, meta, dist in zip(
                    res_amintiri["documents"][0],
                    res_amintiri["metadatas"][0],
                    res_amintiri["distances"][0],
                ):
                    rezultate.append({
                        "text":     doc,
                        "tip":      meta.get("tip", "amintire"),
                        "scor":     round(1 - dist, 3),  # cosine distance → similaritate
                        "metadata": meta,
                        "sursa":    "amintiri",
                    })
        except Exception as e:
            print(f"[RAG] Eroare căutare amintiri: {e}")

        # Căutare în conversații
        try:
            if self._col_conversatii.count() > 0:
                res_conv = self._col_conversatii.query(
                    query_texts=[interogare],
                    n_results=min(n, self._col_conversatii.count()),
                )
                for doc, meta, dist in zip(
                    res_conv["documents"][0],
                    res_conv["metadatas"][0],
                    res_conv["distances"][0],
                ):
                    rezultate.append({
                        "text":     doc,
                        "tip":      f"conversatie_{meta.get('rol', '?')}",
                        "scor":     round(1 - dist, 3),
                        "metadata": meta,
                        "sursa":    "conversatii",
                    })
        except Exception as e:
            print(f"[RAG] Eroare căutare conversații: {e}")

        # Sortăm după scor descrescător
        rezultate.sort(key=lambda x: x["scor"], reverse=True)
        return rezultate[:n]

    def cauta_ca_text(self, interogare: str, n: int = N_REZULTATE_DEFAULT) -> str:
        """
        Returnează rezultatele căutării ca bloc de text,
        gata de injectat în prompt.
        """
        rezultate = self.cauta(interogare, n)

        if not rezultate:
            return ""

        linii = [f"=== Context relevant pentru '{interogare}' ==="]
        for r in rezultate:
            if r["scor"] >= 0.3:  # filtrăm rezultatele prea puțin relevante
                linii.append(f"[{r['tip']} | relevanță: {r['scor']}] {r['text']}")

        return "\n".join(linii) if len(linii) > 1 else ""

    # ── Statistici ────────────────────────────────────────────────────────────

    def statistici(self) -> dict:
        return {
            "conversatii_indexate": self._col_conversatii.count(),
            "amintiri_indexate":    self._col_amintiri.count(),
            "cale_chroma":          str(CHROMA_DIR),
        }


# ── Instanță globală ──────────────────────────────────────────────────────────

rag = RAGManager()


if __name__ == "__main__":
    print("\n=== Test RAG ===\n")

    # Sincronizăm amintirile existente din SQLite
    rag.sincronizeaza_din_db()

    # Indexăm câteva mesaje de test
    rag.indexeaza_mesaj("user", "Lucrez la un proiect Jarvis în Python cu Gemini API", "test-001")
    rag.indexeaza_mesaj("user", "Folosesc Arch Linux cu PipeWire pentru audio", "test-001")
    rag.indexeaza_mesaj("user", "Am 18 ani și sunt programator", "test-001")
    rag.indexeaza_mesaj("assistant", "Înțeles, Vasea. Proiectul tău Jarvis progresează bine.", "test-001")

    print(f"\nStatistici după indexare: {rag.statistici()}")

    # Test căutare semantică
    interogari_test = [
        "ce vârstă are utilizatorul",
        "sistem de operare și audio",
        "proiect Python AI",
    ]

    for interogare in interogari_test:
        print(f"\nCăutare: '{interogare}'")
        rezultate = rag.cauta(interogare, n=3)
        for r in rezultate:
            print(f"  [{r['scor']:.3f}] [{r['tip']}] {r['text'][:80]}")

    print("\n✅ RAG funcționează.")