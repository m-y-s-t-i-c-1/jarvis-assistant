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

import re
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
        # FIX: `rag = RAGManager()` rulează la import. Dacă modelul de
        # embedding nu se poate descărca (prima rulare offline) sau
        # sentence-transformers/chromadb lipsește, TOATĂ aplicația (main.py,
        # server.py) pica la pornire. Acum RAG-ul se dezactivează elegant:
        # restul lui Jarvis merge, doar fără căutare semantică.
        self._activ = False
        try:
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
            self._activ = True
        except Exception as e:
            print(f"[RAG] DEZACTIVAT — inițializare eșuată: {type(e).__name__}: {e}")
            return

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
        if not self._activ or not continut or len(continut.strip()) < 10:
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
        if not self._activ or not continut or len(continut.strip()) < 5:
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
                    "amintire_id": int(amintire_id),
                    "timestamp": datetime.now().isoformat(),
                }],
                ids=[doc_id],
            )
            return doc_id
        except Exception as e:
            print(f"[RAG] Eroare indexare amintire: {e}")
            return ""

    def _id_uri_amintiri(self) -> list[str]:
        """Toate ID-urile din colecția de amintiri (fără a încărca documentele)."""
        try:
            return list(self._col_amintiri.get(include=[])["ids"])
        except Exception:
            return list(self._col_amintiri.get()["ids"])

    def sterge_amintire(self, amintire_id: int) -> int:
        """
        FIX: consolidare.py ștergea amintiri doar din SQLite; în ChromaDB
        rămâneau pentru totdeauna ("amintiri fantomă" returnate de cauta()).
        Șterge din vectori toate documentele amintirii cu acest ID.
        """
        if not self._activ:
            return 0
        try:
            prefix = f"mem_{int(amintire_id)}_"
            de_sters = [i for i in self._id_uri_amintiri() if i.startswith(prefix)]
            if de_sters:
                self._col_amintiri.delete(ids=de_sters)
            return len(de_sters)
        except Exception as e:
            print(f"[RAG] Eroare ștergere amintire {amintire_id}: {e}")
            return 0

    def sincronizeaza_din_db(self):
        """
        Importă toate amintirile existente din SQLite în ChromaDB.
        Util la prima rulare sau după import manual de date.
        """
        if not self._activ:
            return 0

        from src.core.database import db

        # FIX 1: limita era 500 -> amintirile peste #500 (după relevanță)
        # nu ajungeau niciodată în RAG. Acum le luăm pe toate.
        amintiri = db.cauta_memorie(limita=1_000_000)
        ids_db = {a["id"] for a in amintiri}

        # FIX 2: înainte se făcea un `get()` separat PER amintire (N interogări
        # la fiecare pornire). Acum citim ID-urile existente o singură dată.
        existente = set(self._id_uri_amintiri())

        # FIX 3: curățăm "amintirile fantomă" — vectori ale căror rânduri
        # au fost șterse din SQLite (inclusiv cele rămase de la versiunile vechi).
        orfane = []
        for doc_id in existente:
            m = re.match(r"^mem_(\d+)_", doc_id)
            if m and int(m.group(1)) not in ids_db:
                orfane.append(doc_id)
        if orfane:
            try:
                self._col_amintiri.delete(ids=orfane)
                existente -= set(orfane)
            except Exception as e:
                print(f"[RAG] Eroare la curățarea orfanilor: {e}")

        indexate = 0
        for a in amintiri:
            if f"mem_{a['id']}_{a['tip']}" in existente:
                continue
            doc_id = self.indexeaza_amintire(
                tip=a["tip"],
                continut=a["continut"],
                sursa=a.get("sursa", ""),
                amintire_id=a["id"],
            )
            if doc_id:
                indexate += 1

        print(f"[RAG] Sincronizat: {indexate} amintiri noi, {len(orfane)} orfane șterse.")
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
        if not self._activ:
            return []

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
        if not self._activ:
            return {
                "conversatii_indexate": 0,
                "amintiri_indexate":    0,
                "cale_chroma":          str(CHROMA_DIR),
                "activ":                False,
            }
        return {
            "conversatii_indexate": self._col_conversatii.count(),
            "amintiri_indexate":    self._col_amintiri.count(),
            "cale_chroma":          str(CHROMA_DIR),
            "activ":                True,
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