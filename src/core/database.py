"""
Configurarea Stocării Persistente (Task 4.1)

Baza de date SQLite locală care stochează tot ce Jarvis trebuie să țină
minte între sesiuni: conversații, profil utilizator, preferințe, memorie
episodică și semantică.

Fișierul DB e creat automat la prima rulare în:
    ~/.jarvis/jarvis.db

Schema tabelelor:
    conversatii     — istoricul complet al mesajelor (user + assistant)
    sesiuni         — gruparea mesajelor pe sesiuni de lucru
    profil          — chei/valori pentru profilul utilizatorului
    memorie         — fapte și informații extrase din conversații
    joburi_log      — log-ul joburilor rulate în fundal

Utilizare:
    from src.core.database import db
    db.salveaza_mesaj(sesiune_id, "user", "Salut Jarvis")
    db.salveaza_mesaj(sesiune_id, "assistant", "Salut, Vasea.")
    mesaje = db.incarca_sesiune(sesiune_id)
"""

import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


# ── Calea bazei de date ───────────────────────────────────────────────────────

JARVIS_DIR = Path.home() / ".jarvis"
DB_PATH    = JARVIS_DIR / "jarvis.db"


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
-- Sesiuni de conversație (o sesiune = o rulare a programului)
CREATE TABLE IF NOT EXISTS sesiuni (
    id          TEXT PRIMARY KEY,
    creat_la    TEXT NOT NULL,
    inchis_la   TEXT,
    rezumat     TEXT          -- rezumat auto-generat al sesiunii (Task 4.3)
);

-- Mesajele individuale din conversații
CREATE TABLE IF NOT EXISTS conversatii (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sesiune_id  TEXT NOT NULL REFERENCES sesiuni(id),
    rol         TEXT NOT NULL CHECK(rol IN ('user', 'assistant', 'tool')),
    continut    TEXT NOT NULL,
    creat_la    TEXT NOT NULL
);

-- Profilul utilizatorului (chei/valori simple)
CREATE TABLE IF NOT EXISTS profil (
    cheie       TEXT PRIMARY KEY,
    valoare     TEXT NOT NULL,
    actualizat  TEXT NOT NULL
);

-- Memoria semantică: fapte extrase din conversații (Task 4.4, 4.5)
CREATE TABLE IF NOT EXISTS memorie (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tip         TEXT NOT NULL,   -- 'fapt', 'preferinta', 'context', 'corectie'
    continut    TEXT NOT NULL,
    sursa       TEXT,            -- sesiune_id de unde provine
    relevanta   REAL DEFAULT 1.0,
    creat_la    TEXT NOT NULL,
    actualizat  TEXT NOT NULL
);

-- Log joburi în fundal (Task 2.8 — persistent între sesiuni)
CREATE TABLE IF NOT EXISTS joburi_log (
    id          TEXT PRIMARY KEY,
    descriere   TEXT NOT NULL,
    comanda     TEXT NOT NULL,
    status      TEXT NOT NULL,
    rezultat    TEXT,
    creat_la    TEXT NOT NULL,
    finalizat   TEXT
);

-- Indecși pentru interogări frecvente
CREATE INDEX IF NOT EXISTS idx_conv_sesiune ON conversatii(sesiune_id);
CREATE INDEX IF NOT EXISTS idx_conv_creat   ON conversatii(creat_la);
CREATE INDEX IF NOT EXISTS idx_mem_tip      ON memorie(tip);
CREATE INDEX IF NOT EXISTS idx_mem_relev    ON memorie(relevanta DESC);
"""


# ── Clasa principală ──────────────────────────────────────────────────────────

class JarvisDB:
    """
    Interfața principală pentru baza de date Jarvis.
    Folosește connection-per-call cu context manager, thread-safe.
    """

    def __init__(self, cale: Path = DB_PATH):
        self.cale = cale
        self._initializeaza()

    def _initializeaza(self):
        """Creează directorul și tabelele dacă nu există."""
        self.cale.parent.mkdir(parents=True, exist_ok=True)

        with self._conexiune() as con:
            con.executescript(SCHEMA)

        print(f"[DB] Baza de date inițializată: {self.cale}")

    def _conexiune(self) -> sqlite3.Connection:
        """Returnează o conexiune cu row_factory pentru acces prin nume de coloană."""
        con = sqlite3.connect(self.cale)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")   # mai rapid la scrieri concurente
        con.execute("PRAGMA foreign_keys=ON")
        return con

    # ── Sesiuni ───────────────────────────────────────────────────────────────

    def incepe_sesiune(self) -> str:
        """Creează o sesiune nouă și returnează ID-ul ei."""
        sesiune_id = str(uuid.uuid4())[:8]
        acum = datetime.now().isoformat()

        with self._conexiune() as con:
            con.execute(
                "INSERT INTO sesiuni (id, creat_la) VALUES (?, ?)",
                (sesiune_id, acum)
            )

        print(f"[DB] Sesiune nouă: {sesiune_id}")
        return sesiune_id

    def inchide_sesiune(self, sesiune_id: str, rezumat: str = ""):
        """Marchează sesiunea ca închisă și salvează rezumatul opțional."""
        with self._conexiune() as con:
            con.execute(
                "UPDATE sesiuni SET inchis_la=?, rezumat=? WHERE id=?",
                (datetime.now().isoformat(), rezumat, sesiune_id)
            )

    def listeaza_sesiuni(self, limita: int = 10) -> list[dict]:
        """Returnează ultimele N sesiuni."""
        with self._conexiune() as con:
            rows = con.execute(
                "SELECT * FROM sesiuni ORDER BY creat_la DESC LIMIT ?",
                (limita,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Mesaje ────────────────────────────────────────────────────────────────

    def salveaza_mesaj(self, sesiune_id: str, rol: str, continut: str) -> int:
        """
        Salvează un mesaj în conversație.

        Parametri:
            sesiune_id: ID-ul sesiunii curente
            rol:        'user', 'assistant' sau 'tool'
            continut:   textul mesajului

        Returnează:
            ID-ul rândului inserat
        """
        with self._conexiune() as con:
            cursor = con.execute(
                "INSERT INTO conversatii (sesiune_id, rol, continut, creat_la) VALUES (?,?,?,?)",
                (sesiune_id, rol, continut, datetime.now().isoformat())
            )
            return cursor.lastrowid

    def incarca_sesiune(self, sesiune_id: str) -> list[dict]:
        """Returnează toate mesajele dintr-o sesiune, în ordine cronologică."""
        with self._conexiune() as con:
            rows = con.execute(
                "SELECT * FROM conversatii WHERE sesiune_id=? ORDER BY creat_la",
                (sesiune_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def incarca_istoric_recent(self, limita_mesaje: int = 20) -> list[dict]:
        """
        Returnează ultimele N mesaje din toate sesiunile.
        Util pentru a reconstrui contextul la pornire (Task 4.3).
        """
        with self._conexiune() as con:
            rows = con.execute(
                """SELECT c.*, s.creat_la as sesiune_data
                   FROM conversatii c
                   JOIN sesiuni s ON c.sesiune_id = s.id
                   ORDER BY c.creat_la DESC
                   LIMIT ?""",
                (limita_mesaje,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]  # cronologic

    # ── Profil utilizator ─────────────────────────────────────────────────────

    def seteaza_profil(self, cheie: str, valoare: str):
        """Setează sau actualizează o valoare în profilul utilizatorului."""
        with self._conexiune() as con:
            con.execute(
                """INSERT INTO profil (cheie, valoare, actualizat)
                   VALUES (?, ?, ?)
                   ON CONFLICT(cheie) DO UPDATE SET valoare=excluded.valoare,
                   actualizat=excluded.actualizat""",
                (cheie, valoare, datetime.now().isoformat())
            )

    def get_profil(self, cheie: str, default: str = "") -> str:
        """Returnează o valoare din profil, sau default dacă nu există."""
        with self._conexiune() as con:
            row = con.execute(
                "SELECT valoare FROM profil WHERE cheie=?", (cheie,)
            ).fetchone()
        return row["valoare"] if row else default

    def get_tot_profilul(self) -> dict:
        """Returnează întreg profilul ca dicționar."""
        with self._conexiune() as con:
            rows = con.execute("SELECT cheie, valoare FROM profil").fetchall()
        return {r["cheie"]: r["valoare"] for r in rows}

    # ── Memorie semantică ─────────────────────────────────────────────────────

    def salveaza_memorie(
        self,
        tip: str,
        continut: str,
        sursa: str = "",
        relevanta: float = 1.0,
    ) -> int:
        """
        Salvează un fapt sau o informație în memoria semantică.

        Parametri:
            tip:       'fapt', 'preferinta', 'context', 'corectie'
            continut:  textul faptului/preferinței
            sursa:     sesiune_id de unde provine
            relevanta: scor de importanță [0.0-1.0]
        """
        acum = datetime.now().isoformat()
        with self._conexiune() as con:
            cursor = con.execute(
                """INSERT INTO memorie (tip, continut, sursa, relevanta, creat_la, actualizat)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (tip, continut, sursa, relevanta, acum, acum)
            )
            return cursor.lastrowid

    def cauta_memorie(self, tip: str = "", limita: int = 10) -> list[dict]:
        """
        Returnează amintiri filtrate după tip, ordonate după relevanță.

        Parametri:
            tip:    filtru opțional ('fapt', 'preferinta' etc.) — gol = toate
            limita: numărul maxim de rezultate
        """
        with self._conexiune() as con:
            if tip:
                rows = con.execute(
                    """SELECT * FROM memorie WHERE tip=?
                       ORDER BY relevanta DESC, actualizat DESC LIMIT ?""",
                    (tip, limita)
                ).fetchall()
            else:
                rows = con.execute(
                    """SELECT * FROM memorie
                       ORDER BY relevanta DESC, actualizat DESC LIMIT ?""",
                    (limita,)
                ).fetchall()
        return [dict(r) for r in rows]

    def sterge_memorie(self, memorie_id: int):
        """Șterge o amintire specifică după ID."""
        with self._conexiune() as con:
            con.execute("DELETE FROM memorie WHERE id=?", (memorie_id,))

    # ── Statistici ────────────────────────────────────────────────────────────

    def statistici(self) -> dict:
        """Returnează statistici generale despre baza de date."""
        with self._conexiune() as con:
            nr_sesiuni  = con.execute("SELECT COUNT(*) FROM sesiuni").fetchone()[0]
            nr_mesaje   = con.execute("SELECT COUNT(*) FROM conversatii").fetchone()[0]
            nr_amintiri = con.execute("SELECT COUNT(*) FROM memorie").fetchone()[0]
            prima_sesiune = con.execute(
                "SELECT MIN(creat_la) FROM sesiuni"
            ).fetchone()[0]

        return {
            "sesiuni":       nr_sesiuni,
            "mesaje_total":  nr_mesaje,
            "amintiri":      nr_amintiri,
            "prima_sesiune": prima_sesiune or "N/A",
            "cale_db":       str(self.cale),
        }


# ── Instanță globală ──────────────────────────────────────────────────────────

db = JarvisDB()


if __name__ == "__main__":
    # Test rapid al tuturor operațiilor
    print("\n=== Test JarvisDB ===\n")

    # Sesiune
    sid = db.incepe_sesiune()

    # Mesaje
    db.salveaza_mesaj(sid, "user", "Salut Jarvis, cum ești?")
    db.salveaza_mesaj(sid, "assistant", "Bine, Vasea. Cu ce pot ajuta?")
    db.salveaza_mesaj(sid, "user", "Spune-mi ora curentă.")

    mesaje = db.incarca_sesiune(sid)
    print(f"Mesaje salvate în sesiunea {sid}:")
    for m in mesaje:
        print(f"  [{m['rol']}] {m['continut']}")

    # Profil
    db.seteaza_profil("nume", "Vasea")
    db.seteaza_profil("director_proiect", "/home/vaseoc/Downloads/jarvis-assistant-main")
    db.seteaza_profil("tema_editor", "dark")
    print(f"\nProfil: {db.get_tot_profilul()}")

    # Memorie
    db.salveaza_memorie("preferinta", "Vasea preferă răspunsuri scurte și la obiect", sid)
    db.salveaza_memorie("fapt", "Vasea lucrează la un proiect Jarvis în Python", sid)
    db.salveaza_memorie("context", "Sistemul rulează pe Arch Linux cu PipeWire", sid)

    amintiri = db.cauta_memorie()
    print(f"\nAmintiri salvate:")
    for a in amintiri:
        print(f"  [{a['tip']}] {a['continut']}")

    # Statistici
    print(f"\nStatistici: {db.statistici()}")

    # Închide sesiunea
    db.inchide_sesiune(sid, rezumat="Sesiune de test a bazei de date.")
    print(f"\n✅ Toate testele au trecut. DB la: {DB_PATH}")