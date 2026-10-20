"""
Persistența unei ture de conversație pentru modul vocal.

Modul text (main.py) salvează fiecare tură în DB + RAG, extrage amintiri episodice și pornește
periodic consolidarea. Bucla vocală (audio_loop.py) nu făcea nimic din toate astea, deci
conversațiile vorbite se pierdeau și Jarvis nu învăța din ele. Funcția de aici face același lucru,
dar sigur pentru fundal: orice eroare e înghițită, nu oprește conversația.
"""

import threading

_lock_consolidare = threading.Lock()
_ture = 0

# Consolidarea memoriei rulează la fiecare N ture vocale (main.py folosește „la 10 mesaje" din DB,
# condiție care nu mai pornea dacă totalul devenea impar).
CONSOLIDARE_LA_N_TURE = 5


def _consolidare_in_fundal(client, model: str) -> None:
    if not _lock_consolidare.acquire(blocking=False):
        return  # deja rulează una
    try:
        from src.core.consolidare import consolidare
        consolidare.ruleaza(client=client, model=model)
    except Exception as e:
        print(f"[Memorie] Consolidare eșuată: {e}")
    finally:
        _lock_consolidare.release()


def salveaza_tura(sesiune_id: str, text_user: str, raspuns: str, client=None,
                  model: str = "gemini-3.6-flash") -> None:
    """Salvează tura în DB și RAG, extrage amintiri și, din când în când, consolidează memoria."""
    global _ture
    if not sesiune_id or not text_user or not raspuns:
        return

    try:
        from src.core.database import db
        db.salveaza_mesaj(sesiune_id, "user", text_user)
        db.salveaza_mesaj(sesiune_id, "assistant", raspuns)
    except Exception as e:
        print(f"[Memorie] Salvare în DB eșuată: {e}")
        return

    try:
        from src.core.rag import rag
        rag.indexeaza_mesaj("user", text_user, sesiune_id)
        rag.indexeaza_mesaj("assistant", raspuns, sesiune_id)
    except Exception as e:
        print(f"[Memorie] Indexare RAG eșuată: {e}")

    if client is None:
        return

    try:
        from src.core.memory import memorie
        memorie.extrage_si_salveaza(text_user, raspuns, sesiune_id, client, model)
    except Exception as e:
        print(f"[Memorie] Extragere omisă: {e}")

    _ture += 1
    if _ture % CONSOLIDARE_LA_N_TURE == 0:
        threading.Thread(target=_consolidare_in_fundal, args=(client, model), daemon=True).start()
