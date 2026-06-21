"""
Profilul Utilizatorului și Preferințele (Task 4.2)

Strat de abstractizare peste tabelul `profil` din JarvisDB.
Definește cheile standard ale profilului, valorile implicite și
o funcție care construiește un bloc de context injectabil în system prompt.

Cheile de profil sunt constante tipizate — nu string-uri magice împrăștiate
prin cod. Dacă adaugi o cheie nouă, o adaugi aici și e disponibilă peste tot.

Utilizare:
    from src.core.profile import profil
    profil.seteaza("editor", "code")
    print(profil.get("editor"))
    print(profil.bloc_context())   # injectat în system prompt
"""

from src.core.database import db


# ── Cheile standard ale profilului ───────────────────────────────────────────
# Modifică valorile implicite ca să corespundă cu sistemul tău.

CHEI_PROFIL = {
    # Date personale
    "nume":                 "Vasea",
    "limba":                "română",

    # Sistem
    "os":                   "Arch Linux",
    "terminal":             "alacritty",
    "editor":               "code",
    "browser":              "vivaldi",
    "file_manager":         "nautilus",
    "audio_server":         "PipeWire",

    # Proiect curent
    "director_proiect":     "/home/vaseoc/Downloads/jarvis-assistant-main",
    "repo_git":             "jarvis-assistant",
    "limbaj_principal":     "Python",
    "venv_path":            "/home/vaseoc/Downloads/jarvis-assistant-main/venv",

    # Preferințe Jarvis
    "stil_raspuns":         "concis",        # concis / detaliat
    "mod_implicit":         "text",          # text / voce
    "confirmare_comenzi":   "da",            # da / nu
}


class ProfilUtilizator:
    """
    Interfața pentru profilul utilizatorului.
    La inițializare, populează cheile lipsă cu valorile implicite din CHEI_PROFIL.
    """

    def __init__(self):
        self._initializeaza_valori_implicite()

    def _initializeaza_valori_implicite(self):
        """
        Pune valorile implicite din CHEI_PROFIL pentru orice cheie
        care nu există încă în DB. Nu suprascrie valorile existente.
        """
        for cheie, valoare_implicita in CHEI_PROFIL.items():
            existent = db.get_profil(cheie)
            if not existent:
                db.seteaza_profil(cheie, valoare_implicita)

    def get(self, cheie: str, default: str = "") -> str:
        """Returnează valoarea unei chei din profil."""
        return db.get_profil(cheie, default)

    def seteaza(self, cheie: str, valoare: str):
        """Setează sau actualizează o valoare în profil."""
        db.seteaza_profil(cheie, valoare)
        print(f"[Profil] {cheie} = {valoare}")

    def tot(self) -> dict:
        """Returnează întreg profilul."""
        return db.get_tot_profilul()

    def bloc_context(self) -> str:
        """
        Construiește un bloc de text cu informațiile din profil,
        gata de injectat în system prompt-ul lui Jarvis.

        Exemplu output:
            === Profil utilizator ===
            Nume: Vasea | OS: Arch Linux | Editor: VS Code
            Director proiect: /home/vaseoc/...
            Preferințe: răspunsuri concise, mod text
        """
        p = self.tot()

        bloc = "=== Profil utilizator ===\n"
        bloc += f"Nume: {p.get('nume', 'Vasea')} | "
        bloc += f"OS: {p.get('os', 'Linux')} | "
        bloc += f"Terminal: {p.get('terminal', 'N/A')} | "
        bloc += f"Editor: {p.get('editor', 'N/A')} | "
        bloc += f"Browser: {p.get('browser', 'N/A')}\n"
        bloc += f"Director proiect: {p.get('director_proiect', 'N/A')}\n"
        bloc += f"Limbaj principal: {p.get('limbaj_principal', 'Python')} | "
        bloc += f"Venv: {p.get('venv_path', 'N/A')}\n"
        bloc += f"Stil răspuns: {p.get('stil_raspuns', 'concis')} | "
        bloc += f"Mod implicit: {p.get('mod_implicit', 'text')} | "
        bloc += f"Confirmare comenzi: {p.get('confirmare_comenzi', 'da')}\n"

        # Adaugă orice cheie extra care nu face parte din setul standard
        chei_standard = set(CHEI_PROFIL.keys())
        chei_extra = {k: v for k, v in p.items() if k not in chei_standard}
        if chei_extra:
            bloc += "Extra: " + ", ".join(f"{k}={v}" for k, v in chei_extra.items()) + "\n"

        return bloc.strip()


# ── Instanță globală ──────────────────────────────────────────────────────────

profil = ProfilUtilizator()


if __name__ == "__main__":
    print("\n=== Test ProfilUtilizator ===\n")
    print("Profil complet:")
    for cheie, valoare in profil.tot().items():
        print(f"  {cheie}: {valoare}")

    print("\nBloc context pentru system prompt:")
    print(profil.bloc_context())

    # Test modificare
    profil.seteaza("stil_raspuns", "detaliat")
    print(f"\nStil răspuns actualizat: {profil.get('stil_raspuns')}")
    profil.seteaza("stil_raspuns", "concis")  # resetăm