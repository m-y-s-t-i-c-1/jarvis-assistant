"""
Router de Cereri — Clasificare Euristică Locală (Task 6.4)

Decide, ÎNAINTE de orice apel API, în ce categorie se încadrează o cerere
a lui Vasea, ca main.py să știe cărui provider să i-o trimită întâi:

    "unelte"      -> cererea pare să aibă nevoie de o unealtă reală
                      (control sistem, ecran, git, calendar, hardware,
                      vreme, căutare) -> trebuie neapărat Gemini, pentru
                      că tool-calling-ul e legat de formatul lui nativ.
    "cod"         -> întrebare tehnică / de programare, fără unealtă
                      -> NVIDIA (cheia dedicată de coding)
    "conversatie" -> orice altceva, chat/cunoștințe generale
                      -> NVIDIA (cheile dedicate de conversație)

IMPORTANT: e o euristică pe cuvinte cheie, gratuită și instantă, NU un
clasificator AI — deci va greși ocazional (fals negativ/pozitiv). Dacă
observi routări greșite frecvente, extinde listele de mai jos sau spune-mi
să trecem la clasificare printr-un apel LLM rapid.
"""

# Cuvinte care indică nevoie de o unealtă reală (control sistem, ecran,
# git, calendar, hardware, vreme, căutare) — dacă apar, mergem direct pe
# Gemini, care e singurul cu tool-calling funcțional în acest proiect.
CUVINTE_UNELTE = (
    # ecran / vedere / control
    "ecran", "ecranul", "screenshot", "click", "apasă", "apasa",
    "mută mouse", "muta mouse", "tastează", "tasteaza", "deruleaza", "derulează",
    # aplicații / sistem
    "deschide", "lansează", "lanseaza", "pornește", "porneste",
    "oprește", "opreste", "închide aplicația", "inchide aplicatia",
    "volum", "luminozitate", "mute",
    # info sistem / hardware
    "cpu", "ram", "memorie ram", "spațiu pe disc", "spatiu pe disc",
    "temperatura", "procese active", "uptime",
    # dezvoltare / git
    "git", "commit", "push", "branch", "stash", "server http", "vs code",
    "vscode",
    # calendar
    "calendar", "eveniment", "programează", "programeaza", "întâlnire",
    "intalnire",
    # externe
    "vremea", "vreme e", "prognoza", "caută pe internet", "cauta pe internet",
    # timp
    "ce oră", "ce ora", "ce dată", "ce data", "cât e ceasul", "cat e ceasul",
    # joburi fundal
    "job", "rulează în fundal", "ruleaza in fundal",
)

# Cuvinte care indică o întrebare tehnică / de cod (fără unealtă)
CUVINTE_COD = (
    "cod", "codul", "funcție", "functie", "funcția", "functia",
    "bug", "eroare", "debug", "exceptie", "excepție", "traceback",
    "python", "javascript", "typescript", "java", "c++", "c#", "sql",
    "algoritm", "clasă", "clasa", "variabilă", "variabila",
    "compilare", "compileaza", "compilează", "sintaxă", "sintaxa",
    "api", "backend", "frontend", "regex", "librărie", "libraria",
    "biblioteca", "framework", "refactor", "optimizează", "optimizeaza",
)


def clasifica(text: str) -> str:
    """
    Clasifică o cerere text în "unelte", "cod" sau "conversatie".

    Prioritate: "unelte" > "cod" > "conversatie" — dacă mesajul pare
    să aibă nevoie de o acțiune reală, asta câștigă indiferent de alte
    cuvinte prezente (ex: "scrie codul care deschide browserul" ar trebui
    tot pe Gemini, pentru că implică o acțiune reală).
    """
    text_lower = text.lower()

    if any(cuvant in text_lower for cuvant in CUVINTE_UNELTE):
        return "unelte"

    if any(cuvant in text_lower for cuvant in CUVINTE_COD):
        return "cod"

    return "conversatie"