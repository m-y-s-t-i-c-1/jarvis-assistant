import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Nu am găsit GEMINI_API_KEY în .env. Verifică fișierul .env.")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """Tu ești Jarvis, un asistent AI personal inteligent, eficient și loial.

Reguli de comportament:
- Te adresezi întotdeauna utilizatorului cu "Vasea".
- Tonul tău este calm, profesionist, dar cu un strop de umor sec, britanic.
- Răspunzi concis și la obiect, fără să divaghezi inutil.
- Ești proactiv: dacă observi o problemă sau o soluție mai bună, o menționezi.
- Nu ești servil sau exagerat de politicos - ești un consilier de încredere, nu un servitor.
- Vorbești în limba română, cu un vocabular elevat dar natural.
"""

# ---- Funcțiile reale disponibile pentru Jarvis ----
def get_ora_curenta():
    """Returnează ora curentă a sistemului, ca text."""
    acum = datetime.now()
    return acum.strftime("%H:%M, %d.%m.%Y")


# ---- Descrierile funcțiilor pentru Gemini ----
unealta_ora = types.FunctionDeclaration(
    name="get_ora_curenta",
    description="Returnează data și ora curentă a sistemului. Folosește această funcție când utilizatorul întreabă ce oră este sau ce dată este.",
    parameters=types.Schema(type="OBJECT", properties={}),
)

unelte = types.Tool(function_declarations=[unealta_ora])

REGISTRU_FUNCTII = {
    "get_ora_curenta": get_ora_curenta,
}


def ruleaza_functie(nume_functie: str, argumente: dict) -> dict:
    """
    Rulează o funcție din registru în siguranță.
    Dacă funcția nu există sau crapă, returnăm o eroare clară,
    ca Jarvis să poată reacționa inteligent, nu să se blocheze.
    """
    if nume_functie not in REGISTRU_FUNCTII:
        return {"eroare": f"Funcția '{nume_functie}' nu există în registru."}

    try:
        functie = REGISTRU_FUNCTII[nume_functie]
        rezultat = functie(**argumente) if argumente else functie()
        return {"rezultat": rezultat}
    except Exception as e:
        return {"eroare": f"Funcția a generat o eroare: {str(e)}"}


def agent_loop(istoric: list) -> str:
    """
    Orchestratorul principal. Trimite istoricul la model și, cât timp
    modelul cere apeluri de funcții, le execută și continuă bucla.
    Se termină când modelul dă un răspuns text final (fără apel de funcție).
    """
    MAX_PASI = 5  # plasă de siguranță, ca să nu rămânem blocați într-o buclă infinită

    for pas in range(MAX_PASI):
        raspuns = client.models.generate_content(
            model=MODEL,
            contents=istoric,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[unelte],
            )
        )

        continut_raspuns = raspuns.candidates[0].content
        apeluri_functii = [
            parte.function_call
            for parte in continut_raspuns.parts
            if parte.function_call
        ]

        if not apeluri_functii:
            # Nu mai sunt funcții de rulat - avem răspunsul final
            istoric.append(continut_raspuns)
            return raspuns.text

        # Adăugăm cererea modelului în istoric
        istoric.append(continut_raspuns)

        # Rulăm fiecare funcție cerută și adăugăm rezultatul în istoric
        for apel in apeluri_functii:
            print(f"[Jarvis cere să ruleze funcția: {apel.name}]")
            rezultat = ruleaza_functie(apel.name, dict(apel.args) if apel.args else {})

            istoric.append(
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=apel.name,
                            response=rezultat
                        )
                    )]
                )
            )

    return "Domnule, am întâmpinat o buclă neobișnuit de lungă de procesare. Vă recomand să reformulați cererea."


# ---- Bucla principală de conversație ----
istoric = []

print("Jarvis este activ. Scrie 'exit' pentru a încheia conversația.\n")

while True:
    mesaj_utilizator = input("Tu: ")

    if mesaj_utilizator.lower() in ("exit", "quit", "stop"):
        print("Jarvis: La revedere, Vasea.")
        break

    istoric.append(
        types.Content(role="user", parts=[types.Part(text=mesaj_utilizator)])
    )

    raspuns_text = agent_loop(istoric)
    print("Jarvis:", raspuns_text)