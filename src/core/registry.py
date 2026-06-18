"""
Registrul de Unelte (Task 2.1)

Aici trăiește puntea dintre Gemini și codul Python real.

Cum funcționează:
1. Fiecare funcție din tools/ se înregistrează cu decoratorul @unealta.
2. Decoratorul construiește automat FunctionDeclaration-ul pentru Gemini
   din metadatele pe care le dai explicit (description + parameters),
   fără să mai scrii tu manual types.FunctionDeclaration() de fiecare dată.
3. REGISTRU_FUNCTII ține legătura nume -> funcție Python reală.
4. get_unelte_pentru_gemini() construiește lista de Tool-uri de trimis la model.

Adăugarea unei unelte noi = scrii funcția în tools/, o decorezi cu @unealta,
și apare automat în sistem. Nu trebuie să modifici nimic în main.py sau agent.py.
"""

from google.genai import types

# Nume funcție -> funcția Python reală
REGISTRU_FUNCTII = {}

# Nume funcție -> types.FunctionDeclaration (pentru a construi Tool-urile)
DECLARATII_FUNCTII = {}


def unealta(description: str, parameters: dict | None = None):
    """
    Decorator pentru a transforma o funcție Python într-o unealtă
    pe care Jarvis o poate apela prin function calling.

    Parametri:
        description: explicația pe care o citește Gemini, ca să decidă
                      CÂND să folosească unealta. Scrie-o clar și specific.
        parameters:  schema parametrilor, în format dict simplu, ex:

                      {
                          "cale": {
                              "type": "STRING",
                              "description": "Calea fișierului de citit"
                          },
                          "linii": {
                              "type": "INTEGER",
                              "description": "Câte linii să citească",
                              "optional": True
                          }
                      }

                      Dacă funcția nu are parametri, lasă None sau {}.

    Exemplu de utilizare (într-un fișier din tools/):

        @unealta(
            description="Returnează data și ora curentă a sistemului.",
        )
        def get_ora_curenta():
            return datetime.now().strftime("%H:%M, %d.%m.%Y")
    """
    def decorator(func):
        nume = func.__name__

        if nume in REGISTRU_FUNCTII:
            raise ValueError(
                f"Unealta '{nume}' este deja înregistrată. "
                f"Nume duplicat în tools/ — verifică fișierele."
            )

        # Construim schema parametrilor pentru Gemini
        params = parameters or {}
        properties = {}
        required = []

        for nume_param, info in params.items():
            properties[nume_param] = types.Schema(
                type=info["type"],
                description=info.get("description", ""),
            )
            if not info.get("optional", False):
                required.append(nume_param)

        schema_parametri = types.Schema(
            type="OBJECT",
            properties=properties,
            required=required if required else None,
        )

        declaratie = types.FunctionDeclaration(
            name=nume,
            description=description,
            parameters=schema_parametri,
        )

        REGISTRU_FUNCTII[nume] = func
        DECLARATII_FUNCTII[nume] = declaratie

        # Returnăm funcția originală neschimbată, ca să poată fi
        # apelată normal și din alte părți de cod (teste, etc.)
        return func

    return decorator


def get_unelte_pentru_gemini() -> list[types.Tool]:
    """
    Construiește lista de Tool-uri de trimis la Gemini, din toate
    funcțiile înregistrate până în acest moment.

    Trebuie apelată DUPĂ ce toate modulele din tools/ au fost importate
    (vezi tools/__init__.py), altfel DECLARATII_FUNCTII e încă goală.
    """
    if not DECLARATII_FUNCTII:
        return []

    return [types.Tool(function_declarations=list(DECLARATII_FUNCTII.values()))]


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