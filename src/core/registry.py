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

Task 6.6 — filtrare de unelte: get_unelte_pentru_gemini() acceptă acum un
parametru opțional `doar_functiile`, care restricționează lista de unelte
trimise la Gemini la un subset specific. Folosit de src/core/subagenti.py
ca să dea fiecărui sub-agent specializat (Software/DevOps/Cercetare) DOAR
uneltele relevante lui, nu tot registrul complet.
"""

from google.genai import types

# Nume funcție -> funcția Python reală
REGISTRU_FUNCTII = {}

# Nume funcție -> types.FunctionDeclaration (pentru a construi Tool-urile)
DECLARATII_FUNCTII = {}

# Nume funcție -> bool: True dacă unealta necesită confirmare înainte de execuție
CONFIRMARE_FUNCTII = {}

# Nume funcție -> int | None: numărul maxim de linii de returnat (None = fără limită)
MAX_LINII_FUNCTII = {}

# Limita globală de caractere per rezultat trimis la model (~2000 tokens siguri)
LIMITA_GLOBALA_CHARS = 8000


def _trunchează_output(text: str, max_linii: int | None) -> str:
    """
    Trunchiază un output lung păstrând header-ul și primele N linii utile.
    Header-ul (prima linie) e întotdeauna păstrat — conține numele coloanelor.
    """
    if not isinstance(text, str):
        return text

    # Aplicăm mai întâi limita globală de caractere
    if len(text) > LIMITA_GLOBALA_CHARS:
        text = text[:LIMITA_GLOBALA_CHARS]
        truncheat_chars = True
    else:
        truncheat_chars = False

    linii = text.splitlines()

    if max_linii is not None and len(linii) > max_linii + 1:
        # +1 pentru header
        header = linii[0]
        date = linii[1:max_linii + 1]
        total_original = len(linii)
        text = "\n".join([header] + date)
        text += f"\n[... truncheat: afișate {max_linii} din {total_original} linii]"
    elif truncheat_chars:
        text += "\n[... truncheat: output prea lung]"

    return text


def unealta(
    description: str,
    parameters: dict | None = None,
    necesita_confirmare: bool = False,
    max_linii: int | None = None,
):
    """
    Decorator pentru a transforma o funcție Python într-o unealtă
    pe care Jarvis o poate apela prin function calling.

    Parametri:
        description:          explicația pe care o citește Gemini, ca să decidă
                              CÂND să folosească unealta. Scrie-o clar și specific.
        parameters:           schema parametrilor, în format dict simplu, ex:

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
        necesita_confirmare:  dacă True, Jarvis va cere aprobare explicită
                              înainte de a executa unealta. Folosește pentru
                              acțiuni ireversibile sau cu impact mare.
        max_linii:            numărul maxim de linii de returnat la model.
                              Folosește pentru comenzi cu output lung (ps aux,
                              ls -la pe directoare mari etc.). None = fără limită.
                              Header-ul (prima linie) e întotdeauna păstrat.

    Exemplu:
        @unealta(
            description="Listează procesele active.",
            max_linii=30,   # ps aux poate fi 150+ linii, trimitem doar 30
        )
        def procese_active(): ...
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
        CONFIRMARE_FUNCTII[nume] = necesita_confirmare
        MAX_LINII_FUNCTII[nume] = max_linii

        # Returnăm funcția originală neschimbată, ca să poată fi
        # apelată normal și din alte părți de cod (teste, etc.)
        return func

    return decorator


def get_unelte_pentru_gemini(doar_functiile: list[str] | None = None) -> list[types.Tool]:
    """
    Construiește lista de Tool-uri de trimis la Gemini, din toate
    funcțiile înregistrate până în acest moment.

    Trebuie apelată DUPĂ ce toate modulele din tools/ au fost importate
    (vezi tools/__init__.py), altfel DECLARATII_FUNCTII e încă goală.

    Parametri:
        doar_functiile: dacă e furnizată, restricționează lista returnată
                        DOAR la numele din această listă (nume necunoscute
                        sunt ignorate silențios). None (default) = toate
                        uneltele înregistrate, comportament neschimbat.

                        Folosit de sub-agenții specializați (subagenti.py)
                        ca să limiteze ce poate apela fiecare (ex: agentul
                        de Cercetare nu ar trebui să poată rula git push).
    """
    if not DECLARATII_FUNCTII:
        return []

    if doar_functiile is None:
        declaratii = list(DECLARATII_FUNCTII.values())
    else:
        declaratii = [
            DECLARATII_FUNCTII[nume]
            for nume in doar_functiile
            if nume in DECLARATII_FUNCTII
        ]

    if not declaratii:
        return []

    return [types.Tool(function_declarations=declaratii)]


def ruleaza_functie(nume_functie: str, argumente: dict) -> dict:
    """
    Rulează o funcție din registru în siguranță, cu truncare automată
    a output-urilor lungi pentru a nu depăși token limit-ul modelului.
    """
    if nume_functie not in REGISTRU_FUNCTII:
        return {"eroare": f"Funcția '{nume_functie}' nu există în registru."}

    try:
        functie = REGISTRU_FUNCTII[nume_functie]
        rezultat = functie(**argumente) if argumente else functie()

        # Truncăm outputul dacă unealta are max_linii setat
        if isinstance(rezultat, str):
            max_linii = MAX_LINII_FUNCTII.get(nume_functie)
            rezultat = _trunchează_output(rezultat, max_linii)

        return {"rezultat": rezultat}
    except Exception as e:
        return {"eroare": f"Funcția a generat o eroare: {str(e)}"}