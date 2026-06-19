"""Orchestratorul principal (Task 1.6 / Task 2.4 — bucla agentului)."""

from google.genai import types
from src.core.registry import get_unelte_pentru_gemini, ruleaza_functie, CONFIRMARE_FUNCTII
from src.core.security import trece_prin_securitate

MAX_PASI = 5  # plasă de siguranță, ca să nu rămânem blocați într-o buclă infinită

# Cuvinte care indică faptul că răspunsul ar fi trebuit să se bazeze pe o
# unealtă reală (oră, dată, sistem), nu pe ce "crede" modelul. Dacă apar în
# răspunsul final FĂRĂ ca niciun tool să fi fost apelat în pasul respectiv,
# tratăm asta ca semnal de halucinație și forțăm o reverificare.
SEMNALE_POSIBILA_HALUCINATIE = (
    "ora", "ore", "data", "dată", "ziua", "luna",
    "am pornit", "am lansat", "am deschis", "am rulat",
)


def _raspunsul_pare_suspect(text: str) -> bool:
    text_lower = text.lower()
    return any(semnal in text_lower for semnal in SEMNALE_POSIBILA_HALUCINATIE)


def agent_loop(client, model: str, system_prompt: str, istoric: list) -> str:
    """
    Trimite istoricul la model și, cât timp modelul cere apeluri de funcții,
    le execută și continuă bucla. Se termină când modelul dă un răspuns
    text final (fără apel de funcție).

    Plasă de siguranță împotriva halucinațiilor: dacă unelte sunt disponibile
    dar modelul răspunde direct cu text ce sugerează că ar fi trebuit să
    apeleze o unealtă (oră/dată/acțiune de sistem), repetăm cererea o singură
    dată, forțând explicit modelul (mode=ANY) să aleagă o unealtă.
    """
    unelte = get_unelte_pentru_gemini()
    a_facut_deja_retry_fortat = False

    for pas in range(MAX_PASI):
        raspuns = client.models.generate_content(
            model=model,
            contents=istoric,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=unelte if unelte else None,
            )
        )

        continut_raspuns = raspuns.candidates[0].content
        apeluri_functii = [
            parte.function_call
            for parte in continut_raspuns.parts
            if parte.function_call
        ]

        if not apeluri_functii:
            text_raspuns = raspuns.text or ""

            # Plasa de siguranță: răspuns suspect + unelte disponibile +
            # nu am încercat deja un retry forțat -> reverificăm o dată
            if (
                unelte
                and not a_facut_deja_retry_fortat
                and _raspunsul_pare_suspect(text_raspuns)
            ):
                print(
                    "[Avertisment: răspuns posibil halucinat fără tool call. "
                    "Reverific forțând function calling.]"
                )
                a_facut_deja_retry_fortat = True

                raspuns_fortat = client.models.generate_content(
                    model=model,
                    contents=istoric,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=unelte,
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY"
                            )
                        ),
                    )
                )

                continut_fortat = raspuns_fortat.candidates[0].content
                apeluri_fortate = [
                    parte.function_call
                    for parte in continut_fortat.parts
                    if parte.function_call
                ]

                if apeluri_fortate:
                    # Modelul a "recunoscut" că avea nevoie de o unealtă.
                    # Continuăm bucla normal cu acest răspuns în loc de cel vechi.
                    continut_raspuns = continut_fortat
                    apeluri_functii = apeluri_fortate
                else:
                    # Chiar și forțat, nu a ales nicio unealtă -- acceptăm
                    # răspunsul original, nu mai insistăm.
                    istoric.append(continut_raspuns)
                    return text_raspuns

            if not apeluri_functii:
                # Nu mai sunt funcții de rulat - avem răspunsul final
                istoric.append(continut_raspuns)
                return text_raspuns

        # Adăugăm cererea modelului în istoric
        istoric.append(continut_raspuns)

        # Rulăm fiecare funcție cerută și adăugăm rezultatul în istoric
        for apel in apeluri_functii:
            print(f"[Jarvis cere să ruleze funcția: {apel.name}]")
            argumente = dict(apel.args) if apel.args else {}

            # Security check — blacklist + confirmare dacă e marcată
            necesita_confirmare = CONFIRMARE_FUNCTII.get(apel.name, False)
            aprobat, motiv = trece_prin_securitate(apel.name, argumente, necesita_confirmare)

            if not aprobat:
                print(f"[SECURITATE: execuție blocată — {motiv}]")
                rezultat = {"eroare": motiv}
            else:
                rezultat = ruleaza_functie(apel.name, argumente)

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