"""Orchestratorul principal (Task 1.6 / Task 2.4 — bucla agentului)."""

from google.genai import types
from src.core.registry import get_unelte_pentru_gemini, ruleaza_functie

MAX_PASI = 5  # plasă de siguranță, ca să nu rămânem blocați într-o buclă infinită


def agent_loop(client, model: str, system_prompt: str, istoric: list) -> str:
    """
    Trimite istoricul la model și, cât timp modelul cere apeluri de funcții,
    le execută și continuă bucla. Se termină când modelul dă un răspuns
    text final (fără apel de funcție).
    """
    unelte = get_unelte_pentru_gemini()

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