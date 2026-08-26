"""Orchestratorul principal (Task 1.6 / Task 2.4 — bucla agentului)."""

import re
from google.genai import types
from src.core.registry import get_unelte_pentru_gemini, ruleaza_functie, CONFIRMARE_FUNCTII
from src.core.security import trece_prin_securitate

MAX_PASI = 5  # plasă de siguranță, ca să nu rămânem blocați într-o buclă infinită

# Cuvinte care indică faptul că răspunsul ar fi trebuit să se bazeze pe o
# unealtă reală (oră, dată, sistem), nu pe ce "crede" modelul. Dacă apar în
# răspunsul final FĂRĂ ca niciun tool să fi fost apelat în pasul respectiv,
# tratăm asta ca semnal de halucinație și forțăm o reverificare.
#
# IMPORTANT: matching pe CUVINTE ÎNTREGI (\b), nu substring brut — altfel
# "dată" prinde orice cuvânt care conține literele alea, iar "luna" prinde
# "lunar", "lunatic" etc. Substring brut a cauzat retry-uri false-pozitive
# chiar și pe răspunsuri corecte (ex: descrieri de ecran care menționau
# incidental "deschise").
SEMNALE_POSIBILA_HALUCINATIE = (
    "ora", "ore", "data", "dată", "ziua", "luna",
    "am pornit", "am lansat", "am deschis", "am rulat",
)
_PATTERN_HALUCINATIE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in SEMNALE_POSIBILA_HALUCINATIE) + r")\b",
    re.IGNORECASE,
)


def _raspunsul_pare_suspect(text: str) -> bool:
    return bool(_PATTERN_HALUCINATIE.search(text))


def _extrage_continut(raspuns):
    """
    Extrage în siguranță conținutul dintr-un răspuns Gemini.

    Gemini poate întoarce un răspuns fără conținut valid în câteva cazuri:
    - candidates gol (prompt blocat complet de filtrele de siguranță)
    - content.parts == None (finish_reason SAFETY, RECITATION, MAX_TOKENS,
      sau pur și simplu un răspuns gol după un apel de funcție cu context
      vizual/mare, cum e cazul cu screenshot-urile din vedere.py)

    Returnează:
        (continut, apeluri_functii, motiv_esec)
        - dacă totul e OK: (Content, listă apeluri, None)
        - dacă răspunsul e invalid/gol: (None, [], "text explicativ pentru Vasea")
    """
    if not raspuns.candidates:
        return None, [], "Vasea, Gemini a blocat complet cererea (posibil filtru de siguranță pe conținut)."

    candidat = raspuns.candidates[0]
    continut = candidat.content

    finish_reason = getattr(candidat, "finish_reason", None)

    if continut is None or not continut.parts:
        motiv = f" (finish_reason: {finish_reason})" if finish_reason else ""
        return (
            None,
            [],
            f"Vasea, am primit un răspuns gol de la Gemini{motiv}. "
            f"Încearcă să reformulezi cererea, sau dacă persistă, "
            f"probabil e legat de conținutul trimis (ex: un screenshot "
            f"care a lovit un filtru de siguranță).",
        )

    apeluri_functii = [
        parte.function_call
        for parte in continut.parts
        if parte.function_call
    ]

    return continut, apeluri_functii, None


def agent_loop(
    client,
    model: str,
    system_prompt: str,
    istoric: list,
    unelte_permise: list[str] | None = None,
) -> str:
    """
    Trimite istoricul la model și, cât timp modelul cere apeluri de funcții,
    le execută și continuă bucla. Se termină când modelul dă un răspuns
    text final (fără apel de funcție).

    Parametri:
        unelte_permise: (Task 6.6) dacă e furnizat, restricționează uneltele
                        disponibile lui Gemini la DOAR această listă de nume.
                        None (default) = toate uneltele înregistrate, exact
                        comportamentul de dinainte. Folosit de subagenti.py
                        ca să dea fiecărui sub-agent specializat doar
                        uneltele lui, nu tot registrul.

    Plasă de siguranță împotriva halucinațiilor: dacă unelte sunt disponibile
    dar modelul răspunde direct cu text ce sugerează că ar fi trebuit să
    apeleze o unealtă (oră/dată/acțiune de sistem), repetăm cererea o singură
    dată, forțând explicit modelul (mode=ANY) să aleagă o unealtă.

    Plasă de siguranță împotriva răspunsurilor goale: dacă Gemini întoarce
    un candidat fără conținut valid (parts=None), NU crăpăm — returnăm un
    mesaj explicativ către Vasea, ca ruleaza_cu_fallback din main.py să
    poată decide dacă încearcă altă cheie sau afișează eroarea.
    """
    unelte = get_unelte_pentru_gemini(doar_functiile=unelte_permise)
    a_facut_deja_retry_fortat = False
    a_folosit_tool_in_aceasta_tura = False

    for pas in range(MAX_PASI):
        # Construim o sesiune de chat pe baza istoricului (toate mesajele
        # anterioare, mai puțin ultimul) și trimitem ultimul mesaj prin
        # `send_message`. Astfel respectăm recomandarea SDK-ului pentru Chat.
        history = istoric[:-1] if len(istoric) > 1 else []
        chat = client.chats.create(
            model=model,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=unelte if unelte else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        # Extragem part-urile din obiectul Content pentru send_message
        ultimul_mesaj = istoric[-1]
        if hasattr(ultimul_mesaj, "parts"):
            mesaj_trimis = ultimul_mesaj.parts
        else:
            mesaj_trimis = str(ultimul_mesaj)

        raspuns = chat.send_message(mesaj_trimis)

        continut_raspuns, apeluri_functii, motiv_esec = _extrage_continut(raspuns)

        if motiv_esec is not None:
            # Răspuns gol/blocat — nu avem ce adăuga în istoric, returnăm direct
            return motiv_esec

        if not apeluri_functii:
            text_raspuns = raspuns.text or ""

            # Plasa de siguranță: răspuns suspect + unelte disponibile +
            # nu am încercat deja un retry forțat + NU am folosit deja
            # un tool real în tura asta (dacă am folosit, răspunsul e deja
            # bazat pe date reale, nu are sens să forțăm alt tool arbitrar)
            if (
                unelte
                and not a_facut_deja_retry_fortat
                and not a_folosit_tool_in_aceasta_tura
                and _raspunsul_pare_suspect(text_raspuns)
            ):
                print(
                    "[Avertisment: răspuns posibil halucinat fără tool call. "
                    "Reverific forțând function calling.]"
                )
                a_facut_deja_retry_fortat = True

                # Re-creăm sesiunea de chat forțat, forțând modelul să aleagă
                # o funcție (mode="ANY") dacă este cazul.
                history = istoric[:-1] if len(istoric) > 1 else []
                chat_fortat = client.chats.create(
                    model=model,
                    history=history,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=unelte,
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY"
                            )
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )

                ultimul_mesaj = istoric[-1]
                if hasattr(ultimul_mesaj, "parts"):
                    mesaj_trimis = ultimul_mesaj.parts
                else:
                    mesaj_trimis = str(ultimul_mesaj)

                raspuns_fortat = chat_fortat.send_message(mesaj_trimis)

                continut_fortat, apeluri_fortate, motiv_esec_fortat = _extrage_continut(raspuns_fortat)

                if motiv_esec_fortat is not None:
                    # Retry-ul forțat a eșuat și el — acceptăm răspunsul
                    # original (posibil suspect), mai bine decât un crash.
                    istoric.append(continut_raspuns)
                    return text_raspuns

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
                if "eroare" not in rezultat:
                    a_folosit_tool_in_aceasta_tura = True

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


# ==============================================================
# VARIANTA STREAMING — pentru modul vocal (Task 6.9)
# ==============================================================
# Diferă de agent_loop de mai sus: folosește chat.send_message_stream()
# în loc de send_message(), și taie textul pe propoziții complete pe
# măsură ce vine, trimițându-le imediat la TTS — nu se așteaptă tot
# răspunsul înainte să înceapă vorbirea.
#
# Compromisuri asumate față de agent_loop normal:
#   - NU are plasa anti-halucinație (retry forțat pe function calling) —
#     ar necesita să aștepte tot răspunsul oricum, anulând beneficiul
#     de streaming. Pentru conversație vocală fluidă, prioritizăm
#     latența joasă față de acea plasă suplimentară.
#   - Generarea și vorbirea NU rulează în paralel (thread-uri separate)
#     în această versiune — Jarvis generează o propoziție, o rostește,
#     apoi continuă generarea următoarei. Tot e mult mai rapid perceput
#     decât varianta blocantă completă, dar nu e suprapunere reală.
#     (Poate fi adăugată ulterior cu un thread producător/consumator.)

# Potrivește sfârșitul unei propoziții: unul sau mai multe din .!? —
# opțional urmate de ghilimele/paranteză de închidere — apoi spațiu
# (sau sfârșitul textului acumulat până acum).
_PATTERN_SFARSIT_PROPOZITIE = re.compile(r'[.!?]+[”"\')\]]?(?:\s+|$)')


def agent_loop_streaming(
    client,
    model: str,
    system_prompt: str,
    istoric: list,
    unelte_permise: list[str] | None = None,
    la_propozitie_gata=None,
    flag_intrerupere=None,
) -> str:
    """
    Variantă de agent_loop care STREAMEAZĂ răspunsul, tăind textul pe
    propoziții complete și trimițându-le imediat către `la_propozitie_gata`
    (de obicei `spune()` din tts.py), pe măsură ce Gemini le generează —
    nu se așteaptă tot răspunsul.

    Parametri:
        unelte_permise:     identic cu agent_loop — restricționează uneltele
                            disponibile (folosit și de sub-agenți, dacă
                            vreodată au nevoie de variantă streaming).
        la_propozitie_gata: callback(text: str), apelat pentru fiecare
                            propoziție completă detectată în stream. Dacă
                            e None, funcționează ca un agent_loop obișnuit,
                            doar streamează intern fără să facă nimic cu
                            propozițiile (echivalent funcțional, dar fără
                            beneficiul de latență).
        flag_intrerupere:   threading.Event opțional (Task 6.10 — barge-in).
                            Verificat înainte de fiecare propoziție nouă —
                            dacă e setat, oprim consumul stream-ului și
                            generarea de propoziții noi imediat, nu doar
                            redarea audio curentă (care se oprește separat,
                            prin tts.opreste(), apelat de barge_in.py).

    Returnează textul complet al răspunsului final (pentru istoric/log),
    identic ca rezultat cu agent_loop — doar livrarea către utilizator
    diferă (bucată cu bucată, nu tot odată).
    """
    unelte = get_unelte_pentru_gemini(doar_functiile=unelte_permise)

    for pas in range(MAX_PASI):
        history = istoric[:-1] if len(istoric) > 1 else []
        chat = client.chats.create(
            model=model,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=unelte if unelte else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        ultimul_mesaj = istoric[-1]
        if hasattr(ultimul_mesaj, "parts"):
            mesaj_trimis = ultimul_mesaj.parts
        else:
            mesaj_trimis = str(ultimul_mesaj)

        buffer_text = ""
        text_complet = ""
        apeluri_functii = []
        continut_final = None
        intrerupt = False

        for chunk in chat.send_message_stream(mesaj_trimis):
            if flag_intrerupere is not None and flag_intrerupere.is_set():
                intrerupt = True
                break

            if not chunk.candidates:
                continue

            continut = chunk.candidates[0].content
            if continut is None or not continut.parts:
                continue

            continut_final = continut

            for parte in continut.parts:
                if parte.function_call:
                    apeluri_functii.append(parte.function_call)
                elif parte.text:
                    buffer_text += parte.text
                    text_complet += parte.text

                    # Tăiem propoziții complete din buffer pe măsură ce apar
                    while True:
                        if flag_intrerupere is not None and flag_intrerupere.is_set():
                            intrerupt = True
                            break

                        potrivire = _PATTERN_SFARSIT_PROPOZITIE.search(buffer_text)
                        if not potrivire:
                            break
                        capat = potrivire.end()
                        propozitie = buffer_text[:capat].strip()
                        buffer_text = buffer_text[capat:]
                        if propozitie and la_propozitie_gata:
                            la_propozitie_gata(propozitie)

                if intrerupt:
                    break
            if intrerupt:
                break

        if intrerupt:
            # Am fost întrerupți — nu mai rostim restul din buffer, nu mai
            # cerem alt pas din buclă. Salvăm în istoric ce am generat până
            # acum (dacă avem conținut valid), ca și context pentru tura
            # următoare, și returnăm ce s-a apucat să genereze.
            if continut_final is not None:
                istoric.append(continut_final)
            print("[Agent] Generare întreruptă de barge-in.")
            return text_complet or "[întrerupt]"

        # Ce a rămas în buffer (ultima propoziție, fără spațiu final tăiat)
        if buffer_text.strip() and la_propozitie_gata:
            la_propozitie_gata(buffer_text.strip())

        if apeluri_functii:
            if continut_final is not None:
                istoric.append(continut_final)

            for apel in apeluri_functii:
                print(f"[Jarvis cere să ruleze funcția: {apel.name}]")
                argumente = dict(apel.args) if apel.args else {}

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
            continue  # continuăm bucla — cerem răspunsul următor, pe baza rezultatelor uneltelor

        # Fără apeluri de funcții -> răspunsul e complet
        if not text_complet.strip():
            return "Vasea, nu am primit niciun răspuns valid. Poți reformula?"

        if continut_final is not None:
            istoric.append(continut_final)
        return text_complet

    return "Domnule, am întâmpinat o buclă neobișnuit de lungă de procesare. Vă recomand să reformulați cererea."