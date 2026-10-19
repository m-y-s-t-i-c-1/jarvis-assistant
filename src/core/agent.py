"""Orchestratorul principal (Task 1.6 / Task 2.4 — bucla agentului)."""

import re
import time
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
    "ora exactă", "ora exacta", "este ora", "e ora", "sunt orele",
    "data de azi", "data de astăzi", "data de astazi", "astăzi este", "astazi este",
    "azi este", "suntem în", "suntem in",
    "am pornit", "am lansat", "am deschis", "am rulat",
)
_PATTERN_HALUCINATIE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in SEMNALE_POSIBILA_HALUCINATIE) + r")\b",
    re.IGNORECASE,
)


# FIX: \"nu am rulat comanda\" / \"ai refuzat\" conțin \"am rulat\" și forțau un
# apel de unealtă (mode=ANY) exact după ce utilizatorul REFUZASE confirmarea.
_PATTERN_NEGATIE = re.compile(r"\b(nu|n-am|n-a|anulat|refuzat|blocat|imposibil)\b", re.IGNORECASE)


def _raspunsul_pare_suspect(text: str) -> bool:
    if _PATTERN_NEGATIE.search(text):
        return False
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

        # Rulăm fiecare funcție cerută. FIX: toate function_response-urile
        # dintr-o tură merg într-UN singur Content (Gemini dă 400 dacă
        # numărul lor nu se potrivește cu cel al function_call-urilor).
        parti_raspuns = []
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

            parti_raspuns.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=apel.name,
                        response=rezultat
                    )
                )
            )

        istoric.append(types.Content(role="user", parts=parti_raspuns))

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
# FIX: înainte se accepta și `$` (sfârșit de buffer) ca delimitator. În streaming,
# un chunk poate să se termine exact după un punct dintr-un număr ("3." + "14"),
# iar Jarvis rostea "3." ca propoziție separată. Cerem spațiu după punctuație;
# ultima propoziție rămâne oricum rostită din `buffer_text` la finalul stream-ului.
_PATTERN_SFARSIT_PROPOZITIE = re.compile(r'[.!?]+[”"\')\]]?\s+')


def _uneste_parti_stream(parti: list):
    """
    FIX: în streaming, fiecare chunk conține doar o BUCATĂ din răspuns.
    Înainte se păstra doar conținutul ultimului chunk (`continut_final =
    continut`), deci în istoric ajungea doar ultima bucată de text, iar un
    function_call venit într-un chunk anterior se pierdea — apoi
    function_response-ul fără apel în față făcea Gemini să dea 400.
    Aici adunăm părțile din toate chunk-urile într-un singur Content,
    lipind textul simplu adiacent.
    """
    if not parti:
        return None
    unite = []
    for parte in parti:
        text_simplu = (
            getattr(parte, "text", None)
            and not getattr(parte, "function_call", None)
            and not getattr(parte, "thought", None)
        )
        if (
            text_simplu
            and unite
            and getattr(unite[-1], "text", None)
            and not getattr(unite[-1], "function_call", None)
            and not getattr(unite[-1], "thought", None)
        ):
            unite[-1] = types.Part(text=unite[-1].text + parte.text)
        else:
            unite.append(parte)
    return types.Content(role="model", parts=unite)


def _fara_function_calls(continut):
    """Returnează un Content fără part-urile function_call (sau None dacă nu rămâne nimic)."""
    if continut is None:
        return None
    parti = [p for p in (continut.parts or []) if not getattr(p, "function_call", None)]
    return types.Content(role="model", parts=parti) if parti else None


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
        parti_acumulate = []
        intrerupt = False

        _start_gemini = time.perf_counter()
        _primul_chunk_masurat = False

        for chunk in chat.send_message_stream(mesaj_trimis):
            if not _primul_chunk_masurat:
                print(f"[TIMP] Gemini — până la primul chunk: {time.perf_counter() - _start_gemini:.2f}s")
                _primul_chunk_masurat = True

            if flag_intrerupere is not None and flag_intrerupere.is_set():
                intrerupt = True
                break

            if not chunk.candidates:
                continue

            continut = chunk.candidates[0].content
            if continut is None or not continut.parts:
                continue

            parti_acumulate.extend(continut.parts)

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

        print(f"[TIMP] Gemini — generare completă (acest pas): {time.perf_counter() - _start_gemini:.2f}s")
        continut_final = _uneste_parti_stream(parti_acumulate)

        if intrerupt:
            # Am fost întrerupți — nu mai rostim restul din buffer, nu mai
            # cerem alt pas din buclă. Salvăm în istoric ce am generat până
            # acum (dacă avem conținut valid), ca și context pentru tura
            # următoare, și returnăm ce s-a apucat să genereze.
            # FIX: scoatem function_call-urile fără răspuns — un call
            # rămas în istoric fără function_response strică toate turele
            # următoare (400).
            continut_curat = _fara_function_calls(continut_final)
            if continut_curat is not None:
                istoric.append(continut_curat)
            print("[Agent] Generare întreruptă de barge-in.")
            return text_complet or "[întrerupt]"

        # Ce a rămas în buffer (ultima propoziție, fără spațiu final tăiat)
        if buffer_text.strip() and la_propozitie_gata:
            la_propozitie_gata(buffer_text.strip())

        if apeluri_functii:
            if continut_final is not None:
                istoric.append(continut_final)

            parti_raspuns = []  # FIX: un singur Content pentru toate răspunsurile
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

                parti_raspuns.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=apel.name,
                            response=rezultat
                        )
                    )
                )
            istoric.append(types.Content(role="user", parts=parti_raspuns))
            continue  # continuăm bucla — cerem răspunsul următor, pe baza rezultatelor uneltelor

        # Fără apeluri de funcții -> răspunsul e complet
        if not text_complet.strip():
            # FIX: audio_loop ignoră valoarea returnată, deci mesajul
            # trebuie rostit prin callback, altfel Jarvis rămâne mut.
            mesaj = "Vasea, nu am primit niciun răspuns valid. Poți reformula?"
            if la_propozitie_gata:
                la_propozitie_gata(mesaj)
            return mesaj

        if continut_final is not None:
            istoric.append(continut_final)
        return text_complet

    # FIX: înainte, acest `return` venea PRIMUL, iar blocul de dedesubt (care
    # rostea mesajul prin callback) era cod mort — în modul vocal, după
    # MAX_PASI, Jarvis rămânea mut. Acum mesajul se rostește și apoi se returnează.
    mesaj = "Domnule, am întâmpinat o buclă neobișnuit de lungă de procesare. Vă recomand să reformulați cererea."
    if la_propozitie_gata:
        la_propozitie_gata(mesaj)
    return mesaj
