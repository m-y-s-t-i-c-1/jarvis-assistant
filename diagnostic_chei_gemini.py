"""
Script de diagnostic — testează fiecare cheie GEMINI_API_KEY din .env,
individual, și afișează eroarea COMPLETĂ (netrunchiată) dacă vreuna eșuează.

Rulare:
    python diagnostic_chei_gemini.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

chei = [
    os.getenv(f"GEMINI_API_KEY{'' if i == 0 else f'_{i+1}'}")
    for i in range(5)
]
chei = [k for k in chei if k]

print(f"Număr chei găsite în .env: {len(chei)}\n")

if not chei:
    print("Nicio cheie găsită! Verifică .env — trebuie GEMINI_API_KEY, GEMINI_API_KEY_2 etc.")

for i, cheie in enumerate(chei):
    print(f"--- Cheia #{i+1} ({cheie[:12]}...) ---")
    try:
        client = genai.Client(api_key=cheie)
        raspuns = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Răspunde doar cu cuvântul: OK",
        )
        print(f"  ✅ FUNCȚIONEAZĂ — răspuns: {raspuns.text.strip()!r}")
    except Exception as e:
        print(f"  ❌ EROARE COMPLETĂ:\n{str(e)}\n")
    print()

print("Diagnostic terminat.")