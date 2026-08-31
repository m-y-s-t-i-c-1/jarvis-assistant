"""
Stare partajată, minimă, între audio_loop.py și main.py — un singur
threading.Event care spune dacă Jarvis e în mijlocul unui răspuns vocal
(de la prima până la ultima propoziție a turei curente).

De ce există: spune() are propriul _redare_lock (tts.py), dar acela
protejează doar O SINGURĂ propoziție. Între propoziția N și N+1 ale
aceluiași răspuns (cât timp Gemini/NVIDIA generează încă text), lock-ul
e liber o clipă — suficient ca o alertă de ecran (main.py) să se
strecoare exact acolo și să sune incoerent, suprapusă peste conversație.

conversatie_activa.set()   — la începutul turei (înainte de prima propoziție)
conversatie_activa.clear() — la finalul turei (după ultima propoziție,
                              indiferent dacă a reușit sau a picat cu eroare)

Alertele (main.py) așteaptă ca acest flag să fie clear() înainte să
vorbească, ca să nu se suprapună niciodată peste o tură de conversație.
"""

import threading

conversatie_activa = threading.Event()