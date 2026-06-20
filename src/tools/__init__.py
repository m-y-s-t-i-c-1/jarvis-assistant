"""
Importă toate modulele din tools/, ca decoratorul @unealta din fiecare
fișier să se execute și funcțiile să se înregistreze în registry.py.

IMPORTANT: când adaugi un fișier nou în tools/ (ex: tools/sistem.py),
adaugă-l și aici, altfel uneltele din el nu vor exista pentru Jarvis,
indiferent câte @unealta ai scris în el.
"""

from src.tools import timp             # noqa: F401
from src.tools import sistem           # noqa: F401
from src.tools import developer        # noqa: F401
from src.tools import hardware         # noqa: F401
from src.tools import external         # noqa: F401
from src.tools import calendar_google  # noqa: F401
from src.tools import joburi           # noqa: F401