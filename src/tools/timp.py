"""Unelte legate de timp și dată."""

from datetime import datetime
from src.core.registry import unealta

# FIX: get_ora_curenta întorcea doar "HH:MM, ZZ.LL.AAAA". Router-ul și descrierea
# uneltei o prezintă ca singura sursă pentru "ce zi e azi" / "ce lună e", dar
# ziua săptămânii și numele lunii lipseau — modelul trebuia să le calculeze din
# cap (exact ce i se interzice). Le oferim explicit, în română.
_ZILE = ("luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică")
_LUNI = (
    "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie",
    "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie",
)


@unealta(
    description=(
        "OBLIGATORIU: apelează această funcție de fiecare dată când "
        "utilizatorul întreabă ce oră, ce dată, ce zi a săptămânii sau ce "
        "lună este. Nu ai cunoștințe proprii despre data sau ora curentă — "
        "modelul tău de limbaj NU are acces la ceasul de sistem și NU "
        "trebuie să ghicească sau să estimeze data, nici aproximativ. "
        "Singura sursă validă pentru data/ora curentă este rezultatul "
        "acestei funcții."
    ),
)
def get_ora_curenta():
    """Returnează ora, ziua săptămânii și data curentă a sistemului, ca text."""
    acum = datetime.now()
    return (
        f"{acum:%H:%M}, {_ZILE[acum.weekday()]}, {acum.day} "
        f"{_LUNI[acum.month - 1]} {acum.year} ({acum:%d.%m.%Y})"
    )
