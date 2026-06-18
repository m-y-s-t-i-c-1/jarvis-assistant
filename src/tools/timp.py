"""Unelte legate de timp și dată."""

from datetime import datetime
from src.core.registry import unealta


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
    """Returnează ora curentă a sistemului, ca text."""
    acum = datetime.now()
    return acum.strftime("%H:%M, %d.%m.%Y")