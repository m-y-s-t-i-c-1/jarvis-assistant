"""
Unelte de Delegare către Sub-Agenți Specializați (Task 6.6)

Expune cei 3 sub-agenți (Software, DevOps, Cercetare) din subagenti.py
către Jarvis principal, ca unelte normale de function-calling.

Jarvis alege SINGUR când merită să delege — de obicei pentru sarcini mai
complexe sau specifice unui domeniu, care ar beneficia de un system prompt
și un set de unelte mai restrâns/specializat decât conversația generală.

IMPORTANT: aceste 3 unelte NU apar în listele "unelte permise" ale
sub-agenților (vezi AGENTI_SPECIALIZATI din subagenti.py) — deci un
sub-agent nu poate delega mai departe la alt sub-agent. Previne recursia.
"""

from src.core.registry import unealta
from src.core.subagenti import ruleaza_subagent


@unealta(
    description=(
        "Deleagă o sarcină de DEZVOLTARE SOFTWARE (scriere/verificare cod, "
        "Git, VS Code, server local de test) unui sub-agent specializat, "
        "care are propriul buget limitat de pași și doar uneltele relevante "
        "lui. Folosește pentru sarcini tehnice de dezvoltare mai complexe, "
        "care merită context specializat — nu pentru întrebări simple de "
        "genul 'ce e un for loop', la alea răspunzi direct."
    ),
    parameters={
        "sarcina": {
            "type": "STRING",
            "description": "Descrierea completă a sarcinii de dezvoltare, cu tot contextul necesar.",
        }
    },
)
def deleaga_agent_software(sarcina: str):
    """Deleagă o sarcină la sub-agentul specializat pe dezvoltare software."""
    return ruleaza_subagent("software", sarcina)


@unealta(
    description=(
        "Deleagă o sarcină de DEVOPS / SISTEM (monitorizare hardware, "
        "control periferice, joburi în fundal, comenzi de sistem read-only) "
        "unui sub-agent specializat, cu propriul buget limitat de pași. "
        "Folosește pentru sarcini de sistem mai complexe cu mai mulți pași "
        "— nu pentru o simplă întrebare 'cât e CPU-ul', la aia răspunzi "
        "direct cu unealta obișnuită."
    ),
    parameters={
        "sarcina": {
            "type": "STRING",
            "description": "Descrierea completă a sarcinii DevOps/sistem, cu tot contextul necesar.",
        }
    },
)
def deleaga_agent_devops(sarcina: str):
    """Deleagă o sarcină la sub-agentul specializat pe DevOps/sistem."""
    return ruleaza_subagent("devops", sarcina)


@unealta(
    description=(
        "Deleagă o sarcină de CERCETARE (căutare informații externe, vreme, "
        "documentare pe un subiect) unui sub-agent specializat, cu propriul "
        "buget limitat de pași. Folosește pentru cercetări cu mai mulți pași "
        "sau surse — nu pentru o singură căutare simplă, la aia folosești "
        "direct unealta de căutare."
    ),
    parameters={
        "sarcina": {
            "type": "STRING",
            "description": "Descrierea completă a sarcinii de cercetare, cu tot contextul necesar.",
        }
    },
)
def deleaga_agent_cercetare(sarcina: str):
    """Deleagă o sarcină la sub-agentul specializat pe cercetare."""
    return ruleaza_subagent("cercetare", sarcina)