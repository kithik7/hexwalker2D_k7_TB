"""
modes.py

One place that says what each walking mode is comprised of i.e. which kind of
leg, and which mode of interleg coordination.  
The walker, the GUI, the recorder and the run scripts only name a mode.
"""

from dataclasses import dataclass

from hexwalker.interleg.cruse_rules import ClockedCruseRules, CruseRules
from hexwalker.interleg.phase_coupling import PhaseCoupling
from hexwalker.legs.clocked_cruse_leg import ClockedCruseLeg
from hexwalker.legs.cruse_leg import CruseLeg
from hexwalker.legs.phase_leg import PhaseLeg


@dataclass(frozen=True)
class Mode:
    #A recipe containing the leg type and the coordination that drives it.
    name: str
    leg_class: type
    coordination_class: type


MODES = {
    "phase": Mode("phase", PhaseLeg, PhaseCoupling),
    "cruse": Mode("cruse", CruseLeg, CruseRules),
    "cruse_clock": Mode("cruse_clock", ClockedCruseLeg, ClockedCruseRules),
}

PLANNED_MODES = ()


def get_mode(name: str) -> Mode:
    if name in MODES:
        return MODES[name]
    if name in PLANNED_MODES:
        raise NotImplementedError(f"Mode '{name}' is planned, not built yet")
    raise ValueError(f"Unknown mode '{name}'. Available: {sorted(MODES)}")