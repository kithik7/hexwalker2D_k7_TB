"""
clocked_cruse_leg.py

ClockedCruseLeg: a Cruse leg that also carries a maintenance clock.

The foot works like CruseLeg: it steps by itself and lifts off
at its threshold. Alongside it runs a Clock (phi). The foot's own
progress around the lap is psi (the leg's phase).
Here: 
    g pulls the clock towards the foot:   phi' = omega + g sin(psi - phi)
    h lets the clock lean on the foot:    shift += h sin(phi - psi)

If the clock is ahead of the foot, h makes the foot lift a little
earlier; if behind, a little later. With g = h = 0 the foot behaves
like a plain CruseLeg.
"""

import numpy as np

from hexwalker.legs.clock import Clock
from hexwalker.legs.cruse_leg import CruseLeg
from hexwalker.legs.leg import FULL_CYCLE


class ClockedCruseLeg(CruseLeg):
    #Cruse foot plus a clock that listens (g) and leans (h).

    def _set_default_parameters(self) -> None:
        self.clock = Clock(omega=0.0)
        super()._set_default_parameters()

    def _update_timing(self) -> None:
        super()._update_timing()
        # the clock's own tempo: one lap per step cycle
        self.clock.omega = FULL_CYCLE / (self.cycle_ticks * self.dt)

    def _tarsus_from_phase(self) -> None:
        super()._tarsus_from_phase()
        self.clock.phase = self.phase  # clock starts in agreement

    @property
    def clock_phase(self) -> float:
        return self.clock.phase

    def update(self, liftoff_shift: float = 0.0, g: float = 0.0,
               h: float = 0.0) -> bool:
        #One tick for foot and clock, both from the same start state
        psi = self.phase
        phi = self.clock.phase
        lean = h * np.sin(phi - psi)
        pull = g * np.sin(psi - phi)
        landed = super().update(liftoff_shift + lean)
        self.clock.advance(self.dt, pull)
        return landed