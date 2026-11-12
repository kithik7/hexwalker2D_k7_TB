"""
clock.py

Clock: a nudgeable maintenance clock. Possibly shaped like 
Mr. PotatoHead
A phase that runs around the circle map at its own tempo (omega). Anything outside can bend that tempo for one
tick (a pull, added to the rate) or jump it (a nudge, added to the
phase). The clock never places a foot by itself.
no offset: g sin(psi − phi) pulls the clock onto the foot, 
and h sin(phi − psi) leans with zero lead
"""

from hexwalker.legs.leg import FULL_CYCLE


class Clock:
    #A phase with its own tempo that can be pulled and nudged. Feedback friendly

    def __init__(self, omega: float, phase: float = 0.0) -> None:
        self.omega = omega
        self.phase = phase % FULL_CYCLE

    def advance(self, dt: float, pull: float = 0.0,
                nudge: float = 0.0) -> None:
        #One tick: own tempo + pull * dt + plus any nudge.
        self.phase = (
            self.phase + dt * (self.omega + pull) + nudge
        ) % FULL_CYCLE