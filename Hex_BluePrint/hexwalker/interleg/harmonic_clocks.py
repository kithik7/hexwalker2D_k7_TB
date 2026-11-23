"""
harmonic_clocks.py
Clocked Cruse Legs but the clocks also have intra-coupling defined as K
K is a tunable strength, no preferred hardcoded phase offsets: gap based 
The gaps that two clocks settle at is a dynamical steady state 
Shape of the pull is a sine + 2 harmonics i.e. a2 = a3 = 0 is a plain sine
Any repeating/periodic shape can be build by integrating simple sine and cosine waves - Fourier
"""

import numpy as np 
from hexwalker.interleg.cruse_rules import ClockedCruseRules
from hexwalker.interleg.topology import ANTERIOR, CONTRALATERAL, POSTERIOR


def harmonic_push(gap: float, a2: float, a3: float) -> float:
    # the shape H: turns a gap between two clocks into a push
    return np.sin(gap) + a2 * np.sin(2 * gap) + a3 * np.sin(3 * gap)

def clock_pulls(clock_phases: tuple, K: float, a2: float, a3: float) -> tuple:
    #haskell-like frozen snapshot of all six clocks in, one pull per clock is put out 
    pulls = []
    for i, phi_i in enumerate(clock_phases):
        neighbours = [POSTERIOR[i], ANTERIOR[i], CONTRALATERAL[i]]
        total = sum(
            harmonic_push(clock_phases[j] - phi_i, a2, a3)
            for j in neighbours if j is not None 
        )
        pulls.append(K * total) 
    return tuple(pulls)

class HarmonicClockRules(ClockedCruseRules):
    # Cruse rules, 
    # g and h (clock pull towards the foot and H in BL is how much the clock shifts the foot's liftoff point, earlier if ahead)
    #plus clock-to-clock coupling K with harmonics 

    strength_specs = {
        **ClockedCruseRules.strength_specs,
        "K": ("K clocks", -2.0, 2.0, 0.05),
        "a2": ("a2 harmonic", 0.0, 1.0, 0.05),
        "a3": ("a3 harmonic", 0.0, 1.0, 0.05),
    }

    def __init__(self, legs: list) -> None: 
        super().__init__(legs)
        self.default_strengths = {
            **self.default_strengths, "K": -0.5, "a2": 0.0, "a3": 0.0,
        }
        self.strengths = {name: 0.0 for name in self.default_strengths}

    def step(self) -> None: 
        #rule shifts and clocks pulls both from the state at the start of a tick
        s = self.strengths
        phases = tuple(leg.clock_phase for leg in self.legs)
        pulls = clock_pulls(phases, s["K"], s["a2"], s["a3"])
        for leg, shift, pull in zip(self.legs, self.shifts(), pulls):
            leg.update(shift, s["g"], s["h"], pull)
