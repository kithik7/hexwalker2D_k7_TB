"""
PhaseCoupling: rules that define how the kuramoto oscillators pull on each other
K1-K4, preferred gaps, and neighbours are contained here. It reads 
leg phases and returns phase nudge. Contains no information about the body or simulation
"""

import numpy as np

from hexwalker.interleg.topology import ANTERIOR, CONTRALATERAL, POSTERIOR

FIXED_IPSILATERAL_OFFSET = 2 * np.pi / 3


class PhaseCoupling:
    #Coupling strengths, preferred gaps, and the two coupling steps
    # name: (slider label, lowest, highest, slider step) : class attribute
    strength_specs = {
        "K1": ("K1 behind", 0.0, 1.0, 0.01),
        "K2": ("K2 nudge", 0.0, 0.2, 0.01),
        "K3": ("K3 ahead", 0.0, 1.0, 0.01),
        "K4": ("K4 contralateral", 0.0, 1.0, 0.01),
    }
    def __init__(self, legs: list) -> None:
        self.legs = legs
        self.default_coupling = {"K1": 0.3, "K2": 0.1, "K3": 0.3, "K4": 0.6}
        self.K1 = 0.0
        self.K2 = 0.0
        self.K3 = 0.0
        self.K4 = 0.0
        # True: same-side gap follows the swing share (speed dependent)
        # False: fixed same-side gap of a third of a lap
        self.offset_follows_speed = True
        self.contralateral_offset = np.pi
        self._pending_nudges = (0.0,) * len(legs)

    @property
    def ipsilateral_offset(self) -> float:
        if self.offset_follows_speed:
            return min(self.legs[0].touchdown_phase, np.pi)
        return FIXED_IPSILATERAL_OFFSET

    def compute_coupling_signals(self) -> list:
        # Snapshot all six phases, then each leg's pull from neighbours.
        phases = [leg.phase for leg in self.legs]
        offset = self.ipsilateral_offset
        signals = []
        for i, phase in enumerate(phases):
            signal = 0.0
            behind = POSTERIOR[i]
            if behind is not None:
                signal += self.K1 * np.sin(phases[behind] - phase - offset)
            front = ANTERIOR[i]
            if front is not None:
                signal += self.K3 * np.sin(phases[front] - phase + offset)
            opposite = CONTRALATERAL[i]
            signal += self.K4 * np.sin(
                phases[opposite] - phase - self.contralateral_offset
            )
            signals.append(signal)
        return signals

    def compute_nudges(self, touched_down: list) -> tuple:
        #Nudge each leg gets next tick: K2 per touchdown of the leg behind.
        nudges = [0.0] * len(touched_down)
        for i, landed in enumerate(touched_down):
            front = ANTERIOR[i]
            if landed and front is not None:
                nudges[front] += self.K2
        return tuple(nudges)

    @property
    def default_strengths(self) -> dict:
        return self.default_coupling

    @property
    def strengths(self) -> dict:
        return {"K1": self.K1, "K2": self.K2, "K3": self.K3, "K4": self.K4}

    def step(self, dt: float) -> None:
        # One tick: signals and last tick's nudges, then the legs move.
        signals = self.compute_coupling_signals()
        touched_down = [
            leg.update_phase(dt, signal, nudge)
            for leg, signal, nudge in zip(
                self.legs, signals, self._pending_nudges
            )
        ]
        self._pending_nudges = self.compute_nudges(touched_down)

    def set_strength(self, name: str, value: float) -> None:
        if name not in self.strength_specs:
            raise ValueError(f"Unknown coupling: {name}")
        setattr(self, name, value)