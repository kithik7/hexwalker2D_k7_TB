"""
phase_leg.py

PhaseLeg: a leg driven by its own clock.

The phase runs is a circle map with its own tempo plus any pull from
neighbouring clocks. Crossing touchdown_phase
starts stance, wrapping past 2 pi starts swing, 
Effect on the foot from this mode: foot is placed from the phase every tick.
"""

import numpy as np

from hexwalker.legs.leg import FULL_CYCLE, Leg


class PhaseLeg(Leg):
    """Leg whose clock decides liftoff and touchdown and places the foot."""

    def _set_default_parameters(self) -> None:
        self.phase = np.pi  # start in stance
        super()._set_default_parameters()

    def _update_timing(self) -> None:
        super()._update_timing()
        # steady clock rate: one full cycle per (cycle_ticks x dt)
        self.base_omega = FULL_CYCLE / (self.cycle_ticks * self.dt)
        self.omega_current = self.base_omega

    @property
    def ground_contact(self) -> bool:
        return self.touchdown_phase <= self.phase < FULL_CYCLE

    def set_stance_step(self, new_step: float) -> None:
        """Change stance speed and remap phase to keep foot progress."""
        was_stance = self.ground_contact
        stance_share = FULL_CYCLE - self.touchdown_phase
        if was_stance:
            progress = (self.phase - self.touchdown_phase) / stance_share
        else:
            progress = self.phase / self.touchdown_phase
        self.stance_step_size = new_step
        self._update_timing()
        if was_stance:
            stance_share = FULL_CYCLE - self.touchdown_phase
            self.phase = self.touchdown_phase + progress * stance_share
        else:
            self.phase = progress * self.touchdown_phase

    def update_phase(
        self, dt: float, coupling_signal: float, nudge: float = 0.0
    ) -> bool:
        """Advance the clock one tick and return True on touchdown."""
        old_phase = self.phase
        old_tarsus = self.tarsus_position.copy()
        self._advance_phase(dt, coupling_signal, nudge)
        crossed_liftoff, crossed_touchdown = self._detect_crossings(old_phase)
        if crossed_liftoff:
            self._on_liftoff()
        if crossed_touchdown:
            self._on_touchdown()
        self._tarsus_from_phase()
        self._record_step(old_tarsus, crossed_liftoff or crossed_touchdown)
        return crossed_touchdown

    def _advance_phase(
        self, dt: float, coupling_signal: float, nudge: float
    ) -> None:
        """Own rate plus coupling pull plus any stored kick."""
        d_theta = self.omega_current + coupling_signal
        self.phase = (
            self.phase + dt * d_theta + nudge
        ) % FULL_CYCLE

    def _detect_crossings(self, old_phase: float) -> tuple:
        """Report whether this tick crossed liftoff or touchdown.

        Liftoff is detected as the phase going down (wrapping past 2 pi).
        This assumes the clock rate stays positive.
        """
        crossed_liftoff = (
            self.phase < old_phase and old_phase > self.touchdown_phase
        )
        crossed_touchdown = old_phase < self.touchdown_phase <= self.phase
        return crossed_liftoff, crossed_touchdown

    def _on_liftoff(self) -> None:
        super()._on_liftoff()
        self._refresh_omega()

    def _on_touchdown(self) -> None:
        super()._on_touchdown()
        self._refresh_omega()

    def _refresh_omega(self) -> None:
        """Reset clock rate to its steady value plus fresh timing noise."""
        noise = 0.0
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level)
        self.omega_current = self.base_omega + noise