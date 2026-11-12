"""
cruse_leg.py

CruseLeg: a leg whose foot has its own position and moves event-driven
In stance the foot is carried back towards its noisy PEP at a stance amplitude
and lifts off at the liftoff threshold located at the PEP. 
The PEP is moved by the coordination rules - positive shift implies early liftoff,
a negative shift moves it backwards past the PEP. The tarsus can travel past the PEP
but there is a cap ie max_overshoot which is the limit of its workspace. 
Swing: the tarsus travels forward to its noisy AEP and lands there.
The tarsi decide the events and have NO clocks, the rule shift is computed 
outside the leg and is passed in here each tick.
"""

import numpy as np

from hexwalker.legs.leg import FULL_CYCLE, Leg

MAX_OVERSHOOT = 0.1  # how far past PEP the foot can reach (stride = 0.3)


class CruseLeg(Leg):

    def _set_default_parameters(self) -> None:
        self._contact = True
        self._placed = False
        self._start_phase = np.pi
        self._swing_start = np.zeros(3)
        super()._set_default_parameters()

    # phase psi (just translated into radians without any effect)
    @property
    def phase(self) -> float:
        """Foot progress mapped onto the cycle (psi).

        Before the foot is placed, this returns the requested start phase.
        """
        if not self._placed:
            return self._start_phase
        if self._contact:
            stance_share = FULL_CYCLE - self.touchdown_phase
            progress = min(self._stance_travelled() / self._stance_length(),
                           1.0)
            return self.touchdown_phase + progress * stance_share
        return self._swing_progress() * self.touchdown_phase

    @phase.setter
    def phase(self, value: float) -> None:
        self._start_phase = value % FULL_CYCLE
        self._placed = False

    @property
    def ground_contact(self) -> bool:
        return self._contact

    def set_stance_step(self, new_step: float) -> None:
        # change stance speed and remap psi phase to maintain tarsal progression
        self.stance_step_size = new_step
        self._update_timing()

    
    def update(self, liftoff_shift: float = 0.0) -> bool:
        """Advance the foot by one tick and return True on touchdown.

        liftoff_shift: summed cruse rule influence, in stride units.
            Positive moves the liftoff threshold forwards (lift earlier),
            negative moves it backwards past the PEP (lift later).
        """
        old_tarsus = self.tarsus_position.copy()
        lifted = False
        landed = False

        if self._contact:
            if self._reached_threshold(liftoff_shift):
                self._contact = False
                self._swing_start = self.tarsus_position.copy()
                self._on_liftoff()
                lifted = True
        elif self._near_aep():
            self._contact = True
            self._on_touchdown()
            landed = True

        self._move_tarsus()
        self._record_step(old_tarsus, lifted or landed)
        return landed

    # stance geometry

    def _stance_direction(self) -> np.ndarray:
        line = (self.noisy_posterior_extreme_position
                - self.noisy_anterior_extreme_position)
        return line / np.linalg.norm(line)

    def _stance_length(self) -> float:
        return float(np.linalg.norm(
            self.noisy_posterior_extreme_position
            - self.noisy_anterior_extreme_position
        ))

    def _stance_travelled(self) -> float:
        """Distance moved along the stance line since the noisy AEP."""
        offset = self.tarsus_position - self.noisy_anterior_extreme_position
        return float(np.dot(offset, self._stance_direction()))

    def _reached_threshold(self, liftoff_shift: float) -> bool:
        threshold = self._stance_length() - liftoff_shift
        threshold = min(threshold, self._stance_length() + MAX_OVERSHOOT)
        remaining = threshold - self._stance_travelled()
        return remaining <= self.stance_step_size

    # swing geometry

    def _near_aep(self) -> bool:
        distance = np.linalg.norm(
            self.noisy_anterior_extreme_position - self.tarsus_position
        )
        return distance <= self.swing_step_size

    def _swing_progress(self) -> float:
        total = np.linalg.norm(
            self.noisy_anterior_extreme_position - self._swing_start
        )
        if total == 0:
            return 0.0
        travelled = np.linalg.norm(self.tarsus_position - self._swing_start)
        return float(np.clip(travelled / total, 0.0, 1.0))


    def _move_tarsus(self) -> None:
        if self._contact:
            step = self._stance_direction() * self.stance_step_size
            self.tarsus_position = self.tarsus_position + step
            return
        gap = self.noisy_anterior_extreme_position - self.tarsus_position
        distance = np.linalg.norm(gap)
        if distance > 0:
            step = min(self.swing_step_size, distance)
            self.tarsus_position = self.tarsus_position + gap / distance * step


    def _tarsus_from_phase(self) -> None:
        # Place the foot at the start phase only at the outset.
        self._contact = (
            self.touchdown_phase <= self._start_phase < FULL_CYCLE
        )
        super()._tarsus_from_phase()
        self._swing_start = self.noisy_posterior_extreme_position.copy()
        self._placed = True

    def _liftoff_point(self) -> np.ndarray:
        return self.tarsus_position

    def _touchdown_point(self) -> np.ndarray:
        return self.tarsus_position