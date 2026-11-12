"""
leg.py

Leg: everything a leg of the walker has independent of type,
and anything that decides the timing is decided by the subclasses.

Identity and workspace, AEP and PEP, noisy targets, landing and liftoff
histories, the step just taken, and the geometry handed to the
simulation. Subclasses decide when the leg lifts off and lands:
PhaseLeg by a clock, CruseLeg by the foot reaching its target.
No information is contained about other legs or the body.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from hexwalker.core.coordinate_mapper import CoordinateMapper

FULL_CYCLE = 2 * np.pi  # constant


@dataclass
class LegGeometry:
    leg_id: int
    leg_label: str
    ground_contact: bool
    tarsus: np.ndarray
    anterior_extreme: np.ndarray
    posterior_extreme: np.ndarray
    noisy_anterior_extreme: np.ndarray
    noisy_posterior_extreme: np.ndarray
    landing_position_history: list
    liftoff_position_history: list
    workspace_centre: np.ndarray
    workspace_radius: float
    origin: np.ndarray


class Leg(ABC):

    def __init__(
        self,
        leg_id: int,
        leg_label: str,
        origin: np.ndarray,
        workspace_centre: np.ndarray,
        workspace_radius: float,
        dt: float = 0.02,
    ) -> None:
        self.dt = dt
        self._store_identity(
            leg_id, leg_label, origin, workspace_centre, workspace_radius
        )
        self._set_default_parameters()
        self._initialise_tarsus_position()
        self._calculate_stride_endpoints()
        self._tarsus_from_phase()

    @property
    @abstractmethod
    def ground_contact(self) -> bool:
        """ True when the foot is on the ground """

    @property
    def stretch_progress(self) -> float:
        # How far through the current stance or swing, from 0 to 1.
        if self.ground_contact:
            stance_share = FULL_CYCLE - self.touchdown_phase
            return (self.phase - self.touchdown_phase) / stance_share
        return self.phase / self.touchdown_phase

    @abstractmethod
    def set_stance_step(self, new_step: float) -> None:
        """"""
    def place_at_phase(self, phase: float) -> None:
        self.phase = phase
        self._tarsus_from_phase()

    def _store_identity(
        self,
        leg_id: int,
        leg_label: str,
        origin: np.ndarray,
        workspace_centre: np.ndarray,
        workspace_radius: float,
    ) -> None:
        self.leg_id = leg_id
        self.leg_label = leg_label
        self.origin = origin
        self.workspace_centre = workspace_centre
        self.workspace_radius = workspace_radius

    def _set_default_parameters(self) -> None:
        self.stride_amplitude = 0.3
        self.stance_step_size = 0.02
        self.swing_step_size = 0.03
        self.stride_orientation = 0.0
        self.noise_level = 0.0
        self.frame = CoordinateMapper(parent=None)
        self.landing_position_history = []
        self.liftoff_position_history = []
        self.last_step = np.zeros(3)
        self._update_timing()

    def _update_timing(self) -> None:
        stance_ticks = self.stride_amplitude / self.stance_step_size
        swing_ticks = self.stride_amplitude / self.swing_step_size
        self.cycle_ticks = stance_ticks + swing_ticks
        self.touchdown_phase = FULL_CYCLE * swing_ticks / self.cycle_ticks

    def _initialise_tarsus_position(self) -> None:
        self.tarsus_position = self.workspace_centre.copy()

    def _calculate_stride_endpoints(self) -> None:
        rotation_matrix = self._build_rotation_matrix()
        stride_vector = self._calculate_stride_vector(rotation_matrix)
        self.anterior_extreme_position = self.workspace_centre + stride_vector
        self.posterior_extreme_position = (
            self.workspace_centre - stride_vector
        )
        self.noisy_anterior_extreme_position = (
            self.anterior_extreme_position.copy()
        )
        self.noisy_posterior_extreme_position = (
            self.posterior_extreme_position.copy()
        )

    def _build_rotation_matrix(self) -> np.ndarray:
        rotation = Rotation.from_euler('z', self.stride_orientation)
        return rotation.as_matrix()

    def _calculate_stride_vector(self, rotation_matrix) -> np.ndarray:
        forward_direction = np.array([0, 1, 0])
        rotated_forward_direction = rotation_matrix @ forward_direction
        return rotated_forward_direction * self.stride_amplitude / 2

    def _on_liftoff(self) -> None:
        self.noisy_anterior_extreme_position = self._noisy(
            self.anterior_extreme_position
        )
        self.liftoff_position_history.append(
            (self._liftoff_point() - self.workspace_centre).copy()
        )

    def _on_touchdown(self) -> None:
        self.noisy_posterior_extreme_position = self._noisy(
            self.posterior_extreme_position
        )
        self.landing_position_history.append(
            (self._touchdown_point() - self.workspace_centre).copy()
        )

    def _liftoff_point(self) -> np.ndarray:
        # where liftoff is recorded ie the noisy pep 
        return self.noisy_posterior_extreme_position

    def _touchdown_point(self) -> np.ndarray:
        # where touch down is recorded ie noisy aep 
        return self.noisy_anterior_extreme_position

    def _record_step(self, old_tarsus: np.ndarray, crossed: bool) -> None:
        if crossed:
            self.last_step = np.zeros(3)
        else:
            self.last_step = self.tarsus_position - old_tarsus

    def _tarsus_from_phase(self) -> None:
        """Place the foot between its noisy targets according to phase."""
        if not self.ground_contact:
            fraction = self.phase / self.touchdown_phase
            start = self.noisy_posterior_extreme_position
            end = self.noisy_anterior_extreme_position
        else:
            stance_share = FULL_CYCLE - self.touchdown_phase
            fraction = (self.phase - self.touchdown_phase) / stance_share
            start = self.noisy_anterior_extreme_position
            end = self.noisy_posterior_extreme_position
        self.tarsus_position = start + fraction * (end - start)

    def _noisy(self, position: np.ndarray) -> np.ndarray:
        # (SD = noise_level)
        if self.noise_level == 0.0:
            return position.copy()
        noise = np.random.normal(0.0, self.noise_level, size=2)
        return position + np.append(noise, 0.0)

    def get_geometry(self) -> LegGeometry:
        #value object is returned
        return LegGeometry(
            leg_id=self.leg_id,
            leg_label=self.leg_label,
            ground_contact=self.ground_contact,
            tarsus=self.tarsus_position.copy(),
            anterior_extreme=self.anterior_extreme_position.copy(),
            posterior_extreme=self.posterior_extreme_position.copy(),
            noisy_anterior_extreme=(
                self.noisy_anterior_extreme_position.copy()
            ),
            noisy_posterior_extreme=(
                self.noisy_posterior_extreme_position.copy()
            ),
            landing_position_history=self.landing_position_history,
            liftoff_position_history=self.liftoff_position_history,
            workspace_centre=self.workspace_centre.copy(),
            workspace_radius=self.workspace_radius,
            origin=self.origin.copy(),
        )