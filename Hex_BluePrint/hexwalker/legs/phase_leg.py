"""
phase_leg.py 

Defines a single leg of a six legged walking agent. 
Each leg maintains its own stepping state, position, and geometry.
No information about other legs or the body is known to the leg. 
"""

import numpy as np 
from scipy.spatial.transform import Rotation 
from hexwalker.core.coordinate_mapper import CoordinateMapper
from dataclasses import dataclass 

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

class PhaseLeg:

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
            leg_id,
            leg_label, 
            origin, 
            workspace_centre, 
            workspace_radius, 
        )
        self._set_default_parameters()
        self._initialise_tarsus_position()
        self._calculate_stride_endpoints()
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
        self.landing_position_history  = []
        self.liftoff_position_history = []
        self.phase = np.pi #start in stance 
        self._update_timing()
        self.last_step = np.zeros(3)
        self.phase_kick = 0.0

    def _update_timing(self) -> None:
        stance_ticks = self.stride_amplitude / self.stance_step_size
        swing_ticks = self.stride_amplitude / self.swing_step_size
        cycle_ticks = stance_ticks + swing_ticks
        self.base_omega = 2 * np.pi / (cycle_ticks * self.dt)
        self.touchdown_phase = 2 * np.pi * swing_ticks / cycle_ticks
        self.omega_current = self.base_omega #base omega is 2 pi / cycle duration (ticks x dt) : steady clock rate

    def set_stance_step(self, new_step: float) -> None:
        was_stance = self.ground_contact
        if was_stance:
            progress = (self.phase - self.touchdown_phase) / (2 * np.pi - self.touchdown_phase)
        else:
            progress = self.phase / self.touchdown_phase
        self.stance_step_size = new_step
        self._update_timing()
        if was_stance:
            self.phase = self.touchdown_phase + progress * (2 * np.pi - self.touchdown_phase)
        else:
            self.phase = progress * self.touchdown_phase

    def _initialise_tarsus_position(self) -> None: 
        self.tarsus_position = self.workspace_centre.copy()

    def _calculate_stride_endpoints(self) -> None: 
        rotation_matrix = self._build_rotation_matrix()
        stride_vector = self._calculate_stride_vector(rotation_matrix)
        self.anterior_extreme_position = self.workspace_centre + stride_vector
        self.posterior_extreme_position = self.workspace_centre - stride_vector
        self.noisy_anterior_extreme_position = self.anterior_extreme_position.copy()
        self.noisy_posterior_extreme_position = self.posterior_extreme_position.copy()

    def _build_rotation_matrix(self) -> np.ndarray: 
        rotation = Rotation.from_euler('z', self.stride_orientation)
        return rotation.as_matrix()

    def _calculate_stride_vector(self, rotation_matrix) -> np.ndarray: 
        forward_direction = np.array([0,1,0])
        rotated_forward_direction = rotation_matrix @ forward_direction
        stride_vector = rotated_forward_direction * self.stride_amplitude / 2 
        return stride_vector 

    @property
    def ground_contact(self) -> bool:
        return self.touchdown_phase <= self.phase < 2 * np.pi

   
    def update_phase(
        self,
        dt: float,
        coupling_signal: float,
    ) -> bool:
        old_phase = self.phase
        old_tarsus = self.tarsus_position.copy()
        self._advance_phase(dt, coupling_signal)
        crossed_liftoff, crossed_touchdown = self._detect_crossings(old_phase)
        if crossed_liftoff:
            self._on_liftoff()
        if crossed_touchdown:
            self._on_touchdown()
        self._tarsus_from_phase()
        self._record_step(old_tarsus, crossed_liftoff or crossed_touchdown)
        return crossed_touchdown

    def _advance_phase(self, dt: float, coupling_signal: float) -> None:
        #moves phase clock by one step : own rate + coupling pull and stored nudge
        d_theta = self.omega_current + coupling_signal
        self.phase = (self.phase + dt * d_theta + self.phase_kick) % (2 * np.pi)
        self.phase_kick = 0.0

    def _detect_crossings(self, old_phase: float) -> tuple:
        #to report whether this tick crossed the lift-off or touchdown boundary
        crossed_liftoff = self.phase < old_phase and old_phase > self.touchdown_phase
        crossed_touchdown = old_phase < self.touchdown_phase <= self.phase
        return crossed_liftoff, crossed_touchdown

    def _on_liftoff(self) -> None:
        self.noisy_anterior_extreme_position = self._noisy(self.anterior_extreme_position)
        self._refresh_omega()
        self.liftoff_position_history.append(
            (self.noisy_posterior_extreme_position - self.workspace_centre).copy()
        )

    def _on_touchdown(self) -> None:
        self.noisy_posterior_extreme_position = self._noisy(self.posterior_extreme_position)
        self._refresh_omega()
        self.landing_position_history.append(
            (self.noisy_anterior_extreme_position - self.workspace_centre).copy()
        )

    def _refresh_omega(self) -> None:
        #reset clock rate to its steady value + fresh timing noise
        noise = np.random.normal(0, self.noise_level) if self.noise_level > 0 else 0.0
        self.omega_current = self.base_omega + noise

    def _record_step(self, old_tarsus: np.ndarray, crossed: bool) -> None:
        if crossed:
            self.last_step = np.zeros(3)
        else:
            self.last_step = self.tarsus_position - old_tarsus
    
    def _tarsus_from_phase(self) -> None: 
        if not self.ground_contact:
            fraction = self.phase / self.touchdown_phase
            self.tarsus_position = (
                self.noisy_posterior_extreme_position 
                + fraction * (
                    self.noisy_anterior_extreme_position - self.noisy_posterior_extreme_position
                )
            )
        else:
            fraction = (self.phase - self.touchdown_phase) / (2 * np.pi - self.touchdown_phase)
            self.tarsus_position = (
                self.noisy_anterior_extreme_position
                + fraction * (
                    self.noisy_posterior_extreme_position
                    - self.noisy_anterior_extreme_position
                )
            )
    def _noisy(self, position: np.ndarray) -> np.ndarray:
        #return copy of target position shifted by random noise if the noise level is > 0
        if self.noise_level == 0.0:
            return position.copy()
        noise = np.random.normal(0.0, self.noise_level, size=2)
        return position + np.append(noise, 0.0)

    def get_geometry(self) -> 'LegGeometry':
        """Return current leg state as a value object."""
        return LegGeometry(
            leg_id=self.leg_id,
            leg_label=self.leg_label,
            ground_contact=self.ground_contact,
            tarsus=self.tarsus_position.copy(),
            anterior_extreme=self.anterior_extreme_position.copy(),
            posterior_extreme=self.posterior_extreme_position.copy(),
            noisy_anterior_extreme=self.noisy_anterior_extreme_position.copy(),
            noisy_posterior_extreme=self.noisy_posterior_extreme_position.copy(),
            landing_position_history=self.landing_position_history,
            liftoff_position_history=self.liftoff_position_history,
            workspace_centre=self.workspace_centre.copy(),
            workspace_radius=self.workspace_radius,
            origin=self.origin.copy(),
        )
    