"""
hex_walker_leg.py 

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

class HexWalkerLeg:

    def __init__(
        self, 
        leg_id: int, 
        leg_label: str, 
        origin: np.ndarray, 
        workspace_centre: np.ndarray,
        workspace_radius: float, 
    ) -> None: 
        
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
        # self.ground_contact = True 
        self.received_excitation =False #Cruse Rule 2 
        self.ticks_in_swing = 0 
        self.min_swing_duration = 0 #in ticks 
        self.stride_orientation = 0.0
        self.noise_level = 0.0
        self.frame = CoordinateMapper(parent=None)
        self.landing_position_history  = []
        self.liftoff_position_history = []
        self.phase = np.pi #start in stance 
        self.omega_current = self.stance_step_size

    def _initialise_tarsus_position(self) -> None: 
        self.tarsus_position = self.workspace_centre.copy()

    def _calculate_stride_endpoints(self) -> None: 
        rotation_matrix = self._build_rotation_matrix()
        stride_vector = self._calculate_stride_vector(rotation_matrix)
        self.anterior_extreme_position = self.tarsus_position + stride_vector
        self.posterior_extreme_position = self.tarsus_position - stride_vector
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
        return np.pi <= self.phase < 2 * np.pi

    @ground_contact.setter 
    def ground_contact(self, value: bool) -> None:
        if value: 
            self.phase = np.pi 
        else: self.phase = 0.0
    @property
    def distance_to_posterior_extreme(self) -> float:
        return np.linalg.norm(self.posterior_extreme_position - self.tarsus_position)

    @property 
    def distance_to_anterior_extreme(self) -> float:
        return np.linalg.norm(self.anterior_extreme_position - self.tarsus_position)

    @property
    def vector_to_posterior_extreme(self) -> np.ndarray:
        return self.posterior_extreme_position - self.tarsus_position

    @property
    def vector_to_anterior_extreme(self) -> np.ndarray:
        return self.anterior_extreme_position - self.tarsus_position

    @property
    def normalised_vector_to_posterior_extreme(self) -> np.ndarray:
        if self.distance_to_posterior_extreme == 0: 
            return np.zeros(3)
        return self.vector_to_posterior_extreme / self.distance_to_posterior_extreme

    @property
    def normalised_vector_to_anterior_extreme(self) -> np.ndarray:
        if self.distance_to_anterior_extreme == 0: 
            return np.zeros(3)
        return self.vector_to_anterior_extreme / self.distance_to_anterior_extreme

    @property
    def vector_to_noisy_anterior_extreme(self) -> np.ndarray:
        return self.noisy_anterior_extreme_position - self.tarsus_position

    @property
    def normalised_vector_to_noisy_anterior_extreme(self) -> np.ndarray:
        dist = np.linalg.norm(self.vector_to_noisy_anterior_extreme)
        if dist == 0:
            return np.zeros(3)
        return self.vector_to_noisy_anterior_extreme / dist

    @property
    def distance_to_noisy_anterior_extreme(self) -> float:
        return np.linalg.norm(self.vector_to_noisy_anterior_extreme)

    @property
    def vector_to_noisy_posterior_extreme(self) -> np.ndarray:
        return self.noisy_posterior_extreme_position - self.tarsus_position

    @property
    def normalised_vector_to_noisy_posterior_extreme(self) -> np.ndarray:
        dist = np.linalg.norm(self.vector_to_noisy_posterior_extreme)
        if dist == 0:
            return np.zeros(3)
        return self.vector_to_noisy_posterior_extreme / dist

    @property
    def distance_to_noisy_posterior_extreme(self) -> float:
        return np.linalg.norm(self.vector_to_noisy_posterior_extreme)
    
    def update(
            self, 
            dt: float, 
            posterior_in_swing: bool = False, 
            contralateral_swing_progress: float = 0.0, 
            rule3_signal: float = 0.0, 
            contralateral_coupling_strength: float=0.3, 
            rule3_coupling_strength: float = 0.3,
    ) -> None: 
        self._check_state_transition(
            posterior_in_swing, 
            contralateral_swing_progress, 
            rule3_signal,
            contralateral_coupling_strength,
            rule3_coupling_strength)
        self._move_tarsus(dt)

    def _check_state_transition(
        self, 
        posterior_in_swing: bool = False, 
        contralateral_swing_progress: float = 0.0, 
        rule3_signal: float = 0.0,
        contralateral_coupling_strength: float = 0.3,
        rule3_coupling_strength: float = 0.3,
    ) -> None:

        if self.ground_contact:

        #gated rule 1 (cannot be vetoed)
            if posterior_in_swing:
                return
            if contralateral_swing_progress > 0.5:
                return

            # Gate 2: timing. Only reached if Gate 1 passes.
            # Rule 3 extends the threshold proportionally to the incoming signal.
            extended_threshold = self.stance_step_size * (1.0 + rule3_signal * rule3_coupling_strength)
            if self.distance_to_noisy_posterior_extreme <= extended_threshold:
                self.ground_contact = False
                self.liftoff_position_history.append(
                    (self.tarsus_position - self.workspace_centre).copy()
                )
                self._draw_noisy_anterior_extreme_position()

        else:

            # Landing: touch down when close to noisy AEP.
            if (
                self.distance_to_noisy_anterior_extreme <= self.swing_step_size
                or (self.received_excitation and self.distance_to_noisy_anterior_extreme <= self.swing_step_size * 1.5)
            ):
                self.ground_contact = True
                self.received_excitation = False
                self.landing_position_history.append(
                    (self.tarsus_position - self.workspace_centre).copy()
                )
                self._draw_noisy_posterior_extreme_position()
            
    def _tarsus_from_phase(self) -> None: 
        if not self.ground_contact:
            fraction = self.phase / np.pi 
            self.tarsus_position = (
                self.noisy_posterior_extreme_position 
                + fraction * (
                    self.noisy_anterior_extreme_position - self.noisy_posterior_extreme_position
                )
            )
        else:
            fraction = (self.phase - np.pi) / np.pi 
            self.tarsus_position = (
                self.noisy_anterior_extreme_position
                + fraction * (
                    self.noisy_posterior_extreme_position
                    - self.noisy_anterior_extreme_position
                )
            )
    def _draw_noisy_anterior_extreme_position(self) -> None:
        if self.noise_level == 0.0:
            self.noisy_anterior_extreme_position = self.anterior_extreme_position.copy()
            return
        noise = np.random.multivariate_normal(
            mean=[0.0, 0.0],
            cov=[[self.noise_level, 0.0], [0.0, self.noise_level]],
        )
        noise_offset = np.append(noise, 0.0)
        self.noisy_anterior_extreme_position = self.anterior_extreme_position + noise_offset

    def _draw_noisy_posterior_extreme_position(self) -> None: 
        if self.noise_level == 0.0:
                self.noisy_posterior_extreme_position = self.posterior_extreme_position.copy()
                return
        noise = np.random.multivariate_normal(
            mean=[0.0, 0.0],
            cov=[[self.noise_level, 0.0], [0.0, self.noise_level]],
        )
        noise_offset = np.append(noise, 0.0)
        self.noisy_posterior_extreme_position = self.posterior_extreme_position + noise_offset

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
    