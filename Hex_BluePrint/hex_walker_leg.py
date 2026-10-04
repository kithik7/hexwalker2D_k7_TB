"""
hex_walker_leg.py 

Defines a single leg of a six legged walking agent. 
Each leg maintains its own stepping state, position, and geometry.
No information about other legs or the body is known to the leg. 
"""

import numpy as np 
from scipy.spatial.transform import Rotation 


class HexWalkerLeg:

    def __init__(
        self, 
        leg_id: int, 
        leg_label: str, 
        origin: np.ndarray, 
        workspace_center: np.ndarray,
        workspace_radius: float, 
    ) -> None: 
        
        self._store_identity(
            leg_id,
            leg_label, 
            origin, 
            workspace_center, 
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
        workspace_center: np.ndarray,
        workspace_radius: float, 
    ) -> None:
        self.leg_id = leg_id
        self.leg_label = leg_label
        self.origin = origin 
        self.workspace_center = workspace_center
        self.workspace_radius = workspace_radius

    def _set_default_parameters(self) -> None: 
        self.stride_amplitude = 0.3
        self.stance_step_size = 0.02 
        self.swing_step_size = 0.03 
        self.ground_contact = True 
        self.stride_orientation = 0.0
        self.noise_level = 0.0

    def _initialise_tarsus_position(self) -> None: 
        self.tarsus_position = self.origin + self.workspace_center

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
    def normalized_vector_to_posterior_extreme(self) -> np.ndarray:
        if self.distance_to_posterior_extreme == 0: 
            return np.zeros(3)
        return self.vector_to_posterior_extreme / self.distance_to_posterior_extreme

    @property
    def normalized_vector_to_anterior_extreme(self) -> np.ndarray:
        if self.distance_to_anterior_extreme == 0: 
            return np.zeros(3)
        return self.vector_to_anterior_extreme / self.distance_to_anterior_extreme
    
    def update(self, dt: float) -> None: 
        self._check_state_transition(dt)
        self._move_tarsus(dt)

    def _check_state_transition(self, dt: float) -> None:
        if self.ground_contact and self.distance_to_posterior_extreme <= self.stance_step_size * dt:
            self.ground_contact = False 
            self._draw_noisy_anterior_extreme_position()
        elif not self.ground_contact and self.distance_to_anterior_extreme <= self.swing_step_size * dt:
            self.ground_contact = True
            self._draw_noisy_posterior_extreme_position()
            
    def _move_tarsus(self, dt: float) -> None:
        if self.ground_contact:
            self.tarsus_position += self.normalized_vector_to_posterior_extreme * self.stance_step_size * dt
        else:
            self.tarsus_position += self.normalized_vector_to_anterior_extreme * self.swing_step_size * dt   

    def _draw_noisy_anterior_extreme_position(self) -> None:
        noise = np.random.multivariate_normal(
            mean=[0.0, 0.0],
            cov=[[self.noise_level, 0.0], [0.0, self.noise_level]],
        )
        noise_offset = np.append(noise, 0.0)
        self.noisy_anterior_extreme_position = self.anterior_extreme_position + noise_offset

    def _draw_noisy_posterior_extreme_position(self) -> None: 
        noise = np.random.multivariate_normal(
            mean=[0.0, 0.0],
            cov=[[self.noise_level, 0.0], [0.0, self.noise_level]],
        )
        noise_offset = np.append(noise, 0.0)
        self.noisy_posterior_extreme_position = self.posterior_extreme_position + noise_offset

        #TODO: create pytest framework to test HexWalkerLeg 
        
#temporary test - remove when TODO is complete 
if __name__ == "__main__":
    origin = np.array([0.0, 0.0, 0.0])
    workspace_center = np.array([0.0, 0.2, 0.0])
    leg = HexWalkerLeg(
        leg_id=1,
        leg_label='L1',
        origin=origin,
        workspace_center=workspace_center,
        workspace_radius=0.2,
    )

    # test AEP is 0.15 units ahead of tarsus position
    expected_aep_y = workspace_center[1] + 0.15
    assert abs(leg.anterior_extreme_position[1] - expected_aep_y) < 1e-10, \
        f"AEP y expected {expected_aep_y}, got {leg.anterior_extreme_position[1]}"

    # test update moves tarsus toward PEP
    initial_y = leg.tarsus_position[1]
    leg.update(dt=0.1)
    assert leg.tarsus_position[1] < initial_y, "Tarsus should move toward PEP"

    print("All tests passed.")