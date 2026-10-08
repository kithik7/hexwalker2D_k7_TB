"""
coordinate_mapper.py

CoordinateMapper is a node in a kinematic tree stucture. It is a morphism that 
maps any local position to world space through a chain of transformations. 


Has no information about legs, the walker and the simulation
"""
import numpy as np
from typing import Optional

class CoordinateMapper:
    def __init__(
            self,
            parent: Optional['CoordinateMapper'] = None,
    ) -> None:
        self.parent = parent 
        self.local_transform = np.eye(4) # is an identity matrix 4x4

    def world_transform(self) -> np.ndarray:
        if self.parent is None: 
            return self.local_transform
        return self.parent.world_transform() @ self.local_transform

    def to_world(self, point: np.ndarray) -> np.ndarray:
        """Map a local point to world coordinates through the transform chain"""
        homogenous = np.append(point[:3], 1.0)
        return (self.world_transform() @ homogenous)[:3]

    def set_transform(
        self,
        translation: np.ndarray,
        angle: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        """Set transform with optional scale and rotation."""
        c = np.cos(angle)
        s = np.sin(angle)
        self.local_transform = np.array([
            [scale * c, -scale * s, 0, translation[0]],
            [scale * s,  scale * c, 0, translation[1]],
            [0,          0,         1, translation[2]],
            [0,          0,         0, 1             ],
        ])