"""
hex_walker.py 

HexWalker: Receives an InsectBody object and creates six HexWalkerLeg objects 
from the attachment points. It stores body_position and body_orientation 
It gets tarsus positions, gets support polygon and calculates displacement in its update function
No information known about the simulation or leg geometry.
"""

import numpy as np 
from scipy.spatial import ConvexHull  
from insect_body import InsectBody
from hex_walker_leg import HexWalkerLeg 

class HexWalker: 
    def __init__(
            self, 
            walker_id: int, 
            insect_body: InsectBody, 
            body_position: np.ndarray,
            body_orientation: float, 
            dt: float, 

    ) -> None: 
        self.walker_id = walker_id 
        self.insect_body = insect_body
        self.body_position = body_position 
        self.body_orientation = body_orientation 
        self.dt = dt 
        self.legs = []
        self._attach_legs()
        self._set_tripod_pattern()

    def _attach_legs(self) -> None: 
        """Create six HexWalkerLeg Objects from InsectBody attachment points"""
        attachment_points = self.insect_body.attachment_points
        workspace_centres = self._calculate_workspace_centres(attachment_points)
        leg_labels = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

        for i in range(6): 
            leg = HexWalkerLeg(
                leg_id=i+1,
                leg_label=leg_labels[i],
                origin=attachment_points[i],
                workspace_centre=workspace_centres[i],
                workspace_radius=0.1
            )
            self.legs.append(leg)

    def _calculate_workspace_centres(
            self, 
            attachment_points: np.ndarray, 
    ) -> list:
        """Calculate workspace centre in the protruding outward direction for each attachment point"""
        body_edge_to_workspace_centre_distance = 0.15
        centres = []
        for point in attachment_points: 
            outward_direction = point / np.linalg.norm(point)
            centre = outward_direction * body_edge_to_workspace_centre_distance
            centres.append(centre)
        return centres 

    def _set_tripod_pattern(self) -> None: 
        """Set alternating tripod starting pattern. Legs 0, 2, 4 start in swing."""
        self.legs[0].ground_contact = False 
        self.legs[2].ground_contact = False 
        self.legs[4].ground_contact = False 

    def update(
            self,
            dt: float
    ) -> None:
        """Update body position and orientation from displacement calculated from grounded legs."""
        translation, rotation = self._calculate_displacement()
        for leg in self.legs: 
            leg.update(dt)
        self.body_position[0] += translation[0]
        self.body_position[1] += translation[1]
        self.body_orientation += rotation