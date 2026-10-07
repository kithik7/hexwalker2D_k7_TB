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
            body_scale: float = 6.0
    ) -> None: 
        self.walker_id = walker_id 
        self.insect_body = insect_body
        self.body_position = body_position 
        self.body_orientation = body_orientation 
        self.dt = dt 
        self.body_scale = body_scale
        self.legs = []
        self._attach_legs()
        self._set_tripod_pattern()

    def _attach_legs(self) -> None: 
        """Create six HexWalkerLeg Objects from InsectBody attachment points"""
        attachment_points = self.insect_body.attachment_points * self.body_scale
        # sort into left and right, then front to back
        left_points = sorted([p for p in attachment_points if p[0] < 0], key=lambda p: -p[1])
        right_points = sorted([p for p in attachment_points if p[0] > 0], key=lambda p: -p[1])
        attachment_points = np.array(left_points + right_points)
        workspace_centres = self._calculate_workspace_centres(attachment_points)
        leg_labels = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

        for i in range(6): 
            origin = np.append(attachment_points[i], 0.0)
            leg = HexWalkerLeg(
                leg_id=i+1,
                leg_label=leg_labels[i],
                origin=origin,
                workspace_centre=workspace_centres[i],
                workspace_radius=0.1
            )
            self.legs.append(leg)

    def _calculate_workspace_centres(
            self, 
            attachment_points: np.ndarray, 
    ) -> list:
        """Calculate workspace centre in the protruding outward direction for each attachment point"""
        body_edge_to_workspace_centre_distance = 0.08 * self.body_scale
        centres = []
        for point in attachment_points: 
            outward_direction = point / np.linalg.norm(point)
            centre = np.append(outward_direction * body_edge_to_workspace_centre_distance, 0.0)
            centres.append(centre) #Leg class expects 3 element arrays, 
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

    def _calculate_displacement(self) -> tuple:
        displacement = np.zeros(3)
        grounded_legs_count = 0
        for leg in self.legs: 
            if leg.ground_contact:
                displacement += -leg.normalised_vector_to_posterior_extreme * leg.stance_step_size * self.dt
                grounded_legs_count += 1
        if grounded_legs_count > 0:
            displacement /= grounded_legs_count
        return displacement[:2], 0.0

    def get_tarsus_positions(self) -> np.ndarray:
        """Return tarsus positions in world coordinates"""
        positions = []
        for leg in self.legs: 
            positions.append(leg.tarsus_position + self.body_position)
        return np.array(positions)

    def get_support_polygon_grounded_positions(self) -> np.ndarray: 
        """Returns tarsus position of grounded legs"""
        support_polygon_leg_positions = []
        for leg in self.legs: 
            if leg.ground_contact == True:
                support_polygon_leg_positions.append(leg.tarsus_position + self.body_position)
        return np.array(support_polygon_leg_positions)
 
    def set_noise_level(
            self,
            noise_level: float,
    ) -> None: 
        for leg in self.legs: 
            leg.noise_level = noise_level 

    def set_curve_radius(
            self,
            curve_radius: float,
    ) -> None: 
        """Set stride orientation for each leg so walker can start curve walking"""
        CURVE_FACTORS = [1, 0, -1, 1, 0, -1]
        base_angle = 0.0 if curve_radius == 0 else 1.0 / curve_radius 
        for i, leg in enumerate(self.legs):
            leg.stride_orientation = CURVE_FACTORS[i] * base_angle
            leg._calculate_stride_endpoints() #to recalculate AEP PEP based on new orientation



