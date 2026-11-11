"""
hex_walker.py 

HexWalker: Receives an InsectBody object and creates six HexWalkerLeg objects 
from the attachment points. It stores body_position and body_orientation 
It gets tarsus positions, gets support polygon and calculates displacement in its update function
No information known about the simulation or leg geometry.
"""

import numpy as np 
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.coordinate_mapper import CoordinateMapper
from hexwalker.core.hex_walker_leg import HexWalkerLeg, LegGeometry
from hexwalker.core.phase_coupling import PhaseCoupling

class HexWalker: 
    def __init__(
            self, 
            walker_id: int, 
            insect_body: InsectBody, 
            body_position: np.ndarray,
            body_orientation: float, 
            dt: float, 
            body_scale: float = 6.0,
            random_start_phases: bool = True,
    ) -> None: 
        self.walker_id = walker_id 
        self.insect_body = insect_body
        self.body_position = body_position 
        self.body_orientation = body_orientation 
        self.dt = dt 
        self.body_scale = body_scale
        self.legs = []
        
        self.world_frame = CoordinateMapper(parent=None)
        self.body_frame = CoordinateMapper(parent=self.world_frame)
        self.body_frame.set_transform(self.body_position, angle=self.body_orientation, scale=self.body_scale) 
 
        self._attach_legs()
        #self._set_tripod_pattern()
        self.random_start_phases = random_start_phases
        if self.random_start_phases: 
            self._set_random_phases()
        else:
            self._set_tripod_pattern()
        
        self.coupling = PhaseCoupling(self.legs)

       
    @property
    def default_coupling(self) -> dict:
        return self.coupling.default_coupling
    
    def _attach_legs(self) -> None: 
        """Create six HexWalkerLeg Objects from InsectBody attachment points"""
        attachment_points = self.insect_body.attachment_points 
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
                workspace_radius=0.2,
                dt = self.dt,
            )
            self.legs.append(leg)
            leg.frame.parent = self.body_frame
            leg.frame.set_transform(np.append(attachment_points[i], 0.0), scale=1.0)
       

    def _calculate_workspace_centres(self, attachment_points: np.ndarray) -> list:
        body_edge_to_workspace_centre_distance = 0.15
        centres = []
        for point in attachment_points:
            outward_direction = point / np.linalg.norm(point)
            centre = np.append(outward_direction * body_edge_to_workspace_centre_distance, 0.0)
            centres.append(centre)
        return centres
            
    def _set_tripod_pattern(self) -> None: 
        """Set alternating tripod starting pattern. Legs 0, 2, 4 start in swing."""
        tripod_A = [0, 2, 4] #swing: phase in 0, pi 
        tripod_B = [1, 3, 5] #stance: phase in pi, 2pi
        for idx in tripod_A: 
            self.legs[idx].phase = self.legs[idx].touchdown_phase / 2
        for idx in tripod_B:
            self.legs[idx].phase = self.legs[idx].touchdown_phase / 2 + np.pi
        for leg in self.legs: 
            leg._tarsus_from_phase()

    #set a random phase draw for each leg at t=0 to test 

    def _set_random_phases(self) -> None:
        for leg in self.legs:
            leg.phase = np.random.uniform(0.0, 2 * np.pi)
            leg._tarsus_from_phase()

    def update(self, dt: float) -> None:
        coupling_signals = self.coupling.compute_coupling_signals()
        touched_down = [leg.update_phase(dt, coupling_signals[i]) for i, leg in enumerate(self.legs)]
        self.coupling.apply_nudges(touched_down)
        self._move_body()

    
    def _move_body(self) -> None:
        """move body position and orientation based on grounded legs"""
        translation, rotation = self._calculate_displacement()
        self.body_position[0] += translation[0]
        self.body_position[1] += translation[1]
        self.body_orientation += rotation
        self.body_frame.set_transform(
            self.body_position,
            angle=self.body_orientation,
            scale=self.body_scale
        )


    def _calculate_displacement(self) -> tuple:
        """Calculate body displacement using grounded leg pair."""
        grounded_legs = [leg for leg in self.legs if leg.ground_contact and np.linalg.norm(leg.last_step) > 0]
        if len(grounded_legs) < 2:
            return np.zeros(2), 0.0

        all_translations = []
        all_rotations = []
        pairs = [(grounded_legs[i], grounded_legs[j])
                for i in range(len(grounded_legs))
                for j in range(i + 1, len(grounded_legs))]

        for leg_a, leg_b in pairs:
            p1 = leg_a.origin - leg_a.last_step
            p2 = leg_b.origin - leg_b.last_step

            v_original = leg_a.origin - leg_b.origin
            norm_orig = np.linalg.norm(v_original)
            if norm_orig < 1e-10:
                continue
            v_original = v_original / norm_orig

            v_new = p1 - p2
            norm_new = np.linalg.norm(v_new)
            if norm_new < 1e-10:
                continue
            v_new = v_new / norm_new

            dp = np.clip(np.dot(v_original, v_new), -1.0, 1.0)
            angle = np.arccos(dp)
            cross = np.cross(v_original, v_new)
            if cross[2] < 0:
                angle = -angle

            com = np.zeros(3)
            t1 = (np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle),  np.cos(angle), 0],
                         [0, 0, 1]]) @ (com - leg_a.origin)) + leg_a.origin - leg_a.last_step
            t2 = (np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle),  np.cos(angle), 0],
                         [0, 0, 1]]) @ (com - leg_b.origin)) + leg_b.origin - leg_b.last_step

            all_translations.append(t1[:2])
            all_translations.append(t2[:2])
            all_rotations.append(angle)
            all_rotations.append(angle)

        if not all_translations:
            return np.zeros(2), 0.0

        mean_position = np.mean(all_translations, axis=0)
        mean_rotation = np.mean(all_rotations)
        displacement = mean_position * self.body_scale
        c = np.cos(self.body_orientation)
        s = np.sin(self.body_orientation)
        world_x = c * displacement[0] - s * displacement[1]
        world_y = s * displacement[0] + c * displacement[1]
        return np.array([world_x, world_y]), mean_rotation   
       
    
    def get_tarsus_positions(self) -> np.ndarray:
        """Return tarsus positions in world coordinates via kinematic chain."""
        return np.array([
            leg.frame.to_world(leg.tarsus_position) 
            for leg in self.legs
        ])

    def get_support_polygon_grounded_positions(self) -> np.ndarray:
        """Return grounded tarsus positions in world coordinates via kinematic chain."""
        return np.array([
            leg.frame.to_world(leg.tarsus_position)
            for leg in self.legs
            if leg.ground_contact
        ])

    def get_world_geometry(self) -> list:
        """Return all leg geometry converted to world coordinates."""
        result = []
        for leg in self.legs:
            g = leg.get_geometry()
            result.append(LegGeometry(
                leg_id=g.leg_id,
                leg_label=g.leg_label,
                ground_contact=g.ground_contact,
                tarsus=leg.frame.to_world(g.tarsus),
                anterior_extreme=leg.frame.to_world(g.anterior_extreme),
                posterior_extreme=leg.frame.to_world(g.posterior_extreme),
                noisy_anterior_extreme=leg.frame.to_world(g.noisy_anterior_extreme),
                noisy_posterior_extreme=leg.frame.to_world(g.noisy_posterior_extreme),
                landing_position_history=list(g.landing_position_history),
                liftoff_position_history=list(g.liftoff_position_history),
                workspace_centre=leg.frame.to_world(g.workspace_centre),
                workspace_radius=g.workspace_radius * self.body_scale,
                origin=leg.frame.to_world(np.zeros(3)),
            ))
        return result
 
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
        
        for i, leg in enumerate(self.legs):
            leg.stride_orientation = CURVE_FACTORS[i] * curve_radius
            leg._calculate_stride_endpoints() #to recalculate AEP PEP based on new orientation

    def set_coupling(self, name: str, value: float) -> None:
        self.coupling.set_coupling(name, value)

    def set_speed(self, speed: float) -> None:
        for leg in self.legs:
            leg.set_stance_step(speed)
            #only stance step changes with speed 


