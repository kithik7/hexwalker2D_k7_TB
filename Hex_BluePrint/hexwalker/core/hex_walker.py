"""
hex_walker.py 

HexWalker: Receives an InsectBody object and creates six HexWalkerLeg objects 
from the attachment points. It stores body_position and body_orientation 
It gets tarsus positions, gets support polygon and calculates displacement in its update function
No information known about the simulation or leg geometry.
"""

import numpy as np 
from scipy.spatial import ConvexHull  
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker_leg import HexWalkerLeg
from hexwalker.core.coordinate_mapper import CoordinateMapper
from hexwalker.core.hex_walker_leg import HexWalkerLeg, LegGeometry

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
        
        self.world_frame = CoordinateMapper(parent=None)
        self.body_frame = CoordinateMapper(parent=self.world_frame)
        self.body_frame.set_transform(self.body_position, angle=self.body_orientation, scale=self.body_scale) 
        print("calling set_transform with scale:", self.body_scale)
        self.body_frame.set_transform(self.body_position, angle=self.body_orientation, scale=self.body_scale)
        print("body frame after set:", self.body_frame.local_transform)
        self._attach_legs()
        for leg in self.legs:
            print(leg.leg_label, leg.origin)
        self._set_tripod_pattern()
        #ipsilateral_posterior_neighbour_connections 
        self.posterior_neighbours = {
            0: 1, #L1 -> L2 
            1: 2,  #L2 -> L3
            2: None, #L3 has no posterior neighbour
            3: 4, #R1 -> R2 
            4: 5, #R2 -> R3
            5: None, #R3 has no posterior neighbour 
        }
        #ipsilateral anterior neighbour connections 
        self.anterior_neighbours = {
            0: None, #L1 has no anterior neighbour 
            1: 0, #L2 -> L1
            2: 1, #L3 -> L2
            3: None, #R1 has no anterior neighbour
            4: 3, #R2 -> R1
            5: 4, # R3 -> R2 
        }

        #weak minimal rule 1 contraateral coupling 
        self.contralateral_neighbours = {
            0: 3,
            1: 4,
            2: 5,
            3: 0,
            4: 1,
            5: 2,
        }

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
                workspace_radius=0.2
            )
            self.legs.append(leg)
            leg.frame.parent = self.body_frame
            leg.frame.set_transform(np.append(attachment_points[i], 0.0), scale=1.0)
        self.ipsilateral_rule3_strength = 0.3
        self.contralateral_coupling_strength = 0.3
    def _calculate_workspace_centres(
            self, 
            attachment_points: np.ndarray, 
    ) -> list:
        """Calculate workspace centre in the protruding outward direction for each attachment point"""
        body_edge_to_workspace_centre_distance = 0.15
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
        print([leg.ground_contact for leg in self.legs])

        contralateral_progress_snapshot = []
        for leg_index in range(6):
            contralateral_leg = self.legs[self.contralateral_neighbours[leg_index]]
            if not contralateral_leg.ground_contact:
                swing_direction = (contralateral_leg.anterior_extreme_position - contralateral_leg.posterior_extreme_position)
                swing_direction = swing_direction / np.linalg.norm(swing_direction)
                tarsus_displacement = contralateral_leg.tarsus_position - contralateral_leg.posterior_extreme_position
                progress = float(np.clip(
                    np.dot(tarsus_displacement, swing_direction) / contralateral_leg.stride_amplitude,
                    0.0, 1.0
                ))
            else:
                progress = 0.0
            contralateral_progress_snapshot.append(progress)


        for leg_index, leg in enumerate(self.legs): 
            posterior_neighbour_index = self.posterior_neighbours[leg_index]
            anterior_neighbour_index = self.anterior_neighbours[leg_index]
            contralateral_index = self.contralateral_neighbours[leg_index]

            if posterior_neighbour_index is not None: 
                posterior_in_swing = not self.legs[posterior_neighbour_index].ground_contact 
            else: 
                posterior_in_swing = False 

           
           
            #rule 3 runs in all ticks 
            # Rule 2 excitatory pulse boolean tag on groundcontact bool

            rule3_signal = 0.0
            if anterior_neighbour_index is not None:
                anterior_leg = self.legs[anterior_neighbour_index]
                if anterior_leg.ground_contact: 
                    stance_direction = anterior_leg.posterior_extreme_position - anterior_leg.anterior_extreme_position
                    stance_direction = stance_direction / np.linalg.norm(stance_direction)
                    tarsus_displacement = anterior_leg.tarsus_position - anterior_leg.anterior_extreme_position
                    rule3_signal = float(np.clip(
                        np.dot(tarsus_displacement, stance_direction) / anterior_leg.stride_amplitude,
                        0.0, 1.0
                    ))

            was_in_swing = not leg.ground_contact
            
            leg.update(
                dt, 
                posterior_in_swing=posterior_in_swing, 
                contralateral_swing_progress=contralateral_progress_snapshot[leg_index], 
                rule3_signal=rule3_signal,
                contralateral_coupling_strength=self.contralateral_coupling_strength,
                rule3_coupling_strength=self.ipsilateral_rule3_strength)
            just_landed = was_in_swing and leg.ground_contact 
            if just_landed and anterior_neighbour_index is not None: 
                self.legs[anterior_neighbour_index].received_excitation = True

        
        self.body_position[0] += translation[0]
        self.body_position[1] += translation[1]
        self.body_orientation += rotation
        self.body_frame.set_transform(self.body_position, angle=self.body_orientation, scale=self.body_scale)
       
    
    def _calculate_displacement(self) -> tuple:
        """Calculate body displacement using grounded leg pair."""
        grounded_legs = [leg for leg in self.legs if leg.ground_contact]
    
        if len(grounded_legs) < 2:
            return np.zeros(2), 0.0
    
        all_translations = []
        all_rotations = []
    
        pairs = [(grounded_legs[i], grounded_legs[j]) 
                for i in range(len(grounded_legs)) 
                for j in range(i + 1, len(grounded_legs))]
    
        for leg_a, leg_b in pairs:
        
            # new positions of origins after one stance step
            p1 = leg_a.origin + leg_a.normalised_vector_to_noisy_posterior_extreme * -1 * leg_a.stance_step_size 
            p2 = leg_b.origin + leg_b.normalised_vector_to_noisy_posterior_extreme * -1 * leg_b.stance_step_size 
        
            # unit vector between origins before and after
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
        
            # rotation matrix
            c = np.cos(angle)
            s = np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
            # where COM ends up according to each leg in pair
            com = np.zeros(3)
        
            t1 = (R @ (com - leg_a.origin)) + leg_a.origin + leg_a.normalised_vector_to_noisy_posterior_extreme * -1 * leg_a.stance_step_size 
            t2 = (R @ (com - leg_b.origin)) + leg_b.origin + leg_b.normalised_vector_to_noisy_posterior_extreme * -1 * leg_b.stance_step_size 
        
            all_translations.append(t1[:2])
            all_translations.append(t2[:2])
            all_rotations.append(angle)
            all_rotations.append(angle)
    
        if not all_translations:
            return np.zeros(2), 0.0
    
        mean_position = np.mean(all_translations, axis=0)
        mean_rotation = np.mean(all_rotations)
    
        # displacement is difference between new COM position and current
        displacement = mean_position * self.body_scale
        c = np.cos(self.body_orientation)
        s = np.sin(self.body_orientation)
        world_x = c * displacement[0] - s * displacement[1]
        world_y = s * displacement[0] + c * displacement[1]
        #print(f"mean_position: {mean_position}, displacement: {np.array([world_x, world_y])}")
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
            # print("noisy_anterior local:", g.noisy_anterior_extreme)
            # print("noisy_anterior world:", leg.frame.to_world(g.noisy_anterior_extreme))
            # print("anterior local:", g.anterior_extreme)
            # print("anterior world:", leg.frame.to_world(g.anterior_extreme))
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

    def set_coupling_strength(self, strength: float) -> None:
        self.ipsilateral_rule3_strength = strength
        self.contralateral_coupling_strength = strength

    def set_speed(self, speed: float) -> None:
        for leg in self.legs:
            leg.stance_step_size = speed
            #only stance step changes with speed 


