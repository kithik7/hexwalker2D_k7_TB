"""
De Capo 

"""
import numpy as np
from HexWalker_leg import Leg  
import os

class Walker: 
    def __init__(self, ID, label, body_pos, body_orient, body_scale = 1.0):
        self.ID = ID 
        self.label = label 
        self.body_pos = body_pos 
        self.body_orient = body_orient
        self.body_scale = body_scale 
        self.legs = [] #list to hold leg objects)
        self.attach_legs() #six legs six labels and IDs six workspace centres, six workspace radii 
        # for visualisation and 6 diff origin points 

    def attach_legs(self): 
        base_dir = os.path.dirname(__file__)
        origins = np.load(os.path.join(base_dir, 'leg_origins.npy')) * self.body_scale
        
        workspace_centers = [
        np.array([-0.1,  0.2,  0.0]),  # L1
        np.array([-0.2,  0.0,  0.0]),  # L2
        np.array([-0.1, -0.25, 0.0]),  # L3
        np.array([ 0.1,  0.2,  0.0]),  # R1
        np.array([ 0.2,  0.0,  0.0]),  # R2
        np.array([ 0.1, -0.25, 0.0]),  # R3
        ]
        workspace_centers = [ws * self.body_scale for ws in workspace_centers]
        
        leg_labels = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
        workspace_radius = 0.2 * self.body_scale

        for i in range(6):
            leg = Leg(i+1, leg_labels[i], origins[i], workspace_centers[i], workspace_radius)
            self.legs.append(leg)

            # alternating tripod starting pattern
            # legs 0, 2, 4 start in swing
        self.legs[0].GroundContact = False
        self.legs[2].GroundContact = False
        self.legs[4].GroundContact = False

    def calculate_displacement(self): 
        displacement = np.zeros(3)
        grounded_count = 0
        for leg in self.legs: 
            if leg.GroundContact: 
                displacement += -leg.VectorToPEP_normal *leg.StanceStep 
                grounded_count += 1 
        if grounded_count > 0: 
            displacement /= grounded_count
        return displacement

#without rotation (no net turning yet) 
# ||: for each leg 
    def update(self):
        for leg in self.legs:
            leg.update() 

        #calculate displacement 
        displacement = self.calculate_displacement()

        #apply displacement to body position and orientation 
        self.body_pos += displacement
        self.body_orient += displacement[2]

#test_block        
if __name__ == "__main__": 
    import numpy as np 
    body_pos = np.array([0.0, 0.0, 0.0])
    walker = Walker(1, "Walker_01", body_pos, 0.0)

    walker.update()
    print("body position after update 1:", walker.body_pos)

    walker.update()
    print("body position after update 2:", walker.body_pos)

    walker.update()
    print("body position after update 3:", walker.body_pos)

"""
Coda 

"""
"""
29/09: updates: body_scale, scaling in attach legs and workspace radius 
the body scaling passes  onto __init__ in HexSim while creating the walker 

TO DO: Body scaling needs to be modular and not condition specific, attachment points and workspace centers
need to be expressed 
as fractions of the body that is defined and normalized to have a unit length of 1, and multiplying the whole
with a scaling factor will scale them all, like item.children basically 


"""
        