"""
De Capo 

"""
import numpy as np
from HexWalker_leg import Leg  
import os

class Walker: 
    def __init__(self, ID, label, body_pos, body_orient):
        self.ID = ID 
        self.label = label 
        self.body_pos = body_pos 
        self.body_orient = body_orient
        self.legs = [] #list to hold leg objects)
        self.attach_legs() #six legs six labels and IDs six workspace centres, six workspace radii 
        # for visualisation and 6 diff origin points 

    def attach_legs(self): 
        base_dir = os.path.dirname(__file__)
        origins = np.load(os.path.join(base_dir, 'leg_origins.npy'))
        print(origins)

        workspace_centers = [
        np.array([-0.1,  0.2,  0.0]),  # L1
        np.array([-0.2,  0.0,  0.0]),  # L2
        np.array([-0.1, -0.25, 0.0]),  # L3
        np.array([ 0.1,  0.2,  0.0]),  # R1
        np.array([ 0.2,  0.0,  0.0]),  # R2
        np.array([ 0.1, -0.25, 0.0]),  # R3
        ]

        leg_labels = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
        workspace_radius = 0.2

        for i in range(6):
            leg = Leg(i+1, leg_labels[i], origins[i], workspace_centers[i], workspace_radius)
            self.legs.append(leg)

            # alternating tripod starting pattern
            # legs 0, 2, 4 start in swing
        self.legs[0].GroundContact = False
        self.legs[2].GroundContact = False
        self.legs[4].GroundContact = False




#test_block        
if __name__ == "__main__": 
    import numpy as np 
    body_pos = np.array([0.0, 0.0, 0.0])
    walker = Walker(1, "Walker_01", body_pos, 0.0)
    for leg in walker.legs:
        print(leg.ID, leg.leg_label, leg.TarsusPosition)
        