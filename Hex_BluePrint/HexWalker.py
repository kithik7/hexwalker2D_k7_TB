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




#test_block        
if __name__ == "__main__": 
    import numpy as np 
    body_pos = np.array([0.0, 0.0, 0.0])
    walker = Walker(1, "Walker_01", body_pos, 0.0)
        