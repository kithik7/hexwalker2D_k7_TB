"""
De Capo 

"""
import numpy as np
from HexWalker_leg import Leg  

class Walker: 
    def __init__(self, ID, label, body_pos, body_orient):
        self.ID = ID 
        self.label = label 
        self.body_pos = body_pos 
        self.body_orient = body_orient
        self.legs = [] #list to hold leg objects)
        self.attach_legs() #six legs six labels and IDs six workspace centres, six workspace radii 
        # for visualisation and 6 diff origin points 

        