import numpy as np 
import matplotlib.pyplot as plt

#file contains one class called HexWalker Leg
#class method = __init__ the constructor (receives info: ID, leg label, origin point, 
#workspace center - offset from origin to the middle leg of the reacahable zone
#workspace radius - how large the reachable zone is

#INSIDE CLASS leg, init stores each leg on the object it"self" 
class Leg:
    def __init__(self, ID, leg_label, origin_pt, workspace_center, workspace_radius):
        self.ID = ID
        self.leg_label = leg_label 
        self.origin_pt = origin_pt 
        self.workspace_center = workspace_center
        self.workspace_radius = workspace_radius


        # default values are needed so __init__ sets these values that every leg 
        # starts with
        #regardless of what is passed in 
        self.StanceAmp = 0.3 
        self.StanceSpeed = 0.02
        self.SwingSpeed = 0.03
        self.GroundContact = True
        self.StanceOrientation = 0.0
        self.TarsusPosition = origin_pt + workspace_center 

# TEST if name = main keeps testing block tied to this hexleg code ,  wont be run if another script calls it
if __name__ == "__main__":
    origin = np.array([0.1, 0.2, 0.0])
    workspace_center = np.array([-0.1, 0.2, 0.0])
    leg = Leg(1, "L1", origin, workspace_center, 0.2) #call init to create leg with prescribed syntax param 
    print(leg.ID)
    print(leg.leg_label)
    print(leg.TarsusPosition)

    