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
        self.StanceStep = 0.02
        self.SwingStep = 0.03
        self.GroundContact = True
        self.StanceOrientation = 0.0
        self.TarsusPosition = origin_pt + workspace_center 
        #calculate stde, AEP, PEP, anmd implement rotaion matrix M 
        M = np.array([[np.cos(self.StanceOrientation), -np.sin(self.StanceOrientation), 0],
                      [np.sin(self.StanceOrientation), np.cos(self.StanceOrientation), 0],
                      [0, 0, 1]])
        fwd = np.array([0,1,0])
        fwd_rotated = M @ fwd #matrix multiplied w vector resulting in rotated stride axis 
        stride_vector = fwd_rotated * self.StanceAmp/2 
        self.AEP = self.TarsusPosition + stride_vector 
        #fwd_rotated + stance_amp/2 is the resultant stride vector post rotation , each EP is #half a stride length
        self.PEP = self.TarsusPosition - stride_vector

    #define vector and distance dependent properties using @property that acts as an attribute but iteration friendly
    # distance to PEP = vector whihc is the distance from tarsus pos to PEP 
    @property 
    def DistToPEP(self):
        return np.linalg.norm(self.PEP - self.TarsusPosition)
    @property 
    def DistToAEP(self):
        return np.linalg.norm(self.AEP - self.TarsusPosition)
    @property 
    def VectorToPEP(self):
        return self.PEP - self.TarsusPosition
    @property
    def VectorToAEP(self):
        return self.AEP - self.TarsusPosition
    @property 
    def VectorToPEP_normal(self):
        if self.DistToPEP == 0: 
            return np.array([0.0, 0.0, 0.0])
        return self.VectorToPEP / self.DistToPEP
    @property 
    def VectorToAEP_normal(self):
        if self.DistToAEP == 0: 
            return np.array([0.0, 0.0, 0.0])
        return self.VectorToAEP / self.DistToAEP


# TEST if name = main keeps testing block tied to this hexleg code ,  wont be run if another script calls it
if __name__ == "__main__":
    origin = np.array([0.1, 0.2, 0.0])
    workspace_center = np.array([-0.1, 0.2, 0.0])
    leg = Leg(1, "L1", origin, workspace_center, 0.2) #call init to create leg with prescribed syntax param 
    print(leg.ID)
    print(leg.leg_label)
    print(leg.TarsusPosition)
    print(leg.AEP)
    print(leg.PEP)
    print(leg.DistToPEP)
    print(leg.DistToAEP)
    print(leg.VectorToPEP)
    print(leg.VectorToAEP)
    print(leg.VectorToPEP_normal)   
    print(leg.VectorToAEP_normal)