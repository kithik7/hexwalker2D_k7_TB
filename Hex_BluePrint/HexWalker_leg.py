"""
Da Capo (D.C.)

"""

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
        self.workspace_radius = workspace_radius #draws the outline for visualisation 


        # default values are needed so __init__ sets these values that every leg 
        # starts with
        #regardless of what is passed in 
        self.StanceAmp = 0.3 
        self.StanceStep = 0.02 #andante
        self.SwingStep = 0.03 #allegro
        self.GroundContact = True #Fermata (hold if true) 
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

    """
    def update method to elicit stepwise sharps # AEPSharp and PEPSharp 
    1) check if leg should switch stance depending on disttoAEP or PEP <= stance or swing step , 2) update tarsus position depending on groundcontact True or False and initiate movement by adding norm vectors * stancestep or swingstep to tarsus position

    """
    def update(self): 
        if self.DistToPEP <= self.StanceStep and self.GroundContact: #Fermata
            self.GroundContact = False #switch to swing and then add normAEP x swing to tarsus
        elif self.DistToAEP <= self.SwingStep and not self.GroundContact: #Tacet
            self.GroundContact = True #switch to stance and then add normPEP x stance to tarsus 
        if self.GroundContact:
            self.TarsusPosition += self.VectorToPEP_normal * self.StanceStep
        else:
            self.TarsusPosition += self.VectorToAEP_normal * self.SwingStep

# CODA :|| rhythmic leg ostinato 
# if name = main keeps testing block tied to this hexleg code ,  wont be run if another script calls it
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
    leg.update()
    print(leg.TarsusPosition)
    print(leg.GroundContact)
# ||: for each leg 

    """
    1
L1 = label
[0.  0.4 . 0] Tarsus position
[0.   0.55 0.  ] AEP
[0.   0.25 0.  ] PEP
0.15000000000000002 dist to PEP
0.15000000000000002 dist to AEP
[ 0.   -0.15  0.  ] raw dist to PEP vector
[0.   0.15 0.  ] raw dist to AEP vector
[ 0. -1.  0.] PEP vector normalized for constant stance step size (dir independent of dist)
[0. 1. 0.] AEP vector nornalized for constant swing step size (dir independent of dist)
[0.   0.38 0.  ] updated tarsus position after update method 
True updated ground conhtact status 

above = 1 iteration tarsus position moved from 0.4 iniital to 0.38 ie 0.02 stance step backwards towards pep and therefore ground contact = true because dist to pep has 0.13 units to go before switching to swing and updating tarsus position with normAEP * swing step 

therefore test works and leg class functions as intended 

"""