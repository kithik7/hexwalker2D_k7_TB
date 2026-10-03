"""
da capo
"""
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os 
from HexWalker import Walker

class Hex_Sim: 
    def __init__(self, world_size, dt, body_scale):
        self.world_size = world_size
        self.dt = dt
        self.body_scale = body_scale
        
        self.walker = Walker(1, 'Klaus', np.array([0.0, 0.0, 0.0]), 0.0, body_scale=6.0, reach_factor=0.3)
        self.fig, self.ax = plt.subplots(figsize=(12,12))
        self.ax.set_xlim(-self.world_size/2, self.world_size/2)
        self.ax.set_ylim(-self.world_size/2, self.world_size/2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f0')
        self.ax.grid(True, alpha=0.2)
        self._setup_painter()
        #load body_only_outline.npy
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.body_only = np.load(os.path.join(base_dir, 'body_only_outline.npy')) * self.body_scale
    """
    all matplotlib objects animated in my simulation is now a method called internally by the class
    it comprises 6 tarsus dots, six leg lines, body outline, trajectory line 

    """
    def _setup_painter(self): #underscore before setup implies the method is only internal.class 
        #tarsus dots 
        self.tarsus_dots = []
        for i in range(6):
            dot, = self.ax.plot([], [], 'o', color='#e8000b', markersize=8)
            self.tarsus_dots.append(dot)

            #leg lines
        self.leg_lines = []
        for i in range(6): 
            line, = self.ax.plot([], [], "-", color='#7a7a8a', linewidth=2.0)
            self.leg_lines.append(line)

        #body outline 
        self.body_line, = self.ax.plot([], [], '-', color='#4a4a6a', linewidth=1.5)

        #trajectory 
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_line, = self.ax.plot([], [], '-', color='#3888D4', linewidth=1.0, alpha=0.8)


    def animate(self, frame): 
        self.walker.update()
        for leg in self.walker.legs:
            print(leg.leg_label, leg.origin_pt, leg.TarsusPosition)
        body_x = self.body_only[:, 0] + self.walker.body_pos[0]
        body_y = self.body_only[:, 1] + self.walker.body_pos[1]
        self.body_line.set_data(body_x, body_y)
    
        for i, leg in enumerate(self.walker.legs):
            world_x = leg.TarsusPosition[0] + self.walker.body_pos[0]
            world_y = leg.TarsusPosition[1] + self.walker.body_pos[1]
            self.tarsus_dots[i].set_data([world_x], [world_y])
            ox = leg.origin_pt[0] + self.walker.body_pos[0]
            oy = leg.origin_pt[1] + self.walker.body_pos[1]
            self.leg_lines[i].set_data([ox, world_x], [oy, world_y])
    
        self.trajectory_x.append(self.walker.body_pos[0])
        self.trajectory_y.append(self.walker.body_pos[1])
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
    
        return self.tarsus_dots + self.leg_lines + [self.body_line, self.trajectory_line]
        
        #now the run is also a method 
    def run(self):
        self.anim = FuncAnimation(self.fig, self.animate, frames=2000, interval=20, blit=False)
        plt.show()

#separate class from opening the animation itself with __name__ == '__main__":
if __name__ == "__main__":
    sim = Hex_Sim(world_size=20, dt=0.1, body_scale=6.0)
    sim.run()
    #only runs when the file is executed directly alike test blocks in class leg and walker 

            
"""
Coda
"""

"""
render version 1: static legs that didnt move with the tarsus position dots, therefore, 
she cut off its legs in extract_fly_sil and updated os.path.dir to body_only_outline
body outline sits roughly between x = -0.3 +0.3 and legs extend beyond this range"


updates
making hex_sim a class
adding body scale to HexWalker.py Walker class init , then in attach legs - multiply by the scale that i set - so the attachment points and the workspace center scaling is consistent 

workspace radius is therefore also scaled 

call walker for HexSim init which will have bodyscale 
animate is a class, run is a class, and this script is only executed when run directly


TO DO: Body scaling needs to be modular and not condition specific, attachment points and workspace centers
need to be expressed 
as fractions of the body that is defined and normalized to have a unit length of 1, and multiplying the whole
with a scaling factor will scale them all, like item.children basically 


"""