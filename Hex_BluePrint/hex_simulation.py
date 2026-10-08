"""
hex_simulation.py 

HexSimulation is a class that contains the world, the visualisation and animation loop for the hexapod walker
It managers all drawing and has public methods accesible by the user 
Has no information about HexWalkerLeg or computing parameters such as displacement
"""
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.spatial import ConvexHull 
import os 
from insect_body import InsectBody
from hex_walker import HexWalker

class HexSimulation: 
    def __init__(
            self,
            world_size: float = 20, 
            dt: float =0.05, 
            body_scale: float =8.0, 
            
    ) -> None: 
        self.world_size = world_size
        self.dt = dt
        self.body_scale = body_scale
        print(f"body_scale: {self.body_scale}")
        self.insect_body = InsectBody('Hex_BluePrint/flySilhouette.mat')
        self.walker = HexWalker(
            walker_id=1,
            insect_body=self.insect_body,
            body_position=np.array([0.0, 0.0, 0.0]), 
            body_orientation=0.0, 
            dt=self.dt,
            body_scale=self.body_scale
        )
        for i, leg in enumerate(self.walker.legs):
            print(i, leg.leg_label, leg.origin[:2])

        #create figure and its axes 
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.ax.set_xlim(-self.world_size / 2, self.world_size / 2)
        self.ax.set_ylim(-self.world_size / 2, self.world_size / 2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f0')
        self.ax.grid(True, alpha= 0.2)

        plt.subplots_adjust(bottom=0.25) #scale layout to fit parameter slides

        #store x and y trajectories 
        self.trajectory_x = []
        self.trajectory_y = []

        self._setup_simulation_painter()
        
        
        

    def _setup_simulation_painter(self) -> None:
        """Create all matplotlib artist objects for the simulation"""
        print("setting up def)")
        #update tarsus markers with ground contact color scheme switching 
        self.tarsus_markers = [
            self.ax.plot([], [], 'o', markersize=8, 
                     markerfacecolor='#00e64d', markeredgecolor='black')[0] 
            for _ in range(6)
        ]
    
        # leg segments from origin to tarsus
        self.leg_segments = [
            self.ax.plot([], [], '-', color='#888888', linewidth=1.5)[0] 
            for _ in range(6)
        ]

        #body outline 
        self.body_line, = self.ax.plot([], [], '-', color='#888888', linewidth=1.5)

        #platonic anterior extreme position (aep) markets as red filled circles
        self.platonic_aep_markers = [
            self.ax.plot([], [], 'o', color='#e8000b', markersize=6)[0]
            for _ in range(6)
        ]

        #platonic posterior extreme position (pep) markers as empty squares 
        self.platonic_pep_markers = [
            self.ax.plot([], [], 's', markerfacecolor='white',
                         markeredgecolor='black', markersize=6)[0]
            for _ in range(6)
        ]

        #noisy AEP markers, smaller open red circles
        self.noisy_aep_markers = [
            self.ax.plot([], [], 'o', markerfacecolor='none',
                         markeredgecolor='#e8000b', markersize=4)[0]
            for _ in range(6)
        ]

        #noisy PEP markers, smaller open squares
        self.noisy_pep_markers = [
            self.ax.plot([], [], 's', markerfacecolor='none',
                         markeredgecolor='black', markersize=4)[0]
            for _ in range(6)
        ]

        #workspace circles 
        self.workspace_circles = [ 
            plt.Circle((0,0), 0.08 * self.body_scale, fill=False,
                       color='#3B8BD4', linewidth=1.5, alpha=0.6)
            for _ in range(6)

        ]
        for circle in self.workspace_circles: 
            self.ax.add_patch(circle)

        #stability polygon 
        self.stability_polygon = plt.Polygon(
            [[0, 0]], closed=True, fill=True,
            facecolor='#e8000b', alpha=0.15, edgecolor='#e8000b', linewidth=1.5
        )
        self.ax.add_patch(self.stability_polygon)

        #body coordinate axes, red x and green y 
        self.body_x_axis, = self.ax.plot([], [], '-', color='#e8000b', linewidth=2.0)
        self.body_y_axis, = self.ax.plot([], [], '-', color='#00e64d', linewidth=2.0)

        #trace trajectory line 
        self.trajectory_line, = self.ax.plot([], [], '-', color='#3B8BD4',
                                             linewidth=1.0, alpha=0.8)

        #add parameter adjustment sliders 
        slider_ax_noise = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.noise_slider = Slider(slider_ax_noise, 'Noise', 0.0, 0.5, valinit=0.0)
        self.noise_slider.on_changed(self._noise_slider_callback)

        slider_ax_curve = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.curve_slider = Slider(slider_ax_curve, 'Curve radius', 0.1, 5.0, valinit=0.1)
        self.curve_slider.on_changed(self._curve_slider_callback)

    def _noise_slider_callback(self, val: float) -> None:
        """Update noise level on all legs when slider changes."""
        self.walker.set_noise_level(val)

    def _curve_slider_callback(self, val: float) -> None:
        """Update curve radius when slider changes."""
        self.walker.set_curve_radius(val)
        
    def animate(self, frame: int) -> list:
        """Update all visual elements for one timestep dt."""
        
        self.walker.update(self.dt)
    
        #toroidal wrapping
        half = self.world_size / 2
        self.walker.body_position[0] = (self.walker.body_position[0] + half) % self.world_size - half
        self.walker.body_position[1] = (self.walker.body_position[1] + half) % self.world_size - half
    
        body_pos = self.walker.body_position
        tarsus_positions = self.walker.get_tarsus_positions()
    
        #tarsus markers with ground contact colour switching
        for i, leg in enumerate(self.walker.legs):
            colour = '#e8000b' if leg.ground_contact else '#00e64d'
            self.tarsus_markers[i].set_markerfacecolor(colour)
            self.tarsus_markers[i].set_data([tarsus_positions[i][0]], [tarsus_positions[i][1]])
    
        #leg segments
        for i, leg in enumerate(self.walker.legs):
            ox = leg.origin[0] + body_pos[0]
            oy = leg.origin[1] + body_pos[1]
            self.leg_segments[i].set_data([ox, tarsus_positions[i][0]], [oy, tarsus_positions[i][1]])
    
        #body outline
        outline = self.insect_body.normalised_outline * self.body_scale + body_pos[:2]
        self.body_line.set_data(outline[:, 0], outline[:, 1])
    
        #platonic aep pep markers 
        for i, leg in enumerate(self.walker.legs):
            self.platonic_aep_markers[i].set_data(
                [leg.anterior_extreme_position[0]], 
                [leg.anterior_extreme_position[1]]
            )
            self.platonic_pep_markers[i].set_data(
                [leg.posterior_extreme_position[0]], 
                [leg.posterior_extreme_position[1]]
            )

        # noisy AEP and PEP markers
        for i, leg in enumerate(self.walker.legs):
            self.noisy_aep_markers[i].set_data(
                [leg.noisy_anterior_extreme_position[0]],
                [leg.noisy_anterior_extreme_position[1]]
        )
            self.noisy_pep_markers[i].set_data(
                [leg.noisy_posterior_extreme_position[0]],
                [leg.noisy_posterior_extreme_position[1]]
        )
    
        #workspace circles
        for i, leg in enumerate(self.walker.legs):
            cx = leg.workspace_centre[0] + body_pos[0]
            cy = leg.workspace_centre[1] + body_pos[1]
            self.workspace_circles[i].center = (cx, cy)
            self.workspace_circles[i].radius = leg.workspace_radius * self.body_scale
    
        #stability polygon
        grounded = self.walker.get_support_polygon_grounded_positions()
        if len(grounded) >= 3:
            hull = ConvexHull(grounded[:, :2])
            self.stability_polygon.set_xy(grounded[hull.vertices, :2])
    
        #body coordinate axes
        angle = self.walker.body_orientation
        axis_length = 0.5 * self.body_scale
        self.body_x_axis.set_data(
            [body_pos[0], body_pos[0] + axis_length * np.cos(angle)],
            [body_pos[1], body_pos[1] + axis_length * np.sin(angle)]
        )
        self.body_y_axis.set_data(
            [body_pos[0], body_pos[0] + axis_length * np.cos(angle + np.pi / 2)],
            [body_pos[1], body_pos[1] + axis_length * np.sin(angle + np.pi / 2)]
        )
    
        # trajectory
        self.trajectory_x.append(body_pos[0])
        self.trajectory_y.append(body_pos[1])
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
    
        return (self.tarsus_markers + self.leg_segments + self.platonic_aep_markers +
                self.platonic_pep_markers + self.noisy_aep_markers + self.noisy_pep_markers + [self.body_line, self.trajectory_line,
                self.body_x_axis, self.body_y_axis])

    def run(self) -> None:
        """Start playing all of the above"""
        self.anim = FuncAnimation(
            self.fig, self.animate, frames=2000, interval=20, blit=False
        )
        plt.show()

if __name__ == "__main__":
    sim = HexSimulation(world_size=20,dt=0.1,body_scale=6.0)
    sim.run()
        













































#         self.walker = HexWalker(1, 'Klaus', np.array([0.0, 0.0, 0.0]), 0.0, body_scale=6.0, reach_factor=0.3)
#         self.fig, self.ax = plt.subplots(figsize=(12,12))
#         self.ax.set_xlim(-self.world_size/2, self.world_size/2)
#         self.ax.set_ylim(-self.world_size/2, self.world_size/2)
#         self.ax.set_aspect('equal')
#         self.ax.set_facecolor('#f5f5f0')
#         self.ax.grid(True, alpha=0.2)
#         self._setup_painter()
#         #load body_only_outline.npy
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         self.body_only = np.load(os.path.join(base_dir, 'body_only_outline.npy')) * self.body_scale
#     """
#     all matplotlib objects animated in my simulation is now a method called internally by the class
#     it comprises 6 tarsus dots, six leg lines, body outline, trajectory line 

#     """
#     def _setup_painter(self): #underscore before setup implies the method is only internal.class 
#         #tarsus dots 
#         self.tarsus_dots = []
#         for i in range(6):
#             dot, = self.ax.plot([], [], 'o', color='#e8000b', markersize=8)
#             self.tarsus_dots.append(dot)

#             #leg lines
#         self.leg_lines = []
#         for i in range(6): 
#             line, = self.ax.plot([], [], "-", color='#7a7a8a', linewidth=2.0)
#             self.leg_lines.append(line)

#         #body outline 
#         self.body_line, = self.ax.plot([], [], '-', color='#4a4a6a', linewidth=1.5)

#         #trajectory 
#         self.trajectory_x = []
#         self.trajectory_y = []
#         self.trajectory_line, = self.ax.plot([], [], '-', color='#3888D4', linewidth=1.0, alpha=0.8)


#     def animate(self, frame): 
#         self.walker.update()
#         for leg in self.walker.legs:
#             print(leg.leg_label, leg.origin_pt, leg.TarsusPosition)
#         body_x = self.body_only[:, 0] + self.walker.body_pos[0]
#         body_y = self.body_only[:, 1] + self.walker.body_pos[1]
#         self.body_line.set_data(body_x, body_y)
    
#         for i, leg in enumerate(self.walker.legs):
#             world_x = leg.TarsusPosition[0] + self.walker.body_pos[0]
#             world_y = leg.TarsusPosition[1] + self.walker.body_pos[1]
#             self.tarsus_dots[i].set_data([world_x], [world_y])
#             ox = leg.origin_pt[0] + self.walker.body_pos[0]
#             oy = leg.origin_pt[1] + self.walker.body_pos[1]
#             self.leg_lines[i].set_data([ox, world_x], [oy, world_y])
    
#         self.trajectory_x.append(self.walker.body_pos[0])
#         self.trajectory_y.append(self.walker.body_pos[1])
#         self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
    
#         return self.tarsus_dots + self.leg_lines + [self.body_line, self.trajectory_line]
        
#         #now the run is also a method 
#     def run(self):
#         self.anim = FuncAnimation(self.fig, self.animate, frames=2000, interval=20, blit=False)
#         plt.show()

# #separate class from opening the animation itself with __name__ == '__main__":
# if __name__ == "__main__":
#     sim = Hex_Sim(world_size=20, dt=0.1, body_scale=6.0)
#     sim.run()
#     #only runs when the file is executed directly alike test blocks in class leg and walker 

            
# """
# Coda
# """

# """
# render version 1: static legs that didnt move with the tarsus position dots, therefore, 
# she cut off its legs in extract_fly_sil and updated os.path.dir to body_only_outline
# body outline sits roughly between x = -0.3 +0.3 and legs extend beyond this range"


# updates
# making hex_sim a class
# adding body scale to HexWalker.py Walker class init , then in attach legs - multiply by the scale that i set - so the attachment points and the workspace centre scaling is consistent 

# workspace radius is therefore also scaled 

# call walker for HexSim init which will have bodyscale 
# animate is a class, run is a class, and this script is only executed when run directly


# TO DO: Body scaling needs to be modular and not condition specific, attachment points and workspace centres
# need to be expressed 
# as fractions of the body that is defined and normalised to have a unit length of 1, and multiplying the whole
# with a scaling factor will scale them all, like item.children basically 


# """