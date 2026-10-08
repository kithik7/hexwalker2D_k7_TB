"""
hex_simulation.py 

HexSimulation is a class that contains the world, the visualisation and animation loop for the hexapod walker
It managers all drawing and has public methods accesible by the user 
Has no information about HexWalkerLeg or computing parameters such as displacement
"""
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.spatial import ConvexHull 
import os 
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker import HexWalker

class HexSimulation: 
    def __init__(
            self,
            world_size: float = 20, 
            dt: float =0.05, 
            body_scale: float =8.0,
            source_path: str = None,         
    ) -> None: 
        self.world_size = world_size
        self.dt = dt
        self.body_scale = body_scale

        #default to mat file for outline if no other source provided 
        if source_path is None:
            source_path = str(Path(__file__).parent.parent / 'body' / 'flySilhouette.mat')

        self.insect_body = InsectBody(source_path)
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
        body_pos = self.walker.body_position
        half = self.world_size / 2
        self.walker.body_position[0] = (self.walker.body_position[0] + half) % self.world_size - half
        self.walker.body_position[1] = (self.walker.body_position[1] + half) % self.world_size - half
        self.walker.body_frame.set_transform(self.walker.body_position, self.walker.body_orientation, scale=self.body_scale)
    
        # tarsus markers
        for i, leg in enumerate(self.walker.legs):
            colour = '#e8000b' if leg.ground_contact else '#00e64d'
            tarsus_world = leg.frame.to_world(leg.tarsus_position)
            self.tarsus_markers[i].set_markerfacecolor(colour)
            self.tarsus_markers[i].set_data([tarsus_world[0]], [tarsus_world[1]])
    
        # leg segments
        # for i, leg in enumerate(self.walker.legs):
        #     origin_world = leg.frame.to_world(np.zeros(3))
        #     tarsus_world = leg.frame.to_world(leg.tarsus_position)
        #     self.leg_segments[i].set_data(
        #         [origin_world[0], tarsus_world[0]],
        #         [origin_world[1], tarsus_world[1]]
        #     )
        for i, leg in enumerate(self.walker.legs):
            colour = '#e8000b' if leg.ground_contact else '#00e64d'
            tarsus_world = leg.frame.to_world(leg.tarsus_position)
            if i == 0:
                print("leg world transform: \n", leg.frame.world_transform())
                print("body frame local:\n", self.walker.body_frame.local_transform)
                print("body frame parent:", self.walker.body_frame.parent)
            self.tarsus_markers[i].set_markerfacecolor(colour)
            self.tarsus_markers[i].set_data([tarsus_world[0]], [tarsus_world[1]])
    
        # body outline
        outline_local = np.column_stack([
            self.insect_body.normalised_outline,
            np.zeros(len(self.insect_body.normalised_outline))
        ])
        outline_world = np.array([
            self.walker.body_frame.to_world(pt) for pt in outline_local
        ])
        self.body_line.set_data(outline_world[:, 0], outline_world[:, 1])
        
        # AEP and PEP markers
        for i, leg in enumerate(self.walker.legs):
            aep_world = leg.frame.to_world(leg.anterior_extreme_position)
            pep_world = leg.frame.to_world(leg.posterior_extreme_position)
            self.platonic_aep_markers[i].set_data([aep_world[0]], [aep_world[1]])
            self.platonic_pep_markers[i].set_data([pep_world[0]], [pep_world[1]])
    
        # noisy AEP and PEP markers
        for i, leg in enumerate(self.walker.legs):
            noisy_aep_world = leg.frame.to_world(leg.noisy_anterior_extreme_position)
            noisy_pep_world = leg.frame.to_world(leg.noisy_posterior_extreme_position)
            self.noisy_aep_markers[i].set_data([noisy_aep_world[0]], [noisy_aep_world[1]])
            self.noisy_pep_markers[i].set_data([noisy_pep_world[0]], [noisy_pep_world[1]])
    
        # workspace circles
        for i, leg in enumerate(self.walker.legs):
            centre_world = leg.frame.to_world(leg.workspace_centre)
            self.workspace_circles[i].center = (centre_world[0], centre_world[1])
            self.workspace_circles[i].radius = leg.workspace_radius * self.body_scale
    
        # stability polygon
        grounded = self.walker.get_support_polygon_grounded_positions()
        if len(grounded) >= 3:
            hull = ConvexHull(grounded[:, :2])
            self.stability_polygon.set_xy(grounded[hull.vertices, :2])
    
        # body coordinate axes
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
    
        return (self.tarsus_markers + self.leg_segments +
                self.platonic_aep_markers + self.platonic_pep_markers +
                self.noisy_aep_markers + self.noisy_pep_markers +
                [self.body_line, self.trajectory_line,
                self.body_x_axis, self.body_y_axis])

    def run(self) -> None:
        """Start playing all of the above"""
        self.anim = FuncAnimation(
            self.fig, self.animate, frames=2000, interval=20, blit=False
        )
        plt.show()

if __name__ == "__main__":
    sim = HexSimulation(world_size=20,dt=0.5,body_scale=6.0)
    sim.run()
        













































