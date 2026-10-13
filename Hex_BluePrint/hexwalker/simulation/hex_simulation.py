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
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button 
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
        #update tarsus markers with ground contact color scheme switching 
        self.tarsus_markers = [
            self.ax.plot([], [], 'o', markersize=8, 
                     markerfacecolor='#00e64d', markeredgecolor='black')[0] 
            for _ in range(6)
        ]
    
        # leg segments from origin to tarsus
        self.leg_segments = [
            self.ax.plot([], [], '-', color='#333333', linewidth=1.5)[0] 
            for _ in range(6)
        ]

        #body outline 
        self.body_line, = self.ax.plot([], [], '-', color='#888888', linewidth=1.5)

        #platonic anterior extreme position (aep) markets as red filled circles
        self.platonic_aep_markers = [
            self.ax.plot([], [], 'o', markerfacecolor='white' ,markeredgecolor='#e8000b', markersize=8)[0]
            for _ in range(6)
        ]

        #platonic posterior extreme position (pep) markers as empty squares 
        self.platonic_pep_markers = [
            self.ax.plot([], [], 's', markerfacecolor='white',
                         markeredgecolor='#e8000b', markersize=6)[0]
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

        # noisy AEP and PEP distribution clouds 
        self.aep_scatter = [
            self.ax.plot([], [], 'o', color='#e8000b', markersize=2, alpha=0.4)[0]
            for _ in range(6)
        ]
        self.pep_scatter = [
            self.ax.plot([], [], 's', color='#3B8BD4', markersize=2, alpha=0.4)[0]
            for _ in range(6)
        ]
        self.aep_ellipses = [
            Ellipse((0, 0), width=0.1, height=0.1, angle=0,
                fill=False, edgecolor='#e8000b', linewidth=1.5, alpha=0.8)
            for _ in range(6)
        ]
        self.pep_ellipses = [
        Ellipse((0, 0), width=0.1, height=0.1, angle=0,
            fill=False, edgecolor='#3B8BD4', linewidth=1.5, alpha=0.8)
        for _ in range(6)
        ]
        for e in self.aep_ellipses + self.pep_ellipses:
            self.ax.add_patch(e)

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
            [[0, 0]], closed=True, fill=False,
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
        self.noise_slider = Slider(slider_ax_noise, 'Noise', 0.0, 0.02, valinit=0.0)
        self.noise_slider.on_changed(self._noise_slider_callback)

        slider_ax_curve = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.curve_slider = Slider(slider_ax_curve, 'Curve radius', 0.0, 5.0, valinit=0.0)
        self.curve_slider.on_changed(self._curve_slider_callback)

        slider_ax_zoom = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.zoom_slider = Slider(slider_ax_zoom, 'Zoom', 0.5, 5.0, valinit=1.0)
        self.zoom_slider.on_changed(self._zoom_slider_callback) 

        button_ax = plt.axes([0.85, 0.15, 0.1, 0.04])
        self.ellipse_toggle = Button(button_ax, 'Ellipses: ON')
        self.ellipse_toggle.on_clicked(self._toggle_ellipses)
        self.show_ellipses = True

    def _noise_slider_callback(self, val: float) -> None:
        """Update noise level on all legs when slider changes."""
        self.walker.set_noise_level(val)

    def _curve_slider_callback(self, val: float) -> None:
        print('curve slider called with:', val)
        """Update curve radius when slider changes."""
        self.walker.set_curve_radius(val)

    def _zoom_slider_callback(self, val: float) -> None:
        """Zoom in around the walker."""
        half = (self.world_size / 2) / val
        bx, by = self.walker.body_position[0], self.walker.body_position[1]
        self.ax.set_xlim(bx - half, bx + half)
        self.ax.set_ylim(by - half, by + half)

    def _toggle_ellipses(self, event) -> None:
        """Toggle aep/pep distribution visibility"""
        self.show_ellipses = not self.show_ellipses 
        label = 'Ellipses: On' if self.show_ellipses else 'Ellipses:off'
        self.ellipse_toggle.label.set_text(label)
        for e in self.aep_ellipses + self.pep_ellipses:
            e.set_visible(self.show_ellipses)   

    def animate(self, frame: int) -> list:
        """Update all visual elements for one timestep dt."""

        self.walker.update(self.dt)
        body_pos = self.walker.body_position

        # toroidal wrapping
        half = self.world_size / 2
        self.walker.body_position[0] = (self.walker.body_position[0] + half) % self.world_size - half
        self.walker.body_position[1] = (self.walker.body_position[1] + half) % self.world_size - half
        self.walker.body_frame.set_transform(
            self.walker.body_position,
            angle=self.walker.body_orientation,
            scale=self.body_scale
        )

        # all leg geometry in the same loop
        geometries = self.walker.get_world_geometry()
        for i, geom in enumerate(geometries):
            colour = '#e8000b' if geom.ground_contact else '#00e64d'
            self.tarsus_markers[i].set_markerfacecolor(colour)
            self.tarsus_markers[i].set_data([geom.tarsus[0]], [geom.tarsus[1]])
            self.leg_segments[i].set_data(
                [geom.origin[0], geom.tarsus[0]],
                [geom.origin[1], geom.tarsus[1]]
            )
            self.platonic_aep_markers[i].set_data([geom.anterior_extreme[0]], [geom.anterior_extreme[1]])
            self.platonic_pep_markers[i].set_data([geom.posterior_extreme[0]], [geom.posterior_extreme[1]])
            self.noisy_aep_markers[i].set_data([geom.noisy_anterior_extreme[0]], [geom.noisy_anterior_extreme[1]])
            self.noisy_pep_markers[i].set_data([geom.noisy_posterior_extreme[0]], [geom.noisy_posterior_extreme[1]])
            
            # fit and draw ellipses once enough data exists
            if len(geom.landing_position_history) >= 2:
                aep_arr = np.array([
                    self.walker.legs[i].frame.to_world(
                        np.append(p[:2] + self.walker.legs[i].workspace_centre[:2], 0.0))[:2]
                    for p in geom.landing_position_history
                ])
                self.aep_scatter[i].set_data(aep_arr[:, 0], aep_arr[:, 1])
    
                if self.show_ellipses and len(aep_arr) >= 5:
                    mean = aep_arr.mean(axis=0)
                    cov = np.cov(aep_arr.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                    width, height = 2 * np.sqrt(np.abs(eigenvalues)) 
                    self.aep_ellipses[i].set_center(mean)
                    self.aep_ellipses[i].set_width(width)
                    self.aep_ellipses[i].set_height(height)
                    self.aep_ellipses[i].angle = angle

            if self.show_ellipses and len(geom.liftoff_position_history) >= 5:
                pep_arr = np.array([
                    self.walker.legs[i].frame.to_world(
                        np.append(p[:2] + self.walker.legs[i].workspace_centre[:2], 0.0))[:2]
                    for p in geom.liftoff_position_history
                ])
                self.pep_scatter[i].set_data(pep_arr[:, 0],pep_arr[:, 1])

                if self.show_ellipses and len(pep_arr) >= 5:
                    mean = pep_arr.mean(axis=0)
                    cov = np.cov(pep_arr.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                    width, height = 2 * np.sqrt(np.abs(eigenvalues)) 
                    self.pep_ellipses[i].set_center(mean)
                    self.pep_ellipses[i].set_width(width)
                    self.pep_ellipses[i].set_height(height)
                    self.pep_ellipses[i].angle = angle
            self.workspace_circles[i].center = (geom.workspace_centre[0], geom.workspace_centre[1])
            self.workspace_circles[i].radius = geom.workspace_radius

        # body outline
        outline_local = np.column_stack([
            self.insect_body.normalised_outline,
            np.zeros(len(self.insect_body.normalised_outline))
        ])
        outline_world = np.array([
            self.walker.body_frame.to_world(pt) for pt in outline_local
        ])
        self.body_line.set_data(outline_world[:, 0], outline_world[:, 1])

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
                self.aep_scatter + self.pep_scatter +
                [self.body_line, self.trajectory_line,
                self.body_x_axis, self.body_y_axis])

    def run(self) -> None:
        """Start playing all of the above"""
        self.anim = FuncAnimation(
            self.fig, self.animate, frames=2000, interval=20, blit=False
        )
        plt.show()

if __name__ == "__main__":
    sim = HexSimulation(world_size=40,dt=0.05,body_scale=9.0)
    sim.run()
        













































