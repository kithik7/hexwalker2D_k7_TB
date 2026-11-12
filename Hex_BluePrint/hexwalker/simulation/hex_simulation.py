"""
hex_simulation.py 

HexSimulation is a class that contains the world, the visualisation and animation loop for the hexapod walker
It manages all drawing and has public methods accesible by the user 
Has no information about the legs or computing parameters such as displacement
"""
from matplotlib.widgets import RadioButtons, TextBox
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button 
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker import HexWalker
from hexwalker.analysis.recorder import Recorder
from hexwalker.analysis.run import RUNS_DIR, free_path
from hexwalker.modes import MODES

class HexSimulation: 
    def __init__(
            self,
            world_size: float = 20, 
            dt: float =0.05, 
            body_scale: float =8.0,
            source_path: str = None,
            mode: str = "phase",         
    ) -> None: 
        self.world_size = world_size
        self.dt = dt
        self.body_scale = body_scale

        #default to mat file for outline if no other source provided 
        if source_path is None:
            source_path = str(Path(__file__).parent.parent / 'body' / 'flySilhouette.mat')

        self.insect_body = InsectBody(source_path)
        self.baseline_noise = 0.005
        self.walker = None
        self.recorder = None
        self.paused = True
        self.tick_count = 0
        #create figure and its axes 
        self.fig = plt.figure(figsize=(14, 8))
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.3, 1])
        self.ax = self.fig.add_subplot(grid[:, 0])
        self.traj_ax = self.fig.add_subplot(grid[0, 1])
        self.footfall_ax = self.fig.add_subplot(grid[1, 1])
        self.ax.set_xlim(-self.world_size / 2, self.world_size / 2)
        self.ax.set_ylim(-self.world_size / 2, self.world_size / 2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f0')
        self.ax.grid(True, alpha= 0.2)
        self.ax.set_autoscale_on(False)
        self.ax.set_xlabel(f'world units (1 body length = {self.body_scale:g})', fontsize=8)
        self.traj_ax.set_xlabel(f'world units (1 body length = {self.body_scale:g})', fontsize=8)
        plt.subplots_adjust(bottom=0.33, top=0.91, hspace=0.3)

        #store x and y trajectories 
        self.trajectory_x = []
        self.trajectory_y = []

        self.unwrapped_x = []
        self.unwrapped_y = []
        self.wrap_offset = np.zeros(2)
        self.map_unwrapped = False

        #history table for footfall
        self.footfall_length = 100
        self.footfall_buffer = np.zeros((11, self.footfall_length))
        self.footfall_order = [0, 1, 2, 3, 4, 5] 
        self.footfall_black_is_stance = False

        self._setup_simulation_painter()       
        self._build_walker(mode)

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

        #platonic anterior extreme position (aep) markets as red open circles
        self.platonic_aep_markers = [
            self.ax.plot([], [], 'o', markerfacecolor='white' ,markeredgecolor='#e8000b', markersize=8)[0]
            for _ in range(6)
        ]

        #platonic posterior extreme position (pep) markers as empty squares 
        self.platonic_pep_markers = [
            self.ax.plot([], [], 's', markerfacecolor='white',
                         markeredgecolor='black', markersize=6)[0]
            for _ in range(6)
        ]

        #noisy AEP markers, smaller filled red circles
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
                fill=False, edgecolor='#e8000b', linewidth=1.5, alpha=0.8, visible=False)
            for _ in range(6)
        ]
        self.pep_ellipses = [
        Ellipse((0, 0), width=0.1, height=0.1, angle=0,
            fill=False, edgecolor='#3B8BD4', linewidth=1.5, alpha=0.8, visible=False)
        for _ in range(6)
        ]
        for e in self.aep_ellipses + self.pep_ellipses:
            self.ax.add_patch(e)

        #workspace circles 
        self.workspace_circles = [ 
            plt.Circle((0,0), 0.06, fill=False,
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

        #body coordinate axes, red x and greegit add n y 
        self.body_x_axis, = self.ax.plot([], [], '-', color='#e8000b', linewidth=2.0)
        self.body_y_axis, = self.ax.plot([], [], '-', color='#00e64d', linewidth=2.0)

        #trace trajectory line 
        self.trajectory_line, = self.ax.plot([], [], '-', color='#3B8BD4',
                                             linewidth=1.0, alpha=0.8)
        # minimap trajectory view
        self.traj_ax.set_aspect('equal')
        self.traj_ax.set_facecolor('#f5f5f0')
        self.traj_ax.grid(True, alpha=0.2)
        self.traj_ax.set_title('Trajectory Map', fontsize=9)
        self.traj_ax.set_xlim(-self.world_size / 2, self.world_size / 2)
        self.traj_ax.set_ylim(-self.world_size / 2, self.world_size / 2)

        self.minimap_line, = self.traj_ax.plot([], [], '-', color='#3B8BD4', linewidth=1.0, alpha=0.8)
        self.minimap_dot, = self.traj_ax.plot([], [], 'o', color='#e8000b', markersize=6)

        #add footfall image 
        #add footfall image 
        self.footfall_image = self.footfall_ax.imshow(
            self.footfall_buffer, aspect='auto', interpolation='nearest',
            cmap=ListedColormap(['white', 'black']), vmin=0, vmax=1,
            extent=(0, self.footfall_length, 10.5, -0.5))
        self.footfall_ax.set_yticks(range(0, 11, 2))
        self.footfall_ax.set_yticklabels(['L1', 'L2', 'L3', 'R1', 'R2', 'R3'], fontsize=8)
        self.footfall_ax.set_xlabel('Time [frames]', fontsize=8)
        self.footfall_ax.tick_params(axis='x', labelsize=8)
        self.footfall_ax.spines[['top', 'right', 'left']].set_visible(False)
        for k in range(6):
            self.footfall_ax.axhspan(2 * k - 0.5, 2 * k + 0.5, facecolor='none', edgecolor='black', linewidth=0.8)

        #add parameter adjustment sliders 

        slider_ax_speed = plt.axes([0.2, 0.09, 0.3, 0.03]) #
        self.speed_slider = Slider(slider_ax_speed, 'Speed', 0.005, 0.05, valinit=0.02) #speed capped
        self.speed_slider.on_changed(self._speed_slider_callback)

        slider_ax_noise = plt.axes([0.2, 0.14, 0.3, 0.03])
        self.noise_slider = Slider(slider_ax_noise, 'Noise', 0.0, 0.05, valinit=self.baseline_noise, valfmt='%.3f', valstep=0.001)
        self.noise_slider.on_changed(self._noise_slider_callback)

        slider_ax_curve = plt.axes([0.2, 0.19, 0.3, 0.03])
        self.curve_slider = Slider(slider_ax_curve, 'Curve radius', 0.0, 0.3, valinit=0.0, valfmt='%.3f', valstep=0.005)
        self.curve_slider.on_changed(self._curve_slider_callback)

        slider_ax_zoom = plt.axes([0.2, 0.24, 0.3, 0.03])
        self.zoom_slider = Slider(slider_ax_zoom, 'Zoom', 0.5, 5.0, valinit=2.5)
        self.zoom_slider.on_changed(self._zoom_slider_callback) 

        self.strength_sliders = {}
        self.mode_label = self.fig.text(0.99, 0.98, "", ha='right', va='top',
                                        fontsize=10)
        defaults_ax = plt.axes([0.35, 0.03, 0.15, 0.04])
        self.defaults_button = Button(defaults_ax, 'Default')
        self.defaults_button.on_clicked(self._apply_default_strengths)
        map_button_ax = plt.axes([0.52, 0.03, 0.11, 0.04])
        self.map_button = Button(map_button_ax, 'Map: wrapped')
        self.map_button.on_clicked(self._toggle_map)
        button_ax = plt.axes([0.2, 0.03, 0.13, 0.04])
        self.ellipse_toggle = Button(button_ax, 'Ellipses: OFF')
        self.ellipse_toggle.on_clicked(self._toggle_ellipses)
        self.show_ellipses = False

        #saving buttons 
        name_ax = plt.axes([0.13, 0.94, 0.15, 0.04])
        self.run_name_box = TextBox(name_ax, '', initial='Run name')
        save_ax = plt.axes([0.29, 0.94, 0.08, 0.04])
        self.save_button = Button(save_ax, 'Save run')
        self.save_button.on_clicked(self._save_run)

            # mode choice and play / pause
        mode_ax = plt.axes([0.42, 0.915, 0.08, 0.075])
        mode_names = list(MODES)
        self.mode_buttons = RadioButtons(mode_ax, mode_names)
        self.mode_buttons.on_clicked(self._on_mode_selected)
        play_ax = plt.axes([0.52, 0.94, 0.08, 0.04])
        self.play_button = Button(play_ax, 'Play')
        self.play_button.on_clicked(self._toggle_play)

    def _build_walker(self, mode: str) -> None:
        """Fresh walker for this mode. Resets traces, recording, sliders."""
        self.walker = HexWalker(
            walker_id=1,
            insect_body=self.insect_body,
            body_position=np.array([0.0, 0.0, 0.0]),
            body_orientation=0.0,
            dt=self.dt,
            body_scale=self.body_scale,
            random_start_phases=False,
            mode=mode,
        )
        self.walker.set_noise_level(self.noise_slider.val)
        self.walker.set_speed(self.speed_slider.val)
        self.walker.set_curve_radius(self.curve_slider.val)
        self.recorder = Recorder(self.walker, seed=None)
        self.recorder.record()
        self._reset_traces()
        self._build_strength_sliders()
        self.mode_buttons.set_active(list(MODES).index(mode))
        self.paused = True
        self._update_status()

    def _reset_traces(self) -> None:
        """Clear everything drawn from the previous walker."""
        self.tick_count = 0
        self.trajectory_x, self.trajectory_y = [], []
        self.unwrapped_x, self.unwrapped_y = [], []
        self.wrap_offset = np.zeros(2)
        self.footfall_buffer[:] = 0
        for cloud in self.aep_scatter + self.pep_scatter:
            cloud.set_data([], [])
        for ellipse in self.aep_ellipses + self.pep_ellipses:
            ellipse.set_width(0)
            ellipse.set_height(0)

    def _build_strength_sliders(self) -> None:
        #One slider per strength of the current mode; old ones removed 
        for slider in self.strength_sliders.values():
            slider.ax.remove()
        self.strength_sliders = {}
        coupling = self.walker.coupling
        n_sliders = len(coupling.strength_specs)
        gap = min(0.045, 0.21 / max(n_sliders - 1, 1))  # fit any number
        for row, (name, spec) in enumerate(coupling.strength_specs.items()):
            label, low, high, step = spec
            slider_ax = plt.axes([0.68, 0.25 - gap * row, 0.22, 0.025])
            slider = Slider(
                slider_ax, label, low, high,
                valinit=coupling.strengths[name],
                valfmt='%.3f', valstep=step,
            )
            slider.on_changed(self._make_strength_callback(name))
            self.strength_sliders[name] = slider

    def _on_mode_selected(self, mode: str) -> None:
        """Switching mode starts a fresh run; the old one is saved first."""
        if self.walker is None or mode == self.walker.mode.name:
            return
        self._autosave()
        self._build_walker(mode)

    def _autosave(self) -> None:
        if len(self.recorder.ticks) <= 1:
            return
        RUNS_DIR.mkdir(exist_ok=True)
        path = free_path(f"autosave_{self.walker.mode.name}")
        self.recorder.save(path)
        self.recorder.save_csv(path)
        print("autosaved", len(self.recorder.ticks), "ticks to", path.name)

    def _toggle_play(self, event) -> None:
        self.paused = not self.paused
        self._update_status()

    def _update_status(self) -> None:
        self.play_button.label.set_text('Play' if self.paused else 'Pause')
        state = "paused" if self.paused else "running"
        self.mode_label.set_text(f"mode: {self.walker.mode.name} ({state})")

    def _save_run(self, event) -> None:
        name = self.run_name_box.text.strip() or "unnamed"
        RUNS_DIR.mkdir(exist_ok=True)
        path = free_path(name)
        self.recorder.save(path)
        self.recorder.save_csv(path)
        print("saved", len(self.recorder.ticks), "ticks to", path.name)
    
    def _speed_slider_callback(self, val: float) -> None:
        self.walker.set_speed(val)

    def _noise_slider_callback(self, val: float) -> None:
        """Update noise level on all legs when slider changes."""
        self.walker.set_noise_level(val)

    def _curve_slider_callback(self, val: float) -> None:
        """Update curve radius when slider changes."""
        self.walker.set_curve_radius(val)

    def _zoom_slider_callback(self, val: float) -> None:
        """Zoom in around the walker."""
        half = (self.world_size / 2) / val
        bx, by = self.walker.body_position[0], self.walker.body_position[1]
        self.ax.set_xlim(bx - half, bx + half)
        self.ax.set_ylim(by - half, by + half)

    def _make_strength_callback(self, name: str):
        def callback(val: float) -> None:
            self.walker.set_strength(name, val)
        return callback

    def _apply_default_strengths(self, event) -> None:
        for name, value in self.walker.default_strengths.items():
            self.strength_sliders[name].set_val(value)

    def _toggle_map(self, event) -> None:
        self.map_unwrapped = not self.map_unwrapped
        self.map_button.label.set_text('Map: unwrapped' if self.map_unwrapped else 'Map: wrapped')
        if not self.map_unwrapped:
            half = self.world_size / 2
            self.traj_ax.set_xlim(-half, half)
            self.traj_ax.set_ylim(-half, half)

    def _fit_map_to_path(self) -> None:
        x_low, x_high = min(self.unwrapped_x), max(self.unwrapped_x)
        y_low, y_high = min(self.unwrapped_y), max(self.unwrapped_y)
        half_span = max(x_high - x_low, y_high - y_low, 10.0) / 2 * 1.1
        centre_x, centre_y = (x_low + x_high) / 2, (y_low + y_high) / 2
        self.traj_ax.set_xlim(centre_x - half_span, centre_x + half_span)
        self.traj_ax.set_ylim(centre_y - half_span, centre_y + half_span)

    def _toggle_ellipses(self, event) -> None:
        """Toggle aep/pep distribution visibility"""
        self.show_ellipses = not self.show_ellipses 
        label = 'Ellipses: ON' if self.show_ellipses else 'Ellipses: OFF'
        self.ellipse_toggle.label.set_text(label)
        for e in self.aep_ellipses + self.pep_ellipses:
            e.set_visible(self.show_ellipses)   


    def animate(self, frame: int) -> list:
        """Update all visual elements for one timestep dt."""

        if self.paused:
            return []
        self.tick_count += 1
        frame = self.tick_count
        self.walker.update(self.dt)
        self.recorder.record()
        body_pos = self.walker.body_position.copy()
        # all leg geometry in the same loop
        geometries = self.walker.get_world_geometry()

        #update footfall each frame 
        self.footfall_buffer = np.roll(self.footfall_buffer, -1, axis=1)
        self.footfall_buffer[0::2, -1] = [geometries[i].ground_contact == self.footfall_black_is_stance for i in self.footfall_order]
        self.footfall_image.set_extent((frame - self.footfall_length + 1, frame + 1, 10.5, -0.5))
        self.footfall_ax.set_xlim(max(0, frame - self.footfall_length + 1), max(self.footfall_length, frame + 1))
        self.footfall_image.set_data(self.footfall_buffer)

    
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
            if self.walker.legs[0].noise_level > 0:
                self.noisy_aep_markers[i].set_data([geom.noisy_anterior_extreme[0]], [geom.noisy_anterior_extreme[1]])
                self.noisy_pep_markers[i].set_data([geom.noisy_posterior_extreme[0]], [geom.noisy_posterior_extreme[1]])
            else:
                self.noisy_aep_markers[i].set_data([], [])
                self.noisy_pep_markers[i].set_data([], [])
            
            # fit and draw ellipses once enough data exists
            if len(geom.landing_position_history) >= 2:
                aep_arr = np.array([
                    self.walker.legs[i].frame.to_world(
                        np.append(p[:2] + self.walker.legs[i].workspace_centre[:2], 0.0))[:2]
                    for p in geom.landing_position_history[-20:]
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

            if len(geom.liftoff_position_history) >= 2:
                pep_arr = np.array([
                    self.walker.legs[i].frame.to_world(
                        np.append(p[:2] + self.walker.legs[i].workspace_centre[:2], 0.0))[:2]
                    for p in geom.liftoff_position_history[-20:]
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
        # store position before wrapping
        pre_toroid_wrapping_position = self.walker.body_position[:2].copy()
        self.unwrapped_x.append(pre_toroid_wrapping_position[0] + self.wrap_offset[0])
        self.unwrapped_y.append(pre_toroid_wrapping_position[1] + self.wrap_offset[1])

        # toroidal wrapping
        half = self.world_size / 2
        self.walker.body_position[0] = (self.walker.body_position[0] + half) % self.world_size - half
        self.walker.body_position[1] = (self.walker.body_position[1] + half) % self.world_size - half
        self.wrap_offset += pre_toroid_wrapping_position - self.walker.body_position[:2]

        self.walker.body_frame.set_transform(
            self.walker.body_position,
            angle=self.walker.body_orientation,
            scale=self.body_scale
        )

        # camera follow
        half_view = (self.world_size / 2) / self.zoom_slider.val
        self.ax.set_xlim(self.walker.body_position[0] - half_view, self.walker.body_position[0] + half_view)
        self.ax.set_ylim(self.walker.body_position[1] - half_view, self.walker.body_position[1] + half_view)

        # trajectory
        wrap_occured = (
            abs(self.walker.body_position[0] - pre_toroid_wrapping_position[0]) > half or
            abs(self.walker.body_position[1] - pre_toroid_wrapping_position[1]) > half
        )
        self.trajectory_x.append(body_pos[0])
        self.trajectory_y.append(body_pos[1])

        if wrap_occured:
            self.trajectory_x.append(np.nan)
            self.trajectory_y.append(np.nan)
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
        if self.map_unwrapped:
            self.minimap_line.set_data(self.unwrapped_x, self.unwrapped_y)
            self.minimap_dot.set_data([self.unwrapped_x[-1]], [self.unwrapped_y[-1]])
            self._fit_map_to_path()
        else:
            self.minimap_line.set_data(self.trajectory_x, self.trajectory_y)
            self.minimap_dot.set_data([self.walker.body_position[0]], [self.walker.body_position[1]])

        return (self.tarsus_markers + self.leg_segments +
                self.platonic_aep_markers + self.platonic_pep_markers +
                self.noisy_aep_markers + self.noisy_pep_markers +
                self.aep_scatter + self.pep_scatter +
                [self.body_line, self.trajectory_line,
                self.body_x_axis, self.body_y_axis,
                self.minimap_line, self.minimap_dot])

    def run(self) -> None:
        """Start playing all of the above"""
        self.anim = FuncAnimation(
            self.fig, self.animate, interval=100, blit=False
        )
        plt.show()
        #interval = millisec between frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the HexWalker2D viewer.")
    parser.add_argument("--mode", default="phase",
                           help="walking mode: phase or cruse")
    args = parser.parse_args()
    sim = HexSimulation(world_size=40, dt=0.02, body_scale=9.0, mode=args.mode)
    sim.run()













































