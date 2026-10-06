"""
insect_body.py

Defines InsectBody: loads, cleans, and normalises a hexapod body outline 
Supports any format including mat, svg, or image files 

svg formats work only for outlines without legs

Detects leg-attachment points automatically using outline curvature 
No information known about walker and the simulation
"""
from pathlib import Path  
import numpy as np 
from scipy.io import loadmat 
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks 
from svgpathtools import svg2paths
import cv2 

class InsectBody: 
    def __init__(self, source_path: str) -> None: 
        self.source_path = Path(source_path)
        self._load_outline()
        self._remove_duplicates()
        self._extract_outline_without_legs()
        self._normalise()
        self._compute_curvature()
        self._correct_orientation()
        self._find_attachment_points()

    #router function that routes format to its respective private loading methods 
    def _load_outline(self) -> None:
        if not self.source_path.exists():
            raise FileNotFoundError(f'Source file not found: {self.source_path}')
        
        if self.source_path.suffix == '.mat':
            self._load_from_mat()
        elif self.source_path.suffix == '.svg':
            self._load_from_svg()
        elif self.source_path.suffix in ('.jpg', '.png'):
            self._load_from_image()
        else:
            raise ValueError(f'Unsupported format: {self.source_path.suffix}')

    #to avoid hardcoded array stored in .mat keys, use list comprehension for all keys starting without __
    # picks the first one  
    def _load_from_mat(self) -> None:
        raw = loadmat(self.source_path)
        data_key = [k for k in raw.keys() if not k.startswith('__')][0]
        self.outline = raw[data_key].astype(float)
        self.outline[:, 1] *= -1 #flip y axis in image ato correct for inversion

    def _load_from_svg(self) -> None: 
        NUM_SAMPLES = 500
        paths, _ = svg2paths(self.source_path)
        path = paths[0]
    
        points = []
        for i in range(NUM_SAMPLES):
            t = i / NUM_SAMPLES
            pt = path.point(t)
            points.append([pt.real, pt.imag])
    
        self.outline = np.array(points)

    def _load_from_image(self) -> None:
        image = cv2.imread(str(self.source_path), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.outline = contours[0].reshape(-1, 2).astype(float)
        self.outline[:, 1] *= -1 #flip image to correct for inversion

        
    def _remove_duplicates(self) -> None:
        is_not_duplicate = np.any(self.outline[1:] != self.outline[:-1], axis=1)
        is_not_duplicate = np.append(True, is_not_duplicate)
        self.outline = self.outline[is_not_duplicate]

    def _extract_outline_without_legs(self) -> None:
        if self.source_path.suffix not in ('.jpg', '.png'):
            return
        hull = ConvexHull(self.outline)
        edge_equations = hull.equations
        points_homogeneous = np.column_stack([self.outline, np.ones(len(self.outline))])
        distances_to_all_edges = points_homogeneous @ edge_equations.T
        distance_to_hull = np.abs(distances_to_all_edges).min(axis=1)
        body_size = self.outline[:, 0].max() - self.outline[:, 0].min()
        threshold = 0.05 * body_size
        self.outline = self.outline[distance_to_hull < threshold]


    def _normalise(self) -> None:
         """ centre outline and scale to fit """
         x_min = self.outline[:, 0].min()
         x_max = self.outline[:, 0].max()
         y_min = self.outline[:, 1].min()
         y_max = self.outline[:, 1].max()
         scale = max(x_max - x_min, y_max - y_min)
         centred_outline = self.outline.copy()
         centred_outline[:,0] = (self.outline[:, 0] - (x_min + x_max) / 2)
         centred_outline[:,1] = (self.outline[:, 1] - (y_min + y_max) / 2)
         self.normalised_outline = centred_outline / scale 


    def _compute_curvature(self) -> None: 
        tangent_vectors = self.normalised_outline[2:] - self.normalised_outline[:-2]
        theta_tangents = np.arctan2(tangent_vectors[:,1], tangent_vectors[:,0])
        self.curvature = np.diff(theta_tangents)
        # curvature array has reduced length due to subtraction, 
        # find attachment points must use outline index i+2

    def _correct_orientation(self) -> None: 
        centroid_y = self.normalised_outline[:, 1].mean()
        if centroid_y < 0: 
            self.normalised_outline[:, 1] *= -1

    def _find_attachment_points(self) -> None: 
        y_min = self.normalised_outline[:, 1].min()
        y_max = self.normalised_outline[:, 1].max()
        y_range = y_max - y_min 
        y_lower_bound = y_min + 0.3 * y_range
        y_upper_bound = y_max - 0.2 * y_range
       
        in_body_region = (
            (self.normalised_outline[:, 1] > y_lower_bound) &
            (self.normalised_outline[:, 1] < y_upper_bound) 
        )

        body_points = self.normalised_outline[in_body_region]
        y_band_size = (y_upper_bound - y_lower_bound) / 3 
        attachment_point_list = []
        for band in range(3):
            band_lower = y_lower_bound + band * y_band_size
            band_upper = y_lower_bound + (band + 1) * y_band_size 
        
            in_band = (
                (body_points[:, 1] >= band_lower) &
                (body_points[:, 1] < band_upper)
            )
            band_points = body_points[in_band]
            left_band = band_points[band_points[:, 0] < 0]
            right_band = band_points[band_points[:, 0] > 0]
            if len(left_band) > 0:
                lateral_left_most_point = left_band[np.argmin(left_band[:, 0])]
                attachment_point_list.append(lateral_left_most_point)
            if len(right_band) > 0: 
                lateral_right_most_point = right_band[np.argmax(right_band[:, 0])]
                attachment_point_list.append(lateral_right_most_point)
        self.attachment_points = np.array(attachment_point_list)
            

#test 
if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    body = InsectBody('Hex_BluePrint/flySilhouette.mat')
    fig, ax = plt.subplots(figsize=(6,8))
    ax.plot(body.normalised_outline[:, 0], body.normalised_outline[:, 1],
            color='#4a4a6a', linewidth=1.5)
    ax.plot(body.attachment_points[:, 0], body.attachment_points[:,1],
            'o', color='#c0392b', markersize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.show()

#TODO: better attachment point detection and outline extraction for SVGs 