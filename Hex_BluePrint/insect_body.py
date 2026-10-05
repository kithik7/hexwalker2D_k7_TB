"""
insect_body.py

Defines InsectBody: loads, cleans, and normalises a hexapod body outline 
Supports any format including mat, svg, or image files 

Detects leg-attachment points automatically using outline curvature 
No information known about walker and the simulation
"""
from pathlib import Path  
import numpy as np 
from scipy.io import loadmat 
from scipy.signal import find_peaks 
from svgpathtools import svg2paths
import cv2 

class InsectBody: 
    def __init__(self, source_path: str) -> None: 
        self.source_path = Path(source_path)
        self._load_outline()
        self._remove_duplicates()
        self._normalise()
        self._compute_curvature()
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

        
    def _remove_duplicates(self) -> None:
        is_not_duplicate = np.any(self.outline[1:] != self.outline[:-1], axis=1)
        is_not_duplicate = np.append(True, is_not_duplicate)
        self.outline = self.outline[is_not_duplicate]

