"""
insect_body.py

Defines InsectBody: loads, cleans, and normalises a hexapod body outline from a .mat file.
"""
from pathlib import Path  
import numpy as np 
from scipy.io import loadmat 

# head, thorax, abdomen as % of fly weight, weighed in nick's paper (n = 30)
# legs (11 %) and wings (0.5 %) left out
SEGMENT_WEIGHTS = np.array([12.5, 31.0, 45.0])
# middle of each segment along y, area centroid from the outline
# boundaries placed by eye: neck at y = 0.31, waist at y = -0.13
SEGMENT_MIDDLES = np.array([0.39, 0.09, -0.28])
# antennae base to abdomen tip, measured on the outline (head front 0.48, abdomen tip -0.50)
BODY_LENGTH = 0.98


class InsectBody: 
    def __init__(self, source_path: str) -> None: 
        self.source_path = Path(source_path)
        self._load_outline()
        self._remove_duplicates()
        self._normalise()
        self._find_attachment_points()
        self._find_centre_of_mass()
        self._move_to_body_frame()

    #router function that routes format to its respective private loading methods 
    def _load_outline(self) -> None:
        if not self.source_path.exists():
            raise FileNotFoundError(f'Source file not found: {self.source_path}')
        
        if self.source_path.suffix == '.mat':
            self._load_from_mat()
        else:
            raise ValueError(f'Unsupported format: {self.source_path.suffix}')

    #to avoid hardcoded array stored in .mat keys, use list comprehension for all keys starting without __
    # picks the first one  
    def _load_from_mat(self) -> None:
        raw = loadmat(self.source_path)
        data_key = [k for k in raw.keys() if not k.startswith('__')][0]
        self.outline = raw[data_key].astype(float)
        self.outline[:, 1] *= -1 #flip y axis in image to correct for inversion
        
    def _remove_duplicates(self) -> None:
        is_not_duplicate = np.any(self.outline[1:] != self.outline[:-1], axis=1)
        is_not_duplicate = np.append(True, is_not_duplicate)
        self.outline = self.outline[is_not_duplicate]

    def _normalise(self) -> None:
         x_min = self.outline[:, 0].min()
         x_max = self.outline[:, 0].max()
         y_min = self.outline[:, 1].min()
         y_max = self.outline[:, 1].max()
         scale = max(x_max - x_min, y_max - y_min)
         centred_outline = self.outline.copy()
         centred_outline[:,0] = (self.outline[:, 0] - (x_min + x_max) / 2)
         centred_outline[:,1] = (self.outline[:, 1] - (y_min + y_max) / 2)
         self.normalised_outline = centred_outline / scale 

    def _find_attachment_points(self) -> None:
        self.attachment_points = np.array([
            [-0.184,  0.307],   # L1 front
            [-0.232,  0.059],   # L2 middle
            [-0.197, -0.089],   # L3 hind
            [ 0.184,  0.307],   # R1 front
            [ 0.232,  0.059],   # R2 middle
            [ 0.197, -0.089],   # R3 hind
        ])

    def _find_centre_of_mass(self) -> None:
        # x = 0 because the fly body outline is mirror image ie symmetric
        com_y = np.sum(SEGMENT_WEIGHTS * SEGMENT_MIDDLES) / np.sum(SEGMENT_WEIGHTS)
        self.centre_of_mass = np.array([0.0, com_y])

    def _move_to_body_frame(self) -> None:
        # ruler change: one BL=1
        self.normalised_outline = self.normalised_outline / BODY_LENGTH
        self.attachment_points = self.attachment_points / BODY_LENGTH
        self.centre_of_mass = self.centre_of_mass / BODY_LENGTH
        # zero origin pivot is now the com 
        self.normalised_outline = self.normalised_outline - self.centre_of_mass
        self.attachment_points = self.attachment_points - self.centre_of_mass
        self.centre_of_mass = np.zeros(2)
    
#test 
if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    mat_path = Path(__file__).parent / 'flySilhouette.mat'
    body = InsectBody(str(mat_path))
    fig, ax = plt.subplots(figsize=(6,8))
    ax.plot(body.normalised_outline[:, 0], body.normalised_outline[:, 1],
            color='#4a4a6a', linewidth=1.5)
    ax.plot(body.attachment_points[:, 0], body.attachment_points[:,1],
            'o', color='#c0392b', markersize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.show()

