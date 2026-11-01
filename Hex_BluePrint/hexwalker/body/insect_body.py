"""
insect_body.py

Defines InsectBody: loads, cleans, and normalises a hexapod body outline from a .mat file.
"""
from pathlib import Path  
import numpy as np 
from scipy.io import loadmat 


class InsectBody: 
    def __init__(self, source_path: str) -> None: 
        self.source_path = Path(source_path)
        self._load_outline()
        self._remove_duplicates()
        self._normalise()
        self._correct_orientation()
        self._find_attachment_points()

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
        self.outline[:, 1] *= -1 #flip y axis in image ato correct for inversion
        
    def _remove_duplicates(self) -> None:
        is_not_duplicate = np.any(self.outline[1:] != self.outline[:-1], axis=1)
        is_not_duplicate = np.append(True, is_not_duplicate)
        self.outline = self.outline[is_not_duplicate]

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


    def _correct_orientation(self) -> None: 
        centroid_y = self.normalised_outline[:, 1].mean()
        if centroid_y < 0: 
            self.normalised_outline[:, 1] *= -1

    def _find_attachment_points(self) -> None:
        self.attachment_points = np.array([
            [-0.16,  0.29],   # L1 front
            [-0.18,  0.07],   # L2 middle
            [-0.16, -0.08],   # L3 hind
            [ 0.16,  0.29],   # R1 front
            [ 0.18,  0.07],   # R2 middle
            [ 0.16, -0.08],   # R3 hind
        ])
#test 
if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    from pathlib import Path
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

#TODO: better attachment point detection and outline extraction for SVGs 