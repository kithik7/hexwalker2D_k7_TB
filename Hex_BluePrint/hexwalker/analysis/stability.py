"""
stability.py 
Static stability margin: is the signed distance from the COM to the nearest edge 
Parallelogram, counterclockwise leg positioning implies the inside is always on the left 
Considering this, then positive distance = COM inside, negative distance = COM outside 
Follows Szczecinski et al. 2018
"""
import numpy as np 
from scipy.spatial import ConvexHull

def stability_margin(feet: np.ndarray, com: np.ndaarray) -> float:
    #if fewer than three feet down then area = 0
    if len(feet) < 3:
        return 0.0
    hull = ConvexHull(feet)
    corners = feet[hull.vertices] #counterclockwise in two dimensions 
    distances = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        edge = b-a
        to_com = com - a 
        cross_product = edge[0] * to_com[1] - edge[1] * to_com[0]
        distances.append(cross_product / np.linalg.norm(edge))
    return min(distances)

if __name__ == "__main__":
    # known answer: my handworked example from vol 2 page 13, COM just outside the L1-R3 edge
    feet = np.array([[-0.27, 0.616], [0.57, 0.146], [0.385, -0.61]])
    print(round(stability_margin(feet, np.zeros(2)), 4), "expected -0.0521")
    # same frame with L3 down as well: COM should be inside
    feet_l3 = np.vstack([feet, [-0.385, -0.4]])
    print(round(stability_margin(feet_l3, np.zeros(2)), 4), "expected positive")