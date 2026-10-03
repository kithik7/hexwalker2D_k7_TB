"""
de capo

License credits : https://creativecommons.org/licenses/by-sa/3.0/
Dendroctonus ponderosae by Kristina Gagalova

"""

# path: 
# /Users/keerthikesavan/Desktop/Keerthi/MSc_Thesis/hexwalker2D_k7_TB/Hex_BluePrint/hexa_sil.svg
import numpy as np 
from svgpathtools import svg2paths 
import os
#svg2paths returns 2 things stored in paths and attributes 

paths, attributes = svg2paths('/home/kithi_k7/Desktop/Keerthi/MSc_Thesis/HexWalker/Hex_BluePrint/fly_sil.svg')
print(len(paths))

#sample 500 points
path = paths[0]
num_samples = 500
points = []

for i in range(num_samples):
    t = i / num_samples
    pt = path.point(t)
    points.append([pt.real, pt.imag])

points = np.array(points)
print(points.shape)
print(points[:3])

# normalize to simulation space
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

points[:, 0] = (points[:, 0] - (x_min + x_max) / 2) / (x_max - x_min)
points[:, 1] = (points[:, 1] - (y_min + y_max) / 2) / (y_max - y_min)

# flip y because SVG y axis points downward, simulation y points upward
points[:, 1] = -points[:, 1]
points[:, 0] = -points[:, 0]
points[:, 1] = -points[:, 1]
print(points[:3])

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# attachment point midpoints
origins = [
    np.array([-0.262,  0.152, 0.0]),  # L1
    np.array([-0.263,  0.045, 0.0]),  # L2
    np.array([-0.266, -0.162, 0.0]),  # L3
    np.array([ 0.237,  0.162, 0.0]),  # R1
    np.array([ 0.243,  0.045, 0.0]),  # R2
    np.array([ 0.253, -0.174, 0.0]),  # R3
]
labels = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

fig, ax = plt.subplots(figsize=(6, 8))
ax.plot(points[:, 0], points[:, 1], color='#4a4a6a', linewidth=1.5)

for origin, label in zip(origins, labels):
    ax.plot(origin[0], origin[1], 'o', color='#c0392b', markersize=8)
    ax.annotate(label, (origin[0], origin[1]), textcoords="offset points", xytext=(5, 5))

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Hexapod body outline with attachment points')
plt.show()
body_mask = (np.abs(points[:, 0]) < 0.28)
body_only = points[body_mask]
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_only_outline.npy'), body_only)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leg_origins.npy'), np.array(origins))
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'body_outline.npy'), points)
print("to dear Klaus, sorry i erased your legs")
#leg_origins npy contains 6x3 array of attachment points for each leg 
"""
coda

"""

"""
leg attachment coordinates: each leg has a top junction and bottom junction, the mid pt is the attachment pt 
Left Fore : (-0.259, -0.189)(-0.265, 0.114), Right Fore : (0.226, 0.203) (0.237, 0.162)
Left Mid : (-0.259, 0.068)(-0.266, 0.022) , Right Mid : (0.241, 0.071) (0.244, 0.018)
Left Hind : (-0.271, -0.143) (-0.261, -0.181), Right Hind : (-0.253, -0.151) (0.252, -0.197)

update: body outline sits roughly between x = -0.3 +0.3 and legs extend beyond this range
values within this x range are the body, points outside are legs 
so draw leg lines 
"""