"""
de capo

License credits : https://creativecommons.org/licenses/by-sa/3.0/
Dendroctonus ponderosae by Kristina Gagalova

"""

# path: 
# /Users/keerthikesavan/Desktop/Keerthi/MSc_Thesis/hexwalker2D_k7_TB/Hex_BluePrint/hexa_sil.svg
import numpy as np 
from svgpathtools import svg2paths 

#svg2paths returns 2 things stored in paths and attributes 

paths, attributes = svg2paths('fly_sil.svg')
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

fig, ax = plt.subplots(figsize=(6, 8))
ax.plot(points[:, 0], points[:, 1], color='#4a4a6a', linewidth=1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Hexapod body outline')
plt.show()

"""
coda

"""