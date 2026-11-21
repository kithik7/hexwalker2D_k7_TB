"""
outline_landmarks.py
click body landmarks and the two ends of each leg's trace 
save pixel coordinates to landmarks_px.csv
"""
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

LANDMARKS = [
    "antennae_base", "abdomen_tip",
    "L1_aep", "L1_pep", "L2_aep", "L2_pep", "L3_aep", "L3_pep",
    "R1_aep", "R1_pep", "R2_aep", "R2_pep", "R3_aep", "R3_pep",
]

image = mpimg.imread("last_frame.png")
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(image)

points = []
for name in LANDMARKS:
    ax.set_title(f"click: {name}")
    fig.canvas.draw()
    x, y = plt.ginput(1, timeout=0)[0]
    ax.plot(x, y, "+", color="#f0f0f0", markersize=12)
    points.append((name, x, y))
plt.close(fig)

with open("landmarks_px.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "x_px", "y_px"])
    for name, x, y in points:
        writer.writerow([name, round(x, 1), round(y, 1)])
