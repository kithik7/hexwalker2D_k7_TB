import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import matplotlib FuncAnimation 
import os 
from HexWalker import Walker

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1, 1)
ax.setylim(-1, 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)