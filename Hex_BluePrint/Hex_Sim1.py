"""
da capo
"""
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
import os 
from HexWalker import Walker

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

#create walker's form 
body_pos = np.array([0.0, 0.0, 0.0])
walker = Walker(1, 'Klaus', body_pos, 0.0)
colors = ['#4a4a6a', '#7a6a9a', '#5a8a7a', '#c0392b', '#8a6a3a', '#3a6a8a']
tarsus_dots = [ax.plot([], [], 'o', color=colors[i], markersize=10, linestyle='none')[0] 
        for i in range(6)]
#plot tarsus dots w/ six different colors 

#[0] at the end unpacks the line object from list that ax.plot return and live in all frames with different states
#updated rather than recreated 

#carve a trajectory line through this space 
trajectory_x = []
trajectory_y = []
trajectory_line, = ax.plot([], [], '-', color = "#9a9a9a", linewidth = 1, alpha = 0.5)

#load body outline frok extract_fly_sil
base_dir = os.path.dirname(os.path.abspath(__file__))
outline_points = np.load(os.path.join(base_dir, "body_only_outline.npy"))
body_line, = ax.plot([], [], "-", color= "#4a416a", linewidth = 2.0)
leg_lines = [] #glissando 
for i in range(6): 
    line, = ax.plot([], [], '-', color='#7a7a8a', linewidth=2.5)
    leg_lines.append(line)
#comma unpacks line object from list that ax.plot returns else it renders nothing 


#define a method to finally animate 
def animate(frame): 
    walker.update()
    #draw outline by bdy position 
    body_x = outline_points[:, 0] + walker.body_pos[0]
    body_y = outline_points[:, 1] + walker.body_pos[1]
    body_line.set_data(body_x, body_y)
    
    for i, leg in enumerate(walker.legs):
        world_x = leg.TarsusPosition[0] + walker.body_pos[0]
        world_y = leg.TarsusPosition[1] + walker.body_pos[1]
        tarsus_dots[i].set_data([world_x], [world_y])
        ox = leg.origin_pt[0] + walker.body_pos[0]
        oy = leg.origin_pt[1] + walker.body_pos[1]
        leg_lines[i].set_data([ox, world_x], [oy, world_y])
        #enumerate gives you both index i and the item leg simultaneously, set_data updates dot relative to 
        #current tarsus position
    cx, cy = walker.body_pos[0], walker.body_pos[1]
    ax.set_xlim(cx - 0.8, cx + 0.8)
    ax.set_ylim(cy- 0.8, cy+ 0.8)
     #update walker > read current body position > shift axis lim with 0.8 units on each side 
     #to follow the walker 
     #append current body pos to trajectory list and update trajectory 

    trajectory_x.append(walker.body_pos[0])
    trajectory_y.append(walker.body_pos[1])
    trajectory_line.set_data(trajectory_x, trajectory_y)
    
    return tarsus_dots + leg_lines + [body_line, trajectory_line]


anim = FuncAnimation(fig, animate, frames=500, interval=20, blit=False)
plt.show()

"""
render version 1: static legs that didnt move with the tarsus position dots, therefore, 
she cut off its legs in extract_fly_sil and updated os.path.dir to body_only_outline
body outline sits roughly between x = -0.3 +0.3 and legs extend beyond this range"

"""

