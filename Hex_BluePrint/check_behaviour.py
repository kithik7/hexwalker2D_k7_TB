"""
phase refactor must have the same values with noise at 0
"""
from pathlib import Path
import numpy as np
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker import HexWalker

mat = Path(__file__).parent / "hexwalker" / "body" / "flySilhouette.mat"
body = InsectBody(str(mat))
walker = HexWalker(1, body, np.array([0.0, 0.0, 0.0]), 0.0, 0.02, 9.0)
for name, value in walker.default_coupling.items():
    walker.set_coupling(name, value)
for _ in range(1000):
    walker.update(0.02)
print("position:", np.round(walker.body_position[:2], 6))
print("heading :", round(float(walker.body_orientation), 6))
print("phases  :", np.round([leg.phase for leg in walker.legs], 6))

# the phase refactor needs to keep the numbers from the output of this script 
# calling set_transform with scale: 9.0
# sorted attachment points:
#   0: [-0.16  0.29]
#   1: [-0.18  0.07]
#   2: [-0.16 -0.08]
#   3: [0.16 0.29]
#   4: [0.18 0.07]
#   5: [ 0.16 -0.08]
# L1 [-0.16  0.29  0.  ]
# L2 [-0.18  0.07  0.  ]
# L3 [-0.16 -0.08  0.  ]
# R1 [0.16 0.29 0.  ]
# R2 [0.18 0.07 0.  ]
# R3 [ 0.16 -0.08  0.  ]
# position: [  0.493477 180.268904]
# heading : -0.001375
# phases  : [5.272521 0.842628 2.515949 2.183573 4.024627 5.658932]