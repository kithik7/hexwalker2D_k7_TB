"""
topology.py

which legs talk to which 
Legs are indexed 0 to 5 : L1, L2, L3, R1, R2, R3 (front to back ipsilaterally)
None means the leg has no neighbour in that direction.
"""

LEG_LABELS = ("L1", "L2", "L3", "R1", "R2", "R3")

# the leg directly behind, on the same side
POSTERIOR = (1, 2, None, 4, 5, None)

# the leg directly in front, on the same side
ANTERIOR = (None, 0, 1, None, 3, 4)

# the leg opposite, on the other side
CONTRALATERAL = (3, 4, 5, 0, 1, 2)