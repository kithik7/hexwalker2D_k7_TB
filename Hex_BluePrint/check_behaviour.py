"""
phase refactor must have the same values with noise at 0
applies default couplings, sets tripod start and runs it to 1000 ticks numerically 
"""
from pathlib import Path
import numpy as np
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker import HexWalker

#these are values that need to be consistent while refactoring
#noise is 0
#tripod start 
TICKS = 1000
DT = 0.02

# recorded with offset_follows_speed = True (the current default)
EXPECTED_POSITION = [-1.748685, 180.392344]
EXPECTED_HEADING = 0.010893
EXPECTED_PHASES = [4.856999, 0.93946, 2.935258, 1.768443, 4.019465, 6.078605]

mat = Path(__file__).parent / "hexwalker" / "body" / "flySilhouette.mat"
body = InsectBody(str(mat))
walker = HexWalker(1, body, np.array([0.0, 0.0, 0.0]), 0.0, DT, 9.0,
                   random_start_phases=False, mode="phase")
for name, value in walker.default_strengths.items():
    walker.set_strength(name, value)
for _ in range(TICKS):
    walker.update(DT)

position = np.round(walker.body_position[:2], 6)
heading = round(float(walker.body_orientation), 6)
phases = np.round([leg.phase for leg in walker.legs], 6)
print("position:", position)
print("heading :", heading)
print("phases  :", phases)

matches = (
    np.allclose(position, EXPECTED_POSITION, atol=1e-6)
    and abs(heading - EXPECTED_HEADING) <= 1e-6
    and np.allclose(phases, EXPECTED_PHASES, atol=1e-6)
)
print("pass: matches the recorded baseline" if matches
      else "fail: differs from the recorded baseline")