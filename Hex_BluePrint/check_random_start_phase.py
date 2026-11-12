from pathlib import Path
import numpy as np
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker import HexWalker

mat = Path(__file__).parent / "hexwalker" / "body" / "flySilhouette.mat"
body = InsectBody(str(mat))
def phase_gap(a, b):
    return (b - a) % (2 * np.pi)


SAME_SIDE_PAIRS = [(0, 1), (1, 2), (3, 4), (4, 5)]
LEFT_RIGHT_PAIRS = [(0, 3), (1, 4), (2, 5)]

def gaps_from_walker(walker):
    phases = [leg.phase for leg in walker.legs]
    same_side = [phase_gap(phases[i], phases[j]) for i, j in SAME_SIDE_PAIRS]
    left_right = [phase_gap(phases[i], phases[j]) for i, j in LEFT_RIGHT_PAIRS]
    return np.array(same_side), np.array(left_right)

# walker = HexWalker(1, body, np.array([0.0, 0.0, 0.0]), 0.0, 0.02, 9.0, random_start_phases=False)
# same_side, left_right = gaps_from_walker(walker)
# print("same side :", np.round(same_side, 3))
# print("left-right:", np.round(left_right, 3))

def run_one(seed, coupled, ticks=3000):
    np.random.seed(seed)
    walker = HexWalker(1, body, np.array([0.0, 0.0, 0.0]), 0.0, 0.02, 9.0,
                       random_start_phases=True, mode='phase')
    strengths = dict(walker.default_strengths)   # a copy of the defaults
    strengths["K2"] = 0.0                         # this check runs with K2 off
    if not coupled:
        strengths = {name: 0.0 for name in strengths}
    for name, value in strengths.items():
        walker.set_strength(name, value)
    start = np.concatenate(gaps_from_walker(walker))
    for _ in range(ticks):
        walker.update(0.02)
    end = np.concatenate(gaps_from_walker(walker))
    return start, end

for coupled in (False, True):
    print("coupling is on" if coupled else "K = 0")
    for seed in range(5):
        start, end = run_one(seed, coupled)
        print("seed", seed, "start", np.round(start, 2))
        print("       ", "end  ", np.round(end, 2))
              





