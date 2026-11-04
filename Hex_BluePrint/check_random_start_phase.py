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

def run_one(seed, coupled, ticks = 3000):
    np.random.seed(seed)
    walker = HexWalker(1, body, np.array([0.0, 0.0, 0.0]), 0.0, 0.02, 9.0, random_start_phases=True)
    for name, value in walker.default_coupling.items():
        walker.set_coupling("K2", 0.0)
        if coupled:
            walker.set_coupling(name, value)
        else:
            walker.set_coupling(name, 0.0)
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
              





'''
at the momennt phase values indexed ipsilateral legs x6 in check behaviour returned some phase values 
what it did show was that shifting the clocks by the same amount does not change how legs relate to each other 
the gaps showed that both True and False conditions with offset_follows_speed were essentially tight waves 
just half a cycle apart and each side had a stagger and just the stagger differs
but say six random starting phases were used to see if coupling purely pulls them into a particular gait 

at the moment clocks are disconnected from the body 


'''