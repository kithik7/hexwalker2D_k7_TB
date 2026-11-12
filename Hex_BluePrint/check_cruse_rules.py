"""
check_cruse_rules.py

Part 1: rules on leg states.
Part 2: six CruseLegs from random starts, with noise, at three speeds,
with the rules off and on to see how often fewer than three feet were
groundcontact = True, how many times each leg landed, and the 
same-side lag L3 -> L2 (ipsilateral) how long after the back neighbour lands does 
the leg in front of it land and this is measured in:
a fraction of a step (about 0.5 is tripod-like, smaller is wave-like).
"""
import numpy as np

from hexwalker.interleg.cruse_rules import CruseRules, LegState
from hexwalker.interleg.cruse_rules import compute_shifts
from hexwalker.legs.cruse_leg import CruseLeg

TICKS = 3000
NOISE = 0.005
SPEEDS = (0.01, 0.02, 0.03)

# only function
early_stance = LegState(in_stance=True, progress=0.1)
late_stance = LegState(in_stance=True, progress=0.9)
swing = LegState(in_stance=False, progress=0.5)
defaults = CruseRules([]).default_strengths

# L2 (index 1) in swing, everyone else is late in stance
states = (late_stance, swing, late_stance,
          late_stance, late_stance, late_stance)
print("L1 shift while L2 swings (rule 1 wait + rule 3 contralateral go):",
      round(compute_shifts(states, defaults)[0], 4))

# L3 (index 2) has just begun stance
states = (late_stance, late_stance, early_stance,
          late_stance, late_stance, late_stance)
print("L2 shift after L3 lands (rule 2 + rule 3 + contra rule 3):",
      round(compute_shifts(states, defaults)[1], 4))


def run(strengths: dict, speed: float, seed: int) -> tuple:
    np.random.seed(seed)
    legs = []
    for i in range(6):
        leg = CruseLeg(i + 1, f"leg{i}", np.zeros(3), np.zeros(3), 0.2)
        leg.place_at_phase(np.random.uniform(0.0, 2 * np.pi))
        leg.noise_level = NOISE
        leg.set_stance_step(speed)
        legs.append(leg)
    rules = CruseRules(legs)
    for name, value in strengths.items():
        rules.set_strength(name, value)

    landings = [[] for _ in legs]
    too_few_down = 0
    for tick in range(TICKS):
        for leg, shift, record in zip(legs, rules.shifts(), landings):
            if leg.update(shift):
                record.append(tick)
        if sum(leg.ground_contact for leg in legs) < 3:
            too_few_down += 1
    return too_few_down, [len(r) for r in landings], _lag(landings[1],
                                                          landings[2])


def _lag(front: list, back: list) -> float:
    """Median delay from a back-leg landing to the next front-leg landing,
    as a fraction of the front leg's step period."""
    period = np.median(np.diff(front[-20:]))
    delays = []
    for t_back in back[-20:]:
        later = [t for t in front if t >= t_back]
        if later:
            delays.append(((later[0] - t_back) / period) % 1)
    return round(float(np.median(delays)), 2)


off = {name: 0.0 for name in defaults}
for speed in SPEEDS:
    for label, strengths in (("off", off), ("on ", defaults)):
        few, lands, lag = run(strengths, speed, seed=0)
        print(f"speed {speed} rules {label}: fewer than 3 down {few:4d}, "
              f"landings {lands}, lag L3->L2 {lag}")