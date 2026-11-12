"""
check_cruse_leg.py

Builds six CruseLegs with no coordination rules and runs them numerically on their own. 
At noise 0 tripod start should stay that way; with noise the legs
should drift apart, since no coordinating influences act on them yet.
"""
import numpy as np

from hexwalker.legs.cruse_leg import CruseLeg

TICKS = 1000
TRIPOD_A = (0, 2, 4)


def make_legs(noise: float) -> list:
    legs = []
    for i in range(6):
        leg = CruseLeg(i + 1, f"leg{i}", np.zeros(3),
                       np.array([0.0, 0.0, 0.0]), 0.2)
        start = leg.touchdown_phase / 2
        leg.place_at_phase(start if i in TRIPOD_A else start + np.pi)
        leg.noise_level = noise
        legs.append(leg)
    return legs


for noise in (0.0, 0.005):
    np.random.seed(0)
    legs = make_legs(noise)
    landings = [0] * 6
    for _ in range(TICKS):
        for i, leg in enumerate(legs):
            landings[i] += leg.update()
    phases = np.round([leg.phase for leg in legs], 2)
    print(f"noise {noise}: landings {landings}, end phases {phases}")