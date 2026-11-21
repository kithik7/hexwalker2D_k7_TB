"""
slips.py 
a slip counter: how many times a pair of legs lose lock wherein their phase 
goes a whole wrap around. Only reads recorded phases and changes nothing else.
"""
import numpy as np 

from hexwalker.legs.leg import FULL_CYCLE

def count_slips(phase_a: np.ndarray, phase_b: np.ndarray) -> int:
    # gap per tick, with the 2 pi wrap jumps undone into a smooth line to avoid reading them as slips 
    #drifts are an added 2 pi 
    gap = np.unwrap(phase_b - phase_a)
    # whole laps drifted since the start, rounded to the nearest lap
    laps = np.round((gap - gap[0]) / FULL_CYCLE)
    return int(np.sum(np.abs(np.diff(laps)))) #every change of lap is one slip, fwd or backwards 


if __name__ == "__main__":
    # arithmetic paper example for one slip
    raw = np.array([3.1, 3.2, 3.9, 4.8, 5.7, 0.3, 1.4, 2.5, 3.2, 3.1])
    print(count_slips(np.zeros(len(raw)), raw), "expected 1")
    # forward a lap, then back a lap: net 0, but two slips since i used crossings 
    there_and_back = np.concatenate([raw, raw[::-1]])
    print(count_slips(np.zeros(len(there_and_back)), there_and_back), "expected 2")