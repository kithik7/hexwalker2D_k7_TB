"""
recorder.py

Recorder: records a run of the walker, keeps a copy of its state at every tick
it only reads from the walker, no change is effected and nothing is computed
while a copy is made. 
"""

import numpy as np
import json
from pathlib import Path

class Recorder: 
    def __init__(self,
                walker, 
                seed=None
    )  ->  None:
        self.walker = walker
        coordination = walker.coupling
        self.strength_names = list(coordination.strengths)
        self.settings = {
            "mode": walker.mode.name,
            "dt": walker.dt,
            "seed": seed,
            "strength_names": self.strength_names,
            "offset_follows_speed": getattr(
                coordination, "offset_follows_speed", None
            ),
        }
        self.tick = 0
        self.ticks = []
        self.phases = []
        self.clock_phases = []
        self.contacts = []
        self.body_x = []
        self.body_y = []
        self.heading = []
        self.strengths = []
        self.speeds = []
        self.noise_levels = []

    def record(self) -> None:
        self.ticks.append(self.tick)
        self.phases.append([leg.phase for leg in self.walker.legs])
        self.clock_phases.append(
              [getattr(leg, "clock_phase", np.nan) for leg in self.walker.legs]
        )
        self.contacts.append([leg.ground_contact for leg in self.walker.legs])
        self.body_x.append(self.walker.body_position[0])
        self.body_y.append(self.walker.body_position[1])
        self.heading.append(self.walker.body_orientation)
        self.strengths.append(list(self.walker.coupling.strengths.values()))
        self.speeds.append(self.walker.legs[0].stance_step_size)
        self.noise_levels.append(self.walker.legs[0].noise_level)
        self.tick += 1

    def save(self, path: str) -> None:
        np.savez(
            path,
            ticks=np.array(self.ticks),
            phases=np.array(self.phases),
            clock_phases=np.array(self.clock_phases),
            contacts=np.array(self.contacts),
            body_x=np.array(self.body_x),
            body_y=np.array(self.body_y),
            heading=np.array(self.heading),
            strengths=np.array(self.strengths),
            speeds=np.array(self.speeds),
            noise_levels=np.array(self.noise_levels),
            settings=np.array(self.settings, dtype=object),
        )

    def save_csv(self, path) -> None:
        path = Path(path)
        legs = ["L1", "L2", "L3", "R1", "R2", "R3"]
        header = (["tick"]
                  + [f"phase_{leg}" for leg in legs]
                  + [f"clock_{leg}" for leg in legs]
                  + [f"contact_{leg}" for leg in legs]
                  + ["body_x", "body_y", "heading"]
                  + self.strength_names
                  + ["speed", "noise"])
        table = np.column_stack([
            self.ticks,
            self.phases,
            self.clock_phases,
            np.array(self.contacts, dtype=int),
            self.body_x, self.body_y, self.heading,
            self.strengths, self.speeds, self.noise_levels,
        ])
        np.savetxt(path.with_suffix(".csv"), table, delimiter=",",
                   header=",".join(header), comments="", fmt="%.10g")
        with open(path.with_name(path.stem + "_settings.json"), "w") as f:
            json.dump(self.settings, f, indent=2)