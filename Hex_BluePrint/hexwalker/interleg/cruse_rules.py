"""
cruse_rules.py

Event-driven coordination between CruseLegs, follows the Walknet logic
(Cruse et al.). Rules 1, 2 and 3 each give a distance; the distances
for one leg are summed into a single liftoff shift that moves its
liftoff threshold (positive: lift earlier, negative: lift later).

Rules (sender -> receiver):
    rule1: leg behind is in swing -> leg in front waits (negative)
    rule2: leg behind has just begun stance -> leg in front goes
    positive), for the first RULE2_WINDOW of that specific
    stance
    rule3: leg in front is far back in its stance -> leg behind 
    goes (positive), in proportion to how far back
    rule2_contra  rule 2 from the opposite leg
    rule3_contra  rule 3 from the opposite leg

Each tick works by:
haskell like take_snapshot reads every leg once into frozen LegStates.
and then compute_shifts turns the snapshot into one shift per leg. 
functional build that only reads arguments and changes nothing, 
read order is error-proof
"""

from dataclasses import dataclass

from hexwalker.interleg.topology import ANTERIOR, CONTRALATERAL, POSTERIOR

RULE2_WINDOW = 0.2  # rule 2 acts during the first 20 % of the sender's stance


@dataclass(frozen=True)
class LegState:
    # What the rules may know about one leg at the start of a tick.
    in_stance: bool
    progress: float  # 0 to 1 through the current stance or swing


def take_snapshot(legs: list) -> tuple:
    return tuple(
        LegState(in_stance=leg.ground_contact, progress=leg.stretch_progress)
        for leg in legs
    )


def compute_shifts(states: tuple, strengths: dict) -> tuple:
    return tuple(
        _shift_for(i, states, strengths) for i in range(len(states))
    )


def _shift_for(i: int, states: tuple, strengths: dict) -> float:
    shift = 0.0

    behind = POSTERIOR[i]
    if behind is not None:
        shift -= strengths["rule1"] * _in_swing(states[behind])
        shift += strengths["rule2"] * _starting_stance(states[behind])

    front = ANTERIOR[i]
    if front is not None:
        shift += strengths["rule3"] * _stance_progress(states[front])

    opposite = states[CONTRALATERAL[i]]
    shift += strengths["rule2_contra"] * _starting_stance(opposite)
    shift += strengths["rule3_contra"] * _stance_progress(opposite)

    return shift


def _in_swing(state: LegState) -> float:
    return 0.0 if state.in_stance else 1.0


def _starting_stance(state: LegState) -> float:
    return 1.0 if state.in_stance and state.progress < RULE2_WINDOW else 0.0


def _stance_progress(state: LegState) -> float:
    return state.progress if state.in_stance else 0.0


class CruseRules:
    #Holds the rule strengths and turns the legs into shifts each tick.
    # name: (slider label, lowest, highest, slider step)
    strength_specs = {
        "rule1": ("R1 wait", 0.0, 0.2, 0.005),
        "rule2": ("R2 go", 0.0, 0.2, 0.005),
        "rule3": ("R3 go", 0.0, 0.2, 0.005),
        "rule2_contra": ("R2 contra", 0.0, 0.1, 0.005),
        "rule3_contra": ("R3 contra", 0.0, 0.1, 0.005),
    }
    def __init__(self, legs: list) -> None:
        self.legs = legs
        self.default_strengths = {
            "rule1": 0.1,
            "rule2": 0.05,
            "rule3": 0.05,
            "rule2_contra": 0.025,
            "rule3_contra": 0.025,
        }
        self.strengths = {name: 0.0 for name in self.default_strengths}

    def set_strength(self, name: str, value: float) -> None:
        if name not in self.strengths:
            raise ValueError(f"Unknown rule: {name}")
        self.strengths[name] = value

    def shifts(self) -> tuple:
        return compute_shifts(take_snapshot(self.legs), self.strengths)

    def step(self, dt: float) -> None:
    # One tick shifts from one snapshot and then every leg moves
    # dt isnt used here since cruse legs move in steps per tick.
        for leg, shift in zip(self.legs, self.shifts()):
            leg.update(shift)

    

class ClockedCruseRules(CruseRules):
    # Cruse rules plus the clock dial: g (listen) and h (lean).

    strength_specs = {
        **CruseRules.strength_specs,
        "g": ("g listen", 0.0, 10.0, 0.1),
        "h": ("h lean", 0.0, 0.1, 0.005),
    }

    def __init__(self, legs: list) -> None:
        super().__init__(legs)
        self.default_strengths = {
            **self.default_strengths, "g": 2.0, "h": 0.02,
        }
        self.strengths = {name: 0.0 for name in self.default_strengths}

    def step(self, dt: float) -> None:
        # for one tick: rule shifts from one snapshot, then legs and clocks.
        g = self.strengths["g"]
        h = self.strengths["h"]
        for leg, shift in zip(self.legs, self.shifts()):
            leg.update(shift, g, h)

    