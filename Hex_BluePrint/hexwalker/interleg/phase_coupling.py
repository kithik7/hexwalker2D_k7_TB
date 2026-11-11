"""
PhaseCoupling: rules that define how the kuramoto oscillators pull on each other
K1-K4, preferred gaps, and neighbours are contained here. It reads 
leg phases and writes phase_kick. Contains no information about the body or simulation
"""

import numpy as np 


class PhaseCoupling:
    def __init__(self, legs: list) -> None: 
        self.legs = legs
    #ipsilateral_posterior_neighbour_connections 
        self.posterior_neighbours = {
            0: 1, #L1 -> L2 
            1: 2,  #L2 -> L3
            2: None, #L3 has no posterior neighbour
            3: 4, #R1 -> R2 
            4: 5, #R2 -> R3
            5: None, #R3 has no posterior neighbour 
        }
        #ipsilateral anterior neighbour connections 
        self.anterior_neighbours = {
            0: None, #L1 has no anterior neighbour 
            1: 0, #L2 -> L1
            2: 1, #L3 -> L2
            3: None, #R1 has no anterior neighbour
            4: 3, #R2 -> R1
            5: 4, # R3 -> R2 
        }

        #weak minimal rule 1 contraateral coupling 
        self.contralateral_neighbours = {
            0: 3,
            1: 4,
            2: 5,
            3: 0,
            4: 1,
            5: 2,
        }

        #K Coupling strength values 
        self.default_coupling = {"K1": 0.3, "K2": 0.1, "K3": 0.3, "K4": 0.6}
        self.K1 = 0.0
        self.K2 = 0.0
        self.K3 = 0.0
        self.K4 = 0.0        

        #preferred phase offsets between neighbouring legs (radians around the lap)
        self.offset_follows_speed = True #setting this to false gives fixed offset back 
        self.contralateral_offset = np.pi

       
    @property
    def ipsilateral_offset(self) -> float:
        """How far a leg's clock runs ahead of the leg in front of it. With speed-following on, this is
        the swing share of the cycle (up to half a cycle, a tripod), so each leg lifts as the one behind it lands."""
        if self.offset_follows_speed:
            return min(self.legs[0].touchdown_phase, np.pi)
        return 2 * np.pi / 3

    def compute_coupling_signals(self) -> list:
        """Read all six phases once as a snapshot, then compute each legs pull from its neighbours."""
        phase_snapshot = [leg.phase for leg in self.legs]
        coupling_signals = []
        for i in range(6):
            signal = 0.0
    
            post_idx = self.posterior_neighbours[i]
            if post_idx is not None:
                signal += self.K1 * np.sin(
                        phase_snapshot[post_idx] - phase_snapshot[i] - self.ipsilateral_offset
                        )
    
            ant_idx = self.anterior_neighbours[i]
            if ant_idx is not None:
                signal += self.K3 * np.sin(
                    phase_snapshot[ant_idx] - phase_snapshot[i] + self.ipsilateral_offset
                    )
    
            contra_idx = self.contralateral_neighbours[i]
            signal += self.K4 * np.sin(
                    phase_snapshot[contra_idx] - phase_snapshot[i] - self.contralateral_offset
                    )
    
            coupling_signals.append(signal)
        return coupling_signals
    
    def apply_nudges(self, touched_down: list) -> None:
        """A leg that just touched down nudges the leg in front of it fwd inits own cycle."""
        for i, fired in enumerate(touched_down):
            if fired:
                ant_idx = self.anterior_neighbours[i]
                if ant_idx is not None:
                    self.legs[ant_idx].phase_kick += self.K2

    def set_coupling(self, name: str, value: float) -> None:
        if name not in ("K1", "K2", "K3", "K4"):
            raise ValueError(f"Unknown coupling: {name}")
        setattr(self, name, value)
    