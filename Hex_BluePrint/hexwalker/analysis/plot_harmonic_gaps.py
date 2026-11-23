"""
plot_harmonic_gaps.py
Clock gaps and tarsus gaps plotted over time for a saved run - harmonics mode
7 neighbour pairs, 4 ipsilateral, 3 across 
"""

import sys
import matplotlib.pyplot as plt 
import numpy as np

from hexwalker.interleg.topology import CONTRALATERAL, LEG_LABELS, POSTERIOR
from hexwalker.legs.leg import FULL_CYCLE

IPSILATERAL_COLOUR = "#534AB7"
CONTRALATERAL_COLOUR = "#1D9E75"

def neighbour_pairs() -> tuple:
    # each pair once: (leg, hind leg "behind it") and (left leg, right leg)
    side = [(i, POSTERIOR[i]) for i in range(6) if POSTERIOR[i] is not None]
    across = [(i, CONTRALATERAL[i]) for i in range(3)]
    return side, across

def gaps(phases: np.ndarray, pairs:list ) -> np.ndarray:
    #phases: ticks x 6 -> ticks * no. of pairs, each wrapped 0-2pi
    return np.stack(
        [(phases[:, j] - phases[:, i]) % FULL_CYCLE for i, j in pairs], axis=1
    )

def plot_gaps(path:str) -> None: 
    run = np.load(path, allow_pickle=True)
    side, across = neighbour_pairs()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for ax, key, title in zip(axes, ["clock_phases", "phases"],
                              ["clock gaps", "foot gaps"]):
        phases = run[key]
        for pairs, colour, style in [(side, IPSILATERAL_COLOUR, "-"),
                                     (across, CONTRALATERAL_COLOUR, "--")]:
            for (i, j), series in zip(pairs, gaps(phases, pairs).T):
                ax.plot(run["ticks"], series, style, color=colour, lw=0.8,
                        label=f"{LEG_LABELS[i]}-{LEG_LABELS[j]}")
        for level in (0, np.pi, FULL_CYCLE):
            ax.axhline(level, color="0.8", lw=0.5, ls=":")
        ax.set_yticks([0, np.pi, FULL_CYCLE], ["0", "π", "2π"])
        ax.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("tick")
    axes[0].legend(ncol=7, fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_gaps(sys.argv[1])