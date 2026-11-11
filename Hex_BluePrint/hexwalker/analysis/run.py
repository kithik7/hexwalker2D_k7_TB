from pathlib import Path
import numpy as np
from hexwalker.body.insect_body import InsectBody
from hexwalker.core.hex_walker import HexWalker
from hexwalker.analysis.recorder import Recorder

PROJECT = Path(__file__).resolve().parents[2]
BODY_FILE = PROJECT / "hexwalker" / "body" / "flySilhouette.mat"
RUNS_DIR = PROJECT / "runs"


def run_and_record(name, start="tripod", seed=0, n_ticks=1000, dt=0.02,
                   couplings="default", speed=None, noise=0.0):
    np.random.seed(seed)
    body = InsectBody(str(BODY_FILE))
    walker = HexWalker(1, body, np.array([0.0, 0.0, 0.0]), 0.0, dt, 9.0,
                       random_start_phases=(start == "random"))
    if isinstance(start, (list, tuple)):
        for leg, p in zip(walker.legs, start):
            leg.phase = p
            leg._tarsus_from_phase()
    if couplings == "default":
        couplings = walker.default_coupling
    for k, v in couplings.items():
        walker.set_coupling(k, v)
    if speed is not None:
        walker.set_speed(speed)
    walker.set_noise_level(noise)

    recorder = Recorder(walker, seed=seed)
    recorder.record()
    for _ in range(n_ticks):
        walker.update(dt)
        recorder.record()

    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{name}.npz"
    recorder.save(path)
    recorder.save_csv(path)
    return path

def free_path(name: str) -> Path:
    path = RUNS_DIR / f"{name}.npz"
    n = 2
    while path.exists():
        path = RUNS_DIR / f"{name}_{n}.npz"
        n += 1
    return path