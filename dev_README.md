# hexwalker2D_k7_TB
An OOP based modular software to simulate walking in a six legged agent. (Kithi's MSc Neuroscience Thesis Project)

## Env
```
conda activate hexwalker-env
python: 3.11
dependencies: scipy, numpy, matplotlib, svgpathtools
```

## Branches
- `main` clean stable code
- `dev` baustelle :  (everything you would like to see awaits you in branch : dev)

## As of 28-09-2026

## fn(scripts)
- `HexWalker_leg.py` = Leg class (step cycle)
- `HexWalker.py` = Walker Class, six legs, displacement (only fwd heading)
- `Hex_Sim1.py` = animation and visualisation
- `extract_fly_sil.py` = Prelude: SVG outline extraction for walker form (only run 1x, produces 3 .npy files with attachment point and outline binaries)
- sources: fly_sil.svg, leg_origins.npy, body_outline.npy, body_only_outline(without legs)

## Leg class
- AEP-PEP step cycle, stance and swing phase.
- Vector and Distance properties for leg movement
- Euler integration for tarsus movement ie switching states

## Walker Class
- Form with six leg attachment points extracted from a beetle SVG (phylonet)
- Alternating tripod gait from local leg rules - emergent from: lift off when you reach PEP, swing back to AEP, touch down (loop)
- displacement calculated from how much a grounded leg contributes to pushing the body
- FuncAnimation from matplotlib.animation
- walker is followed and a trajectory is traced through the 2D space
- added glissando-ing legs, so my walker does not strut
- only has a fwd heading direction in version 1.

## To do and revisit
- support polygon
- feedback and refining (obviously)
- linear walking has been done ie we flipped the coordinates from local leg to world , but there's a net turning force to be accounted for in the displacement computation so we can explore curve walking
- noise , interleg coordination
- speed heading control, stability margin (nick's paper)
- support polygon
