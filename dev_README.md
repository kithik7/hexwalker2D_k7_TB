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
- `dev` baustelle 

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

## To do and revisit> 29/09/26
- **visualisation**
_scale walker and world inversely proportionate to each other, change tarsusdots colors, pick a simpler silhouette maybe, add time step dt since animation moves according to default funcanimation frame rate i set (necessary for speed and heading control)_

- **AEP PEP workspace "bloble"** :
  _polygon with arbitrary extension pts from mounting pts to centre and then to the boundary of bloble_
- **noise to cause system perturbation** _(bivariate normal distribution, platonic aep pep, noisy aep pep)_
  
- **support and stability polygon**
- **speed, heading control**
- **turning** :
_(noise causes drift: do legs work against each other, if yes, can it be optimised), stance pep to next aep world coordinate memory integrated < resetting pos_

- **inter leg coordination** :
-_in the face of noise, at what time step do they decide to update and walk towards a target that is not the next obvious state (curve walking) : stick insect| pole_
- _is AEP in this case updated instantaneously (constrained: Current PEP is noted and used for computing AEP in the next step, not the current one)_ 
- **parallel plotting and visualisation**of all the things i plan to implement because a **results section**is due 
**- feedback and refining (obviously)**

