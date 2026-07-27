# Flying-Wing Sampling

`data/aircraft_dataset_common.py` is the sole implementation of flying-wing
sampling, JSON construction, GRASS tree construction, and OpenVSP geometry
construction.

All sampling ranges are centralized at the beginning of
`data/aircraft_dataset_common.py`. They cover the wing plan and the fuselage
total dimensions, variable station positions, and station
width and height ratios. They are dataset definitions, not runtime arguments.
`util.SECTION_COUNT_RANGE` is the shared variable-length sequence definition
used by fuselage and wing generation, conversion, and training.

Each sampled fuselage has `2..8` strictly increasing stations and positive
elliptical dimensions. The same stations form the `[x,y,z,width,height]`
sequence payload and are written identically to the OpenVSP fuselage after
linear skinning is cleared.

The wing half-span is parameterized by `eta` in `[0, 1]`. Its quarter-chord
x position and chord use root-relative upward-opening quadratic functions:

```text
x_25(eta) = x_25_root + a_x * eta^2 + b_x * eta
chord(eta) = chord_root + a_c * eta^2 + b_c * eta
```

The quadratic constant terms are deliberately zero: `x_25_root` and
`chord_root` retain their physical root ranges, while `a` and `b` describe
only spanwise variation. Both `a` ranges are non-negative. The chord curve
samples `eta_c = -b_c / (2 * a_c)` from `WING_CHORD_VERTEX_FRACTION_RANGE` and derives
`b_c = -2 * a_c * eta_c`; it does not sample `b_c` independently. Sampled curves are
accepted only when the entire chord curve satisfies the chord range and the
tip quarter-chord range. No adjacent-section variation limit is applied. The
stored leading edge is derived as `x_LE = x_25 - 0.25 * chord`.

After a valid curve is sampled, `N` is independently and uniformly selected
from `2..8`. Section positions then use uniform normalized span locations
`eta_i = i / (N - 1)`. A quadratic has constant curvature, so this minimizes
the largest piecewise-linear interpolation error without introducing an
independent error tolerance or adaptive sampling rule.

`data/visualize_quadratic_wing_sampling.py` writes 30 randomly generated full
planforms to `data/quadratic_wing_planforms.png` and then opens an interactive
Matplotlib window by default. It uses the same environment-selected backend as
`data/visualize_dataset.py` and fails if it is non-interactive. The blue outline
is the discrete leading/trailing edge shape and the dashed orange line is the
original quadratic quarter-chord curve. Use the `myml` environment, for example:

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\visualize_quadratic_wing_sampling.py
```

JSON field names and the airfoil leading-edge y coordinate are fixed literals
at their use sites. They are not exported through `util.py`.
