# OpenVSP Geometry and Visualization

OpenVSP generation is used by the dataset generators; typed dataset
visualization is handled separately by `data/visualize_dataset.py`.

## Dataset generation

`data/generate_flying_wing_dataset.py` and
`data/generate_conventional_canard_dataset.py` run in the `vsppytools`
environment because they import `infrastructure.py`, which loads the shared
`aircraft_tools.openvsp_infrastructure` package from the repository sibling
directory configured by `project_paths.SHARED_AIRCRAFT_TOOLS_ROOT`.

The shared generator code creates a multi-station `SUPER_ELLIPSE` fuselage,
applies its parameters, resets skinning with `ResetXSecSkinParms`, creates
wing stations from processed `.dat` files, mirrors right-side generators
through the GRASS `SYM` topology, and writes one `.vsp3` plus one validated
intermediate `.json` per sample.

All generated files are placed under the selected directory in `data/`. There
is no current standalone `conventional_geometry_v1` or old OBB JSON visualizer
in this pipeline.

## Typed dataset visualization

`data/visualize_dataset.py` loads a typed `.pt` file, expands `ADJ` and
reflective `SYM` nodes, and draws fuselage stations as elliptical rings and
wing sections by decoding CST code and applying leading-edge, chord, and
twist geometry. Legacy engine OBB geometry is supported only as a compatibility
component.

For sequence components it reconstructs absolute geometry from
`z_global`/`z_section`. Wing twist is converted from degrees to radians, and
fuselage dimensions are denormalized from global length, width, and height. A
sequence leaf must not contain `geometry`.

Run a deterministic sample from `myml`:

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\visualize_dataset.py --layout flying_wing --index 0
```

Add `--no-open-vsp` to suppress launching OpenVSP. Use `--vsp-exe` when the
default executable path in `data/visualize_dataset.py` is not valid. The
Matplotlib window requires a GUI backend such as `TkAgg`; non-GUI backends
fail fast because this tool is interactive.

## Small panel check

`test/run_simple_wing_panel.py` is an independent OpenVSP/VSPAERO smoke test.
It writes its `.vsp3` and `.polar` files under `outputs/simple_wing_panel/`,
inside this project. It is not part of the typed dataset schema.
