# Aircraft Dataset Generation

This document describes the current aircraft-data pipeline. The authoritative
implementation is in `data/aircraft_dataset_common.py`,
`data/conventional_canard_dataset_common.py`, and `project_paths.py`.

## Pipeline

`data/generate_aircraft_datasets.ps1` is the reproducible end-to-end entry
point. It generates OpenVSP models and geometry JSON in `vsppytools`, then
converts them to typed PyTorch datasets in `myml` and validates every sample.

Run it from the repository root:

```powershell
.\data\generate_aircraft_datasets.ps1
```

The script clears only these approved directories under `data/`:

```text
data/flying_wing_dataset/
data/conventional_canard_dataset/
```

Each layout currently contains `400` samples. The final files are:

```text
data/flying_wing_dataset/flying_wing_dataset.pt
data/conventional_canard_dataset/conventional_canard_dataset.pt
```

The matching `.vsp3` and intermediate `.json` files remain in each layout
directory for OpenVSP inspection.

## Layouts and sampling

The flying-wing layout contains one fuselage and one right main wing mirrored
about the XZ plane. The conventional/canard layout contains a fuselage, a
right main wing, and a right auxiliary wing; both wings are mirrored. Even
sample indices are conventional and odd indices are canard.

Both generators use these fixed payload rules:

- the fuselage has `10` uniformly spaced stations;
- every wing has `6` uniformly spaced root-to-tip stations;
- wing twist is stored in degrees and adjacent changes are bounded;
- wing planforms use root-relative quadratic quarter-chord and chord curves;
- each wing root is placed inside the local fuselage height;
- `.dat` paths exist only in intermediate JSON and become CST codes during
  conversion.

The conventional/canard auxiliary wing has a span ratio of `0.20..0.80` of
the main wing and an edge gap of `0.10..0.45` fuselage lengths. Its root chord
must remain inside the fuselage longitudinal range.

All detailed sampling ranges and geometry validation remain single-sourced in
the two `*_dataset_common.py` modules.

## JSON and PT boundary

Intermediate JSON uses the `*_global_section_v1` schemas and stores
`z_global`, `z_section`, and wing `airfoil_sources`. The converter
`data/json_to_typed_obb_dataset.py` replaces wing source paths with cached 19D
CST codes. The typed sample contains `boxes`, postorder `ops`, and `syms`;
the component contract is defined in
`docs/aircraft_schema_and_codec.md`.

The converter requires `data/cst_airfoil_code_cache.pt`. Rebuild it when the
processed airfoil set or CST configuration changes:

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\precompute_airfoil_codes.py
```

## Visualization

Open one dataset sample in OpenVSP and the reconstructed Matplotlib 3D view:

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\visualize_dataset.py --layout flying_wing --index 0
```

Use `--no-open-vsp` to inspect only Matplotlib. `--layout` is either
`flying_wing` or `conventional_canard`; `--index` is local to that dataset.
The visualizer reconstructs absolute geometry from `z_global` and
`z_section`, not from the old OBB `geometry` payload.

The quadratic planform diagnostic is non-interactive and saves
`data/quadratic_wing_planforms.png`:

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\visualize_quadratic_wing_sampling.py
```
