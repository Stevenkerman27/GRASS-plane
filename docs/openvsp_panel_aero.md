# OpenVSP Panel Aero

`infrastructure.py` is a compatibility shim that imports the shared
`aircraft_tools.openvsp_infrastructure` implementation. The OpenVSP Python
runtime must be available in the `vsppytools` environment.

## Geometry sets

After `ini_geom()`, the shared infrastructure assigns user geometry sets for
fuselage, wing, and propeller geometry. `runaero_panel(...)` analyzes the
fuselage as thick geometry and the wing as thin geometry; propellers are
excluded and no actuator disk is configured. The VLM path uses the wing set
as thin geometry without a thick set.

## Panel API

```python
runaero_panel(CG, alpha, air_spd, wing_cfg, Cl_target, sol_config, angle=[])
```

The function runs one angle of attack (`AlphaStart = AlphaEnd = alpha`) and
returns:

```text
drag, lift, net_drag, power, Cltot, CDtot, CMy
```

Panel mode has no propeller power, so `power` is zero and `net_drag` is the
aerodynamic drag. Lift and drag are computed from the `.polar` coefficients:

```text
q = 0.5 * rho * V^2
lift = q * Sref * Cltot
drag = q * Sref * CDtot
```

## Smoke test

Run the repository-local test from the `vsppytools` environment:

```powershell
<vsppytools-python> test\run_simple_wing_panel.py
```

It writes `outputs/simple_wing_panel/simple_wing_panel.vsp3` and
`outputs/simple_wing_panel/simple_wing_panel.polar`. Open the `.vsp3` file in
OpenVSP to inspect the multi-station fuselage, wing placement, and panel mesh.
