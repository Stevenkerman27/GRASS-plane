import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infrastructure as infra


def main():
    output_dir = REPO_ROOT / "outputs" / "simple_wing_panel"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    infra.case_name = "simple_wing_panel"
    infra.file_name = "simple_wing_panel.vsp3"

    semispan = 0.7
    root_chord = 0.22
    tip_chord = 0.10
    cruise_spd = 12.0
    tess_int = 0.02

    fuselage_pos = {"name": "simple_panel_fuselage", "x": 0.0, "y": 0.0, "z": 0.0, "yr": 0.0}
    fuselage_xsecs = [
        {"x": 0.00, "width": 0.00, "height": 0.00},
        {"x": 0.12, "width": 0.10, "height": 0.09},
        {"x": 0.45, "width": 0.14, "height": 0.12},
        {"x": 0.85, "width": 0.12, "height": 0.11},
        {"x": 1.05, "width": 0.00, "height": 0.00},
    ]

    airfoil_cfg = {
        "filename": "",
        "Camber": 0.02,
        "CamberLoc": 0.4,
        "ThickChord": 0.12,
    }
    wing_pos = {"name": "simple_panel_wing", "x": 0.34, "y": 0, "z": 0, "yr": 0}

    spans = [semispan]
    chords = [root_chord, tip_chord]
    twists = [0.0]
    wing_S = semispan * (root_chord + tip_chord)
    Cl_target = infra.mass * infra.g / (0.5 * infra.density * cruise_spd**2 * wing_S)

    infra.ini_geom()
    infra.create_fuselage(fuselage_pos, fuselage_xsecs, tess_u=8, tess_w=17)
    infra.create_wing(wing_pos, spans, chords, twists, 0.0, tess_int, airfoil_cfg)

    wing_cfg = {
        "wing_S": wing_S,
        "bref": 2.0 * semispan,
        "cref": wing_S / (2.0 * semispan),
    }

    result = infra.runaero_panel(
        CG=0.25 * root_chord,
        alpha=8.0,
        air_spd=cruise_spd,
        wing_cfg=wing_cfg,
        Cl_target=Cl_target,
        sol_config=infra.solver_config0,
    )

    vsp_path = output_dir / infra.file_name
    polar_path = output_dir / f"{infra.case_name}.polar"
    drag, lift, net_drag, power, cltot, cdtot, cmy = result
    print("Panel result:")
    print("  drag:", drag)
    print("  lift:", lift)
    print("  net_drag:", net_drag)
    print("  power:", power)
    print("  Cltot:", cltot)
    print("  CDtot:", cdtot)
    print("  CMy:", cmy)
    print("OpenVSP model:", vsp_path)
    print("Polar file:", polar_path)


if __name__ == "__main__":
    main()
