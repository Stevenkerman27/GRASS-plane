import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openvsp as vsp

import infrastructure as infra


def main():
    output_dir = REPO_ROOT / "outputs" / "simple_fuselage"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    infra.case_name = "simple_fuselage"
    infra.file_name = "simple_fuselage.vsp3"

    infra.ini_geom()

    fuselage_pos = {"name": "simple_ellipse_fuselage", "x": 0.0, "y": 0.0, "z": 0.0, "yr": 0.0}
    fuselage_xsecs = [
        {"x": 0.00, "width": 0.00, "height": 0.00},
        {"x": 0.12, "width": 0.10, "height": 0.09},
        {"x": 0.45, "width": 0.14, "height": 0.12},
        {"x": 0.85, "width": 0.12, "height": 0.11},
        {"x": 1.05, "width": 0.00, "height": 0.00},
    ]
    fuse_id = infra.create_fuselage(fuselage_pos, fuselage_xsecs, tess_u=8, tess_w=17)

    semispan = 0.55
    root_chord = 0.20
    tip_chord = 0.09
    tess_int = 0.02
    wing_pos = {"name": "simple_fuselage_wing", "x": 0.36, "y": 0.0, "z": 0.0, "yr": 0.0}
    airfoil_cfg = {
        "filename": "",
        "Camber": 0.02,
        "CamberLoc": 0.4,
        "ThickChord": 0.12,
    }
    infra.create_wing(
        wing_pos,
        [semispan],
        [root_chord, tip_chord],
        [0.0],
        0.0,
        tess_int,
        airfoil_cfg,
    )

    vsp.Update()
    vsp_path = output_dir / infra.file_name
    vsp.WriteVSPFile(str(vsp_path), vsp.SET_ALL)
    print("Fuselage geom id:", fuse_id)
    print("OpenVSP model:", vsp_path)


if __name__ == "__main__":
    main()
