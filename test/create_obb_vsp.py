import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import util


BOX_OP = 0
ADJ_OP = 1
SYM_OP = 2

FUSELAGE_CLS = [1.0, 0.0, 0.0]
WING_CLS = [0.0, 1.0, 0.0]
ENGINE_CLS = [0.0, 0.0, 1.0]
MIRROR_Y_SYM = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FUSELAGE_STATION_FRACTIONS = (0.00, 0.15, 0.75, 1.00)
FUSELAGE_WIDTH_RATIOS = (5.0 / 13.0, 1.20, 1.20, 6.0 / 13.0)
FUSELAGE_HEIGHT_RATIOS = (3.0 / 7.0, 1.00, 1.00, 0.50)
FUSELAGE_SEGMENT_NAMES = ("fuselage_nose", "fuselage_center", "fuselage_tail")
ENGINE_END_WIDTH_RATIO = 0.80
ENGINE_MID_STATION_FRACTION = 0.45
FUSELAGE_END_CAP_GROUP = "EndCap"
FUSELAGE_NOSE_CAP_PARM = "CapUMinOption"
FUSELAGE_TAIL_CAP_PARM = "CapUMaxOption"
WING_AIRFOIL_PATH = REPO_ROOT / "foildata" / "processed_foil" / "_falcon.dat"
LEGACY_WING_GEOMETRY_SIZE = 8
GEOMETRY_JSON_FILENAME = "conventional_geometry.json"
TYPED_OBB_JSON_FILENAME = "conventional_obb.json"
HALF_BOX_ORDER = [
    *FUSELAGE_SEGMENT_NAMES,
    "main_wing_right",
    "engine_right",
    "vertical_tail",
    "horizontal_tail_right",
]
FULL_DRAW_BOX_ORDER = [
    *HALF_BOX_ORDER,
    "main_wing_left",
    "engine_left",
    "horizontal_tail_left",
]
WING_BOX_LABELS = frozenset(
    {
        "main_wing_right",
        "main_wing_left",
        "vertical_tail",
        "horizontal_tail_right",
        "horizontal_tail_left",
    }
)
ENGINE_BOX_LABELS = frozenset({"engine_right", "engine_left"})


class TreeAssembler:
    def __init__(self):
        self.boxes = []
        self.ops = []
        self.syms = []

    def push_box(self, box):
        self.boxes.append(box)
        self.ops.append(BOX_OP)

    def apply_adj(self):
        self.ops.append(ADJ_OP)

    def apply_sym(self, sym_vector):
        self.syms.append(sym_vector)
        self.ops.append(SYM_OP)


def make_box(x1, y1, z1, x2, y2, z2, root_chord, root_thick, tip_chord, tip_thick, cls):
    return [
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        root_chord,
        root_thick,
        tip_chord,
        tip_thick,
        *cls,
    ]


def make_wing_box(x1, y1, z1, x2, y2, z2, root_chord, tip_chord):
    return [x1, y1, z1, x2, y2, z2, root_chord, tip_chord]


def build_fuselage_xsecs(length, width, height):
    return [
        {
            "x": length * x_fraction,
            "width": width * width_ratio,
            "height": height * height_ratio,
        }
        for x_fraction, width_ratio, height_ratio in zip(
            FUSELAGE_STATION_FRACTIONS,
            FUSELAGE_WIDTH_RATIOS,
            FUSELAGE_HEIGHT_RATIOS,
            strict=True,
        )
    ]


def build_fuselage_boxes(fuselage_xsecs):
    return {
        name: make_box(
            start["x"],
            0.0,
            0.0,
            end["x"],
            0.0,
            0.0,
            start["width"],
            start["height"],
            end["width"],
            end["height"],
            FUSELAGE_CLS,
        )
        for name, start, end in zip(
            FUSELAGE_SEGMENT_NAMES,
            fuselage_xsecs[:-1],
            fuselage_xsecs[1:],
            strict=True,
        )
    }


def build_engine_xsecs(length, width):
    half_length = length / 2.0
    mid_x = half_length * ENGINE_MID_STATION_FRACTION
    end_width = width * ENGINE_END_WIDTH_RATIO
    return [
        {"x": -half_length, "width": end_width, "height": end_width},
        {"x": -mid_x, "width": width, "height": width},
        {"x": mid_x, "width": width, "height": width},
        {"x": half_length, "width": end_width, "height": end_width},
    ]


def set_round_end_caps(vsp, geom_id, geometry_name):
    vsp.SetParmVal(
        geom_id,
        FUSELAGE_NOSE_CAP_PARM,
        FUSELAGE_END_CAP_GROUP,
        vsp.ROUND_END_CAP,
    )
    vsp.SetParmVal(
        geom_id,
        FUSELAGE_TAIL_CAP_PARM,
        FUSELAGE_END_CAP_GROUP,
        vsp.ROUND_END_CAP,
    )
    vsp.Update()

    nose_cap = vsp.GetParmVal(
        geom_id, FUSELAGE_NOSE_CAP_PARM, FUSELAGE_END_CAP_GROUP
    )
    tail_cap = vsp.GetParmVal(
        geom_id, FUSELAGE_TAIL_CAP_PARM, FUSELAGE_END_CAP_GROUP
    )
    if nose_cap != vsp.ROUND_END_CAP or tail_cap != vsp.ROUND_END_CAP:
        raise ValueError(f"OpenVSP did not apply round caps to {geometry_name}")


def build_reference_geometry():
    fuselage_length = 1.20
    fuselage_width = 0.13
    fuselage_height = 0.14
    half_width = fuselage_width / 2.0
    fuselage_xsecs = build_fuselage_xsecs(
        fuselage_length, fuselage_width, fuselage_height
    )

    main_semispan = 0.72
    main_root_chord = 0.26
    main_tip_chord = 0.13
    main_x = 0.40

    engine_length = 0.20
    engine_width = 0.065
    engine_x = 0.50
    engine_y = half_width + main_semispan * 0.45
    engine_z = -0.070

    vtail_span = 0.30
    vtail_root_chord = 0.20
    vtail_tip_chord = 0.10
    vtail_x = 0.93

    htail_semispan = 0.34
    htail_root_chord = 0.18
    htail_tip_chord = 0.09
    htail_x = 0.92

    main_wing_box = make_wing_box(
        main_x,
        half_width,
        0.0,
        main_x,
        half_width + main_semispan,
        0.0,
        main_root_chord,
        main_tip_chord,
    )
    engine_box = make_box(
        engine_x - engine_length / 2.0,
        engine_y,
        engine_z,
        engine_x + engine_length / 2.0,
        engine_y,
        engine_z,
        engine_width,
        engine_width,
        engine_width,
        engine_width,
        ENGINE_CLS,
    )
    vtail_box = make_wing_box(
        vtail_x,
        0.0,
        fuselage_height / 2.0,
        vtail_x,
        0.0,
        fuselage_height / 2.0 + vtail_span,
        vtail_root_chord,
        vtail_tip_chord,
    )
    htail_box = make_wing_box(
        htail_x,
        half_width,
        fuselage_height / 2.0 + vtail_span * 0.70,
        htail_x,
        half_width + htail_semispan,
        fuselage_height / 2.0 + vtail_span * 0.70,
        htail_root_chord,
        htail_tip_chord,
    )

    return {
        "fuselage_length": fuselage_length,
        "fuselage_width": fuselage_width,
        "fuselage_height": fuselage_height,
        "fuselage_xsecs": fuselage_xsecs,
        "half_width": half_width,
        "main_semispan": main_semispan,
        "main_root_chord": main_root_chord,
        "main_tip_chord": main_tip_chord,
        "main_x": main_x,
        "engine_length": engine_length,
        "engine_width": engine_width,
        "engine_x": engine_x,
        "engine_y": engine_y,
        "engine_z": engine_z,
        "vtail_span": vtail_span,
        "vtail_root_chord": vtail_root_chord,
        "vtail_tip_chord": vtail_tip_chord,
        "vtail_x": vtail_x,
        "htail_semispan": htail_semispan,
        "htail_root_chord": htail_root_chord,
        "htail_tip_chord": htail_tip_chord,
        "htail_x": htail_x,
        "boxes": {
            **build_fuselage_boxes(fuselage_xsecs),
            "main_wing_right": main_wing_box,
            "engine_right": engine_box,
            "vertical_tail": vtail_box,
            "horizontal_tail_right": htail_box,
        },
    }


def build_obb_tree(reference):
    assembler = TreeAssembler()
    boxes = reference["boxes"]

    for name in FUSELAGE_SEGMENT_NAMES:
        assembler.push_box(boxes[name])
    assembler.apply_adj()
    assembler.apply_adj()
    assembler.push_box(boxes["main_wing_right"])
    assembler.push_box(boxes["engine_right"])
    assembler.apply_adj()
    assembler.apply_sym(MIRROR_Y_SYM)
    assembler.apply_adj()
    assembler.push_box(boxes["vertical_tail"])
    assembler.apply_adj()
    assembler.push_box(boxes["horizontal_tail_right"])
    assembler.apply_sym(MIRROR_Y_SYM)
    assembler.apply_adj()

    return assembler


def mirror_y_box(box):
    mirrored = list(box)
    mirrored[1] = -mirrored[1]
    mirrored[4] = -mirrored[4]
    return mirrored


def full_aircraft_boxes(reference):
    boxes = reference["boxes"]
    right_only_names = ("main_wing_right", "engine_right", "horizontal_tail_right")
    full_boxes = [
        *(boxes[name] for name in FUSELAGE_SEGMENT_NAMES),
        boxes["main_wing_right"],
        boxes["engine_right"],
        boxes["vertical_tail"],
        boxes["horizontal_tail_right"],
    ]
    for name in right_only_names:
        full_boxes.append(mirror_y_box(boxes[name]))
    return full_boxes


def component_for_box_label(label):
    if label in FUSELAGE_SEGMENT_NAMES:
        return util.COMPONENT_FUSELAGE
    if label in WING_BOX_LABELS:
        return util.COMPONENT_WING
    if label in ENGINE_BOX_LABELS:
        return util.COMPONENT_ENGINE
    raise ValueError(f"Unknown OBB component label: {label}")


def build_geometry_component(label, box):
    component = component_for_box_label(label)
    geometry_size = LEGACY_WING_GEOMETRY_SIZE if component == util.COMPONENT_WING else util.COMPONENT_GEOMETRY_SIZES[component]
    geometry = box if component == util.COMPONENT_WING else box[:geometry_size]
    if len(geometry) != geometry_size:
        raise ValueError(
            f"{label} geometry has {len(geometry)} values, expected {geometry_size}"
        )
    return {
        "name": label,
        "component": component,
        "geometry": geometry,
    }


def build_wing_airfoil_config():
    if not WING_AIRFOIL_PATH.is_file():
        raise FileNotFoundError(f"Wing airfoil file does not exist: {WING_AIRFOIL_PATH}")
    return {
        "filename": str(WING_AIRFOIL_PATH),
        "Camber": 0.02,
        "CamberLoc": 0.4,
        "ThickChord": 0.12,
    }


def create_openvsp_half(reference, output_dir):
    import infrastructure as infra

    output_dir.mkdir(parents=True, exist_ok=True)
    vsp_path = output_dir / "conventional_half.vsp3"
    infra.case_name = "conventional_half"
    infra.file_name = str(vsp_path)

    airfoil_cfg = build_wing_airfoil_config()
    tess_int = 0.025

    infra.ini_geom()
    fuselage_id = infra.create_fuselage(
        {"name": "fuselage", "x": 0.0, "y": 0.0, "z": 0.0, "yr": 0.0},
        reference["fuselage_xsecs"],
        tess_int,
    )
    set_round_end_caps(infra.vsp, fuselage_id, "fuselage")

    infra.create_wing(
        {"name": "main_wing_right", "x": reference["main_x"], "y": reference["half_width"], "z": 0.0, "yr": 0.0},
        [reference["main_semispan"]],
        [reference["main_root_chord"], reference["main_tip_chord"]],
        [0.0],
        0.0,
        tess_int,
        airfoil_cfg,
    )

    engine_xsecs = build_engine_xsecs(
        reference["engine_length"], reference["engine_width"]
    )
    engine_id = infra.create_fuselage(
        {
            "name": "engine_right_as_fuselage",
            "x": reference["engine_x"],
            "y": reference["engine_y"],
            "z": reference["engine_z"],
            "yr": 0.0,
        },
        engine_xsecs,
        tess_int,
    )
    set_round_end_caps(infra.vsp, engine_id, "engine_right_as_fuselage")

    infra.create_wing(
        {
            "name": "vertical_tail",
            "x": reference["vtail_x"],
            "y": 0.0,
            "z": reference["fuselage_height"] / 2.0,
            "yr": 0.0,
        },
        [reference["vtail_span"]],
        [reference["vtail_root_chord"], reference["vtail_tip_chord"]],
        [0.0],
        0.0,
        tess_int,
        airfoil_cfg,
    )
    vtail_id = infra.vsp.FindGeom("vertical_tail", 0)
    if not vtail_id:
        raise ValueError("OpenVSP vertical_tail geometry was not created")
    infra.vsp.SetParmVal(vtail_id, "X_Rel_Rotation", "XForm", 90.0)
    infra.vsp.Update()

    infra.create_wing(
        {
            "name": "horizontal_tail_right",
            "x": reference["htail_x"],
            "y": reference["half_width"],
            "z": reference["fuselage_height"] / 2.0 + reference["vtail_span"] * 0.70,
            "yr": 0.0,
        },
        [reference["htail_semispan"]],
        [reference["htail_root_chord"], reference["htail_tip_chord"]],
        [0.0],
        0.0,
        tess_int,
        airfoil_cfg,
    )

    infra.vsp.Update()
    infra.vsp.WriteVSPFile(str(vsp_path), infra.vsp.SET_ALL)
    return vsp_path


def save_geometry_encoding(reference, assembler, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_boxes = full_aircraft_boxes(reference)
    payload = {
        "schema": "conventional_geometry_v1",
        "airfoil_source": str(WING_AIRFOIL_PATH),
        "half_components": [
            build_geometry_component(label, box)
            for label, box in zip(HALF_BOX_ORDER, assembler.boxes, strict=True)
        ],
        "full_draw_components": [
            build_geometry_component(label, box)
            for label, box in zip(FULL_DRAW_BOX_ORDER, full_boxes, strict=True)
        ],
        "ops": assembler.ops,
        "syms": assembler.syms,
        "box_order": HALF_BOX_ORDER,
        "full_draw_box_order": FULL_DRAW_BOX_ORDER,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def invalidate_typed_obb_encoding(output_dir):
    typed_obb_path = output_dir / TYPED_OBB_JSON_FILENAME
    if typed_obb_path.exists():
        typed_obb_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Create one conventional aircraft in OpenVSP half-model form and full-aircraft OBB drawing form."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "conventional_openvsp_obb",
    )
    args = parser.parse_args()

    print("Python executable:", sys.executable, flush=True)

    reference = build_reference_geometry()
    assembler = build_obb_tree(reference)

    geometry_data_path = args.output_dir / GEOMETRY_JSON_FILENAME
    save_geometry_encoding(reference, assembler, geometry_data_path)
    invalidate_typed_obb_encoding(args.output_dir)
    vsp_path = create_openvsp_half(reference, args.output_dir)

    print("OpenVSP half model:", vsp_path)
    print("OBB geometry data:", geometry_data_path)
    print("OBB ops:", assembler.ops)
    print("OBB syms:", assembler.syms)


if __name__ == "__main__":
    main()
