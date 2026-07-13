"""Shared schema and geometry construction for the fixed conventional dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import airfoil_geometry
import util


BOX_OP = 0
ADJ_OP = 1
SYM_OP = 2
GEOMETRY_SCHEMA = "conventional_twin_engine_dataset_v2"
TOPOLOGY_NAME = "fixed_conventional_twin_engine_t_tail_v1"
LENGTH_UNIT = "m"
MAX_WINGSPAN = 1.0
TESS_INT = 0.025
FUSELAGE_STATION_FRACTIONS = (0.00, 0.15, 0.75, 1.00)
FUSELAGE_WIDTH_RATIOS = (5.0 / 13.0, 1.20, 1.20, 6.0 / 13.0)
FUSELAGE_HEIGHT_RATIOS = (3.0 / 7.0, 1.00, 1.00, 0.50)
FUSELAGE_SEGMENT_NAMES = ("fuselage_nose", "fuselage_center", "fuselage_tail")
WING_AIRFOIL_COMPONENTS = ("main_wing_right", "vertical_tail", "horizontal_tail_right")
ROOT_AIRFOIL_SECTION = "root"
TIP_AIRFOIL_SECTION = "tip"
WING_AIRFOIL_SECTIONS = (ROOT_AIRFOIL_SECTION, TIP_AIRFOIL_SECTION)
HALF_BOX_ORDER = (
    *FUSELAGE_SEGMENT_NAMES,
    "main_wing_right",
    "engine_right",
    "vertical_tail",
    "horizontal_tail_right",
)
FULL_DRAW_BOX_ORDER = (
    *HALF_BOX_ORDER,
    "main_wing_left",
    "engine_left",
    "horizontal_tail_left",
)
MIRROR_Y_SYM = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ROUND_CAP_PARM_NAMES = ("CapUMinOption", "CapUMaxOption")
ROUND_CAP_GROUP = "EndCap"


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


def make_body_box(x1, y1, z1, x2, y2, z2, start_width, start_height, end_width, end_height):
    return [x1, y1, z1, x2, y2, z2, start_width, start_height, end_width, end_height]


def make_wing_box(x1, y1, z1, x2, y2, z2, root_chord, tip_chord):
    return [x1, y1, z1, x2, y2, z2, root_chord, tip_chord]


def build_fuselage_xsecs(length, width, height):
    return [
        {"x": length * x_fraction, "width": width * width_ratio, "height": height * height_ratio}
        for x_fraction, width_ratio, height_ratio in zip(
            FUSELAGE_STATION_FRACTIONS,
            FUSELAGE_WIDTH_RATIOS,
            FUSELAGE_HEIGHT_RATIOS,
            strict=True,
        )
    ]


def build_engine_xsecs(length, width):
    half_length = length / 2.0
    mid_x = half_length * 0.45
    end_width = width * 0.80
    return [
        {"x": -half_length, "width": end_width, "height": end_width},
        {"x": -mid_x, "width": width, "height": width},
        {"x": mid_x, "width": width, "height": width},
        {"x": half_length, "width": end_width, "height": end_width},
    ]


def build_fuselage_boxes(fuselage_xsecs):
    return {
        name: make_body_box(
            start["x"], 0.0, 0.0, end["x"], 0.0, 0.0,
            start["width"], start["height"], end["width"], end["height"],
        )
        for name, start, end in zip(
            FUSELAGE_SEGMENT_NAMES, fuselage_xsecs[:-1], fuselage_xsecs[1:], strict=True
        )
    }


def build_random_reference(rng):
    fuselage_length = rng.uniform(0.75, 1.35)
    fuselage_width = rng.uniform(0.09, 0.16)
    fuselage_height = rng.uniform(0.10, 0.18)
    fuselage_xsecs = build_fuselage_xsecs(fuselage_length, fuselage_width, fuselage_height)
    main_root_chord = rng.uniform(0.18, 0.32)
    main_tip_chord = main_root_chord * rng.uniform(0.40, 0.75)
    main_x = rng.uniform(0.25 * fuselage_length, 0.48 * fuselage_length)
    engine_length = rng.uniform(0.16, 0.28)
    engine_width = rng.uniform(0.038, 0.075)
    engine_x = main_x + rng.uniform(-0.03, 0.08)
    engine_y = rng.uniform(0.22, 0.36)
    engine_z = rng.uniform(-0.08, -0.03)
    vtail_span = rng.uniform(0.18, 0.36)
    vtail_root_chord = rng.uniform(0.12, 0.20)
    vtail_tip_chord = vtail_root_chord * rng.uniform(0.40, 0.75)
    vtail_x = rng.uniform(0.72 * fuselage_length, 0.88 * fuselage_length)
    htail_semispan = rng.uniform(0.175, 0.30)
    htail_root_chord = rng.uniform(0.12, 0.20)
    htail_tip_chord = htail_root_chord * rng.uniform(0.40, 0.75)
    htail_x = rng.uniform(0.72 * fuselage_length, 0.90 * fuselage_length)

    boxes = {
        **build_fuselage_boxes(fuselage_xsecs),
        "main_wing_right": make_wing_box(
            main_x, 0.0, 0.0, main_x, MAX_WINGSPAN / 2.0, 0.0,
            main_root_chord, main_tip_chord,
        ),
        "engine_right": make_body_box(
            engine_x - engine_length / 2.0, engine_y, engine_z,
            engine_x + engine_length / 2.0, engine_y, engine_z,
            engine_width, engine_width, engine_width, engine_width,
        ),
        "vertical_tail": make_wing_box(
            vtail_x, 0.0, fuselage_height / 2.0,
            vtail_x, 0.0, fuselage_height / 2.0 + vtail_span,
            vtail_root_chord, vtail_tip_chord,
        ),
        "horizontal_tail_right": make_wing_box(
            htail_x, 0.0, fuselage_height / 2.0 + vtail_span * 0.70,
            htail_x, htail_semispan, fuselage_height / 2.0 + vtail_span * 0.70,
            htail_root_chord, htail_tip_chord,
        ),
    }
    return {
        "fuselage_length": fuselage_length,
        "fuselage_width": fuselage_width,
        "fuselage_height": fuselage_height,
        "fuselage_xsecs": fuselage_xsecs,
        "main_semispan": MAX_WINGSPAN / 2.0,
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
        "boxes": boxes,
    }


def sample_wing_airfoil_sources(rng, airfoils):
    if not airfoils:
        raise ValueError("airfoils must not be empty")
    return {
        component_name: {
            section_name: str(airfoils[rng.randrange(len(airfoils))].resolve())
            for section_name in WING_AIRFOIL_SECTIONS
        }
        for component_name in WING_AIRFOIL_COMPONENTS
    }


def build_obb_tree(reference):
    boxes = reference["boxes"]
    assembler = TreeAssembler()
    for name in FUSELAGE_SEGMENT_NAMES:
        assembler.push_box(boxes[name])
    assembler.apply_adj()
    assembler.apply_adj()
    assembler.push_box(boxes["main_wing_right"])
    assembler.apply_sym(MIRROR_Y_SYM)
    assembler.apply_adj()
    assembler.push_box(boxes["engine_right"])
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
    return [
        *(boxes[name] for name in HALF_BOX_ORDER),
        *(mirror_y_box(boxes[name]) for name in (
            "main_wing_right", "engine_right", "horizontal_tail_right"
        )),
    ]


def component_for_label(label):
    if label in FUSELAGE_SEGMENT_NAMES:
        return util.COMPONENT_FUSELAGE
    if label.startswith("engine_"):
        return util.COMPONENT_ENGINE
    if label in FULL_DRAW_BOX_ORDER:
        return util.COMPONENT_WING
    raise ValueError(f"Unknown OBB component label: {label}")


def build_geometry_component(label, box):
    component = component_for_label(label)
    geometry_size = util.COMPONENT_GEOMETRY_SIZES[component]
    geometry = box[:geometry_size]
    if len(geometry) != geometry_size:
        raise ValueError(f"{label} geometry has {len(geometry)} values, expected {geometry_size}")
    return {"name": label, util.BOX_COMPONENT_KEY: component, util.BOX_GEOMETRY_KEY: geometry}


def build_geometry_payload(reference, assembler, sample_index, random_seed, wing_airfoil_sources):
    return {
        "schema": GEOMETRY_SCHEMA,
        "topology": TOPOLOGY_NAME,
        "units": LENGTH_UNIT,
        "sample_index": sample_index,
        "random_seed": random_seed,
        "max_wingspan": MAX_WINGSPAN,
        "tess_int": TESS_INT,
        "wing_airfoil_sources": wing_airfoil_sources,
        "half_components": [
            build_geometry_component(label, box)
            for label, box in zip(HALF_BOX_ORDER, assembler.boxes, strict=True)
        ],
        "full_draw_components": [
            build_geometry_component(label, box)
            for label, box in zip(FULL_DRAW_BOX_ORDER, full_aircraft_boxes(reference), strict=True)
        ],
        "ops": assembler.ops,
        "syms": assembler.syms,
        "box_order": list(HALF_BOX_ORDER),
        "full_draw_box_order": list(FULL_DRAW_BOX_ORDER),
    }


def write_geometry_payload(payload, path):
    validate_geometry_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_geometry_payload(payload):
    required = (
        "schema", "topology", "units", "sample_index", "random_seed", "max_wingspan",
        "tess_int", "wing_airfoil_sources", "half_components", "ops", "syms", "box_order",
    )
    for key in required:
        if key not in payload:
            raise KeyError(f"Geometry JSON missing required key: {key}")
    if payload["schema"] != GEOMETRY_SCHEMA:
        raise ValueError(f"Expected schema {GEOMETRY_SCHEMA!r}, got {payload['schema']!r}")
    if payload["topology"] != TOPOLOGY_NAME:
        raise ValueError(f"Unexpected topology: {payload['topology']!r}")
    if payload["units"] != LENGTH_UNIT:
        raise ValueError(f"Unexpected unit: {payload['units']!r}")
    if payload["max_wingspan"] != MAX_WINGSPAN:
        raise ValueError(f"max_wingspan must be {MAX_WINGSPAN}, got {payload['max_wingspan']}")
    if payload["box_order"] != list(HALF_BOX_ORDER):
        raise ValueError("box_order does not match the fixed topology")
    if set(payload["wing_airfoil_sources"]) != set(WING_AIRFOIL_COMPONENTS):
        raise ValueError("wing_airfoil_sources does not match the fixed wing topology")
    for component_name in WING_AIRFOIL_COMPONENTS:
        section_sources = payload["wing_airfoil_sources"][component_name]
        if set(section_sources) != set(WING_AIRFOIL_SECTIONS):
            raise ValueError(f"{component_name} airfoil sections do not match the fixed topology")
        for section_name in WING_AIRFOIL_SECTIONS:
            source = section_sources[section_name]
            if not isinstance(source, str) or not source:
                raise ValueError(f"{component_name}.{section_name} airfoil source must be a non-empty string")
    if len(payload["half_components"]) != len(HALF_BOX_ORDER):
        raise ValueError("half_components does not match the fixed topology")
    if payload["ops"].count(BOX_OP) != len(payload["half_components"]):
        raise ValueError("ops BOX count does not match half_components")
    if payload["ops"].count(SYM_OP) != len(payload["syms"]):
        raise ValueError("ops SYM count does not match syms")
    for index, (label, component_payload) in enumerate(zip(HALF_BOX_ORDER, payload["half_components"], strict=True)):
        if component_payload.get("name") != label:
            raise ValueError(f"half_components[{index}] does not match {label!r}")
        component = component_payload.get(util.BOX_COMPONENT_KEY)
        if component != component_for_label(label):
            raise ValueError(f"half_components[{index}] has an unexpected component type")
        geometry = component_payload.get(util.BOX_GEOMETRY_KEY)
        if len(geometry) != util.COMPONENT_GEOMETRY_SIZES[component]:
            raise ValueError(f"half_components[{index}] has an invalid geometry length")


def load_geometry_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_geometry_payload(payload)
    return payload


def set_round_end_caps(vsp, geom_id, geometry_name):
    for parm_name in ROUND_CAP_PARM_NAMES:
        vsp.SetParmVal(geom_id, parm_name, ROUND_CAP_GROUP, vsp.ROUND_END_CAP)
    vsp.Update()
    for parm_name in ROUND_CAP_PARM_NAMES:
        if vsp.GetParmVal(geom_id, parm_name, ROUND_CAP_GROUP) != vsp.ROUND_END_CAP:
            raise ValueError(f"OpenVSP did not apply round caps to {geometry_name}")


def validate_wing_airfoil_sources(wing_airfoil_sources):
    if set(wing_airfoil_sources) != set(WING_AIRFOIL_COMPONENTS):
        raise ValueError("wing_airfoil_sources does not match the fixed wing topology")
    for component_name in WING_AIRFOIL_COMPONENTS:
        section_sources = wing_airfoil_sources[component_name]
        if set(section_sources) != set(WING_AIRFOIL_SECTIONS):
            raise ValueError(f"{component_name} airfoil sections do not match the fixed topology")
        for section_name in WING_AIRFOIL_SECTIONS:
            source_path = Path(section_sources[section_name])
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{component_name}.{section_name} airfoil file does not exist: {source_path}"
                )


def set_wing_xsec_file_airfoil(vsp, geom_id, xsec_index, airfoil_path):
    xsec_surface_id = vsp.GetXSecSurf(geom_id, 0)
    vsp.ChangeXSecShape(xsec_surface_id, xsec_index, vsp.XS_FILE_AIRFOIL)
    xsec_id = vsp.GetXSec(xsec_surface_id, xsec_index)
    vsp.ReadFileAirfoil(xsec_id, str(airfoil_path))
    vsp.Update()


def create_wing_with_section_airfoils(infra, position, spans, chords, section_sources):
    root_airfoil = Path(section_sources[ROOT_AIRFOIL_SECTION])
    tip_airfoil = Path(section_sources[TIP_AIRFOIL_SECTION])
    airfoil_cfg = {
        "filename": str(root_airfoil),
        "Camber": 0.02,
        "CamberLoc": 0.4,
        "ThickChord": 0.12,
    }
    infra.create_wing(position, spans, chords, [0.0], 0.0, TESS_INT, airfoil_cfg)
    geom_id = infra.vsp.FindGeom(position["name"], 0)
    if not geom_id:
        raise ValueError(f"OpenVSP {position['name']} geometry was not created")
    set_wing_xsec_file_airfoil(infra.vsp, geom_id, 1, tip_airfoil)
    return geom_id


def create_openvsp_aircraft(reference, wing_airfoil_sources, output_path):
    import infrastructure as infra

    validate_wing_airfoil_sources(wing_airfoil_sources)
    infra.case_name = output_path.stem
    infra.file_name = str(output_path)
    infra.ini_geom()
    fuselage_id = infra.create_fuselage(
        {"name": "fuselage", "x": 0.0, "y": 0.0, "z": 0.0, "yr": 0.0},
        reference["fuselage_xsecs"], TESS_INT,
    )
    set_round_end_caps(infra.vsp, fuselage_id, "fuselage")

    create_wing_with_section_airfoils(
        infra,
        {"name": "main_wing_right", "x": reference["main_x"], "y": 0.0, "z": 0.0, "yr": 0.0},
        [reference["main_semispan"]], [reference["main_root_chord"], reference["main_tip_chord"]],
        wing_airfoil_sources["main_wing_right"],
    )

    engine_xsecs = build_engine_xsecs(reference["engine_length"], reference["engine_width"])
    engine_id = infra.create_fuselage(
        {"name": "engine_right", "x": reference["engine_x"], "y": reference["engine_y"], "z": reference["engine_z"], "yr": 0.0},
        engine_xsecs, TESS_INT,
    )
    set_round_end_caps(infra.vsp, engine_id, "engine_right")

    with tempfile.TemporaryDirectory(prefix="grass_vtail_airfoils_") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        vertical_tail_sources = {
            section_name: str(
                airfoil_geometry.write_symmetric_airfoil_dat(
                    wing_airfoil_sources["vertical_tail"][section_name],
                    temporary_directory / f"vertical_tail_{section_name}.dat",
                )
            )
            for section_name in WING_AIRFOIL_SECTIONS
        }
        vtail_id = create_wing_with_section_airfoils(
            infra,
            {"name": "vertical_tail", "x": reference["vtail_x"], "y": 0.0,
             "z": reference["fuselage_height"] / 2.0, "yr": 0.0},
            [reference["vtail_span"]], [reference["vtail_root_chord"], reference["vtail_tip_chord"]],
            vertical_tail_sources,
        )
        infra.vsp.SetParmVal(vtail_id, "X_Rel_Rotation", "XForm", 90.0)
        infra.vsp.Update()

        create_wing_with_section_airfoils(
            infra,
            {"name": "horizontal_tail_right", "x": reference["htail_x"], "y": 0.0,
             "z": reference["fuselage_height"] / 2.0 + reference["vtail_span"] * 0.70, "yr": 0.0},
            [reference["htail_semispan"]], [reference["htail_root_chord"], reference["htail_tip_chord"]],
            wing_airfoil_sources["horizontal_tail_right"],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        infra.vsp.WriteVSPFile(str(output_path), infra.vsp.SET_ALL)
    if not output_path.is_file():
        raise RuntimeError(f"OpenVSP did not write expected file: {output_path}")
