from pathlib import Path
import random
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
for import_path in (PROJECT_ROOT, DATA_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import aircraft_dataset_common as common
import cst_airfoil_codec
import json_to_typed_obb_dataset
import util


def build_code(upper_offset, lower_offset):
    upper = torch.linspace(0.08 + upper_offset, 0.20 + upper_offset, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    lower = torch.linspace(-0.06 - lower_offset, -0.16 - lower_offset, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    return cst_airfoil_codec.pack_cst_airfoil_code(
        upper,
        lower,
        upper_trailing_edge_y=0.001,
        lower_trailing_edge_y=-0.001,
        class_function_n1=0.5,
        class_function_n2=1.0,
    )


def test_fuselage_station_sampling_is_random_and_valid():
    first = common.sample_fuselage_xsecs(random.Random(17), 1.0, 1.0, 1.0)
    second = common.sample_fuselage_xsecs(random.Random(18), 1.0, 1.0, 1.0)

    assert util.MIN_SECTION_COUNT <= len(first) <= util.MAX_SECTION_COUNT
    assert first != second
    for xsec in first:
        assert 0.0 <= xsec["x"] <= 1.0
        assert xsec["width"] > 0.0
        assert xsec["height"] > 0.0
    assert all(
        start["x"] < end["x"]
        for start, end in zip(first[:-1], first[1:], strict=True)
    )


def test_quadratic_wing_plan_uses_uniform_sections_and_configured_bounds():
    section_counts = []
    for seed in range(20):
        reference = common.build_random_reference(random.Random(seed))
        plan = reference["wing_plan"]
        sections = common.build_wing_planform_sections(plan)
        section_counts.append(plan["section_count"])

        assert common.WING_QUARTER_CHORD_A_RANGE[0] <= plan["quarter_chord_a"] <= common.WING_QUARTER_CHORD_A_RANGE[1]
        assert common.WING_CHORD_A_RANGE[0] <= plan["chord_a"] <= common.WING_CHORD_A_RANGE[1]
        assert common.WING_CHORD_VERTEX_FRACTION_RANGE[0] <= plan["chord_vertex_fraction"] <= common.WING_CHORD_VERTEX_FRACTION_RANGE[1]
        assert plan["chord_b"] == -2.0 * plan["chord_a"] * plan["chord_vertex_fraction"]
        assert len(sections) == plan["section_count"]
        for index, section in enumerate(sections):
            fraction = index / (len(sections) - 1)
            assert section["leading_edge_xyz"][1] == fraction * common.MAX_WINGSPAN / 2.0
            assert common.WING_CHORD_RANGE[0] <= section["chord"] <= common.WING_CHORD_RANGE[1]
    assert set(section_counts).issubset(
        set(range(util.MIN_SECTION_COUNT, util.MAX_SECTION_COUNT + 1))
    )


def test_flying_wing_converter_emits_padded_29d_cst_sections(monkeypatch):
    reference_rng = random.Random(17)
    reference = common.build_random_reference(reference_rng)
    airfoil_paths = sorted((PROJECT_ROOT / 'foildata' / 'processed_foil').glob('*.dat'))
    source_paths = common.sample_wing_airfoil_sources(reference, reference_rng, airfoil_paths)
    wing_sections = common.build_wing_sections(reference, source_paths, reference_rng)
    payload = common.build_geometry_payload(
        reference,
        common.build_obb_tree(reference),
        sample_index=0,
        random_seed=17,
        wing_sections=wing_sections,
    )
    source_paths = [
        section["airfoil_source"]
        for component in payload['half_components']
        if component["component"] == util.COMPONENT_WING
        for section in component["sections"]
    ]
    section_codes = {
        source_path: build_code(index * 0.01, index * 0.01)
        for index, source_path in enumerate(source_paths)
    }
    monkeypatch.setattr(
        json_to_typed_obb_dataset.cst_airfoil_code_cache,
        'lookup_code',
        lambda _cache, source_path: section_codes[source_path].tolist(),
    )

    sample = json_to_typed_obb_dataset.build_structured_sample(payload, {})

    wing_boxes = [
        box for box in sample['boxes']
        if box["component"] == util.COMPONENT_WING
    ]
    assert len(wing_boxes) == 1
    assert all(
        box["sections"].shape == (util.MAX_SECTION_COUNT, util.COMPONENT_SECTION_SIZES[util.COMPONENT_WING])
        for box in wing_boxes
    )
    assert [box["section_count"] for box in wing_boxes] == [
        len(payload['half_components'][index]["sections"])
        for index, component in enumerate(payload['half_components'])
        if component["component"] == util.COMPONENT_WING
    ]

    encoded_wing = wing_boxes[0]["sections"]
    source_sections = next(
        component["sections"]
        for component in payload['half_components']
        if component['name'] == 'main_wing_right'
    )
    expected_sections = torch.stack(
        [
            torch.cat([
                section_codes[section["airfoil_source"]],
                torch.tensor(section["leading_edge_xyz"]),
                torch.tensor([section["chord"], section["twist"]]),
            ])
            for section in source_sections
        ]
    )
    section_count = len(source_sections)
    assert torch.allclose(encoded_wing[:section_count], expected_sections)
    assert torch.equal(encoded_wing[section_count:], torch.zeros_like(encoded_wing[section_count:]))

    fuselage_box = next(
        box for box in sample["boxes"] if box["component"] == util.COMPONENT_FUSELAGE
    )
    source_fuselage_sections = next(
        component["sections"]
        for component in payload["half_components"]
        if component["component"] == util.COMPONENT_FUSELAGE
    )
    fuselage_count = len(source_fuselage_sections)
    assert fuselage_box["section_count"] == fuselage_count
    assert torch.allclose(
        fuselage_box["sections"][:fuselage_count],
        torch.tensor(source_fuselage_sections),
    )
    assert torch.equal(
        fuselage_box["sections"][fuselage_count:],
        torch.zeros_like(fuselage_box["sections"][fuselage_count:]),
    )
