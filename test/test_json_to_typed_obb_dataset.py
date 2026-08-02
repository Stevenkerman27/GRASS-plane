from pathlib import Path
import math
import random
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
for import_path in (PROJECT_ROOT, DATA_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import aircraft_dataset_common as common
import conventional_canard_dataset_common as conventional_canard
import cst_airfoil_codec
import json_to_typed_obb_dataset
import util


def build_code(upper_offset, lower_offset):
    upper = torch.linspace(0.08 + upper_offset, 0.20 + upper_offset, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    lower = torch.linspace(-0.06 - lower_offset, -0.16 - lower_offset, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    return cst_airfoil_codec.pack_cst_airfoil_code(
        upper,
        lower,
        trailing_edge_thickness=0.002,
        class_function_n1=0.5,
        class_function_n2=1.0,
    )


def test_fuselage_station_sampling_is_random_and_valid():
    first = common.sample_fuselage_xsecs(random.Random(17), 1.0, 1.0, 1.0)
    second = common.sample_fuselage_xsecs(random.Random(18), 1.0, 1.0, 1.0)

    assert len(first) == util.FUSELAGE_SECTION_COUNT
    assert first != second
    for index, xsec in enumerate(first):
        assert xsec["x"] == index / (util.FUSELAGE_SECTION_COUNT - 1)
        assert xsec["width"] > 0.0
        assert xsec["height"] > 0.0
        assert xsec['z'] == 0.0
        assert common.FUSELAGE_SUPER_MAX_WIDTH_LOC_RANGE[0] <= xsec['Super_MaxWidthLoc'] <= common.FUSELAGE_SUPER_MAX_WIDTH_LOC_RANGE[1]
        assert common.FUSELAGE_SUPER_M_RANGE[0] <= xsec['Super_M'] <= common.FUSELAGE_SUPER_M_RANGE[1]
        assert common.FUSELAGE_SUPER_N_RANGE[0] <= xsec['Super_N'] <= common.FUSELAGE_SUPER_N_RANGE[1]
    assert first[0]['width'] == first[-1]['width'] == common.FUSELAGE_END_SECTION_SCALE
    assert first[0]['height'] == first[-1]['height'] == common.FUSELAGE_END_SECTION_SCALE
    for xsec in first[1:-1]:
        assert common.FUSELAGE_INTERIOR_SECTION_SCALE_RANGE[0] <= xsec['width'] <= common.FUSELAGE_INTERIOR_SECTION_SCALE_RANGE[1]
        assert common.FUSELAGE_INTERIOR_SECTION_SCALE_RANGE[0] <= xsec['height'] <= common.FUSELAGE_INTERIOR_SECTION_SCALE_RANGE[1]
    assert all(
        start["x"] < end["x"]
        for start, end in zip(first[:-1], first[1:], strict=True)
    )
    for left, right in zip(first[:-1], first[1:], strict=True):
        assert abs(right['Super_MaxWidthLoc'] - left['Super_MaxWidthLoc']) <= common.FUSELAGE_SUPER_MAX_WIDTH_LOC_MAX_DELTA
        assert abs(right['Super_M'] - left['Super_M']) <= common.FUSELAGE_SUPER_M_MAX_DELTA
        assert abs(right['Super_N'] - left['Super_N']) <= common.FUSELAGE_SUPER_N_MAX_DELTA


def test_quadratic_wing_plan_uses_uniform_y_and_keeps_z_relative_to_root():
    assert common.WING_CHORD_JITTER_FRACTION == 0.10
    assert common.WING_SWEEP_JITTER_FRACTION == 0.10
    for seed in range(20):
        reference = common.build_random_reference(random.Random(seed))
        plan = reference["wing_plan"]
        sections = common.build_wing_planform_sections(plan)
        lower_z, upper_z = common.wing_root_z_bounds(reference['fuselage_xsecs'], plan)
        assert plan['section_count'] == util.WING_SECTION_COUNT
        assert len(plan['twist_degrees']) == util.WING_SECTION_COUNT
        assert all(common.WING_TWIST_DEG_RANGE[0] <= twist <= common.WING_TWIST_DEG_RANGE[1] for twist in plan['twist_degrees'])
        assert all(
            abs(right - left) <= common.WING_TWIST_MAX_ADJACENT_DELTA_DEG
            for left, right in zip(plan['twist_degrees'][:-1], plan['twist_degrees'][1:], strict=True)
        )
        assert plan['chord_jitter_factors'][0] == 1.0
        assert plan['sweep_jitter_factors'][0] == 1.0
        assert all(
            1.0 - common.WING_CHORD_JITTER_FRACTION <= factor <= 1.0 + common.WING_CHORD_JITTER_FRACTION
            for factor in plan['chord_jitter_factors'][1:]
        )
        assert all(
            1.0 - common.WING_SWEEP_JITTER_FRACTION <= factor <= 1.0 + common.WING_SWEEP_JITTER_FRACTION
            for factor in plan['sweep_jitter_factors'][1:]
        )

        assert common.WING_QUARTER_CHORD_A_RANGE[0] <= plan["quarter_chord_a"] <= common.WING_QUARTER_CHORD_A_RANGE[1]
        assert common.WING_CHORD_A_RANGE[0] <= plan["chord_a"] <= common.WING_CHORD_A_RANGE[1]
        assert common.WING_CHORD_VERTEX_FRACTION_RANGE[0] <= plan["chord_vertex_fraction"] <= common.WING_CHORD_VERTEX_FRACTION_RANGE[1]
        assert plan["chord_b"] == -2.0 * plan["chord_a"] * plan["chord_vertex_fraction"]
        assert len(sections) == plan["section_count"]
        base_leading_edge_x = []
        for index, section in enumerate(sections):
            fraction = index / (len(sections) - 1)
            assert section['leading_edge_xyz'][1] == plan['span_fractions'][index] * common.MAX_WINGSPAN / 2.0
            base_chord = common.evaluate_root_relative_quadratic(
                plan['root_chord'], plan['chord_a'], plan['chord_b'], fraction
            )
            assert section['chord'] == base_chord * plan['chord_jitter_factors'][index]
            base_leading_edge_x.append(
                common.evaluate_root_relative_quadratic(
                    plan['root_quarter_chord_x'], plan['quarter_chord_a'], plan['quarter_chord_b'], fraction
                ) - 0.25 * base_chord
            )
            if index == 0:
                assert section['leading_edge_xyz'][0] == base_leading_edge_x[index]
            else:
                previous = sections[index - 1]['leading_edge_xyz']
                span_delta = section['leading_edge_xyz'][1] - previous[1]
                base_sweep = math.atan2(
                    base_leading_edge_x[index] - base_leading_edge_x[index - 1], span_delta
                )
                expected_x = previous[0] + span_delta * math.tan(
                    base_sweep * plan['sweep_jitter_factors'][index]
                )
                assert section['leading_edge_xyz'][0] == expected_x
            expected_z = plan['root_leading_edge_z'] + fraction * plan['tip_leading_edge_z_delta']
            assert abs(section['leading_edge_xyz'][2] - expected_z) < 1e-12
            assert common.WING_CHORD_RANGE[0] <= section["chord"] <= common.WING_CHORD_RANGE[1]
        assert lower_z <= sections[0]['leading_edge_xyz'][2] <= upper_z
        assert plan['span_fractions'][0] == 0.0
        assert plan['span_fractions'][-1] == 1.0
        assert all(
            start < end
            for start, end in zip(plan['span_fractions'][:-1], plan['span_fractions'][1:], strict=True)
        )
        assert plan['span_fractions'] == [index / (util.WING_SECTION_COUNT - 1) for index in range(util.WING_SECTION_COUNT)]


def test_all_conventional_and_canard_wing_roots_are_inside_local_fuselage_height():
    for layout in conventional_canard.LAYOUTS:
        reference = conventional_canard.build_random_reference(random.Random(23), layout)
        for plan_name in ('main_wing_plan', 'auxiliary_wing_plan'):
            plan = reference[plan_name]
            lower_z, upper_z = common.wing_root_z_bounds(reference['fuselage_xsecs'], plan)
            assert lower_z <= plan['root_leading_edge_z'] <= upper_z


def test_flying_wing_converter_emits_padded_configured_cst_sections(monkeypatch):
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
        source_path
        for component in payload['half_components']
        if component["component"] == util.COMPONENT_WING
        for source_path in component['airfoil_sources']
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
        box["z_section"].shape == (util.WING_SECTION_COUNT, util.WING_SECTION_SIZE)
        for box in wing_boxes
    )

    encoded_wing = wing_boxes[0]["z_section"]
    source_sections = next(
        component
        for component in payload['half_components']
        if component['name'] == 'main_wing_right'
    )
    expected_sections = torch.stack(
        [
            torch.cat([torch.tensor(section), section_codes[source_path]])
            for section, source_path in zip(source_sections['z_section'], source_sections['airfoil_sources'], strict=True)
        ]
    )
    assert torch.allclose(encoded_wing, expected_sections)

    fuselage_box = next(
        box for box in sample["boxes"] if box["component"] == util.COMPONENT_FUSELAGE
    )
    source_fuselage_sections = next(
        component['z_section']
        for component in payload["half_components"]
        if component["component"] == util.COMPONENT_FUSELAGE
    )
    assert torch.allclose(fuselage_box["z_section"], torch.tensor(source_fuselage_sections))
