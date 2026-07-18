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


def test_conventional_converter_emits_two_24d_cst_sections(monkeypatch):
    reference = common.build_random_reference(random.Random(17))
    payload = common.build_geometry_payload(
        reference,
        common.build_obb_tree(reference),
        sample_index=0,
        random_seed=17,
        wing_airfoil_sources={
            component: {
                section: f'{component}_{section}.dat'
                for section in common.WING_AIRFOIL_SECTIONS
            }
            for component in common.WING_AIRFOIL_COMPONENTS
        },
    )
    source_paths = [
        source_path
        for sections in payload['wing_airfoil_sources'].values()
        for source_path in sections.values()
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

    sample = json_to_typed_obb_dataset.build_structured_sample(payload, {}, symmetry_line_y=0.0)

    wing_boxes = [
        box for box in sample['boxes']
        if box[util.BOX_COMPONENT_KEY] == util.COMPONENT_WING
    ]
    assert len(wing_boxes) == len(common.WING_AIRFOIL_COMPONENTS)
    assert all(box[util.BOX_AIRFOIL_KEY].shape == (util.WING_AIRFOIL_CODE_SIZE,) for box in wing_boxes)

    vertical_tail = wing_boxes[1][util.BOX_AIRFOIL_KEY]
    expected_vertical_tail = torch.cat(
        [
            cst_airfoil_codec.symmetric_airfoil_code_from_upper(
                section_codes[payload['wing_airfoil_sources']['vertical_tail'][section]],
                symmetry_line_y=0.0,
            )
            for section in common.WING_AIRFOIL_SECTIONS
        ]
    )
    assert torch.allclose(vertical_tail, expected_vertical_tail)
