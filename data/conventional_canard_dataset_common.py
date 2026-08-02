"""Schema and geometry construction for conventional and canard aircraft."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import aircraft_dataset_common as flying_wing
import util


DATASET_SAMPLE_COUNT = flying_wing.DATASET_SAMPLE_COUNT
GEOMETRY_SCHEMA = 'conventional_canard_global_section_v1'
TOPOLOGY_NAME = 'symmetric_main_auxiliary_wing_fuselage_global_section_v1'
LAYOUT_CONVENTIONAL = 'conventional'
LAYOUT_CANARD = 'canard'
LAYOUTS = (LAYOUT_CONVENTIONAL, LAYOUT_CANARD)
FUSELAGE_COMPONENT_NAME = 'fuselage'
MAIN_WING_COMPONENT_NAME = 'main_wing_right'
AUXILIARY_WING_COMPONENT_NAME = 'auxiliary_wing_right'
BOX_ORDER = (
    FUSELAGE_COMPONENT_NAME,
    MAIN_WING_COMPONENT_NAME,
    AUXILIARY_WING_COMPONENT_NAME,
)
WING_COMPONENTS = (MAIN_WING_COMPONENT_NAME, AUXILIARY_WING_COMPONENT_NAME)
MIN_LONGITUDINAL_SEPARATION_FRACTION = 0.10
MAX_LONGITUDINAL_SEPARATION_FRACTION = 0.45
AUXILIARY_WINGSPAN_RATIO_RANGE = (0.20, 0.80)
LAYOUT_SAMPLING_ATTEMPTS = 1000


def layout_for_sample_index(sample_index):
    if sample_index < 0:
        raise ValueError(f'sample_index must be non-negative, got {sample_index}')
    return LAYOUT_CONVENTIONAL if sample_index % 2 == 0 else LAYOUT_CANARD


def _sample_auxiliary_plan(rng, fuselage_length, main_plan, layout):
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown aircraft layout: {layout}')
    plan = flying_wing._sample_quadratic_wing_plan(
        rng,
        fuselage_length,
        main_plan['root_twist'],
        main_plan['tip_twist'],
        main_plan['semi_span'] * rng.uniform(*AUXILIARY_WINGSPAN_RATIO_RANGE),
    )
    plan['section_count'] = util.WING_SECTION_COUNT
    plan['span_fractions'] = flying_wing.sample_wing_span_fractions(rng, plan['section_count'])
    plan['twist_degrees'] = flying_wing._sample_bounded_sequence(
        rng,
        util.WING_SECTION_COUNT,
        flying_wing.WING_TWIST_DEG_RANGE,
        flying_wing.WING_TWIST_MAX_ADJACENT_DELTA_DEG,
    )
    plan['root_twist'] = math.radians(plan['twist_degrees'][0])
    plan['tip_twist'] = math.radians(plan['twist_degrees'][-1])
    flying_wing.configure_wing_section_jitter(rng, plan)
    minimum_separation = MIN_LONGITUDINAL_SEPARATION_FRACTION * fuselage_length
    maximum_separation = MAX_LONGITUDINAL_SEPARATION_FRACTION * fuselage_length
    main_min_x, main_max_x = wing_longitudinal_envelope(
        flying_wing.build_wing_planform_sections(main_plan)
    )
    auxiliary_sections = flying_wing.build_wing_planform_sections(plan)
    auxiliary_min_x, auxiliary_max_x = wing_longitudinal_envelope(auxiliary_sections)
    auxiliary_root_min_x, auxiliary_root_max_x = root_chord_longitudinal_envelope(auxiliary_sections)
    if layout == LAYOUT_CONVENTIONAL:
        minimum_shift = max(
            main_max_x + minimum_separation - auxiliary_min_x,
            -auxiliary_root_min_x,
        )
        maximum_shift = min(
            main_max_x + maximum_separation - auxiliary_min_x,
            fuselage_length - auxiliary_root_max_x,
        )
    else:
        minimum_shift = max(
            main_min_x - maximum_separation - auxiliary_max_x,
            -auxiliary_root_min_x,
        )
        maximum_shift = min(
            main_min_x - minimum_separation - auxiliary_max_x,
            fuselage_length - auxiliary_root_max_x,
        )
    if minimum_shift > maximum_shift:
        return None
    plan['root_quarter_chord_x'] += rng.uniform(minimum_shift, maximum_shift)
    return plan


def wing_longitudinal_envelope(sections):
    span_x_squared = _wing_span_x_squared(sections)
    x_coordinates = []
    for section in sections:
        x_coordinates.extend(_wing_section_longitudinal_extent(section, span_x_squared))
    return min(x_coordinates), max(x_coordinates)


def root_chord_longitudinal_envelope(sections):
    return _wing_section_longitudinal_extent(sections[0], _wing_span_x_squared(sections))


def _wing_span_x_squared(sections):
    if len(sections) < 2:
        raise ValueError(f'wing requires at least two sections, got {len(sections)}')
    first_leading_edge = sections[0]['leading_edge_xyz']
    last_leading_edge = sections[-1]['leading_edge_xyz']
    span_vector = [
        last_leading_edge[index] - first_leading_edge[index]
        for index in range(3)
    ]
    span_length_squared = sum(value * value for value in span_vector)
    if span_length_squared <= 0.0:
        raise ValueError('wing span axis must have positive length')
    return span_vector[0] ** 2 / span_length_squared


def _wing_section_longitudinal_extent(section, span_x_squared):
    leading_edge_x = section['leading_edge_xyz'][0]
    twist = section['twist']
    chord_x = section['chord'] * (
        math.cos(twist) + span_x_squared * (1.0 - math.cos(twist))
    )
    return min(leading_edge_x, leading_edge_x + chord_x), max(leading_edge_x, leading_edge_x + chord_x)


def build_random_reference(rng, layout):
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown aircraft layout: {layout}')
    for _ in range(LAYOUT_SAMPLING_ATTEMPTS):
        reference = flying_wing.build_random_reference(rng)
        reference['layout'] = layout
        reference['main_wing_plan'] = reference.pop('wing_plan')
        fuselage_length = reference['fuselage_xsecs'][-1]['x']
        auxiliary_plan = _sample_auxiliary_plan(
            rng,
            fuselage_length,
            reference['main_wing_plan'],
            layout,
        )
        if auxiliary_plan is None:
            continue
        auxiliary_root_min_x, auxiliary_root_max_x = root_chord_longitudinal_envelope(
            flying_wing.build_wing_planform_sections(auxiliary_plan)
        )
        if 0.0 <= auxiliary_root_min_x and auxiliary_root_max_x <= fuselage_length:
            flying_wing.sample_wing_root_leading_edge_z(
                rng, reference['fuselage_xsecs'], auxiliary_plan
            )
            reference['auxiliary_wing_plan'] = auxiliary_plan
            return reference
    raise RuntimeError(
        f'Unable to sample a {layout} layout with a fuselage-connected auxiliary root chord '
        f'after {LAYOUT_SAMPLING_ATTEMPTS} attempts'
    )


def _sample_airfoil_sources(rng, airfoils, count, label):
    if not airfoils:
        raise ValueError('airfoils must not be empty')
    return [
        str(airfoils[rng.randrange(len(airfoils))].resolve())
        for _ in range(count)
    ]


def sample_wing_airfoil_sources(reference, rng, airfoils):
    return {
        MAIN_WING_COMPONENT_NAME: _sample_airfoil_sources(
            rng, airfoils, reference['main_wing_plan']['section_count'], MAIN_WING_COMPONENT_NAME
        ),
        AUXILIARY_WING_COMPONENT_NAME: _sample_airfoil_sources(
            rng, airfoils, reference['auxiliary_wing_plan']['section_count'], AUXILIARY_WING_COMPONENT_NAME
        ),
    }


def _build_wing_sections(plan, source_paths, label):
    if len(source_paths) != plan['section_count']:
        raise ValueError(f'{label} airfoil source count does not match its plan')
    sections = flying_wing.build_wing_planform_sections(plan)
    for section, source_path in zip(sections, source_paths, strict=True):
        section['airfoil_source'] = source_path
    return sections


def build_wing_sections(reference, wing_airfoil_sources):
    if set(wing_airfoil_sources) != set(WING_COMPONENTS):
        raise ValueError('wing_airfoil_sources do not match conventional/canard components')
    return {
        MAIN_WING_COMPONENT_NAME: _build_wing_sections(
            reference['main_wing_plan'], wing_airfoil_sources[MAIN_WING_COMPONENT_NAME], MAIN_WING_COMPONENT_NAME
        ),
        AUXILIARY_WING_COMPONENT_NAME: _build_wing_sections(
            reference['auxiliary_wing_plan'], wing_airfoil_sources[AUXILIARY_WING_COMPONENT_NAME], AUXILIARY_WING_COMPONENT_NAME
        ),
    }


def build_obb_tree():
    assembler = flying_wing.TreeAssembler()
    assembler.push_box(FUSELAGE_COMPONENT_NAME)
    assembler.push_box(MAIN_WING_COMPONENT_NAME)
    assembler.apply_sym(flying_wing.MIRROR_Y_SYM)
    assembler.apply_adj()
    assembler.push_box(AUXILIARY_WING_COMPONENT_NAME)
    assembler.apply_sym(flying_wing.MIRROR_Y_SYM)
    assembler.apply_adj()
    return assembler


def sequence_type_for_label(label):
    sequence_types = {
        FUSELAGE_COMPONENT_NAME: 'fuselage',
        MAIN_WING_COMPONENT_NAME: 'wing',
        AUXILIARY_WING_COMPONENT_NAME: 'wing',
    }
    if label not in sequence_types:
        raise ValueError(f'Unknown conventional/canard component label: {label}')
    return sequence_types[label]


def component_for_label(label):
    if label == FUSELAGE_COMPONENT_NAME:
        return util.COMPONENT_FUSELAGE
    if label in WING_COMPONENTS:
        return util.COMPONENT_WING
    raise ValueError(f'Unknown conventional/canard component label: {label}')


def build_geometry_component(label, reference, wings):
    sequence_type = sequence_type_for_label(label)
    if label == FUSELAGE_COMPONENT_NAME:
        latent = reference['fuselage_latent']
        flying_wing.validate_fuselage_latent(latent)
    else:
        latent = flying_wing.build_wing_latent(wings[label])
    return {
        'name': label,
        'component': component_for_label(label),
        'sequence_type': sequence_type,
        **latent,
    }


def build_geometry_payload(reference, assembler, sample_index, random_seed, wings):
    if reference['layout'] != layout_for_sample_index(sample_index):
        raise ValueError('reference layout does not match the balanced sample-index allocation')
    if tuple(assembler.labels) != BOX_ORDER:
        raise ValueError('Tree labels do not match conventional/canard components')
    return {
        'schema': GEOMETRY_SCHEMA,
        'topology': TOPOLOGY_NAME,
        'layout': reference['layout'],
        'units': flying_wing.LENGTH_UNIT,
        'sample_index': sample_index,
        'random_seed': random_seed,
        'max_wingspan': flying_wing.MAX_WINGSPAN,
        'tess_int': flying_wing.TESS_INT,
        'half_components': [build_geometry_component(label, reference, wings) for label in BOX_ORDER],
        'ops': assembler.ops,
        'syms': assembler.syms,
        'box_order': list(BOX_ORDER),
    }


def write_geometry_payload(payload, path):
    validate_geometry_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def validate_wing_sections(label, sections):
    if len(sections) != util.WING_SECTION_COUNT:
        raise ValueError(f'{label} must contain {util.WING_SECTION_COUNT} sections')
    flying_wing.validate_single_wing_sections(sections)


def _validate_layout_positions(payload):
    components = {component['name']: component for component in payload['half_components']}
    fuselage_length = components[FUSELAGE_COMPONENT_NAME]['z_global'][3]
    main_min_x, main_max_x = wing_longitudinal_envelope(
        flying_wing.wing_geometry_from_latent(components[MAIN_WING_COMPONENT_NAME])
    )
    auxiliary_min_x, auxiliary_max_x = wing_longitudinal_envelope(
        flying_wing.wing_geometry_from_latent(components[AUXILIARY_WING_COMPONENT_NAME])
    )
    minimum_separation = MIN_LONGITUDINAL_SEPARATION_FRACTION * fuselage_length
    maximum_separation = MAX_LONGITUDINAL_SEPARATION_FRACTION * fuselage_length
    auxiliary_root_min_x, auxiliary_root_max_x = root_chord_longitudinal_envelope(
        flying_wing.wing_geometry_from_latent(components[AUXILIARY_WING_COMPONENT_NAME])
    )
    if auxiliary_root_min_x < 0.0 or auxiliary_root_max_x > fuselage_length:
        raise ValueError('auxiliary root chord must remain within the fuselage longitudinal range')
    gap = (
        auxiliary_min_x - main_max_x
        if payload['layout'] == LAYOUT_CONVENTIONAL
        else main_min_x - auxiliary_max_x
    )
    if not minimum_separation <= gap <= maximum_separation:
        raise ValueError('auxiliary wing edge gap is outside the configured longitudinal range')


def validate_geometry_payload(payload):
    required = (
        'schema', 'topology', 'layout', 'units', 'sample_index', 'random_seed', 'max_wingspan',
        'tess_int', 'half_components', 'ops', 'syms', 'box_order',
    )
    for key in required:
        if key not in payload:
            raise KeyError(f'Geometry JSON missing required key: {key}')
    if payload['schema'] != GEOMETRY_SCHEMA or payload['topology'] != TOPOLOGY_NAME:
        raise ValueError('Geometry JSON does not use the conventional/canard schema')
    if payload['layout'] != layout_for_sample_index(payload['sample_index']):
        raise ValueError('layout is inconsistent with the required balanced allocation')
    if payload['units'] != flying_wing.LENGTH_UNIT or payload['max_wingspan'] != flying_wing.MAX_WINGSPAN:
        raise ValueError('Geometry JSON has incompatible physical units or wingspan')
    if tuple(payload['box_order']) != BOX_ORDER:
        raise ValueError('box_order must contain fuselage, main wing, then auxiliary wing')
    if len(payload['half_components']) != len(BOX_ORDER):
        raise ValueError('half_components has an unexpected size')
    expected_ops = [flying_wing.BOX_OP, flying_wing.BOX_OP, flying_wing.SYM_OP, flying_wing.ADJ_OP,
                    flying_wing.BOX_OP, flying_wing.SYM_OP, flying_wing.ADJ_OP]
    if payload['ops'] != expected_ops or len(payload['syms']) != 2:
        raise ValueError('ops/syms do not match the approved conventional/canard tree')
    for label, component_payload in zip(BOX_ORDER, payload['half_components'], strict=True):
        if component_payload.get('name') != label:
            raise ValueError('half_components does not match box_order')
        if component_payload.get('component') != component_for_label(label):
            raise ValueError(f'{label} has an unexpected component type')
        if component_payload.get('sequence_type') != sequence_type_for_label(label):
            raise ValueError(f'{label} has an unexpected sequence type')
        if label == FUSELAGE_COMPONENT_NAME:
            flying_wing.validate_fuselage_latent(component_payload)
        else:
            flying_wing.validate_wing_latent(component_payload)
    _validate_layout_positions(payload)


def load_geometry_payload(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_geometry_payload(payload)
    return payload


def create_openvsp_aircraft(reference, wings, output_path):
    import infrastructure as infra

    for label in WING_COMPONENTS:
        validate_wing_sections(label, wings[label])
    infra.case_name = output_path.stem
    infra.file_name = str(output_path)
    infra.ini_geom()
    fuselage_id = flying_wing._create_fuselage_with_correct_xsec_insertion(
        infra,
        {'name': FUSELAGE_COMPONENT_NAME, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'yr': 0.0},
        reference['fuselage_xsecs'],
        flying_wing.TESS_INT,
    )
    flying_wing.set_super_ellipse_fuselage_xsecs(
        infra.vsp, fuselage_id, reference['fuselage_xsecs']
    )
    flying_wing.set_round_end_caps(infra.vsp, fuselage_id, FUSELAGE_COMPONENT_NAME)
    for label in WING_COMPONENTS:
        flying_wing.create_wing_with_section_airfoils(infra, label, wings[label])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    infra.vsp.WriteVSPFile(str(output_path), infra.vsp.SET_ALL)
    if not output_path.is_file():
        raise RuntimeError(f'OpenVSP did not write expected file: {output_path}')
