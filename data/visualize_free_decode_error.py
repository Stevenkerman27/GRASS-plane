"""Visualize geometric errors after free decoding typed aircraft trees."""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cst_airfoil_codec
import grassdata
import grassmodel
import project_paths
import util
from torchfoldext import FoldExt


# Recurrent model checkpoint to visualize: 'rnn' or 'gru'.
MODEL_RNN_TYPE = 'rnn'
REFLECTION_NORMAL_OFFSET = 1
REFLECTION_POINT_OFFSET = 4
GEOMETRY_EPSILON = 1e-8
GENERATED_UNMATCHED_COLOR = 'tab:purple'
TARGET_MATCHED_COLOR = '0.72'
TARGET_UNMATCHED_COLOR = '0.45'
TOTAL_SAMPLES = 4
NON_GUI_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template'}
WING_LEADING_EDGE_SLICE = slice(
    util.CST_AIRFOIL_CODE_SIZE, util.CST_AIRFOIL_CODE_SIZE + 3
)
WING_CHORD_INDEX = util.CST_AIRFOIL_CODE_SIZE + 3
WING_TWIST_INDEX = util.CST_AIRFOIL_CODE_SIZE + 4


@dataclass(frozen=True)
class ComponentGeometry:
    component: int
    sections: np.ndarray
    section_count: int
    transform: np.ndarray
    translation: np.ndarray
    path: str


def parse_args():
    parser = argparse.ArgumentParser(
        description='Encode dataset aircraft, free-decode, and color per-section geometric error.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=REPO_ROOT / 'models' / 'autoencoder' /
        f'best_{MODEL_RNN_TYPE}.pt',
    )
    parser.add_argument('--layout', choices=tuple(project_paths.AIRCRAFT_DATASET_SPECS), default=None)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        raise ValueError('--gpu must be non-negative.')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {args.checkpoint}')
    return args


def choose_device(args):
    if args.no_cuda:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        return torch.device('cpu')
    torch.cuda.set_device(args.gpu)
    return torch.device(f'cuda:{args.gpu}')


def assert_interactive_backend(matplotlib):
    backend = matplotlib.get_backend()
    backend_key = backend.lower()
    if backend_key in NON_GUI_BACKENDS or 'backend_inline' in backend_key:
        raise RuntimeError(f'Matplotlib backend {backend!r} cannot show an interactive error window')


def make_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    required_keys = (
        'encoder_state_dict', 'decoder_state_dict', 'feature_size', 'hidden_size', 'ae_rnn_type',
        'section_statistics',
    )
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f'Checkpoint missing required keys: {missing}')
    config = SimpleNamespace(
        box_code_size=13,
        feature_size=checkpoint['feature_size'],
        hidden_size=checkpoint['hidden_size'],
        symmetry_size=8,
        ae_rnn_type=checkpoint['ae_rnn_type'],
    )
    encoder = grassmodel.GRASSEncoder(config, checkpoint['section_statistics']).to(device)
    decoder = grassmodel.GRASSDecoder(config, checkpoint['section_statistics']).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
    decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, checkpoint


def select_samples(args):
    datasets = {
        layout: grassdata.StructuredGRASSDataset(spec['dataset'])
        for layout, spec in project_paths.AIRCRAFT_DATASET_SPECS.items()
    }
    chooser = random.Random()
    selected = []
    layouts = (args.layout,) if args.layout is not None else tuple(datasets)
    for _ in range(TOTAL_SAMPLES):
        layout = chooser.choice(layouts)
        dataset = datasets[layout]
        index = chooser.randrange(len(dataset))
        selected.append((layout, index, dataset[index]))
    return selected


def identity_transform():
    return np.eye(3, dtype=float), np.zeros(3, dtype=float)


def reflection_transform(symmetry):
    values = torch.as_tensor(symmetry, dtype=torch.float32).detach().cpu().reshape(-1).numpy()
    if values.size != 8:
        raise ValueError(f'Symmetry vector must have 8 values, got {values.size}')
    normal = values[REFLECTION_NORMAL_OFFSET:REFLECTION_NORMAL_OFFSET + 3]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < GEOMETRY_EPSILON:
        raise ValueError('Free-decoded SYM has a zero reflection normal.')
    normal = normal / normal_norm
    point = values[REFLECTION_POINT_OFFSET:REFLECTION_POINT_OFFSET + 3]
    matrix = np.eye(3) - 2.0 * np.outer(normal, normal)
    translation = 2.0 * np.dot(point, normal) * normal
    return matrix, translation


def compose_reflection(component, reflection_matrix, reflection_translation, suffix):
    return replace(
        component,
        transform=reflection_matrix @ component.transform,
        translation=reflection_matrix @ component.translation + reflection_translation,
        path=f'{component.path}{suffix}',
    )


def target_leaf_component(node, transform, translation, path):
    component = int(node.box['component'])
    if component not in grassdata.AE_COMPONENT_TYPES:
        raise ValueError(f'Unsupported component in error visualizer: {util.component_name(component)}')
    return ComponentGeometry(
        component=component,
        sections=node.box['sections'].detach().cpu().squeeze(0).numpy(),
        section_count=int(node.box['section_count'].item()),
        transform=transform,
        translation=translation,
        path=path,
    )


def expand_target_node(node, transform, translation, path='root'):
    if node.is_leaf():
        return [target_leaf_component(node, transform, translation, path)]
    if node.is_adj():
        return [
            *expand_target_node(node.left, transform, translation, f'{path}/L'),
            *expand_target_node(node.right, transform, translation, f'{path}/R'),
        ]
    if node.is_sym():
        original = expand_target_node(node.left, transform, translation, f'{path}/G')
        reflection_matrix, reflection_translation = reflection_transform(node.sym)
        mirrors = [
            compose_reflection(component, reflection_matrix, reflection_translation, '/M')
            for component in original
        ]
        return [*original, *mirrors]
    raise ValueError(f'Unknown target node type: {node.node_type}')


def generated_leaf_component(node, transform, translation, path):
    component = int(node['component'])
    if component not in grassdata.AE_COMPONENT_TYPES:
        raise ValueError(f'Unsupported generated component: {util.component_name(component)}')
    return ComponentGeometry(
        component=component,
        sections=node['sections'].detach().cpu().squeeze(0).numpy(),
        section_count=int(node['section_count'].item()),
        transform=transform,
        translation=translation,
        path=path,
    )


def expand_generated_node(node, transform, translation, path='root'):
    node_type = int(node['node_type'])
    if node_type == grassdata.Tree.NodeType.BOX.value:
        return [generated_leaf_component(node, transform, translation, path)]
    if node_type == grassdata.Tree.NodeType.ADJ.value:
        return [
            *expand_generated_node(node['left'], transform, translation, f'{path}/L'),
            *expand_generated_node(node['right'], transform, translation, f'{path}/R'),
        ]
    if node_type == grassdata.Tree.NodeType.SYM.value:
        original = expand_generated_node(node['generator'], transform, translation, f'{path}/G')
        reflection_matrix, reflection_translation = reflection_transform(node['symmetry'])
        mirrors = [
            compose_reflection(component, reflection_matrix, reflection_translation, '/M')
            for component in original
        ]
        return [*original, *mirrors]
    raise ValueError(f'Unknown generated node type: {node_type}')


def normalize_vector(vector, name):
    norm = np.linalg.norm(vector)
    if norm < GEOMETRY_EPSILON:
        raise ValueError(f'{name} has zero length.')
    return vector / norm


def apply_transform(points, component):
    return points @ component.transform.T + component.translation


def rotate_about_axis(vector, axis, angle):
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def wing_section_points(component, section):
    # Wing sections use the global X/Z airfoil plane: X is the freestream
    # direction and Z is thickness. Sweep and dihedral move section origins,
    # but must not rotate the airfoil profile out of that plane.
    twist_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    chord_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    thickness_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    twist = float(section[WING_TWIST_INDEX])
    chord_axis = rotate_about_axis(chord_axis, twist_axis, twist)
    thickness_axis = rotate_about_axis(thickness_axis, twist_axis, twist)
    curve = cst_airfoil_codec.decode_cst_airfoil_code(
        torch.as_tensor(section[:util.CST_AIRFOIL_CODE_SIZE], dtype=torch.float32),
        num_output_points=util.FREE_DECODE_ERROR_AIRFOIL_POINTS,
    ).squeeze(0).detach().cpu().numpy()
    local_points = (
        section[WING_LEADING_EDGE_SLICE]
        + curve[:, :1] * float(section[WING_CHORD_INDEX]) * chord_axis
        + curve[:, 1:2] * float(section[WING_CHORD_INDEX]) * thickness_axis
    )
    return apply_transform(local_points, component)


def fuselage_section_points(component, section):
    theta = np.linspace(0.0, 2.0 * np.pi, util.FREE_DECODE_ERROR_ELLIPSE_POINTS, endpoint=False)
    local_points = np.column_stack((
        np.full(theta.shape, section[0]),
        section[1] + (section[3] / 2.0) * np.cos(theta),
        section[2] + (section[4] / 2.0) * np.sin(theta),
    ))
    return apply_transform(local_points, component)


def section_points(component, section):
    if component.component == util.COMPONENT_WING:
        return wing_section_points(component, section)
    if component.component == util.COMPONENT_FUSELAGE:
        return fuselage_section_points(component, section)
    raise ValueError(f'Unsupported section component: {component.component}')


def component_points(component):
    sections = component.sections[:component.section_count]
    return np.concatenate([section_points(component, section) for section in sections], axis=0)


def component_bounds(component):
    points = component_points(component)
    return points.min(axis=0), points.max(axis=0)


def aircraft_scale(components):
    points = np.concatenate([component_points(component) for component in components], axis=0)
    scale = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if scale < GEOMETRY_EPSILON:
        raise ValueError('Target aircraft has zero geometric extent.')
    return scale


def component_match_cost(generated, target, scale):
    generated_min, generated_max = component_bounds(generated)
    target_min, target_max = component_bounds(target)
    generated_center = (generated_min + generated_max) / 2.0
    target_center = (target_min + target_max) / 2.0
    generated_extent = generated_max - generated_min
    target_extent = target_max - target_min
    return float(
        np.linalg.norm(generated_center - target_center) / scale
        + np.linalg.norm(generated_extent - target_extent) / scale
    )


def minimum_cost_pairs(generated, targets, scale):
    if not generated or not targets:
        return [], list(generated), list(targets)
    pair_count = min(len(generated), len(targets))
    best = None
    if len(generated) <= len(targets):
        for target_subset in itertools.combinations(targets, pair_count):
            for target_order in itertools.permutations(target_subset):
                pairs = list(zip(generated, target_order, strict=True))
                cost = sum(component_match_cost(item[0], item[1], scale) for item in pairs)
                if best is None or cost < best[0]:
                    best = (cost, pairs)
    else:
        for generated_subset in itertools.combinations(generated, pair_count):
            for target_order in itertools.permutations(targets):
                pairs = list(zip(generated_subset, target_order, strict=True))
                cost = sum(component_match_cost(item[0], item[1], scale) for item in pairs)
                if best is None or cost < best[0]:
                    best = (cost, pairs)
    pairs = best[1]
    paired_generated = {generated.path for generated, _target in pairs}
    paired_targets = {target.path for _generated, target in pairs}
    return (
        pairs,
        [component for component in generated if component.path not in paired_generated],
        [component for component in targets if component.path not in paired_targets],
    )


def match_components(generated, targets):
    scale = aircraft_scale(targets)
    pairs = []
    unmatched_generated = []
    unmatched_targets = []
    for component_type in grassdata.AE_COMPONENT_TYPES:
        generated_of_type = [item for item in generated if item.component == component_type]
        targets_of_type = [item for item in targets if item.component == component_type]
        type_pairs, type_unmatched_generated, type_unmatched_targets = minimum_cost_pairs(
            generated_of_type, targets_of_type, scale
        )
        pairs.extend(type_pairs)
        unmatched_generated.extend(type_unmatched_generated)
        unmatched_targets.extend(type_unmatched_targets)
    return pairs, unmatched_generated, unmatched_targets, scale


def normalized_station_positions(component):
    sections = component.sections[:component.section_count]
    if component.component == util.COMPONENT_FUSELAGE:
        coordinate = sections[:, 0]
    elif component.component == util.COMPONENT_WING:
        coordinate = np.r_[
            0.0,
            np.cumsum(
                np.linalg.norm(
                    np.diff(sections[:, WING_LEADING_EDGE_SLICE], axis=0), axis=1
                )
            ),
        ]
    else:
        raise ValueError(f'Unsupported component: {component.component}')
    span = float(coordinate[-1] - coordinate[0])
    if span < GEOMETRY_EPSILON:
        return np.linspace(0.0, 1.0, component.section_count)
    return (coordinate - coordinate[0]) / span


def interpolate_target_section(target, normalized_position):
    target_sections = target.sections[:target.section_count]
    target_positions = normalized_station_positions(target)
    return np.asarray([
        np.interp(normalized_position, target_positions, target_sections[:, index])
        for index in range(target_sections.shape[1])
    ])


def section_errors(generated, target, scale):
    generated_sections = generated.sections[:generated.section_count]
    generated_positions = normalized_station_positions(generated)
    errors = []
    for generated_section, position in zip(generated_sections, generated_positions, strict=True):
        target_section = interpolate_target_section(target, float(position))
        generated_points = section_points(generated, generated_section)
        target_points = section_points(target, target_section)
        if generated_points.shape != target_points.shape:
            raise RuntimeError('Corresponding section point clouds have incompatible shapes.')
        rms = float(np.sqrt(np.mean(np.sum((generated_points - target_points) ** 2, axis=1))))
        errors.append(rms / scale)
    return np.asarray(errors, dtype=float)


def average_color(first, second):
    from matplotlib.colors import to_rgba

    return np.mean(np.asarray([to_rgba(first), to_rgba(second)]), axis=0)


def draw_fuselage(ax, component, colors, linestyle='-', alpha=1.0):
    sections = component.sections[:component.section_count]
    rings = [fuselage_section_points(component, section) for section in sections]
    for ring, color in zip(rings, colors, strict=True):
        closed = np.vstack((ring, ring[0]))
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linestyle=linestyle, alpha=alpha, linewidth=1.2)
    for first, second, first_color, second_color in zip(rings[:-1], rings[1:], colors[:-1], colors[1:], strict=True):
        edge_color = average_color(first_color, second_color)
        for point_index in range(first.shape[0]):
            ax.plot(
                [first[point_index, 0], second[point_index, 0]],
                [first[point_index, 1], second[point_index, 1]],
                [first[point_index, 2], second[point_index, 2]],
                color=edge_color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=0.7,
            )


def draw_wing(ax, component, colors, linestyle='-', alpha=1.0):
    sections = component.sections[:component.section_count]
    profiles = [wing_section_points(component, section) for section in sections]
    for profile, color in zip(profiles, colors, strict=True):
        ax.plot(profile[:, 0], profile[:, 1], profile[:, 2], color=color, linestyle=linestyle, alpha=alpha, linewidth=0.9)
    for first, second, first_color, second_color in zip(profiles[:-1], profiles[1:], colors[:-1], colors[1:], strict=True):
        edge_color = average_color(first_color, second_color)
        for point_index in (0, first.shape[0] // 2, first.shape[0] - 1):
            ax.plot(
                [first[point_index, 0], second[point_index, 0]],
                [first[point_index, 1], second[point_index, 1]],
                [first[point_index, 2], second[point_index, 2]],
                color=edge_color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=1.3,
            )


def draw_component(ax, component, colors, linestyle='-', alpha=1.0):
    if component.component == util.COMPONENT_FUSELAGE:
        draw_fuselage(ax, component, colors, linestyle=linestyle, alpha=alpha)
        return
    if component.component == util.COMPONENT_WING:
        draw_wing(ax, component, colors, linestyle=linestyle, alpha=alpha)
        return
    raise ValueError(f'Unsupported component: {component.component}')


def set_aircraft_axes(ax, components):
    points = np.concatenate([component_points(component) for component in components], axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) * 0.62, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.45, center[2] + radius * 0.75)
    ax.set_box_aspect((1.0, 1.0, 0.6))


def free_decode_root(encoder, decoder, tree, cuda_enabled):
    encoder_fold = FoldExt(cuda=cuda_enabled)
    encoded = grassmodel.encode_structure_fold(encoder_fold, tree, use_sampler=False)
    root_feature = encoder_fold.apply(encoder, [[encoded]])[0]
    return decoder.decode_free(root_feature)[0]


def free_tree_summary(node):
    node_type = int(node['node_type'])
    if node_type == grassdata.Tree.NodeType.BOX.value:
        return util.component_name(int(node['component']))
    if node_type == grassdata.Tree.NodeType.ADJ.value:
        return f"ADJ({free_tree_summary(node['left'])}, {free_tree_summary(node['right'])})"
    if node_type == grassdata.Tree.NodeType.SYM.value:
        return f"SYM({free_tree_summary(node['generator'])})"
    raise ValueError(f'Unknown generated node type: {node_type}')


def render_sample(layout, index, tree, generated_tree):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D

    transform, translation = identity_transform()
    targets = expand_target_node(tree.root, transform, translation)
    generated = expand_generated_node(generated_tree['root'], transform, translation)
    pairs, unmatched_generated, unmatched_targets, scale = match_components(generated, targets)
    errors_by_generated_path = {
        generated_component.path: section_errors(generated_component, target_component, scale)
        for generated_component, target_component in pairs
    }
    all_errors = np.concatenate(list(errors_by_generated_path.values())) if errors_by_generated_path else np.array([])
    if all_errors.size:
        error_min = float(all_errors.min())
        error_max = float(all_errors.max())
        if error_min == error_max:
            margin = max(abs(error_min) * 0.05, GEOMETRY_EPSILON)
            error_min -= margin
            error_max += margin
    else:
        error_min, error_max = 0.0, 1.0
    normalized = colors.Normalize(vmin=error_min, vmax=error_max, clip=True)
    cmap = plt.get_cmap('RdYlGn_r')

    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection='3d')
    for target in targets:
        target_color = TARGET_UNMATCHED_COLOR if target in unmatched_targets else TARGET_MATCHED_COLOR
        target_style = '--' if target in unmatched_targets else '-'
        draw_component(
            axis,
            target,
            [target_color] * target.section_count,
            linestyle=target_style,
            alpha=0.5,
        )
    for component in generated:
        if component.path in errors_by_generated_path:
            colors_for_sections = [cmap(normalized(value)) for value in errors_by_generated_path[component.path]]
        else:
            colors_for_sections = [GENERATED_UNMATCHED_COLOR] * component.section_count
        draw_component(axis, component, colors_for_sections)

    set_aircraft_axes(axis, [*targets, *generated])
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_zlabel('Z (m)')
    error_text = (
        'no matched components'
        if all_errors.size == 0
        else f'RMS min={all_errors.min():.4f} max={all_errors.max():.4f} mean={all_errors.mean():.4f}'
    )
    axis.set_title(f'{project_paths.AIRCRAFT_DATASET_SPECS[layout]["label"]} sample {index:04d} | {error_text}')
    axis.view_init(elev=22, azim=-58)
    colorbar = figure.colorbar(cm.ScalarMappable(norm=normalized, cmap=cmap), ax=axis, pad=0.08)
    colorbar.set_label('Section geometry RMS for this sample')
    axis.legend(handles=[
        Line2D([0], [0], color=TARGET_MATCHED_COLOR, label='Matched target geometry'),
        Line2D([0], [0], color=TARGET_UNMATCHED_COLOR, linestyle='--', label='Unmatched target'),
        Line2D([0], [0], color=GENERATED_UNMATCHED_COLOR, label='Unmatched generated'),
    ], loc='upper left')
    figure.tight_layout()
    print(f'  free tree: {free_tree_summary(generated_tree["root"])}', flush=True)
    print(f'  matched={len(pairs)} extra_generated={len(unmatched_generated)} missing_target={len(unmatched_targets)}', flush=True)
    for generated_component, target_component in pairs:
        component_errors = errors_by_generated_path[generated_component.path]
        print(
            f'  {generated_component.path} -> {target_component.path} '
            f'({util.component_name(generated_component.component)}): '
            f'mean={component_errors.mean():.4f} max={component_errors.max():.4f}',
            flush=True,
        )
    print(f'Showing sample {index:04d}. Close the window to continue.', flush=True)
    plt.show()
    plt.close(figure)


def main():
    args = parse_args()
    import matplotlib
    if args.backend is not None:
        matplotlib.use(args.backend)
    assert_interactive_backend(matplotlib)
    device = choose_device(args)
    encoder, decoder, checkpoint = make_model(args.checkpoint, device)
    print(
        f'Using {device}; checkpoint epoch={checkpoint.get("epoch", "unknown")}; '
        f'feature_size={checkpoint["feature_size"]}; hidden_size={checkpoint["hidden_size"]}',
        flush=True,
    )
    selected = select_samples(args)
    with torch.no_grad():
        for layout, index, tree in selected:
            generated_tree = free_decode_root(encoder, decoder, tree, device.type == 'cuda')
            render_sample(layout, index, tree, generated_tree)


if __name__ == '__main__':
    main()
