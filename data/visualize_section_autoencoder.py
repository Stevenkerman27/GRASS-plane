"""Render selected-split wing and fuselage free reconstructions for section AEs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matplotlib.lines import Line2D

import grassdata
import section_autoencoder
import util
from data import visualize_free_decode_error as geometry
from train_autoencoder import choose_device
from train_section_autoencoder import make_aircraft_splits, section_ae_checkpoint_dir


TARGET_COLOR = '0.60'
EXTRA_GENERATED_COLOR = 'tab:purple'
DEFAULT_SAMPLES_PER_TYPE = 4
GEOMETRY_EPSILON = 1e-8
NON_GUI_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template'}
DEFAULT_INTERACTIVE_BACKEND = 'TkAgg'


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        '-train', dest='source', action='store_const', const='train',
        help='Visualize samples from the training split.',
    )
    source_group.add_argument(
        '-val', dest='source', action='store_const', const='validation',
        help='Visualize samples from the validation split (default).',
    )
    source_group.add_argument(
        '-overfit', dest='source', action='store_const', const='overfit',
        help='Visualize the deterministic overfit subset.',
    )
    parser.set_defaults(source='validation')
    parser.add_argument('--samples_per_type', type=int, default=DEFAULT_SAMPLES_PER_TYPE)
    parser.add_argument('--sample_seed', type=int, default=None)
    parser.add_argument('--backend', default=None)
    visualization_args, remaining_args = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_args]
        config = util.get_args()
    finally:
        sys.argv = original_argv
    if visualization_args.samples_per_type < 1:
        raise ValueError('--samples_per_type must be at least 1.')
    return visualization_args, config


def configure_interactive_backend(backend):
    if backend is not None:
        matplotlib.use(backend)
    backend_name = matplotlib.get_backend()
    if backend is None and backend_name.lower() in NON_GUI_BACKENDS:
        matplotlib.use(DEFAULT_INTERACTIVE_BACKEND)
        backend_name = matplotlib.get_backend()
    if backend_name.lower() in NON_GUI_BACKENDS or 'backend_inline' in backend_name.lower():
        raise RuntimeError(f'Matplotlib backend {backend_name!r} cannot show an interactive section-AE window.')
    from matplotlib import cm as color_maps
    from matplotlib import colors as color_normalization
    from matplotlib import pyplot as plot

    return plot, color_maps, color_normalization


def make_component(sequence_type, sections, section_count):
    component = grassdata.sequence_spec(sequence_type)['component']
    transform, translation = geometry.identity_transform()
    return geometry.ComponentGeometry(
        component=component,
        sections=sections.detach().cpu().squeeze(0).numpy(),
        section_count=int(section_count.detach().cpu().reshape(()).item()),
        transform=transform,
        translation=translation,
        path=sequence_type,
    )


def component_scale(component):
    points = geometry.component_points(component)
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if extent < GEOMETRY_EPSILON:
        raise ValueError(f'{component.path} target geometry has zero extent.')
    return extent


def section_geometry_errors(target, generated):
    matched_count = min(target.section_count, generated.section_count)
    scale = component_scale(target)
    errors = []
    for index in range(matched_count):
        target_points = geometry.section_points(target, target.sections[index])
        generated_points = geometry.section_points(generated, generated.sections[index])
        if target_points.shape != generated_points.shape:
            raise RuntimeError('Target and generated section point clouds have incompatible shapes.')
        rms = float(np.sqrt(np.mean(np.sum((generated_points - target_points) ** 2, axis=1))))
        errors.append(rms / scale)
    return np.asarray(errors, dtype=float)


def color_range(errors, color_normalization):
    minimum = float(errors.min())
    maximum = float(errors.max())
    if minimum == maximum:
        margin = max(abs(minimum) * 0.05, GEOMETRY_EPSILON)
        minimum -= margin
        maximum += margin
    return color_normalization.Normalize(vmin=minimum, vmax=maximum, clip=True)


def free_reconstruct(model, sample, device):
    target_sections = sample['sections'].unsqueeze(0).to(device)
    target_count = sample['section_count'].reshape(1).to(device)
    feature = model.section_encoder(target_sections, target_count)
    generated_sections, generated_count, _, _ = model.section_decoder.generate(feature)
    return target_sections, target_count, generated_sections, generated_count


def render_reconstruction(
        sequence_type, sample_index, target, generated, errors, source_name,
        plot, color_maps, color_normalization):
    normalized = color_range(errors, color_normalization)
    cmap = plot.get_cmap('RdYlGn_r')
    matched_count = errors.size
    generated_colors = [cmap(normalized(value)) for value in errors]
    generated_colors.extend([EXTRA_GENERATED_COLOR] * (generated.section_count - matched_count))

    figure = plot.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection='3d')
    geometry.draw_component(
        axis, target, [TARGET_COLOR] * target.section_count, linestyle='--', alpha=0.55
    )
    geometry.draw_component(axis, generated, generated_colors)
    geometry.set_aircraft_axes(axis, [target, generated])
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_zlabel('Z (m)')
    axis.set_title(
        f'{sequence_type} {source_name} leaf {sample_index:03d} | '
        f'target={target.section_count} decoded={generated.section_count} | '
        f'RMS mean={errors.mean():.4f} max={errors.max():.4f}'
    )
    axis.view_init(elev=22, azim=-58)
    colorbar = figure.colorbar(color_maps.ScalarMappable(norm=normalized, cmap=cmap), ax=axis, pad=0.08)
    colorbar.set_label('Normalized section geometry RMS')
    axis.legend(handles=[
        Line2D([0], [0], color=TARGET_COLOR, linestyle='--', label='Target geometry'),
        Line2D([0], [0], color=EXTRA_GENERATED_COLOR, label='Extra generated section'),
    ], loc='upper left')
    figure.tight_layout()
    print(f'Matplotlib backend: {matplotlib.get_backend()}', flush=True)
    print(f'Showing {sequence_type} {source_name} leaf {sample_index:03d}. Close the window to continue.', flush=True)
    plot.show()
    plot.close(figure)


def visualize_sequence_type(
        sequence_type, aircraft, samples_per_type, chooser, device, checkpoint_dir,
        source_name, plot, color_maps, color_normalization):
    checkpoint_path = section_autoencoder.final_checkpoint_path(
        checkpoint_dir, sequence_type
    )
    model, checkpoint = section_autoencoder.load_section_autoencoder(
        checkpoint_path, sequence_type, device
    )
    leaves = section_autoencoder.extract_section_leaves(aircraft, sequence_type)
    sample_count = min(samples_per_type, len(leaves))
    selected_indices = chooser.sample(range(len(leaves)), sample_count)
    print(
        f'{sequence_type}: checkpoint epoch={checkpoint["epoch"]}; '
        f'{source_name} leaves={len(leaves)}; selected={sample_count}', flush=True
    )
    with torch.no_grad():
        for leaf_index in selected_indices:
            target_sections, target_count, generated_sections, generated_count = free_reconstruct(
                model, leaves[leaf_index], device
            )
            target = make_component(sequence_type, target_sections, target_count)
            generated = make_component(sequence_type, generated_sections, generated_count)
            errors = section_geometry_errors(target, generated)
            render_reconstruction(
                sequence_type, leaf_index, target, generated, errors, source_name,
                plot, color_maps, color_normalization
            )
            print(
                f'  target_count={target.section_count} decoded_count={generated.section_count} matched={errors.size} '
                f'extra_generated={max(0, generated.section_count - target.section_count)} '
                f'missing_target={max(0, target.section_count - generated.section_count)} '
                f'mean={errors.mean():.4f} max={errors.max():.4f}',
                flush=True,
            )


def main():
    visualization_args, config = parse_args()
    config.overfit = visualization_args.source == 'overfit'
    config.ae_rnn_type = util.validate_ae_rnn_type(config.ae_rnn_type)
    plot, color_maps, color_normalization = configure_interactive_backend(visualization_args.backend)
    device = choose_device(config)
    training_aircraft, validation_aircraft = make_aircraft_splits(config)
    if visualization_args.source == 'train':
        visualization_aircraft = training_aircraft
        source_name = 'train'
    elif visualization_args.source == 'validation':
        visualization_aircraft = validation_aircraft
        source_name = 'validation'
    else:
        visualization_aircraft = training_aircraft
        source_name = 'overfit'
    chooser = random.Random(visualization_args.sample_seed)
    print(f'Using {device}; {source_name} aircraft={len(visualization_aircraft)}', flush=True)
    for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE):
        visualize_sequence_type(
            sequence_type,
            visualization_aircraft,
            visualization_args.samples_per_type,
            chooser,
            device,
            section_ae_checkpoint_dir(config),
            source_name,
            plot,
            color_maps,
            color_normalization,
        )


if __name__ == '__main__':
    main()
