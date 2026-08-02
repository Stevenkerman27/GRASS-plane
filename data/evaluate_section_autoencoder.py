"""Batch diagnostics for independent wing and fuselage section autoencoders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grassdata
import section_autoencoder
import section_autoencoder_evaluation
import section_parameter_codec
import util
from train_autoencoder import choose_device
from train_section_autoencoder import (
    make_aircraft_splits,
    section_ae_checkpoint_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate independent section-AE reconstruction errors on its validation split.'
    )
    parser.add_argument(
        '--mode', choices=('deterministic',), default='deterministic'
    )
    parser.add_argument(
        '--sequence_type', choices=('wing', 'fuselage', 'both'), default='both'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Evaluate only the first N deterministic validation leaves per sequence type.',
    )
    parser.add_argument(
        '--no-overwrite', action='store_false', dest='overwrite',
        help='Fail when evaluation JSON or CSV output files already exist.',
    )
    parser.set_defaults(overwrite=True)
    evaluation_args, remaining_args = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_args]
        config = util.get_args()
    finally:
        sys.argv = original_argv
    if evaluation_args.max_samples is not None and evaluation_args.max_samples < 1:
        raise ValueError('--max_samples must be at least 1 when supplied.')
    return evaluation_args, config


def evaluation_sequence_types(value):
    if value == 'both':
        return (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE)
    return (value,)


def assert_statistics_match(checkpoint_statistics, expected_statistics, sequence_type):
    section_parameter_codec.validate_section_parameter_statistics(
        checkpoint_statistics, sequence_type
    )
    section_parameter_codec.validate_section_parameter_statistics(
        expected_statistics, sequence_type
    )
    fields_match = all(
        torch.equal(
            torch.as_tensor(checkpoint_statistics[f'{prefix}_constant_mask']),
            torch.as_tensor(expected_statistics[f'{prefix}_constant_mask']),
        )
        and torch.allclose(
            torch.as_tensor(checkpoint_statistics[f'{prefix}_mean']),
            torch.as_tensor(expected_statistics[f'{prefix}_mean']), atol=1e-7, rtol=0.0,
        )
        and torch.allclose(
            torch.as_tensor(checkpoint_statistics[f'{prefix}_std']),
            torch.as_tensor(expected_statistics[f'{prefix}_std']), atol=1e-7, rtol=0.0,
        )
        for prefix in ('global', 'section')
    )
    if not fields_match:
        raise ValueError(
            f'{sequence_type} checkpoint parameter statistics do not match the current '
            'training split. Use the exact training data paths, seed, and validation fraction.'
        )


def output_paths(evaluation_dir, mode, sequence_type):
    stem = f'{mode}_{sequence_type}'
    return evaluation_dir / f'{stem}_summary.json', evaluation_dir / f'{stem}_samples.csv'


def assert_output_paths_writable(paths, overwrite):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ', '.join(str(path) for path in existing)
        raise FileExistsError(f'Evaluation output already exists: {formatted}. Pass --overwrite to replace it.')


def write_result(result, summary_path, samples_path):
    rows = result['sample_rows']
    if not rows:
        raise RuntimeError('Evaluation produced no sample rows.')
    summary = {key: value for key, value in result.items() if key != 'sample_rows'}
    with summary_path.open('w', encoding='utf-8', newline='') as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write('\n')
    with samples_path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result, source_name, checkpoint):
    loss = result['normalized_training_loss']['total']['mean']
    parameter_error = result['physical_parameter_error']
    print(
        f'{result["mode"]} {result["sequence_type"]}: checkpoint_epoch={checkpoint["epoch"]}; '
        f'{source_name}_leaves={result["leaf_count"]}; normalized_total={loss:.6g}; '
        f'global_mae={parameter_error["global_mae"]["mean"]:.6g}; '
        f'section_mae={parameter_error["section_mae"]["mean"]:.6g}',
        flush=True,
    )


def main():
    evaluation_args, config = parse_args()
    device = choose_device(config)
    training_aircraft, validation_aircraft = make_aircraft_splits(config)
    evaluation_aircraft = training_aircraft if config.overfit else validation_aircraft
    source_name = 'overfit' if config.overfit else 'validation'
    checkpoint_dir = section_ae_checkpoint_dir(config)
    evaluation_dir = checkpoint_dir / 'evaluation'

    sequence_types = evaluation_sequence_types(evaluation_args.sequence_type)
    modes = (evaluation_args.mode,)
    output_path_pairs = [
        output_paths(evaluation_dir, mode, sequence_type)
        for sequence_type in sequence_types
        for mode in modes
    ]
    assert_output_paths_writable(
        [path for pair in output_path_pairs for path in pair], evaluation_args.overwrite
    )

    loaded = {}
    for sequence_type in sequence_types:
        leaves = section_autoencoder.extract_section_leaves(evaluation_aircraft, sequence_type)
        training_leaves = section_autoencoder.extract_section_leaves(training_aircraft, sequence_type)
        expected_statistics = section_parameter_codec.fit_section_parameter_statistics(
            training_leaves, sequence_type
        )
        checkpoint_path = section_autoencoder.final_checkpoint_path(
            checkpoint_dir, sequence_type
        )
        model, checkpoint = section_autoencoder.load_section_autoencoder(
            checkpoint_path, sequence_type, device
        )
        assert_statistics_match(
            checkpoint['parameter_statistics'], expected_statistics, sequence_type
        )
        if evaluation_args.max_samples is not None:
            leaves = Subset(
                leaves, range(min(evaluation_args.max_samples, len(leaves)))
            )
        if len(leaves) == 0:
            raise RuntimeError(f'{sequence_type} evaluation split has no leaves.')
        loaded[sequence_type] = (
            DataLoader(leaves, batch_size=config.ae_batch_size, shuffle=False), model, checkpoint
        )

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    print(f'Using {device}; {source_name} aircraft={len(evaluation_aircraft)}', flush=True)
    for sequence_type in sequence_types:
        loader, model, checkpoint = loaded[sequence_type]
        for mode in modes:
            result = section_autoencoder_evaluation.evaluate_loader(model, loader, device, mode)
            print_result(result, source_name, checkpoint)
            summary_path, samples_path = output_paths(evaluation_dir, mode, sequence_type)
            write_result(result, summary_path, samples_path)
            print(f'  wrote {summary_path}', flush=True)
            print(f'  wrote {samples_path}', flush=True)


if __name__ == '__main__':
    main()
