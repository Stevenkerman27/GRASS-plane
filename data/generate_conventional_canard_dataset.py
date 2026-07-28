"""Generate balanced conventional/canard OpenVSP and JSON samples."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
DEFAULT_AIRFOIL_DIR = REPO_ROOT / 'foildata' / 'processed_foil'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'conventional_canard_dataset'
DEFAULT_ROOT_SEED = 20260727
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import conventional_canard_dataset_common as common


def parse_args():
    parser = argparse.ArgumentParser(description='Generate conventional/canard OpenVSP and JSON samples.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--airfoil-dir', type=Path, default=DEFAULT_AIRFOIL_DIR)
    parser.add_argument('--count', type=int, default=common.DATASET_SAMPLE_COUNT)
    parser.add_argument('--seed', type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_airfoils(airfoil_dir):
    if not airfoil_dir.is_dir():
        raise NotADirectoryError(f'Airfoil directory does not exist: {airfoil_dir}')
    airfoils = sorted(airfoil_dir.glob('*.dat'))
    if not airfoils:
        raise ValueError(f'Airfoil directory contains no .dat files: {airfoil_dir}')
    return airfoils


def generate_sample(output_dir, airfoils, root_seed, sample_index, overwrite):
    seed = root_seed + sample_index
    rng = random.Random(seed)
    layout = common.layout_for_sample_index(sample_index)
    reference = common.build_random_reference(rng, layout)
    sources = common.sample_wing_airfoil_sources(reference, rng, airfoils)
    wings = common.build_wing_sections(reference, sources)
    assembler = common.build_obb_tree()
    sample_name = f'sample_{sample_index:04d}'
    vsp_path = output_dir / f'{sample_name}.vsp3'
    json_path = output_dir / f'{sample_name}.json'
    if (vsp_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError(f'Sample output already exists; use --overwrite to replace it: {sample_name}')
    common.create_openvsp_aircraft(reference, wings, vsp_path)
    payload = common.build_geometry_payload(reference, assembler, sample_index, seed, wings)
    common.write_geometry_payload(payload, json_path)
    return layout, vsp_path, json_path


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError(f'count must be positive, got {args.count}')
    if args.start_index < 0:
        raise ValueError(f'start-index must be non-negative, got {args.start_index}')
    airfoils = load_airfoils(args.airfoil_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Python executable:', sys.executable, flush=True)
    print('Airfoil candidates:', len(airfoils), flush=True)
    for sample_index in range(args.start_index, args.start_index + args.count):
        layout, vsp_path, json_path = generate_sample(
            args.output_dir, airfoils, args.seed, sample_index, args.overwrite
        )
        print(f'[{sample_index:04d}] {layout}: {vsp_path.name} / {json_path.name}', flush=True)


if __name__ == '__main__':
    main()
