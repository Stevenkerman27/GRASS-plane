"""Generate fixed-topology OpenVSP aircraft and matching geometry JSON files."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import aircraft_dataset_common as common


DEFAULT_AIRFOIL_DIR = REPO_ROOT / "foildata" / "processed_foil"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed conventional OpenVSP aircraft samples.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "conventional_dataset")
    parser.add_argument("--airfoil-dir", type=Path, default=DEFAULT_AIRFOIL_DIR)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_airfoils(airfoil_dir):
    if not airfoil_dir.is_dir():
        raise NotADirectoryError(f"Airfoil directory does not exist: {airfoil_dir}")
    airfoils = sorted(airfoil_dir.glob("*.dat"))
    if not airfoils:
        raise ValueError(f"Airfoil directory contains no .dat files: {airfoil_dir}")
    return airfoils


def sample_seed(root_seed, sample_index):
    return root_seed + sample_index


def generate_sample(output_dir, airfoils, root_seed, sample_index, overwrite):
    seed = sample_seed(root_seed, sample_index)
    rng = random.Random(seed)
    reference = common.build_random_reference(rng)
    wing_airfoil_sources = common.sample_wing_airfoil_sources(rng, airfoils)
    assembler = common.build_obb_tree(reference)
    sample_name = f"sample_{sample_index:04d}"
    vsp_path = output_dir / f"{sample_name}.vsp3"
    json_path = output_dir / f"{sample_name}.json"
    if (vsp_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError(f"Sample output already exists; use --overwrite to replace it: {sample_name}")
    common.create_openvsp_aircraft(reference, wing_airfoil_sources, vsp_path)
    payload = common.build_geometry_payload(
        reference, assembler, sample_index, seed, wing_airfoil_sources
    )
    common.write_geometry_payload(payload, json_path)
    return vsp_path, json_path


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError(f"count must be positive, got {args.count}")
    if args.start_index < 0:
        raise ValueError(f"start-index must be non-negative, got {args.start_index}")
    airfoils = load_airfoils(args.airfoil_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Python executable:", sys.executable, flush=True)
    print("Airfoil candidates:", len(airfoils), flush=True)
    for sample_index in range(args.start_index, args.start_index + args.count):
        vsp_path, json_path = generate_sample(
            args.output_dir, airfoils, args.seed, sample_index, args.overwrite
        )
        print(f"[{sample_index:04d}] {vsp_path.name} / {json_path.name}", flush=True)


if __name__ == "__main__":
    main()
