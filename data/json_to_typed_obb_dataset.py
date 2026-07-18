"""Convert fixed-topology OpenVSP geometry JSON files into a structured GRASS .pt dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cst_airfoil_code_cache
import cst_airfoil_codec
import grassdata
import util
import aircraft_dataset_common as common


def parse_args():
    parser = argparse.ArgumentParser(description="Convert conventional geometry JSON files to structured OBB .pt data.")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "conventional_dataset")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "conventional_dataset" / "conventional_dataset.pt")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cst_airfoil_code_cache.pt")
    parser.add_argument("--expected-count", type=int, default=200)
    return parser.parse_args()


def component_airfoil_code(component_name, section_sources, airfoil_cache, symmetry_line_y):
    section_codes = []
    for section_name in common.WING_AIRFOIL_SECTIONS:
        source_path = section_sources[section_name]
        code = cst_airfoil_code_cache.lookup_code(airfoil_cache, source_path)
        if code is None:
            raise KeyError(
                f"Airfoil cache lacks {source_path} for {component_name}.{section_name}. "
                "Run data/precompute_airfoil_codes.py before converting the dataset."
            )
        code = torch.as_tensor(code, dtype=torch.float32)
        if component_name == "vertical_tail":
            code = cst_airfoil_codec.symmetric_airfoil_code_from_upper(code, symmetry_line_y)
        section_codes.append(code)
    airfoil_code = torch.cat(section_codes)
    if airfoil_code.numel() != util.WING_AIRFOIL_CODE_SIZE:
        raise ValueError(
            f"{component_name} airfoil code has {airfoil_code.numel()} values, "
            f"expected {util.WING_AIRFOIL_CODE_SIZE}"
        )
    return airfoil_code


def build_structured_sample(payload, airfoil_cache, symmetry_line_y):
    boxes = []
    for component_payload in payload["half_components"]:
        component = component_payload[util.BOX_COMPONENT_KEY]
        box = {
            util.BOX_COMPONENT_KEY: component,
            util.BOX_GEOMETRY_KEY: component_payload[util.BOX_GEOMETRY_KEY],
        }
        if component == util.COMPONENT_WING:
            component_name = component_payload["name"]
            box[util.BOX_AIRFOIL_KEY] = component_airfoil_code(
                component_name,
                payload["wing_airfoil_sources"][component_name],
                airfoil_cache,
                symmetry_line_y,
            )
        boxes.append(box)
    return {"boxes": boxes, "ops": payload["ops"], "syms": payload["syms"]}


def load_payloads(input_dir):
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    json_paths = sorted(input_dir.glob("sample_*.json"))
    if not json_paths:
        raise ValueError(f"No sample JSON files found in {input_dir}")
    payloads = [common.load_geometry_payload(path) for path in json_paths]
    sample_indices = [payload["sample_index"] for payload in payloads]
    if sample_indices != sorted(sample_indices) or len(set(sample_indices)) != len(sample_indices):
        raise ValueError("Sample JSON files do not contain unique ascending sample_index values")
    return payloads


def main():
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError(f"expected-count must be positive, got {args.expected_count}")
    print("Python executable:", sys.executable, flush=True)
    payloads = load_payloads(args.input_dir)
    if len(payloads) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} JSON samples, found {len(payloads)}")
    airfoil_cache = cst_airfoil_code_cache.load_cache(args.cache)
    samples = []
    for payload in payloads:
        sample = build_structured_sample(payload, airfoil_cache, symmetry_line_y=0.0)
        grassdata.validate_structured_sample(sample)
        samples.append(sample)
        print(f"[{payload['sample_index']:04d}] encoded six wing sections", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    if len(reloaded) != len(samples):
        raise RuntimeError("Reloaded dataset count does not match saved dataset")
    for sample in reloaded:
        grassdata.validate_structured_sample(sample)
    print(f"Wrote {len(samples)} structured samples to {args.output}", flush=True)


if __name__ == "__main__":
    main()
