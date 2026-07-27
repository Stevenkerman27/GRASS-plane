"""Convert variable flying-wing geometry JSON files into a structured GRASS .pt dataset."""

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
import grassdata
import util
import aircraft_dataset_common as common


def parse_args():
    parser = argparse.ArgumentParser(description="Convert flying-wing geometry JSON files to structured OBB .pt data.")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "flying_wing_dataset")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "flying_wing_dataset" / "flying_wing_dataset.pt")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cst_airfoil_code_cache.pt")
    parser.add_argument("--expected-count", type=int, default=200)
    return parser.parse_args()


def wing_sections(component_name, source_sections, airfoil_cache):
    count = len(source_sections)
    if not util.MIN_SECTION_COUNT <= count <= util.MAX_SECTION_COUNT:
        raise ValueError(
            f'{component_name} section count must be in '
            f'[{util.MIN_SECTION_COUNT}, {util.MAX_SECTION_COUNT}], got {count}'
        )
    sections = []
    for section_index, source_section in enumerate(source_sections):
        source_path = source_section["airfoil_source"]
        code = cst_airfoil_code_cache.lookup_code(airfoil_cache, source_path)
        if code is None:
            raise KeyError(
                f"Airfoil cache lacks {source_path} for {component_name}.sections[{section_index}]. "
                "Run data/precompute_airfoil_codes.py before converting the dataset."
            )
        code = torch.as_tensor(code, dtype=torch.float32)
        position = torch.as_tensor(
            source_section["leading_edge_xyz"], dtype=torch.float32
        )
        chord = torch.tensor([source_section["chord"]], dtype=torch.float32)
        twist = torch.tensor([source_section["twist"]], dtype=torch.float32)
        section = torch.cat([code, position, chord, twist])
        if section.numel() != 29:
            raise ValueError(
                f'{component_name}.sections[{section_index}] has {section.numel()} values, '
                'expected 29'
            )
        sections.append(section)
    padded = torch.zeros((util.MAX_SECTION_COUNT, util.COMPONENT_SECTION_SIZES[util.COMPONENT_WING]), dtype=torch.float32)
    padded[:count] = torch.stack(sections)
    if padded.shape != (util.MAX_SECTION_COUNT, util.COMPONENT_SECTION_SIZES[util.COMPONENT_WING]):
        raise ValueError(
            f'{component_name} sections have unexpected shape {list(padded.shape)}'
        )
    return padded, count


def fuselage_sections(source_sections):
    count = len(source_sections)
    if not util.MIN_SECTION_COUNT <= count <= util.MAX_SECTION_COUNT:
        raise ValueError(
            f'fuselage section count must be in '
            f'[{util.MIN_SECTION_COUNT}, {util.MAX_SECTION_COUNT}], got {count}'
        )
    sections = torch.as_tensor(source_sections, dtype=torch.float32)
    expected_shape = (count, util.FUSELAGE_SECTION_SIZE)
    if tuple(sections.shape) != expected_shape:
        raise ValueError(
            f'fuselage sections must have shape {list(expected_shape)}, got {list(sections.shape)}'
        )
    padded = torch.zeros((util.MAX_SECTION_COUNT, util.FUSELAGE_SECTION_SIZE), dtype=torch.float32)
    padded[:count] = sections
    return padded, count


def build_structured_sample(payload, airfoil_cache):
    boxes = []
    for component_payload in payload["half_components"]:
        component = component_payload["component"]
        if component == util.COMPONENT_WING:
            component_name = component_payload["name"]
            sections, count = wing_sections(
                component_name,
                component_payload["sections"],
                airfoil_cache,
            )
            box = {
                "component": component,
                "sections": sections,
                "section_count": count,
            }
        elif component == util.COMPONENT_FUSELAGE:
            sections, count = fuselage_sections(component_payload["sections"])
            box = {
                "component": component,
                "sections": sections,
                "section_count": count,
            }
        else:
            box = {
                "component": component,
                "geometry": component_payload["geometry"],
            }
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
        sample = build_structured_sample(payload, airfoil_cache)
        grassdata.validate_structured_sample(sample)
        samples.append(sample)
        print(f"[{payload['sample_index']:04d}] encoded wing CST sections", flush=True)
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
