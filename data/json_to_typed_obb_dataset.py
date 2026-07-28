"""Convert variable flying-wing geometry JSON files into a structured GRASS .pt dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cst_airfoil_code_cache
import conventional_canard_dataset_common as conventional_canard
import grassdata
import util
import aircraft_dataset_common as common


def parse_args():
    parser = argparse.ArgumentParser(description="Convert flying-wing geometry JSON files to structured OBB .pt data.")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "conventional_canard_dataset")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "conventional_canard_dataset" / "conventional_canard_dataset.pt")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cst_airfoil_code_cache.pt")
    parser.add_argument("--expected-count", type=int, default=200)
    return parser.parse_args()


def wing_sections(component_name, sequence_type, source_sections, airfoil_cache):
    expected_range = grassdata.sequence_section_count_range(sequence_type)
    count = len(source_sections)
    if not expected_range[0] <= count <= expected_range[1]:
        raise ValueError(
            f'{component_name} section count must be in '
            f'[{expected_range[0]}, {expected_range[1]}], got {count}'
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
    padded = torch.zeros(
        (grassdata.sequence_max_sections(sequence_type), grassdata.sequence_section_size(sequence_type)),
        dtype=torch.float32,
    )
    padded[:count] = torch.stack(sections)
    expected_shape = (grassdata.sequence_max_sections(sequence_type), grassdata.sequence_section_size(sequence_type))
    if tuple(padded.shape) != expected_shape:
        raise ValueError(
            f'{component_name} sections have unexpected shape {list(padded.shape)}'
        )
    return padded, count


def fuselage_sections(source_sections, sequence_type):
    count = len(source_sections)
    expected_range = grassdata.sequence_section_count_range(sequence_type)
    if not expected_range[0] <= count <= expected_range[1]:
        raise ValueError(
            f'fuselage section count must be in '
            f'[{expected_range[0]}, {expected_range[1]}], got {count}'
        )
    sections = torch.as_tensor(source_sections, dtype=torch.float32)
    expected_shape = (count, util.FUSELAGE_SECTION_SIZE)
    if tuple(sections.shape) != expected_shape:
        raise ValueError(
            f'fuselage sections must have shape {list(expected_shape)}, got {list(sections.shape)}'
        )
    padded = torch.zeros(
        (grassdata.sequence_max_sections(sequence_type), grassdata.sequence_section_size(sequence_type)),
        dtype=torch.float32,
    )
    padded[:count] = sections
    return padded, count


def build_structured_sample(payload, airfoil_cache):
    boxes = []
    for component_payload in payload["half_components"]:
        component = component_payload["component"]
        sequence_type = component_payload['sequence_type']
        if component == util.COMPONENT_WING:
            component_name = component_payload["name"]
            sections, count = wing_sections(
                component_name,
                sequence_type,
                component_payload["sections"],
                airfoil_cache,
            )
            box = {
                "component": component,
                'sequence_type': sequence_type,
                "sections": sections,
                "section_count": count,
            }
        elif component == util.COMPONENT_FUSELAGE:
            sections, count = fuselage_sections(component_payload["sections"], sequence_type)
            box = {
                "component": component,
                'sequence_type': sequence_type,
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
    payloads = []
    for path in json_paths:
        schema = json.loads(path.read_text(encoding='utf-8')).get('schema')
        if schema == common.GEOMETRY_SCHEMA:
            payload = common.load_geometry_payload(path)
        elif schema == conventional_canard.GEOMETRY_SCHEMA:
            payload = conventional_canard.load_geometry_payload(path)
        else:
            raise ValueError(f'Unsupported geometry schema in {path}: {schema!r}')
        payloads.append(payload)
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
