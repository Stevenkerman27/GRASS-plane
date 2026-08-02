"""Show random flying-wing planforms sampled from the quadratic plan logic."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
DEFAULT_PLANFORM_COUNT = 30
DEFAULT_ROOT_SEED = 20260727
DEFAULT_OUTPUT_PATH = DATA_DIR / "quadratic_wing_planforms.png"

for import_path in (REPO_ROOT, DATA_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import aircraft_dataset_common as common


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show random planforms from the quadratic flying-wing sampler."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_PLANFORM_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Image path. Its parent directory must already exist.",
    )
    return parser.parse_args()


def draw_planform(axis, plan):
    import numpy as np

    sections = common.build_wing_planform_sections(plan)
    leading_edges = [section["leading_edge_xyz"] for section in sections]
    trailing_edges = [
        [
            section["leading_edge_xyz"][0] + section["chord"],
            section["leading_edge_xyz"][1],
            section["leading_edge_xyz"][2],
        ]
        for section in sections
    ]
    for sign in (1.0, -1.0):
        leading = [(point[0], sign * point[1]) for point in leading_edges]
        trailing = [(point[0], sign * point[1]) for point in trailing_edges]
        polygon = [*leading, *reversed(trailing)]
        axis.fill(
            [point[0] for point in polygon],
            [point[1] for point in polygon],
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.22,
            linewidth=1.0,
        )
        axis.plot(
            [point[0] for point in leading],
            [point[1] for point in leading],
            color="tab:blue",
            linewidth=1.3,
        )
        axis.plot(
            [point[0] for point in trailing],
            [point[1] for point in trailing],
            color="tab:blue",
            linewidth=1.3,
        )
        axis.scatter(
            [point[0] for point in leading],
            [point[1] for point in leading],
            color="tab:blue",
            s=10,
            zorder=3,
        )
        curve_fractions = np.linspace(0.0, 1.0, 201)
        quarter_chord_x = [
            common.evaluate_root_relative_quadratic(
                plan["root_quarter_chord_x"],
                plan["quarter_chord_a"],
                plan["quarter_chord_b"],
                fraction,
            )
            for fraction in curve_fractions
        ]
        axis.plot(
            quarter_chord_x,
            sign * curve_fractions * common.MAX_WINGSPAN / 2.0,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title(
        "sections={}  chord/sweep jitter=+/-{:.0%}\nax={:.3f}, bx={:.3f}, ac={:.3f}, eta_c={:.3f}".format(
            plan["section_count"],
            common.WING_CHORD_JITTER_FRACTION,
            plan["quarter_chord_a"],
            plan["quarter_chord_b"],
            plan["chord_a"],
            plan["chord_vertex_fraction"],
        ),
        fontsize=8,
    )


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError(f"count must be positive, got {args.count}")
    if args.output is not None and not args.output.parent.is_dir():
        raise NotADirectoryError(f"output parent directory does not exist: {args.output.parent}")

    import matplotlib

    import matplotlib.pyplot as plt

    print("Matplotlib backend:", matplotlib.get_backend(), flush=True)
    print(f"Generating {args.count} quadratic wing planforms...", flush=True)
    columns = math.ceil(math.sqrt(args.count))
    rows = math.ceil(args.count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.1 * rows))
    for sample_index, axis in enumerate(axes.flat):
        if sample_index >= args.count:
            axis.remove()
            continue
        reference = common.build_random_reference(
            random.Random(args.seed + sample_index)
        )
        draw_planform(axis, reference["wing_plan"])
    figure.suptitle("Quadratic flying-wing planform samples", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    print(f"Wrote planform visualization: {args.output}", flush=True)


if __name__ == "__main__":
    main()
