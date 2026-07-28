"""Randomly show one full aircraft OBB from a generated structured dataset."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cst_airfoil_codec
import grassdata
import project_paths
import util


NON_GUI_BACKENDS = {"agg", "pdf", "ps", "svg", "template"}
ELLIPSE_SAMPLES = 25
REFLECTION_SYM_TYPE = 1.0
REFLECTION_TOLERANCE = 1e-6
DEFAULT_OPENVSP_EXECUTABLE = Path(r"D:\3D\Projects\OpenVSP-3.51.0-win64\vsp.exe")


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly show one aircraft OBB from a generated dataset.")
    parser.add_argument("--layout", choices=tuple(project_paths.AIRCRAFT_DATASET_SPECS), default=None)
    parser.add_argument("--index", type=int, default=None, help="Sample index within --layout.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for layout and sample selection.")
    parser.add_argument("--backend", default=None, help="Optional interactive matplotlib backend, for example TkAgg.")
    parser.add_argument("--vsp-exe", type=Path, default=DEFAULT_OPENVSP_EXECUTABLE)
    parser.add_argument("--no-open-vsp", dest="open_vsp", action="store_false")
    parser.set_defaults(open_vsp=True)
    return parser.parse_args()


def assert_interactive_backend(matplotlib):
    backend = matplotlib.get_backend()
    backend_key = backend.lower()
    if backend_key in NON_GUI_BACKENDS or "backend_inline" in backend_key:
        raise RuntimeError(f"Matplotlib backend {backend!r} cannot show an interactive OBB window")


def select_aircraft_sample(layout, index, seed):
    if index is not None and layout is None:
        raise ValueError("--index requires --layout because sample indices are dataset-local")
    chooser = random.Random(seed)
    selected_layout = layout
    if selected_layout is None:
        selected_layout = chooser.choice(tuple(project_paths.AIRCRAFT_DATASET_SPECS))
    dataset_spec = project_paths.AIRCRAFT_DATASET_SPECS[selected_layout]
    dataset = grassdata.StructuredGRASSDataset(dataset_spec['dataset'])
    if index is not None:
        if not 0 <= index < len(dataset):
            raise IndexError(f"index must be in [0, {len(dataset) - 1}], got {index}")
        return selected_layout, dataset_spec, index, dataset[index]
    selected_index = chooser.randrange(len(dataset))
    return selected_layout, dataset_spec, selected_index, dataset[selected_index]


def corresponding_vsp_path(vsp_dir, index):
    if not vsp_dir.is_dir():
        raise NotADirectoryError(f"VSP3 directory does not exist: {vsp_dir}")
    vsp_path = vsp_dir / f"sample_{index:04d}.vsp3"
    if not vsp_path.is_file():
        raise FileNotFoundError(f"VSP3 file does not exist for sample {index:04d}: {vsp_path}")
    return vsp_path


def open_vsp(vsp_executable, vsp_path):
    if not vsp_executable.is_file():
        raise FileNotFoundError(f"OpenVSP executable does not exist: {vsp_executable}")
    subprocess.Popen([str(vsp_executable), str(vsp_path)], cwd=str(vsp_executable.parent))
    print(f"Opened OpenVSP model: {vsp_path}", flush=True)


def component_geometry(node):
    component = node.box["component"]
    if component in grassdata.AE_COMPONENT_TYPES:
        return component, {
            "sections": node.box["sections"].detach().cpu().squeeze(0),
            "section_count": int(node.box["section_count"].item()),
        }
    geometry = node.box["geometry"].detach().cpu().reshape(-1).tolist()
    expected_size = util.COMPONENT_GEOMETRY_SIZES[component]
    if len(geometry) != expected_size:
        raise ValueError(f"OBB geometry has {len(geometry)} values, expected {expected_size}")
    return component, geometry


def reflect_geometry(component, geometry, symmetry):
    symmetry_values = symmetry.detach().cpu().reshape(-1).tolist()
    if len(symmetry_values) != 8:
        raise ValueError(f"Symmetry vector has {len(symmetry_values)} values, expected 8")
    if abs(symmetry_values[0] - REFLECTION_SYM_TYPE) > REFLECTION_TOLERANCE:
        raise ValueError("Dataset visualizer supports only reflective SYM nodes")

    normal = torch.tensor(symmetry_values[1:4], dtype=torch.float32)
    normal_norm = torch.linalg.vector_norm(normal)
    if normal_norm == 0:
        raise ValueError("Reflective SYM node has a zero plane normal")
    normal /= normal_norm
    point = torch.tensor(symmetry_values[4:7], dtype=torch.float32)
    if component == util.COMPONENT_WING:
        sections = geometry["sections"].clone()
        count = geometry["section_count"]
        sections[:count, 25] *= -1.0
        return component, {
            "sections": sections,
            "section_count": count,
        }
    if component == util.COMPONENT_FUSELAGE:
        sections = geometry["sections"].clone()
        count = geometry["section_count"]
        positions = sections[:count, :3]
        reflected_positions = positions - 2.0 * torch.matmul(positions - point, normal).unsqueeze(1) * normal
        sections[:count, :3] = reflected_positions
        return component, {"sections": sections, "section_count": count}
    reflected = list(geometry)
    for start_index in (0, 3):
        position = torch.tensor(geometry[start_index:start_index + 3], dtype=torch.float32)
        reflected_position = position - 2.0 * torch.dot(position - point, normal) * normal
        reflected[start_index:start_index + 3] = reflected_position.tolist()
    return component, reflected


def expand_tree(node):
    if node.is_leaf():
        return [component_geometry(node)]
    if node.is_adj():
        return [*expand_tree(node.left), *expand_tree(node.right)]
    if node.is_sym():
        originals = expand_tree(node.left)
        mirrors = [reflect_geometry(component, geometry, node.sym) for component, geometry in originals]
        return [*originals, *mirrors]
    raise ValueError(f"Unknown tree node type: {node.node_type}")


def component_label(component, counters):
    name = util.component_name(component)
    counters[name] = counters.get(name, 0) + 1
    return f"{name}_{counters[name]}"


def draw_fuselage_sections(ax, fuselage, label):
    import numpy as np

    count = fuselage["section_count"]
    sections = fuselage["sections"][:count].numpy()
    theta = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES)
    rings = []
    for section in sections:
        center = section[:3]
        width, height = section[3:5]
        ring = np.vstack((
            center[0] + np.zeros_like(theta),
            center[1] + (width / 2.0) * np.cos(theta),
            center[2] + (height / 2.0) * np.sin(theta),
        ))
        rings.append(ring)
        ax.plot(ring[0], ring[1], ring[2], color="gray", linewidth=1.2)
    for start, end in zip(rings[:-1], rings[1:], strict=True):
        for sample in range(ELLIPSE_SAMPLES):
            ax.plot([start[0, sample], end[0, sample]], [start[1, sample], end[1, sample]], [start[2, sample], end[2, sample]], color="gray", linewidth=0.8)
    center = sections[:, :3].mean(axis=0)
    ax.text(center[0], center[1], center[2], label, color="black", fontsize=9)


def draw_elliptical_body(ax, geometry, label):
    import numpy as np

    start = np.asarray(geometry[0:3], dtype=float)
    end = np.asarray(geometry[3:6], dtype=float)
    axis = end - start
    axis_length = np.linalg.norm(axis)
    if axis_length == 0:
        raise ValueError(f"Elliptical-body OBB {label!r} has coincident endpoint centers")
    axis /= axis_length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(axis, reference)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(lateral, axis)
    theta = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES)
    rings = []
    for center, width, height in ((start, geometry[6], geometry[7]), (end, geometry[8], geometry[9])):
        ring = center[:, None] + lateral[:, None] * (width / 2.0) * np.cos(theta) + vertical[:, None] * (height / 2.0) * np.sin(theta)
        rings.append(ring)
        ax.plot(ring[0], ring[1], ring[2], color="gray", linewidth=1.2)
    for sample in range(ELLIPSE_SAMPLES):
        ax.plot([rings[0][0, sample], rings[1][0, sample]], [rings[0][1, sample], rings[1][1, sample]], [rings[0][2, sample], rings[1][2, sample]], color="gray", linewidth=0.8)
    center = (start + end) / 2.0
    ax.text(center[0], center[1], center[2], label, color="black", fontsize=9)


def rotate_about_axis(vector, axis, angle):
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def wing_section_points(section):
    import numpy as np

    twist_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    chord_axis = rotate_about_axis(
        np.array([1.0, 0.0, 0.0], dtype=float), twist_axis, float(section[28])
    )
    thickness_axis = rotate_about_axis(
        np.array([0.0, 0.0, 1.0], dtype=float), twist_axis, float(section[28])
    )
    curve = cst_airfoil_codec.decode_cst_airfoil_code(
        torch.as_tensor(section[:util.CST_AIRFOIL_CODE_SIZE], dtype=torch.float32),
        num_output_points=util.FREE_DECODE_ERROR_AIRFOIL_POINTS,
    ).squeeze(0).detach().cpu().numpy()
    return (
        section[24:27]
        + curve[:, :1] * float(section[27]) * chord_axis
        + curve[:, 1:2] * float(section[27]) * thickness_axis
    )


def draw_wing_planform(ax, wing, color, label):
    import numpy as np

    count = wing["section_count"]
    sections = wing["sections"][:count].numpy()
    profiles = [wing_section_points(section) for section in sections]
    for profile in profiles:
        ax.plot(profile[:, 0], profile[:, 1], profile[:, 2], color=color, linewidth=1.0)
    for first, second in zip(profiles[:-1], profiles[1:], strict=True):
        for point_index in (0, first.shape[0] // 2, first.shape[0] - 1):
            ax.plot(
                [first[point_index, 0], second[point_index, 0]],
                [first[point_index, 1], second[point_index, 1]],
                [first[point_index, 2], second[point_index, 2]],
                color=color,
                linewidth=1.3,
            )
    center = np.mean(np.concatenate(profiles, axis=0), axis=0)
    ax.text(center[0], center[1], center[2], label, color="black", fontsize=9)


def set_aircraft_axes(ax, components):
    import numpy as np

    points = []
    for component, geometry in components:
        if component in grassdata.AE_COMPONENT_TYPES:
            count = geometry["section_count"]
            if component == util.COMPONENT_WING:
                points.extend(np.concatenate(
                    [wing_section_points(section) for section in geometry["sections"][:count]],
                    axis=0,
                ).tolist())
            else:
                points.extend(geometry["sections"][:count, :3].tolist())
        else:
            points.extend((geometry[0:3], geometry[3:6]))
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    max_range = max((points.max(axis=0) - points.min(axis=0)).max(), 1.0)
    radius = max_range * 0.62
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.45, center[2] + radius * 0.75)
    ax.set_box_aspect((1.0, 1.0, 0.6))


def show_aircraft(layout_label, index, tree):
    import matplotlib
    import matplotlib.pyplot as plt

    assert_interactive_backend(matplotlib)
    components = expand_tree(tree.root)
    if not components:
        raise ValueError("Expanded aircraft topology contains no components")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("jet_r")
    counters = {}
    for component_index, (component, geometry) in enumerate(components):
        label = component_label(component, counters)
        if component == util.COMPONENT_FUSELAGE:
            draw_fuselage_sections(ax, geometry, label)
        elif component == util.COMPONENT_ENGINE:
            draw_elliptical_body(ax, geometry, label)
        elif component == util.COMPONENT_WING:
            draw_wing_planform(ax, geometry, cmap(component_index / len(components)), label)
        else:
            raise ValueError(f"Unknown component type: {component}")
    set_aircraft_axes(ax, components)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{layout_label} OBB - Sample {index:04d}")
    ax.view_init(elev=22, azim=-58)
    print("Matplotlib backend:", matplotlib.get_backend(), flush=True)
    print(f"Showing sample {index:04d}. Close the window to continue.", flush=True)
    plt.show()


def main():
    args = parse_args()
    if args.backend is not None:
        import matplotlib

        matplotlib.use(args.backend)
    _layout, dataset_spec, index, tree = select_aircraft_sample(
        args.layout, args.index, args.seed
    )
    if args.open_vsp:
        vsp_path = corresponding_vsp_path(dataset_spec['directory'], index)
        open_vsp(args.vsp_exe, vsp_path)
    show_aircraft(dataset_spec['label'], index, tree)


if __name__ == "__main__":
    main()
