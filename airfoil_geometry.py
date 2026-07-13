"""Torch-free airfoil coordinate helpers shared by OpenVSP dataset generation."""

from __future__ import annotations

from pathlib import Path


def load_dat_points(dat_path):
    path = Path(dat_path)
    if not path.is_file():
        raise FileNotFoundError(f"Airfoil .dat file not found: {path}")

    points = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.strip().split()
            if len(fields) != 2:
                continue
            try:
                points.append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue
    if not points:
        raise ValueError(f"Airfoil .dat file contains no coordinate pairs: {path}")
    return points


def symmetric_airfoil_points(dat_path):
    """Return TE-to-LE-to-TE points with the supplied upper surface mirrored about y=0."""
    points = load_dat_points(dat_path)
    leading_edge_index = min(range(len(points)), key=lambda index: points[index][0])
    if leading_edge_index == 0 or leading_edge_index == len(points) - 1:
        raise ValueError(
            f"Airfoil coordinates must contain both upper and lower surfaces: {dat_path}"
        )
    upper_surface = points[:leading_edge_index + 1]
    lower_surface = [(x, -y) for x, y in reversed(upper_surface)]
    return [*upper_surface, *lower_surface[1:]]


def write_symmetric_airfoil_dat(source_path, output_path):
    destination = Path(output_path)
    points = symmetric_airfoil_points(source_path)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"Symmetric derivative of {Path(source_path).name}\n")
        for x_value, y_value in points:
            output.write(f"{x_value:.12g} {y_value:.12g}\n")
    if not destination.is_file():
        raise RuntimeError(f"Failed to write symmetric airfoil file: {destination}")
    return destination
