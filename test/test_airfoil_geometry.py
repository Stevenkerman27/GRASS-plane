from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import airfoil_geometry


def test_write_symmetric_airfoil_dat_uses_upper_surface_and_mirrors_it(tmp_path):
    source_path = tmp_path / "cambered.dat"
    source_path.write_text(
        "cambered\n"
        "1.0 0.0\n"
        "0.5 0.08\n"
        "0.0 0.0\n"
        "0.5 -0.02\n"
        "1.0 0.0\n",
        encoding="utf-8",
    )

    output_path = airfoil_geometry.write_symmetric_airfoil_dat(source_path, tmp_path / "symmetric.dat")

    assert airfoil_geometry.load_dat_points(output_path) == [
        (1.0, 0.0),
        (0.5, 0.08),
        (0.0, 0.0),
        (0.5, -0.08),
        (1.0, -0.0),
    ]


def test_symmetric_airfoil_requires_two_surfaces(tmp_path):
    source_path = tmp_path / "invalid.dat"
    source_path.write_text("invalid\n0.0 0.0\n0.5 0.1\n1.0 0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="both upper and lower"):
        airfoil_geometry.symmetric_airfoil_points(source_path)
