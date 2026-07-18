import numpy as np

from foildata.manage_foildata import normalize_airfoil_chord_coordinates, resample_single_airfoil


def test_normalize_airfoil_chord_coordinates_aligns_chord_with_x_axis():
    points = np.array([
        [1.98, 0.22],
        [1.40, 0.35],
        [0.30, 0.10],
        [1.42, -0.05],
        [2.02, 0.18],
    ])

    normalized = normalize_airfoil_chord_coordinates(points, leading_edge_index=2)

    assert normalized[2, 0] == 0.0
    assert normalized[2, 1] == 0.0
    assert normalized[0, 0] == 1.0
    assert normalized[-1, 0] == 1.0
    np.testing.assert_allclose(np.mean(normalized[[0, -1]], axis=0), [1.0, 0.0], atol=1e-12)
    assert normalized[1, 1] > 0.0
    assert normalized[3, 1] < 0.0


def test_resample_single_airfoil_writes_normalized_coordinates(tmp_path):
    source_path = tmp_path / 'source.dat'
    source_path.write_text(
        'test airfoil\n'
        '1.98 0.22\n'
        '1.40 0.35\n'
        '0.30 0.10\n'
        '1.42 -0.05\n'
        '2.02 0.18\n',
        encoding='utf-8',
    )
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    status, _ = resample_single_airfoil(source_path, output_dir, num_points=5, beta=1.3)

    assert status is True
    points = np.loadtxt(output_dir / 'source.dat', skiprows=1)
    np.testing.assert_allclose(points[2], [0.0, 0.0], atol=1e-6)
    assert points[0, 0] == 1.0
    assert points[-1, 0] == 1.0
