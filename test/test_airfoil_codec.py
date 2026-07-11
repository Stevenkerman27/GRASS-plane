import pytest
import sys
import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import airfoil_codec
import util


def build_fit_config(iterations=5):
    return {
        'iterations': iterations,
        'lr': 0.01,
        'loss_scale': 10.0,
        'weight_reg': 0.0,
        'length_penalty': 0.0,
        'leading_edge_window': 2,
        'leading_edge_weight_amplitude': 1.0,
        'point_density_beta': util.AIRFOIL_DEFAULT_POINT_DENSITY_BETA,
        'surface_control_points': util.AIRFOIL_SURFACE_CONTROL_POINTS,
        'scheduler_patience': 2,
        'scheduler_factor': 0.5,
        'log_interval': 1,
    }


def build_simple_code():
    upper_cp = torch.tensor([
        [1.0, 0.0],
        [0.75, 0.08],
        [0.45, 0.10],
        [0.15, 0.05],
        [0.0, 0.0],
    ])
    lower_cp = torch.tensor([
        [0.0, 0.0],
        [0.15, -0.04],
        [0.45, -0.07],
        [0.75, -0.05],
        [1.0, 0.0],
    ])
    weights = torch.ones(util.AIRFOIL_SURFACE_CONTROL_POINTS)
    return airfoil_codec.pack_split_airfoil_code(upper_cp, lower_cp, weights, weights)


def test_pack_unpack_decode_airfoil_code():
    code = build_simple_code()
    assert code.shape == (util.AIRFOIL_BEZIER_CODE_SIZE,)

    unpacked = airfoil_codec.unpack_split_airfoil_code(code)
    assert unpacked[util.AIRFOIL_UPPER_SURFACE]['control_points'].shape == (
        util.AIRFOIL_SURFACE_CONTROL_POINTS,
        2,
    )

    curve = airfoil_codec.decode_airfoil_code(code, num_output_points=41)
    assert curve.shape == (1, 41, 2)
    assert torch.all(torch.isfinite(curve))
    assert torch.allclose(curve[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(curve[0, -1], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_encode_airfoil_dat_runs_short_fit(tmp_path):
    curve = airfoil_codec.decode_airfoil_code(build_simple_code(), num_output_points=31).squeeze(0)
    dat_path = tmp_path / "simple.dat"
    with dat_path.open("w", encoding="utf-8") as f:
        f.write("simple\n")
        for x, y in curve:
            f.write(f"{x.item():.8f} {y.item():.8f}\n")

    result = airfoil_codec.encode_airfoil_dat(dat_path, build_fit_config(iterations=3))

    assert result['code'].shape == (util.AIRFOIL_BEZIER_CODE_SIZE,)
    assert result['curve'].shape == (31, 2)
    assert result['mae'] >= 0.0
    assert torch.all(torch.isfinite(result['code']))


def test_fit_config_requires_all_keys():
    config = build_fit_config()
    del config['lr']

    with pytest.raises(KeyError, match="fit_config missing required key"):
        airfoil_codec.fit_airfoil_points(torch.zeros(10, 2), config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
