import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cst_airfoil_codec
import util


def build_cst_code():
    upper = torch.linspace(0.1, 0.3, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    lower = torch.linspace(-0.08, -0.2, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    return cst_airfoil_codec.pack_cst_airfoil_code(
        upper,
        lower,
        torch.tensor(0.006),
        torch.tensor(0.5),
        torch.tensor(1.0),
    )


def test_cst_code_is_19d_and_round_trips():
    code = build_cst_code()
    assert code.shape == (util.CST_AIRFOIL_CODE_SIZE,)
    unpacked = cst_airfoil_codec.unpack_cst_airfoil_code(code)
    assert unpacked['upper_shape_coefficients'].shape == (
        util.CST_SURFACE_SHAPE_COEFFICIENTS,
    )
    assert unpacked['trailing_edge_thickness'].item() == pytest.approx(0.006)
    assert unpacked['class_function_n1'].item() == pytest.approx(0.5)
    assert unpacked['class_function_n2'].item() == pytest.approx(1.0)
    curve = cst_airfoil_codec.decode_cst_airfoil_code(code)
    assert curve.shape == (1, util.AIRFOIL_DEFAULT_OUTPUT_POINTS, 2)
    assert torch.allclose(curve[0, 0, 0], torch.tensor(util.AIRFOIL_TRAILING_EDGE_X))
    assert torch.allclose(curve[0, -1, 0], torch.tensor(util.AIRFOIL_TRAILING_EDGE_X))
    assert torch.allclose(curve[0, 0, 1], torch.tensor(0.003))
    assert torch.allclose(curve[0, -1, 1], torch.tensor(-0.003))
    assert torch.allclose(
        curve[0, curve.size(1) // 2],
        torch.tensor([util.AIRFOIL_LEADING_EDGE_X, 0.0]),
    )


def test_eight_coefficient_cst_code_round_trips_and_fits(monkeypatch):
    coefficient_count = 8
    upper = torch.linspace(0.1, 0.3, coefficient_count)
    lower = torch.linspace(-0.08, -0.2, coefficient_count)
    code = cst_airfoil_codec.pack_cst_airfoil_code(
        upper,
        lower,
        torch.tensor(0.006),
        torch.tensor(0.5),
        torch.tensor(1.0),
        shape_coefficient_count=coefficient_count,
    )
    assert code.shape == (cst_airfoil_codec.cst_airfoil_code_size(coefficient_count),)
    decoded = cst_airfoil_codec.decode_cst_airfoil_code(
        code, shape_coefficient_count=coefficient_count
    )
    unpacked = cst_airfoil_codec.unpack_cst_airfoil_code(
        code, shape_coefficient_count=coefficient_count
    )
    assert unpacked['upper_shape_coefficients'].shape == (coefficient_count,)

    monkeypatch.setitem(util.CST_FIT_CONFIG, 'iterations', 3)
    monkeypatch.setitem(util.CST_FIT_CONFIG, 'log_interval', 1)
    fitted = cst_airfoil_codec.fit_airfoil_points(
        decoded.squeeze(0), shape_coefficient_count=coefficient_count
    )
    assert fitted['shape_coefficient_count'] == coefficient_count
    assert fitted['code'].shape == (cst_airfoil_codec.cst_airfoil_code_size(coefficient_count),)


def test_cst_fit_uses_shared_physical_leading_edge(monkeypatch):
    monkeypatch.setitem(util.CST_FIT_CONFIG, 'iterations', 3)
    monkeypatch.setitem(util.CST_FIT_CONFIG, 'log_interval', 1)
    target = cst_airfoil_codec.decode_cst_airfoil_code(build_cst_code()).squeeze(0)
    result = cst_airfoil_codec.fit_airfoil_points(target)
    assert result['code'].shape == (util.CST_AIRFOIL_CODE_SIZE,)
    assert result['curve'].shape == (util.AIRFOIL_DEFAULT_OUTPUT_POINTS, 2)
    assert torch.allclose(
        result['leading_edge'],
        torch.tensor([util.AIRFOIL_LEADING_EDGE_X, 0.0]),
    )
    assert torch.all(torch.isfinite(result['code']))
    assert result['class_function_n1'].item() > util.CST_MIN_CLASS_FUNCTION_EXPONENT
    assert result['class_function_n2'].item() > util.CST_MIN_CLASS_FUNCTION_EXPONENT


def test_cst_symmetric_code_mirrors_upper_surface_about_zero():
    unpacked = cst_airfoil_codec.unpack_cst_airfoil_code(build_cst_code())
    symmetric_code = cst_airfoil_codec.symmetric_airfoil_code_from_upper(
        build_cst_code(), symmetry_line_y=0.0
    )
    symmetric = cst_airfoil_codec.unpack_cst_airfoil_code(symmetric_code)
    assert torch.allclose(
        symmetric['lower_shape_coefficients'],
        -unpacked['upper_shape_coefficients'],
    )
    assert torch.allclose(
        symmetric['trailing_edge_thickness'],
        unpacked['trailing_edge_thickness'],
    )


def test_cst_fit_rejects_uncentered_trailing_edge():
    target = cst_airfoil_codec.decode_cst_airfoil_code(build_cst_code()).squeeze(0)
    target[0, 1] += 0.001
    with pytest.raises(ValueError, match='trailing-edge midpoint y must equal 0.0'):
        cst_airfoil_codec.fit_airfoil_points(target)


def test_cst_fit_raises_when_global_config_key_is_missing(monkeypatch):
    monkeypatch.delitem(util.CST_FIT_CONFIG, 'lr')
    target = cst_airfoil_codec.decode_cst_airfoil_code(build_cst_code()).squeeze(0)
    with pytest.raises(KeyError, match='lr'):
        cst_airfoil_codec.fit_airfoil_points(target)
