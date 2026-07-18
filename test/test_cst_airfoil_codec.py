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
        torch.tensor(0.0),
        torch.tensor(0.5),
        torch.tensor(1.0),
    )


def short_fit_config():
    config = cst_airfoil_codec.cst_fit_config()
    config['iterations'] = 3
    config['log_interval'] = 1
    return config


def test_cst_code_is_24d_and_round_trips():
    code = build_cst_code()
    assert code.shape == (util.CST_AIRFOIL_CODE_SIZE,)
    unpacked = cst_airfoil_codec.unpack_cst_airfoil_code(code)
    assert unpacked[util.AIRFOIL_UPPER_SURFACE]['shape_coefficients'].shape == (
        util.CST_SURFACE_SHAPE_COEFFICIENTS,
    )
    assert unpacked['class_function_n1'].item() == pytest.approx(0.5)
    assert unpacked['class_function_n2'].item() == pytest.approx(1.0)
    curve = cst_airfoil_codec.decode_cst_airfoil_code(code)
    assert curve.shape == (1, util.AIRFOIL_DEFAULT_OUTPUT_POINTS, 2)
    assert torch.allclose(curve[0, 0, 0], torch.tensor(util.AIRFOIL_TRAILING_EDGE_X))
    assert torch.allclose(curve[0, -1, 0], torch.tensor(util.AIRFOIL_TRAILING_EDGE_X))
    assert torch.allclose(
        curve[0, curve.size(1) // 2],
        torch.tensor([util.AIRFOIL_LEADING_EDGE_X, util.AIRFOIL_LEADING_EDGE_Y]),
    )


def test_cst_fit_uses_shared_physical_leading_edge():
    target = cst_airfoil_codec.decode_cst_airfoil_code(build_cst_code()).squeeze(0)
    result = cst_airfoil_codec.fit_airfoil_points(target, short_fit_config())
    assert result['code'].shape == (util.CST_AIRFOIL_CODE_SIZE,)
    assert result['curve'].shape == (util.AIRFOIL_DEFAULT_OUTPUT_POINTS, 2)
    assert torch.allclose(
        result['leading_edge'],
        torch.tensor([util.AIRFOIL_LEADING_EDGE_X, util.AIRFOIL_LEADING_EDGE_Y]),
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
        symmetric[util.AIRFOIL_LOWER_SURFACE]['shape_coefficients'],
        -unpacked[util.AIRFOIL_UPPER_SURFACE]['shape_coefficients'],
    )
    assert torch.allclose(
        symmetric[util.AIRFOIL_LOWER_SURFACE]['trailing_edge_y'],
        -unpacked[util.AIRFOIL_UPPER_SURFACE]['trailing_edge_y'],
    )


def test_cst_rejects_missing_fit_config_key():
    config = short_fit_config()
    del config['lr']
    with pytest.raises(KeyError, match='fit_config missing required key'):
        cst_airfoil_codec.fit_airfoil_points(
            torch.zeros(util.AIRFOIL_DEFAULT_OUTPUT_POINTS, 2), config
        )
