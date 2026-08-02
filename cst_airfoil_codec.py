"""Kulfan CST encoding with a fixed physical leading edge at (0, 0)."""

from pathlib import Path
import math

import torch
import torch.nn.functional as functional
import torch.optim as optim

import airfoil_sampling
import util


PROCESSED_AIRFOIL_ENCODING_VERSION = 'fixed_leading_edge_centered_trailing_edge_cst_v3'
CST_FIT_METRIC_KEYS = (
    'mae', 'mse', 'max_point_error', 'leading_edge_mae', 'leading_edge_mse',
)


def cst_fit_config():
    return dict(util.CST_FIT_CONFIG)


def resolve_shape_coefficient_count(shape_coefficient_count=None):
    count = (
        util.CST_SURFACE_SHAPE_COEFFICIENTS
        if shape_coefficient_count is None else shape_coefficient_count
    )
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(
            'shape_coefficient_count must be an integer, '
            f'got {type(count).__name__}'
        )
    if count < 1:
        raise ValueError(f'shape_coefficient_count must be positive, got {count}')
    return count


def cst_airfoil_code_size(shape_coefficient_count=None):
    return 2 * resolve_shape_coefficient_count(shape_coefficient_count) + util.CST_BOUNDARY_CODE_SIZE


def load_dat(dat_path):
    path = Path(dat_path)
    if not path.is_file():
        raise FileNotFoundError(f'Airfoil .dat file not found: {path}')
    points = []
    with path.open('r', encoding='utf-8') as source:
        for line in source:
            values = line.strip().split()
            if len(values) != 2:
                continue
            try:
                points.append([float(values[0]), float(values[1])])
            except ValueError:
                continue
    if not points:
        raise ValueError(f'Airfoil .dat file contains no coordinate pairs: {path}')
    return torch.tensor(points, dtype=torch.float32)


def cosine_leading_edge_weights(target_points, window, amplitude):
    if window < 0:
        raise ValueError(f'leading_edge_window must be non-negative, got {window}')
    if amplitude < 0:
        raise ValueError(f'leading_edge_weight_amplitude must be non-negative, got {amplitude}')
    weights = torch.ones(target_points.shape[:2], dtype=target_points.dtype, device=target_points.device)
    if window == 0 or amplitude == 0:
        return weights
    leading_edge_index = target_points.size(1) // 2
    positions = torch.arange(target_points.size(1), device=target_points.device)
    distance = torch.abs(positions - leading_edge_index)
    mask = distance <= window
    normalized_distance = distance[mask].to(target_points.dtype) / float(window)
    weights[:, mask] = 1.0 + amplitude * 0.5 * (1.0 + torch.cos(torch.pi * normalized_distance))
    return weights


def weighted_mae_per_airfoil(curve, target_points, weights):
    absolute_error = torch.mean(torch.abs(curve - target_points), dim=2)
    return torch.sum(absolute_error * weights, dim=1) / torch.sum(weights, dim=1)


def leading_edge_error_metrics(curve, target_points, window):
    if window < 0:
        raise ValueError(f'leading_edge_window must be non-negative, got {window}')
    leading_edge_index = target_points.size(0) // 2
    start = max(0, leading_edge_index - window)
    end = min(target_points.size(0), leading_edge_index + window + 1)
    error = curve[start:end] - target_points[start:end]
    return {
        'leading_edge_mae': float(torch.mean(torch.abs(error)).detach().cpu().item()),
        'leading_edge_mse': float(torch.mean(error ** 2).detach().cpu().item()),
    }


def _bernstein_basis(x_values, coefficient_count):
    degree = coefficient_count - 1
    basis = torch.zeros(
        (*x_values.shape, coefficient_count), dtype=x_values.dtype, device=x_values.device
    )
    for index in range(coefficient_count):
        basis[..., index] = (
            math.comb(degree, index)
            * (x_values ** index)
            * ((1.0 - x_values) ** (degree - index))
        )
    return basis


def _as_batched_code(code, shape_coefficient_count):
    tensor = torch.as_tensor(code, dtype=torch.float32)
    unbatched = tensor.dim() == 1
    if unbatched:
        tensor = tensor.unsqueeze(0)
    code_size = cst_airfoil_code_size(shape_coefficient_count)
    if tensor.dim() != 2 or tensor.size(1) != code_size:
        raise ValueError(
            f'CST code must have shape [{code_size}] or '
            f'[B, {code_size}], got {list(tensor.size())}'
        )
    if not torch.isfinite(tensor).all():
        raise ValueError('CST code must contain only finite values')
    return tensor, unbatched


def _as_batched_shape_coefficients(coefficients, name, shape_coefficient_count):
    tensor = torch.as_tensor(coefficients, dtype=torch.float32)
    unbatched = tensor.dim() == 1
    if unbatched:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(1) != shape_coefficient_count:
        raise ValueError(
            f'{name} must have shape [{shape_coefficient_count}] or '
            f'[B, {shape_coefficient_count}], got {list(tensor.size())}'
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} must contain only finite values')
    return tensor, unbatched


def _as_batched_scalar(value, name):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 0:
        tensor = tensor.reshape(1, 1)
        unbatched = True
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
        unbatched = True
    else:
        unbatched = False
    if tensor.dim() != 2 or tensor.size(1) != 1:
        raise ValueError(f'{name} must be scalar, [B], or [B, 1], got {list(tensor.size())}')
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} must contain only finite values')
    return tensor, unbatched


def _as_batched_class_exponent(exponent, name):
    tensor, unbatched = _as_batched_scalar(exponent, name)
    if not torch.all(tensor > util.CST_MIN_CLASS_FUNCTION_EXPONENT):
        raise ValueError(
            f'{name} must exceed {util.CST_MIN_CLASS_FUNCTION_EXPONENT}'
        )
    return tensor, unbatched


def pack_cst_airfoil_code(
        upper_shape_coefficients,
        lower_shape_coefficients,
        trailing_edge_thickness,
        class_function_n1,
        class_function_n2,
        shape_coefficient_count=None):
    coefficient_count = resolve_shape_coefficient_count(shape_coefficient_count)
    upper, upper_unbatched = _as_batched_shape_coefficients(
        upper_shape_coefficients, 'upper_shape_coefficients', coefficient_count
    )
    lower, lower_unbatched = _as_batched_shape_coefficients(
        lower_shape_coefficients, 'lower_shape_coefficients', coefficient_count
    )
    trailing_edge_thickness, thickness_unbatched = _as_batched_scalar(
        trailing_edge_thickness, 'trailing_edge_thickness'
    )
    n1, n1_unbatched = _as_batched_class_exponent(class_function_n1, 'class_function_n1')
    n2, n2_unbatched = _as_batched_class_exponent(class_function_n2, 'class_function_n2')
    batch_size = upper.size(0)
    for name, tensor in (
            ('lower_shape_coefficients', lower),
            ('trailing_edge_thickness', trailing_edge_thickness),
            ('class_function_n1', n1),
            ('class_function_n2', n2)):
        if tensor.size(0) != batch_size:
            raise ValueError(f'{name} batch size {tensor.size(0)} does not match {batch_size}')
    code = torch.cat([upper, lower, trailing_edge_thickness, n1, n2], dim=1)
    code_size = cst_airfoil_code_size(coefficient_count)
    if code.size(1) != code_size:
        raise RuntimeError(
            f'Packed CST code has {code.size(1)} values, expected {code_size}'
        )
    if (upper_unbatched and lower_unbatched and thickness_unbatched
            and n1_unbatched and n2_unbatched):
        return code.squeeze(0)
    return code


def unpack_cst_airfoil_code(code, shape_coefficient_count=None):
    coefficient_count = resolve_shape_coefficient_count(shape_coefficient_count)
    code_tensor, unbatched = _as_batched_code(code, coefficient_count)
    offset = 0
    upper = code_tensor[:, offset:offset + coefficient_count]
    offset += coefficient_count
    lower = code_tensor[:, offset:offset + coefficient_count]
    offset += coefficient_count
    trailing_edge_thickness = code_tensor[:, offset:offset + 1]
    offset += 1
    n1 = code_tensor[:, offset:offset + 1]
    offset += 1
    n2 = code_tensor[:, offset:offset + 1]
    offset += 1
    code_size = cst_airfoil_code_size(coefficient_count)
    if offset != code_size:
        raise RuntimeError(f'Expected to consume {code_size} CST values, consumed {offset}')
    _as_batched_class_exponent(n1, 'class_function_n1')
    _as_batched_class_exponent(n2, 'class_function_n2')
    result = {
        'upper_shape_coefficients': upper,
        'lower_shape_coefficients': lower,
        'trailing_edge_thickness': trailing_edge_thickness,
        'class_function_n1': n1,
        'class_function_n2': n2,
    }
    if unbatched:
        unpacked = {
            key: value.squeeze(0)
            for key, value in result.items()
        }
        return unpacked
    return result


def _class_shape_y(
        shape_coefficients, trailing_edge_y, class_function_n1, class_function_n2, x,
        shape_coefficient_count):
    if not torch.all((x >= -1e-5) & (x <= util.AIRFOIL_TRAILING_EDGE_X + 1e-5)):
        raise ValueError('x coordinates must lie between the fixed leading and trailing edges')
    xi = torch.clamp(x, util.AIRFOIL_LEADING_EDGE_X, util.AIRFOIL_TRAILING_EDGE_X)
    basis = _bernstein_basis(xi, shape_coefficient_count)
    class_function = (
        xi ** class_function_n1
        * (1.0 - xi) ** class_function_n2
    )
    shape_function = torch.sum(basis * shape_coefficients.unsqueeze(1), dim=2)
    return xi * trailing_edge_y + class_function * shape_function


def _decode_at_x(
        upper_shape, lower_shape, trailing_edge_thickness, class_function_n1,
        class_function_n2, upper_x, lower_x, shape_coefficient_count):
    upper_te_y = 0.5 * trailing_edge_thickness
    lower_te_y = -upper_te_y
    upper_y = _class_shape_y(
        upper_shape, upper_te_y, class_function_n1, class_function_n2, upper_x,
        shape_coefficient_count
    )
    lower_y = _class_shape_y(
        lower_shape, lower_te_y, class_function_n1, class_function_n2, lower_x,
        shape_coefficient_count
    )
    upper_curve = torch.stack([upper_x, upper_y], dim=2)
    lower_curve = torch.stack([lower_x, lower_y], dim=2)
    return torch.cat([upper_curve, lower_curve[:, 1:, :]], dim=1)


def _sampling_x_values(num_output_points, device, dtype):
    upper_t, lower_t = airfoil_sampling.split_surface_t_values(
        num_output_points, util.AIRFOIL_DEFAULT_POINT_DENSITY_BETA
    )
    return (
        util.AIRFOIL_TRAILING_EDGE_X - torch.as_tensor(upper_t, device=device, dtype=dtype).unsqueeze(0),
        torch.as_tensor(lower_t, device=device, dtype=dtype).unsqueeze(0),
    )


def decode_cst_airfoil_code(
        code, num_output_points=util.AIRFOIL_DEFAULT_OUTPUT_POINTS,
        shape_coefficient_count=None):
    if num_output_points < 3:
        raise ValueError(f'num_output_points must be at least 3, got {num_output_points}')
    coefficient_count = resolve_shape_coefficient_count(shape_coefficient_count)
    code_tensor, _ = _as_batched_code(code, coefficient_count)
    unpacked = unpack_cst_airfoil_code(code_tensor, coefficient_count)
    upper_x, lower_x = _sampling_x_values(num_output_points, code_tensor.device, code_tensor.dtype)
    upper_x = upper_x.expand(code_tensor.size(0), -1)
    lower_x = lower_x.expand(code_tensor.size(0), -1)
    return _decode_at_x(
        unpacked['upper_shape_coefficients'],
        unpacked['lower_shape_coefficients'],
        unpacked['trailing_edge_thickness'],
        unpacked['class_function_n1'],
        unpacked['class_function_n2'],
        upper_x,
        lower_x,
        coefficient_count,
    )


def symmetric_airfoil_code_from_upper(
        airfoil_code, symmetry_line_y, shape_coefficient_count=None):
    if symmetry_line_y != 0.0:
        raise ValueError(
            'fixed-leading-edge CST can only mirror about '
            f'y={0.0}, got {symmetry_line_y}'
        )
    coefficient_count = resolve_shape_coefficient_count(shape_coefficient_count)
    unpacked = unpack_cst_airfoil_code(airfoil_code, coefficient_count)
    return pack_cst_airfoil_code(
        unpacked['upper_shape_coefficients'],
        -unpacked['upper_shape_coefficients'],
        unpacked['trailing_edge_thickness'],
        unpacked['class_function_n1'],
        unpacked['class_function_n2'],
        coefficient_count,
    )


def _validate_target_points(points):
    if points.dim() != 2 or points.size(1) != 2:
        raise ValueError(f'raw_points must have shape [N, 2], got {list(points.size())}')
    if points.size(0) != util.AIRFOIL_DEFAULT_OUTPUT_POINTS:
        raise ValueError(
            f'raw_points must contain {util.AIRFOIL_DEFAULT_OUTPUT_POINTS} points, got {points.size(0)}'
        )
    if not torch.isfinite(points).all():
        raise ValueError('raw_points must contain only finite values')
    leading_edge_index = points.size(0) // 2
    actual_leading_edge_index = int(torch.argmin(points[:, 0]).item())
    if actual_leading_edge_index != leading_edge_index:
        raise ValueError(
            'raw_points leading-edge index must match the configured split-surface layout: '
            f'expected {leading_edge_index}, got {actual_leading_edge_index}'
        )
    fixed_leading_edge = torch.tensor(
        [util.AIRFOIL_LEADING_EDGE_X, 0.0], dtype=points.dtype
    )
    if not torch.allclose(points[leading_edge_index], fixed_leading_edge, atol=1e-6, rtol=0.0):
        raise ValueError(
            'raw_points shared leading-edge point must equal '
            f'({util.AIRFOIL_LEADING_EDGE_X}, {0.0})'
        )
    if not torch.isclose(points[0, 0], torch.tensor(util.AIRFOIL_TRAILING_EDGE_X, dtype=points.dtype)):
        raise ValueError(f'upper trailing-edge x must equal {util.AIRFOIL_TRAILING_EDGE_X}')
    if not torch.isclose(points[-1, 0], torch.tensor(util.AIRFOIL_TRAILING_EDGE_X, dtype=points.dtype)):
        raise ValueError(f'lower trailing-edge x must equal {util.AIRFOIL_TRAILING_EDGE_X}')
    trailing_edge_midpoint_y = 0.5 * (points[0, 1] + points[-1, 1])
    if not torch.isclose(trailing_edge_midpoint_y, torch.tensor(0.0, dtype=points.dtype), atol=1e-6, rtol=0.0):
        raise ValueError('raw_points trailing-edge midpoint y must equal 0.0')
    return leading_edge_index


def fit_airfoil_points(
        raw_points, device=None, verbose=False, shape_coefficient_count=None):
    coefficient_count = resolve_shape_coefficient_count(shape_coefficient_count)
    points = torch.as_tensor(raw_points, dtype=torch.float32)
    leading_edge_index = _validate_target_points(points)
    target_points = points.unsqueeze(0).to(torch.device('cpu') if device is None else torch.device(device))
    upper_x = target_points[:, :leading_edge_index + 1, 0]
    lower_x = target_points[:, leading_edge_index:, 0]
    trailing_edge_thickness = torch.nn.Parameter(
        target_points[:, 0:1, 1] - target_points[:, -1:, 1]
    )
    shape_parameter_shape = (1, coefficient_count)
    upper_shape = torch.nn.Parameter(torch.zeros(shape_parameter_shape, device=target_points.device))
    lower_shape = torch.nn.Parameter(torch.zeros(shape_parameter_shape, device=target_points.device))
    minimum_exponent = util.CST_FIT_CONFIG['minimum_class_function_exponent']
    initial_n1 = torch.tensor(util.CST_FIT_CONFIG['initial_n1'] - minimum_exponent)
    initial_n2 = torch.tensor(util.CST_FIT_CONFIG['initial_n2'] - minimum_exponent)
    raw_n1 = torch.nn.Parameter(torch.log(torch.expm1(initial_n1)).reshape(1, 1).to(target_points.device))
    raw_n2 = torch.nn.Parameter(torch.log(torch.expm1(initial_n2)).reshape(1, 1).to(target_points.device))
    optimizer = optim.Adam(
        [upper_shape, lower_shape, trailing_edge_thickness, raw_n1, raw_n2],
        lr=util.CST_FIT_CONFIG['lr'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=util.CST_FIT_CONFIG['scheduler_patience'],
        factor=util.CST_FIT_CONFIG['scheduler_factor'],
    )
    leading_edge_weights = cosine_leading_edge_weights(
        target_points,
        util.CST_FIT_CONFIG['leading_edge_window'],
        util.CST_FIT_CONFIG['leading_edge_weight_amplitude'],
    )
    for iteration in range(util.CST_FIT_CONFIG['iterations']):
        optimizer.zero_grad()
        class_function_n1 = minimum_exponent + functional.softplus(raw_n1)
        class_function_n2 = minimum_exponent + functional.softplus(raw_n2)
        curve = _decode_at_x(
            upper_shape, lower_shape, trailing_edge_thickness,
            class_function_n1, class_function_n2, upper_x, lower_x, coefficient_count
        )
        fit_loss = torch.sum(
            weighted_mae_per_airfoil(curve, target_points, leading_edge_weights)
        ) * util.CST_FIT_CONFIG['loss_scale']
        regularization = (
            torch.mean(upper_shape ** 2) + torch.mean(lower_shape ** 2)
        ) * util.CST_FIT_CONFIG['coefficient_reg']
        total_loss = fit_loss + regularization
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())
        if verbose and (
                iteration % util.CST_FIT_CONFIG['log_interval'] == 0
                or iteration == util.CST_FIT_CONFIG['iterations'] - 1):
            print(
                f'Iter {iteration}, Total: {total_loss.item():.6f}, '
                f'Weighted MAE: {fit_loss.item():.6f}, Reg: {regularization.item():.6f}'
            )
    with torch.no_grad():
        class_function_n1 = minimum_exponent + functional.softplus(raw_n1)
        class_function_n2 = minimum_exponent + functional.softplus(raw_n2)
        curve = _decode_at_x(
            upper_shape, lower_shape, trailing_edge_thickness,
            class_function_n1, class_function_n2, upper_x, lower_x, coefficient_count
        )
        code = pack_cst_airfoil_code(
            upper_shape, lower_shape, trailing_edge_thickness, class_function_n1,
            class_function_n2, coefficient_count
        ).squeeze(0)
        mae_value = torch.mean(torch.abs(curve - target_points)).detach().cpu().item()
        mse_value = torch.mean((curve - target_points) ** 2).detach().cpu().item()
        max_point_error = torch.sqrt(torch.sum((curve - target_points) ** 2, dim=2)).max()
    return {
        'code': code.detach().cpu(),
        'curve': curve.squeeze(0).detach().cpu(),
        'target_points': target_points.squeeze(0).detach().cpu(),
        'leading_edge': torch.tensor([util.AIRFOIL_LEADING_EDGE_X, 0.0]),
        'class_function_n1': class_function_n1.squeeze(0).detach().cpu(),
        'class_function_n2': class_function_n2.squeeze(0).detach().cpu(),
        'shape_coefficient_count': coefficient_count,
        'mae': float(mae_value),
        'mse': float(mse_value),
        'max_point_error': float(max_point_error.detach().cpu().item()),
    }


def encode_airfoil_dat(
        dat_path, device=None, verbose=False, shape_coefficient_count=None):
    return fit_airfoil_points(
        load_dat(dat_path), device=device, verbose=verbose,
        shape_coefficient_count=shape_coefficient_count
    )


def encode_processed_airfoil_dat(
        dat_path, device=None, verbose=False, shape_coefficient_count=None):
    coefficient_count = resolve_shape_coefficient_count(shape_coefficient_count)
    result = encode_airfoil_dat(
        dat_path, device=device, verbose=verbose,
        shape_coefficient_count=coefficient_count
    )
    result['source_airfoil'] = str(Path(dat_path).resolve())
    result['fit_config'] = cst_fit_config()
    result['device'] = str(torch.device('cpu') if device is None else torch.device(device))
    result['metrics'] = {
        'mae': result['mae'],
        'mse': result['mse'],
        'max_point_error': result['max_point_error'],
        **leading_edge_error_metrics(
            result['curve'], result['target_points'],
            util.CST_FIT_CONFIG['leading_edge_window']
        ),
    }
    return result
