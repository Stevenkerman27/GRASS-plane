"""24D Kulfan CST encoding with a fixed physical leading edge at (0, 0)."""

from pathlib import Path
import math

import torch
import torch.nn.functional as functional
import torch.optim as optim

import airfoil_sampling
import util


PROCESSED_AIRFOIL_ENCODING_VERSION = 'fixed_leading_edge_cst_v2'
CST_FIT_CONFIG_KEYS = (
    'iterations', 'lr', 'loss_scale', 'coefficient_reg', 'leading_edge_window',
    'leading_edge_weight_amplitude', 'scheduler_patience', 'scheduler_factor',
    'log_interval', 'surface_shape_coefficients', 'initial_n1', 'initial_n2',
    'minimum_class_function_exponent',
)
CST_FIT_METRIC_KEYS = (
    'mae', 'mse', 'max_point_error', 'leading_edge_mae', 'leading_edge_mse',
)


def cst_fit_config():
    return dict(util.CST_FIT_CONFIG)


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


def validate_fit_config(fit_config):
    for key in CST_FIT_CONFIG_KEYS:
        if key not in fit_config:
            raise KeyError(f'fit_config missing required key: {key}')
    if fit_config['iterations'] <= 0:
        raise ValueError(f"iterations must be positive, got {fit_config['iterations']}")
    if fit_config['lr'] <= 0:
        raise ValueError(f"lr must be positive, got {fit_config['lr']}")
    if fit_config['loss_scale'] <= 0:
        raise ValueError(f"loss_scale must be positive, got {fit_config['loss_scale']}")
    if fit_config['coefficient_reg'] < 0:
        raise ValueError(f"coefficient_reg must be non-negative, got {fit_config['coefficient_reg']}")
    if fit_config['leading_edge_window'] < 0:
        raise ValueError(
            f"leading_edge_window must be non-negative, got {fit_config['leading_edge_window']}"
        )
    if fit_config['leading_edge_weight_amplitude'] < 0:
        raise ValueError(
            'leading_edge_weight_amplitude must be non-negative, '
            f"got {fit_config['leading_edge_weight_amplitude']}"
        )
    if fit_config['scheduler_patience'] < 0:
        raise ValueError(
            f"scheduler_patience must be non-negative, got {fit_config['scheduler_patience']}"
        )
    if not 0.0 < fit_config['scheduler_factor'] < 1.0:
        raise ValueError(
            f"scheduler_factor must be in (0, 1), got {fit_config['scheduler_factor']}"
        )
    if fit_config['log_interval'] <= 0:
        raise ValueError(f"log_interval must be positive, got {fit_config['log_interval']}")
    if fit_config['surface_shape_coefficients'] != util.CST_SURFACE_SHAPE_COEFFICIENTS:
        raise ValueError(
            'surface_shape_coefficients must equal '
            f'util.CST_SURFACE_SHAPE_COEFFICIENTS ({util.CST_SURFACE_SHAPE_COEFFICIENTS})'
        )
    if fit_config['initial_n1'] <= fit_config['minimum_class_function_exponent']:
        raise ValueError(
            'initial_n1 must exceed minimum_class_function_exponent'
        )
    if fit_config['initial_n2'] <= fit_config['minimum_class_function_exponent']:
        raise ValueError(
            'initial_n2 must exceed minimum_class_function_exponent'
        )
    if fit_config['minimum_class_function_exponent'] != util.CST_MIN_CLASS_FUNCTION_EXPONENT:
        raise ValueError(
            'minimum_class_function_exponent must equal util.CST_MIN_CLASS_FUNCTION_EXPONENT '
            f'({util.CST_MIN_CLASS_FUNCTION_EXPONENT})'
        )


def _as_batched_code(code):
    tensor = torch.as_tensor(code, dtype=torch.float32)
    unbatched = tensor.dim() == 1
    if unbatched:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(1) != util.CST_AIRFOIL_CODE_SIZE:
        raise ValueError(
            f'CST code must have shape [{util.CST_AIRFOIL_CODE_SIZE}] or '
            f'[B, {util.CST_AIRFOIL_CODE_SIZE}], got {list(tensor.size())}'
        )
    if not torch.isfinite(tensor).all():
        raise ValueError('CST code must contain only finite values')
    return tensor, unbatched


def _as_batched_shape_coefficients(coefficients, name):
    tensor = torch.as_tensor(coefficients, dtype=torch.float32)
    unbatched = tensor.dim() == 1
    if unbatched:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(1) != util.CST_SURFACE_SHAPE_COEFFICIENTS:
        raise ValueError(
            f'{name} must have shape [{util.CST_SURFACE_SHAPE_COEFFICIENTS}] or '
            f'[B, {util.CST_SURFACE_SHAPE_COEFFICIENTS}], got {list(tensor.size())}'
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} must contain only finite values')
    return tensor, unbatched


def _as_batched_trailing_edge_y(trailing_edge_y, name):
    tensor = torch.as_tensor(trailing_edge_y, dtype=torch.float32)
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
    tensor, unbatched = _as_batched_trailing_edge_y(exponent, name)
    if not torch.all(tensor > util.CST_MIN_CLASS_FUNCTION_EXPONENT):
        raise ValueError(
            f'{name} must exceed {util.CST_MIN_CLASS_FUNCTION_EXPONENT}'
        )
    return tensor, unbatched


def pack_cst_airfoil_code(
        upper_shape_coefficients,
        lower_shape_coefficients,
        upper_trailing_edge_y,
        lower_trailing_edge_y,
        class_function_n1,
        class_function_n2):
    upper, upper_unbatched = _as_batched_shape_coefficients(
        upper_shape_coefficients, 'upper_shape_coefficients'
    )
    lower, lower_unbatched = _as_batched_shape_coefficients(
        lower_shape_coefficients, 'lower_shape_coefficients'
    )
    upper_te, upper_te_unbatched = _as_batched_trailing_edge_y(
        upper_trailing_edge_y, 'upper_trailing_edge_y'
    )
    lower_te, lower_te_unbatched = _as_batched_trailing_edge_y(
        lower_trailing_edge_y, 'lower_trailing_edge_y'
    )
    n1, n1_unbatched = _as_batched_class_exponent(class_function_n1, 'class_function_n1')
    n2, n2_unbatched = _as_batched_class_exponent(class_function_n2, 'class_function_n2')
    batch_size = upper.size(0)
    for name, tensor in (
            ('lower_shape_coefficients', lower),
            ('upper_trailing_edge_y', upper_te),
            ('lower_trailing_edge_y', lower_te),
            ('class_function_n1', n1),
            ('class_function_n2', n2)):
        if tensor.size(0) != batch_size:
            raise ValueError(f'{name} batch size {tensor.size(0)} does not match {batch_size}')
    code = torch.cat([upper, lower, upper_te, lower_te, n1, n2], dim=1)
    if code.size(1) != util.CST_AIRFOIL_CODE_SIZE:
        raise RuntimeError(
            f'Packed CST code has {code.size(1)} values, expected {util.CST_AIRFOIL_CODE_SIZE}'
        )
    if (upper_unbatched and lower_unbatched and upper_te_unbatched and lower_te_unbatched
            and n1_unbatched and n2_unbatched):
        return code.squeeze(0)
    return code


def unpack_cst_airfoil_code(code):
    code_tensor, unbatched = _as_batched_code(code)
    coefficient_count = util.CST_SURFACE_SHAPE_COEFFICIENTS
    offset = 0
    upper = code_tensor[:, offset:offset + coefficient_count]
    offset += coefficient_count
    lower = code_tensor[:, offset:offset + coefficient_count]
    offset += coefficient_count
    upper_te = code_tensor[:, offset:offset + 1]
    offset += 1
    lower_te = code_tensor[:, offset:offset + 1]
    offset += 1
    n1 = code_tensor[:, offset:offset + 1]
    offset += 1
    n2 = code_tensor[:, offset:offset + 1]
    offset += 1
    if offset != util.CST_AIRFOIL_CODE_SIZE:
        raise RuntimeError(f'Expected to consume {util.CST_AIRFOIL_CODE_SIZE} CST values, consumed {offset}')
    _as_batched_class_exponent(n1, 'class_function_n1')
    _as_batched_class_exponent(n2, 'class_function_n2')
    result = {
        'upper': {'shape_coefficients': upper, 'trailing_edge_y': upper_te},
        'lower': {'shape_coefficients': lower, 'trailing_edge_y': lower_te},
        'class_function_n1': n1,
        'class_function_n2': n2,
    }
    if unbatched:
        unpacked = {
            surface_name: {
                'shape_coefficients': values['shape_coefficients'].squeeze(0),
                'trailing_edge_y': values['trailing_edge_y'].squeeze(0),
            }
            for surface_name, values in result.items()
            if surface_name in ('upper', 'lower')
        }
        unpacked['class_function_n1'] = n1.squeeze(0)
        unpacked['class_function_n2'] = n2.squeeze(0)
        return unpacked
    return result


def _class_shape_y(shape_coefficients, trailing_edge_y, class_function_n1, class_function_n2, x):
    if not torch.all((x >= -1e-5) & (x <= util.AIRFOIL_TRAILING_EDGE_X + 1e-5)):
        raise ValueError('x coordinates must lie between the fixed leading and trailing edges')
    xi = torch.clamp(x, util.AIRFOIL_LEADING_EDGE_X, util.AIRFOIL_TRAILING_EDGE_X)
    basis = _bernstein_basis(xi, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    class_function = (
        xi ** class_function_n1
        * (1.0 - xi) ** class_function_n2
    )
    shape_function = torch.sum(basis * shape_coefficients.unsqueeze(1), dim=2)
    return xi * trailing_edge_y + class_function * shape_function


def _decode_at_x(upper_shape, lower_shape, upper_te_y, lower_te_y, class_function_n1, class_function_n2, upper_x, lower_x):
    upper_y = _class_shape_y(upper_shape, upper_te_y, class_function_n1, class_function_n2, upper_x)
    lower_y = _class_shape_y(lower_shape, lower_te_y, class_function_n1, class_function_n2, lower_x)
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


def decode_cst_airfoil_code(code, num_output_points=util.AIRFOIL_DEFAULT_OUTPUT_POINTS):
    if num_output_points < 3:
        raise ValueError(f'num_output_points must be at least 3, got {num_output_points}')
    code_tensor, _ = _as_batched_code(code)
    unpacked = unpack_cst_airfoil_code(code_tensor)
    upper_x, lower_x = _sampling_x_values(num_output_points, code_tensor.device, code_tensor.dtype)
    upper_x = upper_x.expand(code_tensor.size(0), -1)
    lower_x = lower_x.expand(code_tensor.size(0), -1)
    return _decode_at_x(
        unpacked['upper']['shape_coefficients'],
        unpacked['lower']['shape_coefficients'],
        unpacked['upper']['trailing_edge_y'],
        unpacked['lower']['trailing_edge_y'],
        unpacked['class_function_n1'],
        unpacked['class_function_n2'],
        upper_x,
        lower_x,
    )


def symmetric_airfoil_code_from_upper(airfoil_code, symmetry_line_y):
    if symmetry_line_y != 0.0:
        raise ValueError(
            'fixed-leading-edge CST can only mirror about '
            f'y={0.0}, got {symmetry_line_y}'
        )
    unpacked = unpack_cst_airfoil_code(airfoil_code)
    upper = unpacked['upper']
    return pack_cst_airfoil_code(
        upper['shape_coefficients'],
        -upper['shape_coefficients'],
        upper['trailing_edge_y'],
        -upper['trailing_edge_y'],
        unpacked['class_function_n1'],
        unpacked['class_function_n2'],
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
    return leading_edge_index


def fit_airfoil_points(raw_points, fit_config, device=None, verbose=False):
    validate_fit_config(fit_config)
    points = torch.as_tensor(raw_points, dtype=torch.float32)
    leading_edge_index = _validate_target_points(points)
    target_points = points.unsqueeze(0).to(torch.device('cpu') if device is None else torch.device(device))
    upper_x = target_points[:, :leading_edge_index + 1, 0]
    lower_x = target_points[:, leading_edge_index:, 0]
    upper_te_y = torch.nn.Parameter(target_points[:, 0:1, 1].clone())
    lower_te_y = torch.nn.Parameter(target_points[:, -1:, 1].clone())
    shape_parameter_shape = (1, util.CST_SURFACE_SHAPE_COEFFICIENTS)
    upper_shape = torch.nn.Parameter(torch.zeros(shape_parameter_shape, device=target_points.device))
    lower_shape = torch.nn.Parameter(torch.zeros(shape_parameter_shape, device=target_points.device))
    minimum_exponent = fit_config['minimum_class_function_exponent']
    initial_n1 = torch.tensor(fit_config['initial_n1'] - minimum_exponent)
    initial_n2 = torch.tensor(fit_config['initial_n2'] - minimum_exponent)
    raw_n1 = torch.nn.Parameter(torch.log(torch.expm1(initial_n1)).reshape(1, 1).to(target_points.device))
    raw_n2 = torch.nn.Parameter(torch.log(torch.expm1(initial_n2)).reshape(1, 1).to(target_points.device))
    optimizer = optim.Adam(
        [upper_shape, lower_shape, upper_te_y, lower_te_y, raw_n1, raw_n2], lr=fit_config['lr']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=fit_config['scheduler_patience'],
        factor=fit_config['scheduler_factor'],
    )
    leading_edge_weights = cosine_leading_edge_weights(
        target_points,
        fit_config['leading_edge_window'],
        fit_config['leading_edge_weight_amplitude'],
    )
    for iteration in range(fit_config['iterations']):
        optimizer.zero_grad()
        class_function_n1 = minimum_exponent + functional.softplus(raw_n1)
        class_function_n2 = minimum_exponent + functional.softplus(raw_n2)
        curve = _decode_at_x(
            upper_shape, lower_shape, upper_te_y, lower_te_y,
            class_function_n1, class_function_n2, upper_x, lower_x
        )
        fit_loss = torch.sum(
            weighted_mae_per_airfoil(curve, target_points, leading_edge_weights)
        ) * fit_config['loss_scale']
        regularization = (
            torch.mean(upper_shape ** 2) + torch.mean(lower_shape ** 2)
        ) * fit_config['coefficient_reg']
        total_loss = fit_loss + regularization
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())
        if verbose and (
                iteration % fit_config['log_interval'] == 0
                or iteration == fit_config['iterations'] - 1):
            print(
                f'Iter {iteration}, Total: {total_loss.item():.6f}, '
                f'Weighted MAE: {fit_loss.item():.6f}, Reg: {regularization.item():.6f}'
            )
    with torch.no_grad():
        class_function_n1 = minimum_exponent + functional.softplus(raw_n1)
        class_function_n2 = minimum_exponent + functional.softplus(raw_n2)
        curve = _decode_at_x(
            upper_shape, lower_shape, upper_te_y, lower_te_y,
            class_function_n1, class_function_n2, upper_x, lower_x
        )
        code = pack_cst_airfoil_code(
            upper_shape, lower_shape, upper_te_y, lower_te_y, class_function_n1, class_function_n2
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
        'mae': float(mae_value),
        'mse': float(mse_value),
        'max_point_error': float(max_point_error.detach().cpu().item()),
    }


def encode_airfoil_dat(dat_path, fit_config, device=None, verbose=False):
    return fit_airfoil_points(load_dat(dat_path), fit_config, device=device, verbose=verbose)


def encode_processed_airfoil_dat(dat_path, device=None, verbose=False):
    result = encode_airfoil_dat(dat_path, cst_fit_config(), device=device, verbose=verbose)
    result['source_airfoil'] = str(Path(dat_path).resolve())
    result['fit_config'] = cst_fit_config()
    result['device'] = str(torch.device('cpu') if device is None else torch.device(device))
    result['metrics'] = {
        'mae': result['mae'],
        'mse': result['mse'],
        'max_point_error': result['max_point_error'],
        **leading_edge_error_metrics(
            result['curve'], result['target_points'], result['fit_config']['leading_edge_window']
        ),
    }
    return result
