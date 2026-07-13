import math
from pathlib import Path

import torch
import torch.optim as optim

import util


PROCESSED_AIRFOIL_ENCODING_VERSION = "split_rational_bezier_v1"
FIT_CONFIG_KEYS = (
    'iterations',
    'lr',
    'loss_scale',
    'weight_reg',
    'length_penalty',
    'leading_edge_window',
    'leading_edge_weight_amplitude',
    'point_density_beta',
    'surface_control_points',
    'scheduler_patience',
    'scheduler_factor',
    'log_interval',
)

DEFAULT_NN_ROOT = Path(r'D:\3D\Projects\ML\NN')
DEFAULT_COORD_NORM_PATH = DEFAULT_NN_ROOT / 'model' / 'coord_norm.pt'
OPTIMIZED_SPLIT_SURFACE_FIT_CONFIG = {
    'iterations': 700,
    'lr': 0.005,
    'loss_scale': 100.0,
    'weight_reg': 0.0007792224749560202,
    'length_penalty': 0.005746261451125304,
    'leading_edge_window': 18,
    'leading_edge_weight_amplitude': 4.8,
    'point_density_beta': util.AIRFOIL_DEFAULT_POINT_DENSITY_BETA,
    'surface_control_points': util.AIRFOIL_SURFACE_CONTROL_POINTS,
    'scheduler_patience': 500,
    'scheduler_factor': 0.5,
    'log_interval': 100,
}


def load_dat(dat_path):
    path = Path(dat_path)
    if not path.exists():
        raise FileNotFoundError(f"Airfoil .dat file not found: {path}")

    points = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    if len(points) == 0:
        raise ValueError(f"Airfoil .dat file contains no coordinate pairs: {path}")
    return torch.tensor(points, dtype=torch.float32)


def load_coord_norm(coord_norm_path):
    path = Path(coord_norm_path)
    if not path.exists():
        raise FileNotFoundError(f"Coordinate normalization file not found: {path}")
    coord_norm = torch.load(path, map_location='cpu', weights_only=True)
    validate_coord_norm(coord_norm)
    return coord_norm


def validate_coord_norm(coord_norm):
    for key in ('x_min', 'x_max', 'y_min', 'y_max'):
        if key not in coord_norm:
            raise KeyError(f"coord_norm missing required key: {key}")


def scalar_on_device(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def normalize_points(points, coord_norm):
    validate_coord_norm(coord_norm)
    normalized = points.clone()
    x_min = scalar_on_device(coord_norm['x_min'], points.device, points.dtype)
    x_max = scalar_on_device(coord_norm['x_max'], points.device, points.dtype)
    y_min = scalar_on_device(coord_norm['y_min'], points.device, points.dtype)
    y_max = scalar_on_device(coord_norm['y_max'], points.device, points.dtype)
    normalized[..., 0] = (normalized[..., 0] - x_min) / (x_max - x_min + 1e-8)
    normalized[..., 1] = (normalized[..., 1] - y_min) / (y_max - y_min + 1e-8)
    return normalized


def denormalize_points(points, coord_norm):
    validate_coord_norm(coord_norm)
    denormalized = points.clone()
    x_min = scalar_on_device(coord_norm['x_min'], points.device, points.dtype)
    x_max = scalar_on_device(coord_norm['x_max'], points.device, points.dtype)
    y_min = scalar_on_device(coord_norm['y_min'], points.device, points.dtype)
    y_max = scalar_on_device(coord_norm['y_max'], points.device, points.dtype)
    denormalized[..., 0] = denormalized[..., 0] * (x_max - x_min + 1e-8) + x_min
    denormalized[..., 1] = denormalized[..., 1] * (y_max - y_min + 1e-8) + y_min
    return denormalized


def validate_surface_control_points(surface_control_points):
    if surface_control_points != util.AIRFOIL_SURFACE_CONTROL_POINTS:
        raise ValueError(
            f"surface_control_points must equal util.AIRFOIL_SURFACE_CONTROL_POINTS "
            f"({util.AIRFOIL_SURFACE_CONTROL_POINTS}), got {surface_control_points}"
        )
    return surface_control_points


def validate_fit_config(fit_config):
    for key in FIT_CONFIG_KEYS:
        if key not in fit_config:
            raise KeyError(f"fit_config missing required key: {key}")
    validate_surface_control_points(fit_config['surface_control_points'])
    if fit_config['iterations'] <= 0:
        raise ValueError(f"iterations must be positive, got {fit_config['iterations']}")
    if fit_config['lr'] <= 0:
        raise ValueError(f"lr must be positive, got {fit_config['lr']}")
    if fit_config['loss_scale'] <= 0:
        raise ValueError(f"loss_scale must be positive, got {fit_config['loss_scale']}")
    if fit_config['weight_reg'] < 0:
        raise ValueError(f"weight_reg must be non-negative, got {fit_config['weight_reg']}")
    if fit_config['length_penalty'] < 0:
        raise ValueError(f"length_penalty must be non-negative, got {fit_config['length_penalty']}")
    if fit_config['leading_edge_window'] < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {fit_config['leading_edge_window']}")
    if fit_config['leading_edge_weight_amplitude'] < 0:
        raise ValueError(
            f"leading_edge_weight_amplitude must be non-negative, "
            f"got {fit_config['leading_edge_weight_amplitude']}"
        )
    if fit_config['point_density_beta'] <= 0:
        raise ValueError(f"point_density_beta must be positive, got {fit_config['point_density_beta']}")
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


def optimized_split_surface_fit_config():
    return dict(OPTIMIZED_SPLIT_SURFACE_FIT_CONFIG)


def center_dense_spacing(num_points, s_le=0.5, beta=util.AIRFOIL_DEFAULT_POINT_DENSITY_BETA):
    if num_points < 3:
        raise ValueError(f"num_points must be at least 3, got {num_points}")
    if not 0.0 < s_le < 1.0:
        raise ValueError(f"s_le must be in (0, 1), got {s_le}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    left_count = max(2, int(round(num_points * s_le)))
    right_count = num_points - left_count + 1
    left_u = torch.linspace(0, 1, left_count)
    right_u = torch.linspace(0, 1, right_count)[1:]
    left_t = s_le * (1.0 - (1.0 - left_u) ** beta)
    right_t = s_le + (1.0 - s_le) * (right_u ** beta)
    t_values = torch.cat([left_t, right_t])
    t_values[0] = 0.0
    t_values[-1] = 1.0
    return torch.clamp(t_values, 0.0, 1.0)


def build_bernstein_basis(t_values, num_control_points):
    if num_control_points < 2:
        raise ValueError(f"num_control_points must be at least 2, got {num_control_points}")
    order = num_control_points - 1
    t_double = t_values.to(torch.float64)
    basis = torch.zeros(
        (*t_values.shape, num_control_points),
        dtype=torch.float64,
        device=t_values.device,
    )
    for index in range(num_control_points):
        coeff = math.comb(order, index)
        basis[..., index] = coeff * (t_double ** index) * ((1.0 - t_double) ** (order - index))
    return basis.to(torch.float32)


def rational_bezier_curve_from_basis(control_points, weights, basis):
    weighted_control_points = control_points * weights.unsqueeze(-1)
    numerator = torch.bmm(basis, weighted_control_points)
    denominator = torch.bmm(basis, weights.unsqueeze(-1))
    return numerator / (denominator + 1e-8)


def _as_batched_control_points(control_points, name):
    tensor = torch.as_tensor(control_points, dtype=torch.float32)
    unbatched = tensor.dim() == 2
    if unbatched:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3 or tensor.size(1) != util.AIRFOIL_SURFACE_CONTROL_POINTS or tensor.size(2) != 2:
        raise ValueError(
            f"{name} must have shape "
            f"[{util.AIRFOIL_SURFACE_CONTROL_POINTS}, 2] or "
            f"[B, {util.AIRFOIL_SURFACE_CONTROL_POINTS}, 2], got {list(tensor.size())}"
        )
    return tensor, unbatched


def _as_batched_weights(weights, name):
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    unbatched = tensor.dim() == 1
    if unbatched:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(1) != util.AIRFOIL_SURFACE_CONTROL_POINTS:
        raise ValueError(
            f"{name} must have shape "
            f"[{util.AIRFOIL_SURFACE_CONTROL_POINTS}] or "
            f"[B, {util.AIRFOIL_SURFACE_CONTROL_POINTS}], got {list(tensor.size())}"
        )
    return tensor, unbatched


def pack_split_airfoil_code(upper_control_points, lower_control_points, upper_weights, lower_weights):
    upper_cp, upper_cp_unbatched = _as_batched_control_points(upper_control_points, 'upper_control_points')
    lower_cp, lower_cp_unbatched = _as_batched_control_points(lower_control_points, 'lower_control_points')
    upper_w, upper_w_unbatched = _as_batched_weights(upper_weights, 'upper_weights')
    lower_w, lower_w_unbatched = _as_batched_weights(lower_weights, 'lower_weights')

    batch_size = upper_cp.size(0)
    for name, tensor in (
            ('lower_control_points', lower_cp),
            ('upper_weights', upper_w),
            ('lower_weights', lower_w)):
        if tensor.size(0) != batch_size:
            raise ValueError(f"{name} batch size {tensor.size(0)} does not match {batch_size}")

    code = torch.cat([
        upper_cp.reshape(batch_size, -1),
        lower_cp.reshape(batch_size, -1),
        upper_w,
        lower_w,
    ], dim=1)
    if code.size(1) != util.AIRFOIL_BEZIER_CODE_SIZE:
        raise ValueError(
            f"Packed airfoil code has {code.size(1)} values, "
            f"expected {util.AIRFOIL_BEZIER_CODE_SIZE}"
        )
    if upper_cp_unbatched and lower_cp_unbatched and upper_w_unbatched and lower_w_unbatched:
        return code.squeeze(0)
    return code


def _as_batched_code(code):
    tensor = torch.as_tensor(code, dtype=torch.float32)
    unbatched = tensor.dim() == 1
    if unbatched:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(1) != util.AIRFOIL_BEZIER_CODE_SIZE:
        raise ValueError(
            f"airfoil code must have shape [{util.AIRFOIL_BEZIER_CODE_SIZE}] "
            f"or [B, {util.AIRFOIL_BEZIER_CODE_SIZE}], got {list(tensor.size())}"
        )
    return tensor, unbatched


def unpack_split_airfoil_code(code):
    code_tensor, unbatched = _as_batched_code(code)
    batch_size = code_tensor.size(0)
    cp_values = util.AIRFOIL_SURFACE_CONTROL_POINTS * 2
    upper_cp = code_tensor[:, :cp_values].reshape(batch_size, util.AIRFOIL_SURFACE_CONTROL_POINTS, 2)
    lower_start = cp_values
    lower_end = lower_start + cp_values
    lower_cp = code_tensor[:, lower_start:lower_end].reshape(batch_size, util.AIRFOIL_SURFACE_CONTROL_POINTS, 2)
    upper_weights_start = lower_end
    upper_weights_end = upper_weights_start + util.AIRFOIL_SURFACE_CONTROL_POINTS
    upper_weights = code_tensor[:, upper_weights_start:upper_weights_end]
    lower_weights = code_tensor[:, upper_weights_end:]

    if unbatched:
        return {
            util.AIRFOIL_UPPER_SURFACE: {
                'control_points': upper_cp.squeeze(0),
                'weights': upper_weights.squeeze(0),
            },
            util.AIRFOIL_LOWER_SURFACE: {
                'control_points': lower_cp.squeeze(0),
                'weights': lower_weights.squeeze(0),
            },
        }
    return {
        util.AIRFOIL_UPPER_SURFACE: {
            'control_points': upper_cp,
            'weights': upper_weights,
        },
        util.AIRFOIL_LOWER_SURFACE: {
            'control_points': lower_cp,
            'weights': lower_weights,
        },
    }


def symmetric_airfoil_code_from_upper(airfoil_code, symmetry_line_y):
    """Mirror the encoded upper surface into the lower surface exactly.

    Split codes traverse upper TE-to-LE and lower LE-to-TE. Therefore the
    lower controls and weights must be reversed before reflecting their y
    coordinates about the physical chord line represented in normalized space.
    """
    surfaces = unpack_split_airfoil_code(airfoil_code)
    upper_control_points = surfaces[util.AIRFOIL_UPPER_SURFACE]['control_points']
    upper_weights = surfaces[util.AIRFOIL_UPPER_SURFACE]['weights']
    lower_control_points = torch.flip(upper_control_points, dims=(-2,)).clone()
    lower_control_points[..., 1] = (
        2.0 * torch.as_tensor(
            symmetry_line_y,
            dtype=lower_control_points.dtype,
            device=lower_control_points.device,
        )
        - lower_control_points[..., 1]
    )
    lower_weights = torch.flip(upper_weights, dims=(-1,))
    return pack_split_airfoil_code(
        upper_control_points,
        lower_control_points,
        upper_weights,
        lower_weights,
    )


def normalized_y_coordinate(y_value, coord_norm):
    validate_coord_norm(coord_norm)
    y_min = torch.as_tensor(coord_norm['y_min'], dtype=torch.float32)
    y_max = torch.as_tensor(coord_norm['y_max'], dtype=torch.float32)
    return (torch.as_tensor(y_value, dtype=torch.float32) - y_min) / (y_max - y_min + 1e-8)


def _unpack_batched_split_airfoil_code(code):
    code_tensor, _ = _as_batched_code(code)
    return unpack_split_airfoil_code(code_tensor)


def decode_airfoil_code(
        code,
        num_output_points=util.AIRFOIL_DEFAULT_OUTPUT_POINTS,
        point_density_beta=util.AIRFOIL_DEFAULT_POINT_DENSITY_BETA):
    code_tensor, _ = _as_batched_code(code)
    surfaces = _unpack_batched_split_airfoil_code(code_tensor)
    upper_cp = surfaces[util.AIRFOIL_UPPER_SURFACE]['control_points']
    lower_cp = surfaces[util.AIRFOIL_LOWER_SURFACE]['control_points']
    upper_weights = torch.abs(surfaces[util.AIRFOIL_UPPER_SURFACE]['weights'])
    lower_weights = torch.abs(surfaces[util.AIRFOIL_LOWER_SURFACE]['weights'])

    batch_size = code_tensor.size(0)
    base_t = center_dense_spacing(num_output_points, s_le=0.5, beta=point_density_beta).to(
        device=code_tensor.device,
        dtype=code_tensor.dtype,
    )
    upper_mask = base_t <= 0.5
    upper_t = torch.clamp(base_t / 0.5, 0.0, 1.0)
    lower_t = torch.clamp((base_t - 0.5) / 0.5, 0.0, 1.0)
    upper_basis = build_bernstein_basis(upper_t, util.AIRFOIL_SURFACE_CONTROL_POINTS)
    lower_basis = build_bernstein_basis(lower_t, util.AIRFOIL_SURFACE_CONTROL_POINTS)
    upper_basis = upper_basis.unsqueeze(0).expand(batch_size, -1, -1)
    lower_basis = lower_basis.unsqueeze(0).expand(batch_size, -1, -1)

    upper_curve = rational_bezier_curve_from_basis(upper_cp, upper_weights, upper_basis)
    lower_curve = rational_bezier_curve_from_basis(lower_cp, lower_weights, lower_basis)
    return torch.where(upper_mask.view(1, -1, 1), upper_curve, lower_curve)


def cosine_leading_edge_weights(target_points, window, amplitude):
    if window < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {window}")
    if amplitude < 0:
        raise ValueError(f"leading_edge_weight_amplitude must be non-negative, got {amplitude}")

    weights = torch.ones(
        target_points.shape[:2],
        dtype=target_points.dtype,
        device=target_points.device,
    )
    if window == 0 or amplitude == 0:
        return weights

    index_positions = torch.arange(target_points.size(1), device=target_points.device)
    for sample_index in range(target_points.size(0)):
        leading_edge_index = int(torch.argmin(target_points[sample_index, :, 0]).item())
        distance = torch.abs(index_positions - leading_edge_index)
        mask = distance <= window
        normalized_distance = distance[mask].to(target_points.dtype) / float(window)
        weights[sample_index, mask] = (
            1.0
            + amplitude
            * 0.5
            * (1.0 + torch.cos(torch.pi * normalized_distance))
        )
    return weights


def weighted_mae_per_airfoil(curve, target_points, weights):
    absolute_error = torch.mean(torch.abs(curve - target_points), dim=2)
    return torch.sum(absolute_error * weights, dim=1) / torch.sum(weights, dim=1)


def leading_edge_error_metrics(curve, target_points, window):
    if window < 0:
        raise ValueError(f"leading_edge_window must be non-negative, got {window}")
    leading_edge_index = int(torch.argmin(target_points[:, 0]).item())
    start = max(0, leading_edge_index - window)
    end = min(target_points.size(0), leading_edge_index + window + 1)
    error = curve[start:end] - target_points[start:end]
    return {
        'leading_edge_mae': float(torch.mean(torch.abs(error)).detach().cpu().item()),
        'leading_edge_mse': float(torch.mean(error ** 2).detach().cpu().item()),
    }


def sample_surface_control_points(target_points, leading_edge_indices, count, surface_name):
    batch_points = []
    for sample_index in range(target_points.size(0)):
        leading_edge_index = int(leading_edge_indices[sample_index].item())
        if surface_name == util.AIRFOIL_UPPER_SURFACE:
            indices = torch.linspace(0, leading_edge_index, count, device=target_points.device).long()
        elif surface_name == util.AIRFOIL_LOWER_SURFACE:
            indices = torch.linspace(leading_edge_index, target_points.size(1) - 1, count, device=target_points.device).long()
        else:
            raise ValueError(f"Unknown airfoil surface: {surface_name}")
        batch_points.append(target_points[sample_index, indices, :])
    return torch.stack(batch_points, dim=0)


def build_split_surface_t_values(target_points, point_density_beta):
    batch_size = target_points.size(0)
    num_points = target_points.size(1)
    base_t = center_dense_spacing(num_points, s_le=0.5, beta=point_density_beta).to(
        device=target_points.device,
        dtype=target_points.dtype,
    )
    upper_t = torch.zeros((batch_size, num_points), dtype=target_points.dtype, device=target_points.device)
    lower_t = torch.zeros((batch_size, num_points), dtype=target_points.dtype, device=target_points.device)
    upper_mask = torch.zeros((batch_size, num_points), dtype=torch.bool, device=target_points.device)
    leading_edge_indices = torch.argmin(target_points[:, :, 0], dim=1)

    for sample_index in range(batch_size):
        leading_edge_index = int(leading_edge_indices[sample_index].item())
        split_t = base_t[leading_edge_index]
        upper_denominator = torch.clamp(split_t, min=1e-8)
        lower_denominator = torch.clamp(1.0 - split_t, min=1e-8)
        upper_t[sample_index, :] = torch.clamp(base_t / upper_denominator, 0.0, 1.0)
        lower_t[sample_index, :] = torch.clamp((base_t - split_t) / lower_denominator, 0.0, 1.0)
        upper_mask[sample_index, :leading_edge_index + 1] = True

    return upper_t, lower_t, upper_mask, leading_edge_indices


def fit_airfoil_points(raw_points, fit_config, coord_norm=None, device=None, verbose=False):
    validate_fit_config(fit_config)
    points = torch.as_tensor(raw_points, dtype=torch.float32)
    if points.dim() != 2 or points.size(1) != 2:
        raise ValueError(f"raw_points must have shape [N, 2], got {list(points.size())}")
    if points.size(0) < util.AIRFOIL_SURFACE_CONTROL_POINTS * 2:
        raise ValueError(
            f"raw_points must contain at least {util.AIRFOIL_SURFACE_CONTROL_POINTS * 2} points, "
            f"got {points.size(0)}"
        )
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    raw_target_points = points.unsqueeze(0).to(device)
    if coord_norm is None:
        target_points = raw_target_points.clone()
    else:
        target_points = normalize_points(raw_target_points, coord_norm)

    leading_edge_weights = cosine_leading_edge_weights(
        target_points,
        fit_config['leading_edge_window'],
        fit_config['leading_edge_weight_amplitude'],
    )
    upper_t, lower_t, upper_mask, leading_edge_indices = build_split_surface_t_values(
        target_points,
        fit_config['point_density_beta'],
    )
    upper_basis = build_bernstein_basis(upper_t, util.AIRFOIL_SURFACE_CONTROL_POINTS)
    lower_basis = build_bernstein_basis(lower_t, util.AIRFOIL_SURFACE_CONTROL_POINTS)

    upper_init = sample_surface_control_points(
        target_points,
        leading_edge_indices,
        util.AIRFOIL_SURFACE_CONTROL_POINTS,
        util.AIRFOIL_UPPER_SURFACE,
    )
    lower_init = sample_surface_control_points(
        target_points,
        leading_edge_indices,
        util.AIRFOIL_SURFACE_CONTROL_POINTS,
        util.AIRFOIL_LOWER_SURFACE,
    )
    leading_edge_points = torch.gather(
        target_points,
        1,
        leading_edge_indices.view(1, 1, 1).expand(-1, -1, 2),
    )
    upper_start_points = target_points[:, 0:1, :]
    lower_end_points = target_points[:, -1:, :]

    upper_trainable_control_points = torch.nn.Parameter(upper_init[:, 1:-1, :])
    lower_trainable_control_points = torch.nn.Parameter(lower_init[:, 1:-1, :])
    upper_weights = torch.nn.Parameter(torch.ones((1, util.AIRFOIL_SURFACE_CONTROL_POINTS), device=device))
    lower_weights = torch.nn.Parameter(torch.ones((1, util.AIRFOIL_SURFACE_CONTROL_POINTS), device=device))
    optimizer = torch.optim.Adam(
        [
            upper_trainable_control_points,
            lower_trainable_control_points,
            upper_weights,
            lower_weights,
        ],
        lr=fit_config['lr'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=fit_config['scheduler_patience'],
        factor=fit_config['scheduler_factor'],
    )

    for iteration in range(fit_config['iterations']):
        optimizer.zero_grad()
        full_upper_control_points = torch.cat(
            [upper_start_points, upper_trainable_control_points, leading_edge_points],
            dim=1,
        )
        full_lower_control_points = torch.cat(
            [leading_edge_points, lower_trainable_control_points, lower_end_points],
            dim=1,
        )
        abs_upper_weights = torch.abs(upper_weights)
        abs_lower_weights = torch.abs(lower_weights)
        upper_curve = rational_bezier_curve_from_basis(
            full_upper_control_points,
            abs_upper_weights,
            upper_basis,
        )
        lower_curve = rational_bezier_curve_from_basis(
            full_lower_control_points,
            abs_lower_weights,
            lower_basis,
        )
        curve = torch.where(upper_mask.unsqueeze(-1), upper_curve, lower_curve)
        mae = weighted_mae_per_airfoil(curve, target_points, leading_edge_weights)
        fit_loss = torch.sum(mae) * fit_config['loss_scale']
        reg_loss = (
            torch.sum(torch.mean(abs_upper_weights ** 2, dim=1))
            + torch.sum(torch.mean(abs_lower_weights ** 2, dim=1))
        ) * fit_config['weight_reg']
        upper_cp_diff = full_upper_control_points[:, 1:, :] - full_upper_control_points[:, :-1, :]
        lower_cp_diff = full_lower_control_points[:, 1:, :] - full_lower_control_points[:, :-1, :]
        length_penalty = (
            torch.sum(torch.mean(upper_cp_diff ** 2, dim=(1, 2)))
            + torch.sum(torch.mean(lower_cp_diff ** 2, dim=(1, 2)))
        ) * fit_config['length_penalty']
        total_loss = fit_loss + reg_loss + length_penalty
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        if verbose and (
            iteration % fit_config['log_interval'] == 0
            or iteration == fit_config['iterations'] - 1
        ):
            print(
                f"Iter {iteration}, Total: {total_loss.item():.6f}, "
                f"Weighted MAE: {fit_loss.item():.6f}, Reg: {reg_loss.item():.6f}, "
                f"Len: {length_penalty.item():.6f}"
            )

    with torch.no_grad():
        final_upper_control_points = torch.cat(
            [upper_start_points, upper_trainable_control_points, leading_edge_points],
            dim=1,
        )
        final_lower_control_points = torch.cat(
            [leading_edge_points, lower_trainable_control_points, lower_end_points],
            dim=1,
        )
        final_upper_weights = torch.abs(upper_weights)
        final_lower_weights = torch.abs(lower_weights)
        final_upper_curve = rational_bezier_curve_from_basis(
            final_upper_control_points,
            final_upper_weights,
            upper_basis,
        )
        final_lower_curve = rational_bezier_curve_from_basis(
            final_lower_control_points,
            final_lower_weights,
            lower_basis,
        )
        final_curve = torch.where(upper_mask.unsqueeze(-1), final_upper_curve, final_lower_curve)
        code = pack_split_airfoil_code(
            final_upper_control_points,
            final_lower_control_points,
            final_upper_weights,
            final_lower_weights,
        ).squeeze(0)
        mae_value = torch.mean(torch.abs(final_curve - target_points)).detach().cpu().item()
        mse_value = torch.mean((final_curve - target_points) ** 2).detach().cpu().item()
        max_point_error = torch.sqrt(torch.sum((final_curve - target_points) ** 2, dim=2)).max()

    return {
        'code': code.detach().cpu(),
        'control_points': {
            util.AIRFOIL_UPPER_SURFACE: final_upper_control_points.squeeze(0).detach().cpu(),
            util.AIRFOIL_LOWER_SURFACE: final_lower_control_points.squeeze(0).detach().cpu(),
        },
        'weights': {
            util.AIRFOIL_UPPER_SURFACE: final_upper_weights.squeeze(0).detach().cpu(),
            util.AIRFOIL_LOWER_SURFACE: final_lower_weights.squeeze(0).detach().cpu(),
        },
        'curve': final_curve.squeeze(0).detach().cpu(),
        'target_points': target_points.squeeze(0).detach().cpu(),
        'raw_target_points': raw_target_points.squeeze(0).detach().cpu(),
        'mae': float(mae_value),
        'mse': float(mse_value),
        'max_point_error': float(max_point_error.detach().cpu().item()),
    }


def encode_airfoil_dat(dat_path, fit_config, coord_norm=None, device=None, verbose=False):
    raw_points = load_dat(dat_path)
    return fit_airfoil_points(
        raw_points,
        fit_config,
        coord_norm=coord_norm,
        device=device,
        verbose=verbose,
    )


def encode_processed_airfoil_dat(dat_path, device=None, verbose=False):
    source_path = Path(dat_path)
    coord_norm = load_coord_norm(DEFAULT_COORD_NORM_PATH)
    result = encode_airfoil_dat(
        source_path,
        optimized_split_surface_fit_config(),
        coord_norm=coord_norm,
        device=device,
        verbose=verbose,
    )
    result['source_airfoil'] = str(source_path.resolve())
    result['coord_norm'] = coord_norm
    result['coord_norm_path'] = str(DEFAULT_COORD_NORM_PATH)
    result['fit_config'] = optimized_split_surface_fit_config()
    result['device'] = str(torch.device('cpu') if device is None else torch.device(device))
    result['metrics'] = {
        'mae': result['mae'],
        'mse': result['mse'],
        'max_point_error': result['max_point_error'],
        **leading_edge_error_metrics(
            result['curve'],
            result['target_points'],
            result['fit_config']['leading_edge_window'],
        ),
    }
    return result
