"""Sampling layout shared by processed-airfoil generation and CST decoding."""

import numpy as np


def endpoint_dense_spacing(num_points, endpoint, beta):
    if num_points < 2:
        raise ValueError(f'num_points must be at least 2, got {num_points}')
    if beta <= 0.0:
        raise ValueError(f'beta must be positive, got {beta}')
    values = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    if endpoint == 'start':
        return values ** beta
    if endpoint == 'end':
        return 1.0 - (1.0 - values) ** beta
    raise ValueError(f"endpoint must be 'start' or 'end', got {endpoint}")


def split_surface_t_values(num_output_points, beta):
    if num_output_points < 3:
        raise ValueError(f'num_output_points must be at least 3, got {num_output_points}')
    upper_output_points = num_output_points // 2 + 1
    lower_output_points = num_output_points - upper_output_points + 1
    return (
        endpoint_dense_spacing(upper_output_points, 'end', beta),
        endpoint_dense_spacing(lower_output_points, 'start', beta),
    )
