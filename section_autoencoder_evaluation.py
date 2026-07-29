"""Metrics for diagnosing independent recurrent section autoencoders."""

from __future__ import annotations

import math

import numpy as np
import torch

import grassdata
import section_autoencoder
import util
from data import visualize_free_decode_error as geometry


_EPSILON = 1e-8


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu().item())


def distribution_summary(values):
    """Summarize one non-empty collection of scalar diagnostic values."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError('Cannot summarize an empty diagnostic value collection.')
    if not np.isfinite(array).all():
        raise ValueError('Diagnostic values must be finite.')
    return {
        'mean': float(np.mean(array)),
        'median': float(np.median(array)),
        'p90': float(np.percentile(array, 90)),
        'max': float(np.max(array)),
    }


def _field_spec(sequence_type):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        cst_size = util.CST_AIRFOIL_CODE_SIZE
        return (
            ('cst_shape_coefficients', slice(0, cst_size - 4), False),
            ('cst_trailing_edge_y', slice(cst_size - 4, cst_size - 2), False),
            ('cst_n1_n2', slice(cst_size - 2, cst_size), False),
            ('leading_edge_xyz_m', slice(cst_size, cst_size + 3), False),
            ('chord_m', slice(cst_size + 3, cst_size + 4), True),
            ('twist_rad', slice(cst_size + 4, cst_size + 5), False),
        )
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        return (
            ('station_x_m', slice(0, 1), False),
            ('center_yz_m', slice(1, 3), False),
            ('width_m', slice(3, 4), True),
            ('height_m', slice(4, 5), True),
        )
    raise ValueError(f'Unsupported sequence type: {sequence_type}')


def _component(sequence_type, sections, section_count):
    transform, translation = geometry.identity_transform()
    return geometry.ComponentGeometry(
        component=grassdata.sequence_spec(sequence_type)['component'],
        sections=sections.detach().cpu().numpy(),
        section_count=int(section_count),
        transform=transform,
        translation=translation,
        path=sequence_type,
    )


def _geometry_metrics(sequence_type, predicted_sections, target_sections, target_count):
    """Compare matching true section indices; output per-sample averages."""
    target = _component(sequence_type, target_sections, target_count)
    predicted = _component(sequence_type, predicted_sections, target_count)
    target_points = geometry.component_points(target)
    scale = float(np.max(target_points.max(axis=0) - target_points.min(axis=0)))
    if scale < _EPSILON:
        raise ValueError(f'{sequence_type} target geometry has zero extent.')
    rms_m = []
    normalized_rms = []
    for section_index in range(target_count):
        target_points = geometry.section_points(target, target.sections[section_index])
        predicted_points = geometry.section_points(predicted, predicted.sections[section_index])
        if target_points.shape != predicted_points.shape:
            raise RuntimeError('Target and predicted section point clouds have incompatible shapes.')
        rms = float(np.sqrt(np.mean(np.sum((predicted_points - target_points) ** 2, axis=1))))
        rms_m.append(rms)
        normalized_rms.append(rms / scale)
    return float(np.mean(rms_m)), float(np.mean(normalized_rms))


def _fuselage_length(sections, section_count):
    counts = section_count.reshape(-1).to(dtype=torch.long, device=sections.device)
    batch_index = torch.arange(sections.size(0), device=sections.device)
    return sections[batch_index, counts - 1, 0] - sections[:, 0, 0]


def _append_parameter_errors(state, sequence_type, predicted_sections, target_sections, target_count):
    valid_mask = grassdata.section_mask(
        target_count, device=predicted_sections.device, sequence_type=sequence_type
    )
    for field_name, field_slice, relative in _field_spec(sequence_type):
        field_mask = valid_mask.unsqueeze(2).expand(
            -1, -1, field_slice.stop - field_slice.start
        )
        predicted = predicted_sections[..., field_slice]
        target = target_sections[..., field_slice]
        state['parameter_values'][field_name].extend(
            torch.abs(predicted - target)[field_mask].detach().cpu().tolist()
        )
        state['parameter_squared_values'][field_name].extend(
            torch.square(predicted - target)[field_mask].detach().cpu().tolist()
        )
        if relative:
            relative_values = (
                100.0 * torch.abs(predicted - target) / torch.abs(target).clamp_min(_EPSILON)
            )[field_mask]
            state['parameter_relative_percent'][field_name].extend(
                relative_values.detach().cpu().tolist()
            )
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        predicted_length = _fuselage_length(predicted_sections, target_count)
        target_length = _fuselage_length(target_sections, target_count)
        state['parameter_values']['length_m'].extend(
            torch.abs(predicted_length - target_length).detach().cpu().tolist()
        )
        state['parameter_squared_values']['length_m'].extend(
            torch.square(predicted_length - target_length).detach().cpu().tolist()
        )
        state['parameter_relative_percent']['length_m'].extend(
            (
                100.0 * torch.abs(predicted_length - target_length)
                / torch.abs(target_length).clamp_min(_EPSILON)
            ).detach().cpu().tolist()
        )


def _sample_parameter_errors(sequence_type, predicted_sections, target_sections, target_count):
    """Return per-leaf physical MAE fields for CSV inspection."""
    count = int(target_count.item())
    result = {}
    for field_name, field_slice, relative in _field_spec(sequence_type):
        absolute = torch.abs(
            predicted_sections[:count, field_slice] - target_sections[:count, field_slice]
        )
        result[f'{field_name}_mae'] = _as_float(absolute.mean())
        if relative:
            relative_percent = 100.0 * absolute / torch.abs(
                target_sections[:count, field_slice]
            ).clamp_min(_EPSILON)
            result[f'{field_name}_relative_percent_mean'] = _as_float(relative_percent.mean())
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        predicted_length = _fuselage_length(
            predicted_sections.unsqueeze(0), target_count.reshape(1)
        )
        target_length = _fuselage_length(
            target_sections.unsqueeze(0), target_count.reshape(1)
        )
        absolute = torch.abs(predicted_length - target_length)
        result['length_m_mae'] = _as_float(absolute)
        result['length_m_relative_percent_mean'] = _as_float(
            100.0 * absolute / torch.abs(target_length).clamp_min(_EPSILON)
        )
    return result


def _new_state(sequence_type):
    field_names = [name for name, _field_slice, _relative in _field_spec(sequence_type)]
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        field_names.append('length_m')
    return {
        'loss_values': None,
        'parameter_values': {name: [] for name in field_names},
        'parameter_squared_values': {name: [] for name in field_names},
        'parameter_relative_percent': {name: [] for name in field_names},
        'geometry_rms_m': [],
        'geometry_rms_normalized': [],
        'target_counts': [],
        'predicted_counts': [],
        'rows': [],
    }


def _loss_summaries(loss_values):
    return {
        name: distribution_summary(values)
        for name, values in loss_values.items()
    }


def _parameter_summaries(state):
    result = {}
    for name, values in state['parameter_values'].items():
        summary = distribution_summary(values)
        summary['rmse'] = float(math.sqrt(np.mean(state['parameter_squared_values'][name])))
        if state['parameter_relative_percent'][name]:
            summary['relative_percent'] = distribution_summary(
                state['parameter_relative_percent'][name]
            )
        result[name] = summary
    return result


def _count_summary(target_counts, predicted_counts, sequence_type):
    targets = np.asarray(target_counts, dtype=int)
    predicted = np.asarray(predicted_counts, dtype=int)
    if targets.size == 0:
        raise ValueError('Count metrics require at least one evaluated leaf.')
    minimum, maximum = grassdata.sequence_section_count_range(sequence_type)
    labels = list(range(minimum, maximum + 1))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for target, generated in zip(targets, predicted, strict=True):
        matrix[target - minimum, generated - minimum] += 1
    return {
        'accuracy': float(np.mean(targets == predicted)),
        'mean_absolute_error': float(np.mean(np.abs(targets - predicted))),
        'labels': labels,
        'confusion_matrix': matrix.tolist(),
    }


def _run_model(model, sections, target_count, mode):
    feature = model.section_encoder(sections, target_count)
    if mode == 'teacher_forced':
        probability = torch.ones(sections.size(0), dtype=sections.dtype, device=sections.device)
        predicted_sections, count_logits = model.section_decoder(
            feature, sections, target_count, probability
        )
        predicted_count = torch.argmax(count_logits, dim=1) + model.section_decoder.minimum_sections
        return predicted_sections, count_logits, predicted_count
    if mode == 'free_running':
        predicted_sections, predicted_count, _mask, count_logits = model.section_decoder.generate(feature)
        return predicted_sections, count_logits, predicted_count
    raise ValueError(f'Unsupported evaluation mode: {mode}')


def evaluate_loader(model, loader, device, mode):
    """Evaluate a leaf dataloader and return JSON/CSV-ready diagnostic metrics."""
    sequence_type = model.sequence_type
    state = _new_state(sequence_type)
    model.eval()
    with torch.no_grad():
        leaf_index = 0
        for batch in loader:
            target_sections = batch['sections'].to(device)
            target_count = batch['section_count'].to(device)
            predicted_sections, count_logits, predicted_count = _run_model(
                model, target_sections, target_count, mode
            )
            losses = section_autoencoder.reconstruction_losses(
                model, predicted_sections, count_logits, target_sections, target_count
            )
            if state['loss_values'] is None:
                state['loss_values'] = {name: [] for name in losses}
            for name, value in losses.items():
                state['loss_values'][name].extend(value.detach().cpu().tolist())

            _append_parameter_errors(
                state, sequence_type, predicted_sections, target_sections, target_count
            )
            for batch_index in range(target_sections.size(0)):
                count = int(target_count[batch_index].item())
                geometry_rms_m, geometry_rms_normalized = _geometry_metrics(
                    sequence_type,
                    predicted_sections[batch_index],
                    target_sections[batch_index],
                    count,
                )
                state['geometry_rms_m'].append(geometry_rms_m)
                state['geometry_rms_normalized'].append(geometry_rms_normalized)
                target = int(target_count[batch_index].item())
                generated = int(predicted_count[batch_index].item())
                row = {
                    'leaf_index': leaf_index,
                    'target_section_count': target,
                    'predicted_section_count': generated,
                    'section_count_absolute_error': abs(target - generated),
                    'geometry_rms_m': geometry_rms_m,
                    'geometry_rms_normalized': geometry_rms_normalized,
                }
                row.update({name: _as_float(value[batch_index]) for name, value in losses.items()})
                row.update(_sample_parameter_errors(
                    sequence_type,
                    predicted_sections[batch_index],
                    target_sections[batch_index],
                    target_count[batch_index],
                ))
                state['rows'].append(row)
                state['target_counts'].append(target)
                state['predicted_counts'].append(generated)
                leaf_index += 1

    if state['loss_values'] is None:
        raise RuntimeError('Evaluation dataloader produced no batches.')
    return {
        'sequence_type': sequence_type,
        'mode': mode,
        'leaf_count': len(state['rows']),
        'normalized_training_loss': _loss_summaries(state['loss_values']),
        'physical_parameter_error': _parameter_summaries(state),
        'geometry_error': {
            'section_mean_rms_m': distribution_summary(state['geometry_rms_m']),
            'section_mean_rms_normalized': distribution_summary(
                state['geometry_rms_normalized']
            ),
        },
        'section_count': _count_summary(
            state['target_counts'], state['predicted_counts'], sequence_type
        ),
        'sample_rows': state['rows'],
    }
