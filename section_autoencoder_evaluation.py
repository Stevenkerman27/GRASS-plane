"""Deterministic diagnostics for fixed-section autoencoder reconstructions."""

from __future__ import annotations

import numpy as np
import torch

import section_autoencoder


def distribution_summary(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError('Cannot summarize an empty value collection.')
    return {
        'mean': float(array.mean()),
        'median': float(np.median(array)),
        'max': float(array.max()),
    }


def evaluate_loader(model, loader, device, mode='deterministic'):
    """Evaluate direct reconstructions; fixed payloads have no decode mode distinction."""
    if mode not in ('deterministic', 'teacher_forced', 'free_running'):
        raise ValueError(f'Unsupported evaluation mode: {mode}')
    loss_values = None
    global_mae = []
    section_mae = []
    rows = []
    model.eval()
    with torch.no_grad():
        leaf_index = 0
        for batch in loader:
            target_global = batch['z_global'].to(device)
            target_sections = batch['sections'].to(device)
            predicted_global, predicted_sections = model(target_global, target_sections)
            losses = section_autoencoder.reconstruction_losses(
                model, predicted_global, predicted_sections, target_global, target_sections
            )
            if loss_values is None:
                loss_values = {name: [] for name in losses}
            for name, value in losses.items():
                loss_values[name].extend(value.detach().cpu().tolist())
            per_global_mae = torch.abs(predicted_global - target_global).mean(dim=1)
            per_section_mae = torch.abs(predicted_sections - target_sections).mean(dim=(1, 2))
            global_mae.extend(per_global_mae.detach().cpu().tolist())
            section_mae.extend(per_section_mae.detach().cpu().tolist())
            for index in range(target_global.size(0)):
                row = {
                    'leaf_index': leaf_index,
                    'global_mae': float(per_global_mae[index].item()),
                    'section_mae': float(per_section_mae[index].item()),
                }
                row.update({name: float(value[index].item()) for name, value in losses.items()})
                rows.append(row)
                leaf_index += 1
    if loss_values is None:
        raise RuntimeError('Evaluation dataloader produced no batches.')
    return {
        'sequence_type': model.sequence_type,
        'mode': 'deterministic',
        'leaf_count': len(rows),
        'normalized_training_loss': {
            name: distribution_summary(values) for name, values in loss_values.items()
        },
        'physical_parameter_error': {
            'global_mae': distribution_summary(global_mae),
            'section_mae': distribution_summary(section_mae),
        },
        'sample_rows': rows,
    }
