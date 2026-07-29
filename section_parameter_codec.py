"""Reversible physical/model-space transforms for section sequences."""

from __future__ import annotations

import torch
from torch import nn

import grassdata
import util


SECTION_PARAMETER_CODEC_SCHEMA = 'grass_section_parameter_codec_v1'
_CST_EXPONENT_SLICE = slice(
    util.CST_AIRFOIL_CODE_SIZE - 2, util.CST_AIRFOIL_CODE_SIZE
)
_WING_CHORD_INDEX = util.CST_AIRFOIL_CODE_SIZE + 3
_FUSELAGE_SIZE_SLICE = slice(3, 5)


def _validate_physical_shape(sections, sequence_type):
    expected = (
        grassdata.sequence_max_sections(sequence_type),
        grassdata.sequence_section_size(sequence_type),
    )
    if sections.dim() != 3 or tuple(sections.shape[1:]) != expected:
        raise ValueError(
            f'{sequence_type} sections must have shape [B, {expected[0]}, {expected[1]}], '
            f'got {list(sections.shape)}'
        )
    if not torch.isfinite(sections).all():
        raise ValueError(f'{sequence_type} sections must contain only finite values.')


def _physical_to_transformed(sections, section_count, sequence_type):
    _validate_physical_shape(sections, sequence_type)
    mask = grassdata.section_mask(
        section_count, device=sections.device, sequence_type=sequence_type
    )
    transformed = sections.clone()
    dimension_mask = mask.unsqueeze(2).expand_as(sections).clone()

    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        valid = mask
        shifted_exponents = (
            sections[..., _CST_EXPONENT_SLICE]
            - util.CST_MIN_CLASS_FUNCTION_EXPONENT
        )
        if torch.any(shifted_exponents[valid] <= 0.0):
            raise ValueError(
                'Wing CST N1/N2 must exceed CST_MIN_CLASS_FUNCTION_EXPONENT.'
            )
        if torch.any(sections[..., _WING_CHORD_INDEX][valid] <= 0.0):
            raise ValueError('Wing chord must be positive.')
        safe_exponents = torch.where(
            valid.unsqueeze(2), shifted_exponents, torch.ones_like(shifted_exponents)
        )
        safe_chord = torch.where(
            valid,
            sections[..., _WING_CHORD_INDEX],
            torch.ones_like(sections[..., _WING_CHORD_INDEX]),
        )
        transformed[..., _CST_EXPONENT_SLICE] = torch.log(safe_exponents)
        transformed[..., _WING_CHORD_INDEX] = torch.log(safe_chord)
    elif sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        first_x = sections[:, 0, 0]
        if not torch.allclose(first_x, torch.zeros_like(first_x), atol=1e-6, rtol=0.0):
            raise ValueError('Fuselage first-section x must equal zero.')
        valid = mask
        if torch.any(sections[..., _FUSELAGE_SIZE_SLICE][valid] <= 0.0):
            raise ValueError('Fuselage width and height must be positive.')
        delta_x = sections[:, 1:, 0] - sections[:, :-1, 0]
        delta_valid = mask[:, 1:]
        if torch.any(delta_x[delta_valid] <= 0.0):
            raise ValueError('Fuselage valid x stations must be strictly increasing.')
        safe_delta = torch.where(delta_valid, delta_x, torch.ones_like(delta_x))
        transformed[..., 0] = 0.0
        transformed[:, 1:, 0] = torch.log(safe_delta)
        safe_size = torch.where(
            valid.unsqueeze(2),
            sections[..., _FUSELAGE_SIZE_SLICE],
            torch.ones_like(sections[..., _FUSELAGE_SIZE_SLICE]),
        )
        transformed[..., _FUSELAGE_SIZE_SLICE] = torch.log(safe_size)
        dimension_mask[:, 0, 0] = False
    else:
        raise ValueError(f'Unsupported sequence type: {sequence_type}')

    transformed = torch.where(dimension_mask, transformed, torch.zeros_like(transformed))
    return transformed, dimension_mask


def fit_section_parameter_statistics(dataset, sequence_type):
    """Fit per-dimension model-space statistics from training leaves only."""
    grassdata.sequence_spec(sequence_type)
    section_size = grassdata.sequence_section_size(sequence_type)
    sums = torch.zeros(section_size, dtype=torch.float64)
    squared_sums = torch.zeros(section_size, dtype=torch.float64)
    counts = torch.zeros(section_size, dtype=torch.float64)

    for sample in dataset:
        sections = torch.as_tensor(sample['sections'], dtype=torch.float64).unsqueeze(0)
        section_count = torch.as_tensor(sample['section_count'], dtype=torch.long).reshape(1)
        transformed, dimension_mask = _physical_to_transformed(
            sections, section_count, sequence_type
        )
        mask = dimension_mask.to(dtype=torch.float64)
        sums += (transformed * mask).sum(dim=(0, 1))
        squared_sums += (transformed.square() * mask).sum(dim=(0, 1))
        counts += mask.sum(dim=(0, 1))

    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            f'{sequence_type} normalization has no training values for dimensions {missing}.'
        )
    mean = sums / counts
    variance = torch.clamp(squared_sums / counts - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    constant_mask = std == 0.0
    std = torch.where(constant_mask, torch.ones_like(std), std)
    return {
        'schema': SECTION_PARAMETER_CODEC_SCHEMA,
        'sequence_type': sequence_type,
        'mean': mean.to(dtype=torch.float32),
        'std': std.to(dtype=torch.float32),
        'constant_mask': constant_mask,
    }


def validate_section_parameter_statistics(statistics, sequence_type):
    required = ('schema', 'sequence_type', 'mean', 'std', 'constant_mask')
    missing = [key for key in required if key not in statistics]
    if missing:
        raise KeyError(f'{sequence_type} parameter statistics are missing keys: {missing}')
    if statistics['schema'] != SECTION_PARAMETER_CODEC_SCHEMA:
        raise ValueError(
            f'{sequence_type} parameter statistics have schema={statistics["schema"]!r}; '
            f'expected {SECTION_PARAMETER_CODEC_SCHEMA!r}.'
        )
    if statistics['sequence_type'] != sequence_type:
        raise ValueError(
            f'Parameter statistics are for {statistics["sequence_type"]!r}, '
            f'expected {sequence_type!r}.'
        )
    section_size = grassdata.sequence_section_size(sequence_type)
    mean = torch.as_tensor(statistics['mean'])
    std = torch.as_tensor(statistics['std'])
    constant_mask = torch.as_tensor(statistics['constant_mask'])
    for name, value in (('mean', mean), ('std', std), ('constant_mask', constant_mask)):
        if tuple(value.shape) != (section_size,):
            raise ValueError(
                f'{sequence_type} parameter-statistics {name} must have shape '
                f'[{section_size}], got {list(value.shape)}.'
            )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f'{sequence_type} parameter statistics must be finite.')
    if torch.any(std <= 0.0):
        raise ValueError(f'{sequence_type} parameter-statistics std must be positive.')


class SectionParameterCodec(nn.Module):
    """Map physical padded sections to standardized model space and back."""

    def __init__(self, sequence_type, statistics):
        super().__init__()
        validate_section_parameter_statistics(statistics, sequence_type)
        self.sequence_type = sequence_type
        self.section_size = grassdata.sequence_section_size(sequence_type)
        self.max_sections = grassdata.sequence_max_sections(sequence_type)
        self.register_buffer(
            'mean', torch.as_tensor(statistics['mean'], dtype=torch.float32).clone()
        )
        self.register_buffer(
            'std', torch.as_tensor(statistics['std'], dtype=torch.float32).clone()
        )
        self.register_buffer(
            'constant_mask',
            torch.as_tensor(statistics['constant_mask'], dtype=torch.bool).clone(),
        )

    def export_statistics(self):
        return {
            'schema': SECTION_PARAMETER_CODEC_SCHEMA,
            'sequence_type': self.sequence_type,
            'mean': self.mean.detach().cpu().clone(),
            'std': self.std.detach().cpu().clone(),
            'constant_mask': self.constant_mask.detach().cpu().clone(),
        }

    def normalize(self, physical_sections, section_count):
        transformed, dimension_mask = _physical_to_transformed(
            physical_sections, section_count, self.sequence_type
        )
        normalized = (transformed - self.mean) / self.std
        normalized = torch.where(dimension_mask, normalized, torch.zeros_like(normalized))
        if self.sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
            normalized[:, 0, 0] = 0.0
        return normalized

    def denormalize(self, model_sections):
        expected = (self.max_sections, self.section_size)
        if model_sections.dim() != 3 or tuple(model_sections.shape[1:]) != expected:
            raise ValueError(
                f'{self.sequence_type} model sections must have shape '
                f'[B, {expected[0]}, {expected[1]}], got {list(model_sections.shape)}'
            )
        transformed = model_sections * self.std + self.mean
        physical = transformed.clone()
        if self.sequence_type == grassdata.SEQUENCE_TYPE_WING:
            physical[..., _CST_EXPONENT_SLICE] = (
                torch.exp(transformed[..., _CST_EXPONENT_SLICE])
                + util.CST_MIN_CLASS_FUNCTION_EXPONENT
            )
            physical[..., _WING_CHORD_INDEX] = torch.exp(
                transformed[..., _WING_CHORD_INDEX]
            )
        elif self.sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
            delta_x = torch.cat([
                torch.zeros_like(transformed[:, :1, 0]),
                torch.exp(transformed[:, 1:, 0]),
            ], dim=1)
            physical[..., 0] = torch.cumsum(delta_x, dim=1)
            physical[..., _FUSELAGE_SIZE_SLICE] = torch.exp(
                transformed[..., _FUSELAGE_SIZE_SLICE]
            )
        else:
            raise ValueError(f'Unsupported sequence type: {self.sequence_type}')
        return physical
