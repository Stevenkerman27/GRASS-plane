"""Reversible physical/model-space transforms for fixed component payloads."""

from __future__ import annotations

import torch
from torch import nn

import grassdata
import util


SECTION_PARAMETER_CODEC_SCHEMA = 'grass_component_parameter_codec_v1'


def _spec(sequence_type):
    return grassdata.sequence_spec(sequence_type)


def _global_log_indices(sequence_type):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        return (3, 4)
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        return (3, 4, 5)
    raise ValueError(f'Unsupported sequence type: {sequence_type}')


def _section_log_indices(sequence_type):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        return (1, util.WING_SECTION_SIZE - 2, util.WING_SECTION_SIZE - 1)
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        return (1, 2)
    raise ValueError(f'Unsupported sequence type: {sequence_type}')


def _learned_section_mask(sequence_type):
    spec = _spec(sequence_type)
    mask = torch.ones(
        spec['section_count'], spec['section_size'], dtype=torch.bool
    )
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        mask[:, 0] = False
        mask[0, 1] = False
        mask[0, 3:5] = False
    elif sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        mask[:, 0] = False
        mask[0, 1:3] = False
        mask[-1, 1:3] = False
    else:
        raise ValueError(f'Unsupported sequence type: {sequence_type}')
    return mask


def _canonical_sections(sequence_type, batch_size, dtype, device):
    spec = _spec(sequence_type)
    sections = torch.zeros(
        batch_size, spec['section_count'], spec['section_size'], dtype=dtype, device=device
    )
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        sections[..., 0] = torch.linspace(0.0, 1.0, spec['section_count'], dtype=dtype, device=device)
        sections[..., 1] = 1.0
        sections[..., -2:] = util.CST_MIN_CLASS_FUNCTION_EXPONENT + 1.0
    elif sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        sections[..., 0] = torch.linspace(0.0, 1.0, spec['section_count'], dtype=dtype, device=device)
        sections[..., 1:3] = 1.0
        sections[:, (0, -1), 1:3] = 0.1
    else:
        raise ValueError(f'Unsupported sequence type: {sequence_type}')
    return sections


def _validate_global(global_parameters, sequence_type):
    expected_size = _spec(sequence_type)['global_size']
    if global_parameters.dim() != 2 or global_parameters.size(1) != expected_size:
        raise ValueError(
            f'{sequence_type} global parameters must have shape [B, {expected_size}], '
            f'got {list(global_parameters.shape)}.'
        )
    if not torch.isfinite(global_parameters).all():
        raise ValueError(f'{sequence_type} global parameters must contain only finite values.')
    indices = _global_log_indices(sequence_type)
    if torch.any(global_parameters[:, indices] <= 0.0):
        raise ValueError(f'{sequence_type} positive global parameters must be strictly positive.')


def _validate_sections(sections, sequence_type):
    spec = _spec(sequence_type)
    expected = (spec['section_count'], spec['section_size'])
    if sections.dim() != 3 or tuple(sections.shape[1:]) != expected:
        raise ValueError(
            f'{sequence_type} sections must have shape [B, {expected[0]}, {expected[1]}], '
            f'got {list(sections.shape)}.'
        )
    if not torch.isfinite(sections).all():
        raise ValueError(f'{sequence_type} sections must contain only finite values.')
    indices = _section_log_indices(sequence_type)
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        if torch.any(sections[..., 1] <= 0.0):
            raise ValueError('Wing chord fractions must be strictly positive.')
        if torch.any(sections[..., -2:] <= util.CST_MIN_CLASS_FUNCTION_EXPONENT):
            raise ValueError('Wing CST N1/N2 must exceed CST_MIN_CLASS_FUNCTION_EXPONENT.')
    elif sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        if torch.any(sections[..., indices] <= 0.0):
            raise ValueError('Fuselage width and height fractions must be strictly positive.')


def _transform_global(global_parameters, sequence_type):
    _validate_global(global_parameters, sequence_type)
    transformed = global_parameters.clone()
    transformed[:, _global_log_indices(sequence_type)] = torch.log(
        global_parameters[:, _global_log_indices(sequence_type)]
    )
    return transformed


def _transform_sections(sections, sequence_type):
    _validate_sections(sections, sequence_type)
    transformed = sections.clone()
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        transformed[..., 1] = torch.log(sections[..., 1])
        transformed[..., -2:] = torch.log(
            sections[..., -2:] - util.CST_MIN_CLASS_FUNCTION_EXPONENT
        )
    elif sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        transformed[..., 1:3] = torch.log(sections[..., 1:3])
    else:
        raise ValueError(f'Unsupported sequence type: {sequence_type}')
    return transformed


def _moments(values, dimensions, sequence_type, name):
    sums = torch.zeros(dimensions, dtype=torch.float64)
    squared_sums = torch.zeros(dimensions, dtype=torch.float64)
    count = 0
    for value in values:
        value = value.to(dtype=torch.float64)
        sums += value.sum(dim=0)
        squared_sums += value.square().sum(dim=0)
        count += value.size(0)
    if count == 0:
        raise ValueError(f'No {sequence_type} training values were supplied for {name}.')
    mean = sums / count
    variance = torch.clamp(squared_sums / count - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    constant_mask = std == 0.0
    std = torch.where(constant_mask, torch.ones_like(std), std)
    return mean.to(torch.float32), std.to(torch.float32), constant_mask


def fit_section_parameter_statistics(dataset, sequence_type):
    """Fit component-level statistics from fixed-length training leaves only."""
    spec = _spec(sequence_type)
    global_values = []
    section_values = []
    for sample in dataset:
        global_parameters = torch.as_tensor(sample['z_global'], dtype=torch.float32).reshape(1, -1)
        sections = torch.as_tensor(sample['sections'], dtype=torch.float32).unsqueeze(0)
        global_values.append(_transform_global(global_parameters, sequence_type))
        section_values.append(
            _transform_sections(sections, sequence_type).reshape(-1, spec['section_size'])
        )
    global_mean, global_std, global_constant_mask = _moments(
        global_values, spec['global_size'], sequence_type, 'global parameters'
    )
    section_mean, section_std, section_constant_mask = _moments(
        section_values, spec['section_size'], sequence_type, 'section parameters'
    )
    return {
        'schema': SECTION_PARAMETER_CODEC_SCHEMA,
        'sequence_type': sequence_type,
        'global_mean': global_mean,
        'global_std': global_std,
        'global_constant_mask': global_constant_mask,
        'section_mean': section_mean,
        'section_std': section_std,
        'section_constant_mask': section_constant_mask,
    }


def validate_section_parameter_statistics(statistics, sequence_type):
    spec = _spec(sequence_type)
    required = (
        'schema', 'sequence_type', 'global_mean', 'global_std', 'global_constant_mask',
        'section_mean', 'section_std', 'section_constant_mask',
    )
    missing = [key for key in required if key not in statistics]
    if missing:
        raise KeyError(f'{sequence_type} component statistics are missing keys: {missing}')
    if statistics['schema'] != SECTION_PARAMETER_CODEC_SCHEMA:
        raise ValueError(
            f'{sequence_type} component statistics have schema={statistics["schema"]!r}; '
            f'expected {SECTION_PARAMETER_CODEC_SCHEMA!r}.'
        )
    if statistics['sequence_type'] != sequence_type:
        raise ValueError(
            f'Component statistics are for {statistics["sequence_type"]!r}, '
            f'expected {sequence_type!r}.'
        )
    for prefix, size in (('global', spec['global_size']), ('section', spec['section_size'])):
        mean = torch.as_tensor(statistics[f'{prefix}_mean'])
        std = torch.as_tensor(statistics[f'{prefix}_std'])
        constant_mask = torch.as_tensor(statistics[f'{prefix}_constant_mask'])
        for name, value in ((f'{prefix}_mean', mean), (f'{prefix}_std', std),
                            (f'{prefix}_constant_mask', constant_mask)):
            if tuple(value.shape) != (size,):
                raise ValueError(
                    f'{sequence_type} {name} must have shape [{size}], got {list(value.shape)}.'
                )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(f'{sequence_type} {prefix} statistics must be finite.')
        if torch.any(std <= 0.0):
            raise ValueError(f'{sequence_type} {prefix} standard deviations must be positive.')


class SectionParameterCodec(nn.Module):
    """Map one fixed component payload to normalized model space and back."""

    def __init__(self, sequence_type, statistics):
        super().__init__()
        validate_section_parameter_statistics(statistics, sequence_type)
        spec = _spec(sequence_type)
        self.sequence_type = sequence_type
        self.global_size = spec['global_size']
        self.section_size = spec['section_size']
        self.section_count = spec['section_count']
        self.register_buffer('global_mean', torch.as_tensor(statistics['global_mean'], dtype=torch.float32).clone())
        self.register_buffer('global_std', torch.as_tensor(statistics['global_std'], dtype=torch.float32).clone())
        self.register_buffer('global_constant_mask', torch.as_tensor(statistics['global_constant_mask'], dtype=torch.bool).clone())
        self.register_buffer('section_mean', torch.as_tensor(statistics['section_mean'], dtype=torch.float32).clone())
        self.register_buffer('section_std', torch.as_tensor(statistics['section_std'], dtype=torch.float32).clone())
        self.register_buffer('section_constant_mask', torch.as_tensor(statistics['section_constant_mask'], dtype=torch.bool).clone())
        self.register_buffer('learned_section_mask', _learned_section_mask(sequence_type))

    @property
    def decoder_output_size(self):
        return self.global_size + int(self.learned_section_mask.sum().item())

    def export_statistics(self):
        return {
            'schema': SECTION_PARAMETER_CODEC_SCHEMA,
            'sequence_type': self.sequence_type,
            'global_mean': self.global_mean.detach().cpu().clone(),
            'global_std': self.global_std.detach().cpu().clone(),
            'global_constant_mask': self.global_constant_mask.detach().cpu().clone(),
            'section_mean': self.section_mean.detach().cpu().clone(),
            'section_std': self.section_std.detach().cpu().clone(),
            'section_constant_mask': self.section_constant_mask.detach().cpu().clone(),
        }

    def normalize(self, global_parameters, sections):
        global_model = (_transform_global(global_parameters, self.sequence_type) - self.global_mean) / self.global_std
        section_model = (_transform_sections(sections, self.sequence_type) - self.section_mean) / self.section_std
        return global_model, section_model

    def denormalize(self, global_model, section_model):
        if global_model.dim() != 2 or global_model.size(1) != self.global_size:
            raise ValueError(f'global model parameters must have shape [B, {self.global_size}].')
        expected = (self.section_count, self.section_size)
        if section_model.dim() != 3 or tuple(section_model.shape[1:]) != expected:
            raise ValueError(f'section model parameters must have shape [B, {expected[0]}, {expected[1]}].')
        global_transformed = global_model * self.global_std + self.global_mean
        section_transformed = section_model * self.section_std + self.section_mean
        global_parameters = global_transformed.clone()
        global_parameters[:, _global_log_indices(self.sequence_type)] = torch.exp(
            global_transformed[:, _global_log_indices(self.sequence_type)]
        )
        sections = section_transformed.clone()
        if self.sequence_type == grassdata.SEQUENCE_TYPE_WING:
            sections[..., 1] = torch.exp(section_transformed[..., 1])
            sections[..., -2:] = torch.exp(section_transformed[..., -2:]) + util.CST_MIN_CLASS_FUNCTION_EXPONENT
        elif self.sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
            sections[..., 1:3] = torch.exp(section_transformed[..., 1:3])
        else:
            raise ValueError(f'Unsupported sequence type: {self.sequence_type}')
        return global_parameters, sections

    def target_vector(self, global_parameters, sections):
        global_model, section_model = self.normalize(global_parameters, sections)
        return torch.cat(
            [global_model, section_model[:, self.learned_section_mask]], dim=1
        )

    def decode_vector(self, vector):
        if vector.dim() != 2 or vector.size(1) != self.decoder_output_size:
            raise ValueError(
                f'decoder vector must have shape [B, {self.decoder_output_size}], '
                f'got {list(vector.shape)}.'
            )
        global_model = vector[:, :self.global_size]
        canonical = _canonical_sections(
            self.sequence_type, vector.size(0), vector.dtype, vector.device
        )
        _, section_model = self.normalize(
            torch.ones(
                vector.size(0), self.global_size, dtype=vector.dtype, device=vector.device
            ),
            canonical,
        )
        section_model[:, self.learned_section_mask] = vector[:, self.global_size:]
        global_parameters, sections = self.denormalize(global_model, section_model)
        canonical = _canonical_sections(
            self.sequence_type, vector.size(0), vector.dtype, vector.device
        )
        sections[:, ~self.learned_section_mask] = canonical[:, ~self.learned_section_mask]
        return global_parameters, sections
