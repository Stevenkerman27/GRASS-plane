"""Reusable leaf-section autoencoder components and checkpoint contract."""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

import grassdata
import grassmodel
import section_parameter_codec


SECTION_AUTOENCODER_CHECKPOINT_SCHEMA = 'grass_section_autoencoder_v2'


def final_checkpoint_path(checkpoint_dir, sequence_type):
    """Return the sole section-AE checkpoint selected after training completes."""
    grassdata.sequence_spec(sequence_type)
    return Path(checkpoint_dir) / f'last_{sequence_type}.pt'


class SectionLeafDataset(Dataset):
    """Leaf sequences extracted after the aircraft-level train/validation split."""

    def __init__(self, leaves, sequence_type):
        if not leaves:
            raise ValueError(f'No {sequence_type} leaves were extracted.')
        self.sequence_type = sequence_type
        self.leaves = leaves

    def __getitem__(self, index):
        box = self.leaves[index]
        return {
            'sections': box['sections'].squeeze(0),
            'section_count': box['section_count'].reshape(()),
        }

    def __len__(self):
        return len(self.leaves)


def extract_section_leaves(trees, sequence_type):
    grassdata.sequence_spec(sequence_type)
    leaves = []

    def visit(node):
        if node.is_leaf():
            if isinstance(node.box, dict) and node.box['sequence_type'] == sequence_type:
                leaves.append(node.box)
            return
        if node.is_adj():
            visit(node.left)
            visit(node.right)
            return
        if node.is_sym():
            visit(node.left)
            return
        raise RuntimeError('Tree contains an unsupported node type.')

    for tree in trees:
        visit(tree.root)
    return SectionLeafDataset(leaves, sequence_type)


class SectionAutoencoder(nn.Module):
    def __init__(self, sequence_type, config, parameter_statistics):
        super().__init__()
        grassdata.sequence_spec(sequence_type)
        self.sequence_type = sequence_type
        self.section_encoder = grassmodel.SectionEncoder(
            sequence_type,
            config.feature_size,
            config.ae_rnn_type,
            parameter_statistics,
        )
        self.section_decoder = grassmodel.AutoregressiveSectionDecoder(
            sequence_type,
            config.feature_size,
            config.ae_rnn_type,
            parameter_statistics,
        )

    def forward(self, sections, section_count, teacher_forcing_probability):
        feature = self.section_encoder(sections, section_count)
        return self.section_decoder(
            feature, sections, section_count, teacher_forcing_probability
        )


def reconstruction_losses(model, sections, count_logits, target_sections, target_count):
    sequence_type = model.sequence_type
    parameter_codec = model.section_decoder.parameter_codec
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        return grassmodel.wing_section_reconstruction_losses(
            parameter_codec, sections, count_logits, target_sections, target_count
        )
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        return grassmodel.fuselage_section_reconstruction_losses(
            parameter_codec, sections, count_logits, target_sections, target_count
        )
    raise ValueError(f'Unsupported sequence type: {sequence_type}')


def build_checkpoint(model, epoch, optimizer, scheduler, validation_metrics, config):
    parameter_statistics = model.section_encoder.parameter_codec.export_statistics()
    return {
        'schema': SECTION_AUTOENCODER_CHECKPOINT_SCHEMA,
        'sequence_type': model.sequence_type,
        'section_size': grassdata.sequence_section_size(model.sequence_type),
        'epoch': epoch,
        'section_encoder_state_dict': model.section_encoder.state_dict(),
        'section_decoder_state_dict': model.section_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'validation_metrics': validation_metrics,
        'feature_size': config.feature_size,
        'hidden_size': config.hidden_size,
        'ae_rnn_type': config.ae_rnn_type,
        'parameter_statistics': parameter_statistics,
        'teacher_forcing_schedule': {
            'p_final': config.ae_teacher_forcing_p_final,
            'ramp_start_epoch': config.ae_teacher_forcing_ramp_start_epoch,
            'ramp_end_epoch': config.ae_teacher_forcing_ramp_end_epoch,
        },
    }


def _statistics_match(first, second, sequence_type):
    section_parameter_codec.validate_section_parameter_statistics(first, sequence_type)
    section_parameter_codec.validate_section_parameter_statistics(second, sequence_type)
    return (
        torch.equal(torch.as_tensor(first['constant_mask']), torch.as_tensor(second['constant_mask']))
        and torch.allclose(
            torch.as_tensor(first['mean']), torch.as_tensor(second['mean']), atol=1e-7, rtol=0.0
        )
        and torch.allclose(
            torch.as_tensor(first['std']), torch.as_tensor(second['std']), atol=1e-7, rtol=0.0
        )
    )


def _validate_checkpoint(
        checkpoint, source, sequence_type, config, expected_statistics=None):
    expected = {
        'schema': SECTION_AUTOENCODER_CHECKPOINT_SCHEMA,
        'sequence_type': sequence_type,
        'section_size': grassdata.sequence_section_size(sequence_type),
        'feature_size': config.feature_size,
        'hidden_size': config.hidden_size,
        'ae_rnn_type': config.ae_rnn_type,
    }
    for key, expected_value in expected.items():
        actual_value = checkpoint[key]
        if actual_value != expected_value:
            raise ValueError(
                f'{source} has {key}={actual_value!r}; expected {expected_value!r}.'
            )
    if 'parameter_statistics' not in checkpoint:
        raise KeyError(f'{source} is missing parameter_statistics.')
    section_parameter_codec.validate_section_parameter_statistics(
        checkpoint['parameter_statistics'], sequence_type
    )
    if expected_statistics is not None and not _statistics_match(
            checkpoint['parameter_statistics'], expected_statistics, sequence_type):
        raise ValueError(
            f'{source} parameter statistics do not match the full-AE training split.'
        )


def apply_pretrained_section_autoencoders(full_encoder, full_decoder, checkpoints, config):
    targets = {
        grassdata.SEQUENCE_TYPE_WING: (
            full_encoder.wing_section_encoder,
            full_decoder.wing_section_decoder,
        ),
        grassdata.SEQUENCE_TYPE_FUSELAGE: (
            full_encoder.fuselage_section_encoder,
            full_decoder.fuselage_section_decoder,
        ),
    }
    for sequence_type, (encoder_target, decoder_target) in targets.items():
        checkpoint = checkpoints[sequence_type]
        expected_statistics = encoder_target.parameter_codec.export_statistics()
        if not _statistics_match(
                expected_statistics,
                decoder_target.parameter_codec.export_statistics(),
                sequence_type):
            raise ValueError(
                f'Full-AE {sequence_type} encoder and decoder parameter statistics differ.'
            )
        _validate_checkpoint(
            checkpoint,
            f'{sequence_type} section-AE checkpoint',
            sequence_type,
            config,
            expected_statistics,
        )
        encoder_target.load_state_dict(checkpoint['section_encoder_state_dict'], strict=True)
        decoder_target.load_state_dict(checkpoint['section_decoder_state_dict'], strict=True)


def load_pretrained_section_autoencoders(full_encoder, full_decoder, checkpoint_dir, config, device):
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f'Section-AE checkpoint directory does not exist: {directory}')

    checkpoints = {}
    for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE):
        path = final_checkpoint_path(directory, sequence_type)
        if not path.is_file():
            raise FileNotFoundError(f'Missing required {sequence_type} section-AE checkpoint: {path}')
        checkpoints[sequence_type] = torch.load(path, map_location=device, weights_only=True)
    apply_pretrained_section_autoencoders(full_encoder, full_decoder, checkpoints, config)


def load_section_autoencoder(checkpoint_path, sequence_type, device):
    """Load one independently trained leaf-section autoencoder for evaluation."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Section-AE checkpoint does not exist: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required_keys = ('feature_size', 'hidden_size', 'ae_rnn_type', 'parameter_statistics')
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f'{path} is missing required checkpoint keys: {missing}')
    config = SimpleNamespace(
        feature_size=checkpoint['feature_size'],
        hidden_size=checkpoint['hidden_size'],
        ae_rnn_type=checkpoint['ae_rnn_type'],
    )
    _validate_checkpoint(checkpoint, str(path), sequence_type, config)
    model = SectionAutoencoder(
        sequence_type, config, checkpoint['parameter_statistics']
    ).to(device)
    model.section_encoder.load_state_dict(checkpoint['section_encoder_state_dict'], strict=True)
    model.section_decoder.load_state_dict(checkpoint['section_decoder_state_dict'], strict=True)
    model.eval()
    return model, checkpoint
