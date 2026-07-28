import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grassdata
from grassmodel import GRASSDecoder, GRASSEncoder
import section_autoencoder


def model_config():
    return SimpleNamespace(
        box_code_size=13,
        feature_size=12,
        hidden_size=16,
        symmetry_size=8,
        ae_rnn_type='rnn',
        ae_teacher_forcing_p_final=0.1,
        ae_teacher_forcing_ramp_start_epoch=2,
        ae_teacher_forcing_ramp_end_epoch=4,
    )


@pytest.mark.parametrize(
    ('sequence_type', 'section_size'),
    [
        (grassdata.SEQUENCE_TYPE_WING, 29),
        (grassdata.SEQUENCE_TYPE_FUSELAGE, 5),
    ],
)
def test_section_autoencoder_forward_and_backward(sequence_type, section_size):
    config = model_config()
    model = section_autoencoder.SectionAutoencoder(sequence_type, config)
    sections = torch.randn(2, 8, section_size)
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        sections[..., 22:24] = torch.rand(2, 8, 2) + 0.1
        sections[..., 27:28] = torch.rand(2, 8, 1) + 0.1
    count = torch.tensor([2, 8])

    reconstructed, count_logits = model(sections, count, torch.tensor([1.0, 1.0]))
    losses = section_autoencoder.reconstruction_losses(
        sequence_type, reconstructed, count_logits, sections, count
    )
    losses['total'].mean().backward()

    assert reconstructed.shape == sections.shape
    assert count_logits.shape == (2, 7)
    assert torch.isfinite(losses['total']).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pretrained_section_checkpoints_load_into_full_autoencoder():
    config = model_config()
    checkpoints = {}
    for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE):
        model = section_autoencoder.SectionAutoencoder(sequence_type, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoints[sequence_type] = section_autoencoder.build_checkpoint(
            model, 1, optimizer, scheduler, {'total': 0.0}, config
        )

    full_encoder = GRASSEncoder(config)
    full_decoder = GRASSDecoder(config)
    section_autoencoder.apply_pretrained_section_autoencoders(
        full_encoder, full_decoder, checkpoints, config
    )

    for key, value in full_encoder.wing_section_encoder.state_dict().items():
        assert torch.equal(value, checkpoints[grassdata.SEQUENCE_TYPE_WING]['section_encoder_state_dict'][key])
    for key, value in full_decoder.fuselage_section_decoder.state_dict().items():
        assert torch.equal(value, checkpoints[grassdata.SEQUENCE_TYPE_FUSELAGE]['section_decoder_state_dict'][key])

    checkpoints[grassdata.SEQUENCE_TYPE_WING]['feature_size'] = config.feature_size + 1
    with pytest.raises(ValueError, match='feature_size'):
        section_autoencoder.apply_pretrained_section_autoencoders(
            GRASSEncoder(config), GRASSDecoder(config), checkpoints, config
        )
