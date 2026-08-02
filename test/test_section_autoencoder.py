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
import section_parameter_codec
import train_autoencoder
import util


def model_config():
    return SimpleNamespace(box_code_size=13, feature_size=128, hidden_size=16, symmetry_size=8)


def payload(sequence_type, batch_size=2):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        global_parameters = torch.tensor([[0.2, -0.1, 0.3, 2.0, 6.0]]).repeat(batch_size, 1)
        sections = torch.zeros(batch_size, util.WING_SECTION_COUNT, util.WING_SECTION_SIZE)
        sections[..., 0] = torch.linspace(0.0, 1.0, util.WING_SECTION_COUNT)
        sections[..., 1] = torch.linspace(1.0, 0.5, util.WING_SECTION_COUNT)
        sections[..., -2:] = 0.5
        return global_parameters, sections
    global_parameters = torch.tensor([[0.0, 0.0, 0.0, 12.0, 3.0, 2.0]]).repeat(batch_size, 1)
    sections = torch.zeros(batch_size, util.FUSELAGE_SECTION_COUNT, util.FUSELAGE_SECTION_SIZE)
    sections[..., 0] = torch.linspace(0.0, 1.0, util.FUSELAGE_SECTION_COUNT)
    sections[..., 1:3] = 0.7
    sections[:, (0, -1), 1:3] = 0.1
    sections[..., 4:6] = 2.0
    return global_parameters, sections


def parameter_statistics(sequence_type):
    global_parameters, sections = payload(sequence_type)
    return section_parameter_codec.fit_section_parameter_statistics(
        [
            {'z_global': global_parameters[index], 'sections': sections[index]}
            for index in range(global_parameters.size(0))
        ],
        sequence_type,
    )


@pytest.mark.parametrize('sequence_type', [
    grassdata.SEQUENCE_TYPE_WING,
    grassdata.SEQUENCE_TYPE_FUSELAGE,
])
def test_component_codec_round_trip(sequence_type):
    global_parameters, sections = payload(sequence_type)
    codec = section_parameter_codec.SectionParameterCodec(
        sequence_type, parameter_statistics(sequence_type)
    )
    model_global, model_sections = codec.normalize(global_parameters, sections)
    decoded_global, decoded_sections = codec.denormalize(model_global, model_sections)
    assert torch.allclose(decoded_global, global_parameters, atol=1e-5)
    assert torch.allclose(decoded_sections, sections, atol=1e-5)


@pytest.mark.parametrize('sequence_type', [
    grassdata.SEQUENCE_TYPE_WING,
    grassdata.SEQUENCE_TYPE_FUSELAGE,
])
def test_section_autoencoder_forward_backward_and_canonical_fields(sequence_type):
    model = section_autoencoder.SectionAutoencoder(
        sequence_type, model_config(), parameter_statistics(sequence_type)
    )
    global_parameters, sections = payload(sequence_type)
    reconstructed_global, reconstructed_sections = model(global_parameters, sections)
    losses = section_autoencoder.reconstruction_losses(
        model, reconstructed_global, reconstructed_sections, global_parameters, sections
    )
    losses['total'].mean().backward()
    assert reconstructed_global.shape == global_parameters.shape
    assert reconstructed_sections.shape == sections.shape
    assert torch.isfinite(losses['total']).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        assert torch.equal(reconstructed_sections[..., 0], torch.linspace(0.0, 1.0, 6).expand(2, -1))
        assert torch.equal(reconstructed_sections[:, 0, 1], torch.ones(2))
    else:
        assert torch.equal(reconstructed_sections[..., 0], torch.linspace(0.0, 1.0, 10).expand(2, -1))
        assert torch.equal(reconstructed_sections[:, (0, -1), 1:3], torch.full((2, 2, 2), 0.1))


def test_pretrained_section_checkpoints_load_into_full_autoencoder():
    config = model_config()
    statistics = {
        sequence_type: parameter_statistics(sequence_type)
        for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE)
    }
    checkpoints = {}
    for sequence_type in statistics:
        model = section_autoencoder.SectionAutoencoder(sequence_type, config, statistics[sequence_type])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoints[sequence_type] = section_autoencoder.build_checkpoint(
            model, 1, optimizer, scheduler, {'total': 0.0}, config
        )
    encoder = GRASSEncoder(config, statistics)
    decoder = GRASSDecoder(config, statistics)
    section_autoencoder.apply_pretrained_section_autoencoders(encoder, decoder, checkpoints, config)
    for key, value in encoder.wing_section_encoder.state_dict().items():
        assert torch.equal(value, checkpoints[grassdata.SEQUENCE_TYPE_WING]['section_encoder_state_dict'][key])


def test_tree_reconstruction_uses_fixed_component_payloads():
    config = model_config()
    statistics = {
        sequence_type: parameter_statistics(sequence_type)
        for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE)
    }
    wing_global, wing_sections = payload(grassdata.SEQUENCE_TYPE_WING, batch_size=1)
    fuselage_global, fuselage_sections = payload(grassdata.SEQUENCE_TYPE_FUSELAGE, batch_size=1)
    tree = grassdata.Tree.from_structured_sample({
        'boxes': [
            {
                'component': util.COMPONENT_FUSELAGE,
                'sequence_type': grassdata.SEQUENCE_TYPE_FUSELAGE,
                'z_global': fuselage_global[0],
                'z_section': fuselage_sections[0],
            },
            {
                'component': util.COMPONENT_WING,
                'sequence_type': grassdata.SEQUENCE_TYPE_WING,
                'z_global': wing_global[0],
                'z_section': wing_sections[0],
            },
        ],
        'ops': [0, 0, 1],
        'syms': [],
    })
    encoder = GRASSEncoder(config, statistics)
    decoder = GRASSDecoder(config, statistics)
    losses = train_autoencoder.reconstruction_losses(
        encoder, decoder, [tree], cuda_enabled=False, device=torch.device('cpu')
    )
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
