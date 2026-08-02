import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grassdata
import grassmodel
import section_parameter_codec
import util


def payload(sequence_type, batch_size=2):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        global_parameters = torch.tensor([[0.0, 0.0, 0.0, 2.0, 6.0]]).repeat(batch_size, 1)
        sections = torch.zeros(batch_size, util.WING_SECTION_COUNT, util.WING_SECTION_SIZE)
        sections[..., 0] = torch.linspace(0.0, 1.0, util.WING_SECTION_COUNT)
        sections[..., 1] = 1.0
        sections[..., -2:] = 0.5
        return global_parameters, sections
    global_parameters = torch.tensor([[0.0, 0.0, 0.0, 12.0, 3.0, 2.0]]).repeat(batch_size, 1)
    sections = torch.zeros(batch_size, util.FUSELAGE_SECTION_COUNT, util.FUSELAGE_SECTION_SIZE)
    sections[..., 0] = torch.linspace(0.0, 1.0, util.FUSELAGE_SECTION_COUNT)
    sections[..., 1:3] = 0.7
    sections[:, (0, -1), 1:3] = 0.1
    sections[..., 4:6] = 2.0
    return global_parameters, sections


def statistics(sequence_type):
    global_parameters, sections = payload(sequence_type)
    return section_parameter_codec.fit_section_parameter_statistics(
        [
            {'z_global': global_parameters[index], 'sections': sections[index]}
            for index in range(global_parameters.size(0))
        ],
        sequence_type,
    )


def test_fixed_section_encoder_and_decoder_architecture():
    sequence_type = grassdata.SEQUENCE_TYPE_WING
    encoder = grassmodel.SectionEncoder(sequence_type, 128, statistics(sequence_type))
    decoder = grassmodel.SectionDecoder(sequence_type, 128, statistics(sequence_type))

    assert isinstance(encoder.conv1, torch.nn.Conv1d)
    assert encoder.conv1.in_channels == util.WING_SECTION_SIZE
    assert encoder.conv1.out_channels == util.SECTION_CODEC_CONV_CHANNELS
    assert encoder.hidden.out_features == util.SECTION_CODEC_HIDDEN_SIZE
    assert decoder.hidden.out_features == util.SECTION_CODEC_HIDDEN_SIZE
    assert decoder.output.out_features == 140

    global_parameters, sections = payload(sequence_type)
    features = encoder(global_parameters, sections)
    reconstructed_global, reconstructed_sections = decoder(features)
    assert features.shape == (2, 128)
    assert reconstructed_global.shape == global_parameters.shape
    assert reconstructed_sections.shape == sections.shape
