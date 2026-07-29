import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util
import grassdata
import section_parameter_codec
from grassmodel import GRASSDecoder, GRASSEncoder


def model_config(rnn_type):
    return SimpleNamespace(
        box_code_size=13,
        feature_size=12,
        hidden_size=16,
        symmetry_size=8,
        ae_rnn_type=rnn_type,
    )


def section_statistics():
    return {
        sequence_type: {
            'schema': section_parameter_codec.SECTION_PARAMETER_CODEC_SCHEMA,
            'sequence_type': sequence_type,
            'mean': torch.zeros(grassdata.sequence_section_size(sequence_type)),
            'std': torch.ones(grassdata.sequence_section_size(sequence_type)),
            'constant_mask': torch.zeros(
                grassdata.sequence_section_size(sequence_type), dtype=torch.bool
            ),
        }
        for sequence_type in (
            grassdata.SEQUENCE_TYPE_WING,
            grassdata.SEQUENCE_TYPE_FUSELAGE,
        )
    }


@pytest.mark.parametrize(
    ('rnn_type', 'encoder_cell', 'decoder_cell'),
    [
    ('rnn', torch.nn.RNN, torch.nn.RNNCell),
    ('gru', torch.nn.GRU, torch.nn.GRUCell),
    ],
)
def test_recurrent_type_selects_encoder_and_decoder_cells(
        rnn_type, encoder_cell, decoder_cell):
    config = model_config(rnn_type)
    statistics = section_statistics()
    encoder = GRASSEncoder(config, statistics)
    decoder = GRASSDecoder(config, statistics)

    assert isinstance(encoder.fuselage_section_encoder.rnn, encoder_cell)
    assert isinstance(encoder.wing_section_encoder.rnn, encoder_cell)
    assert isinstance(decoder.fuselage_section_decoder.rnn, decoder_cell)
    assert isinstance(decoder.wing_section_decoder.rnn, decoder_cell)
    assert decoder.fuselage_section_decoder.rnn.input_size == (
        util.FUSELAGE_SECTION_SIZE + config.feature_size + 1
    )
    assert decoder.wing_section_decoder.rnn.input_size == (
        grassdata.sequence_section_size(grassdata.SEQUENCE_TYPE_WING)
        + config.feature_size
        + 1
    )

    fuselage_sections = torch.zeros(2, util.SECTION_COUNT_RANGE[1], util.FUSELAGE_SECTION_SIZE)
    fuselage_sections[..., 0] = torch.arange(util.SECTION_COUNT_RANGE[1])
    fuselage_sections[..., 3:5] = 1.0
    fuselage_count = torch.tensor([2, util.SECTION_COUNT_RANGE[1]])
    fuselage_features = encoder.fuselageSectionEncoder(fuselage_sections, fuselage_count)
    reconstructed, count_logits = decoder.fuselageSectionDecoder(
        fuselage_features, fuselage_sections, fuselage_count, torch.tensor([1.0])
    )
    generated, count, valid_mask, _ = decoder.generateFuselageSections(fuselage_features)

    assert fuselage_features.shape == (2, config.feature_size)
    assert reconstructed.shape == fuselage_sections.shape
    assert count_logits.shape == (2, util.SECTION_COUNT_RANGE[1] - util.SECTION_COUNT_RANGE[0] + 1)
    assert generated.shape == fuselage_sections.shape
    assert count.shape == (2,)
    assert valid_mask.shape == (2, util.SECTION_COUNT_RANGE[1])


def test_recurrent_type_rejects_unknown_value():
    with pytest.raises(ValueError, match='ae_rnn_type'):
        GRASSEncoder(model_config('lstm'), section_statistics())
