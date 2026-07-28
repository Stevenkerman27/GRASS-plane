import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grassdata
import util
from grassmodel import AutoregressiveSectionDecoder


def schedule_config(**overrides):
    values = {
        'ae_teacher_forcing_p_final': 0.1,
        'ae_teacher_forcing_ramp_start_epoch': 80,
        'ae_teacher_forcing_ramp_end_epoch': 140,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_teacher_forcing_probability_has_constant_ramp_constant_stages():
    config = schedule_config()

    assert util.ae_teacher_forcing_probability(1, config) == 1.0
    assert util.ae_teacher_forcing_probability(79, config) == 1.0
    assert util.ae_teacher_forcing_probability(80, config) == 1.0
    assert util.ae_teacher_forcing_probability(110, config) == pytest.approx(0.55)
    assert util.ae_teacher_forcing_probability(140, config) == 0.1
    assert util.ae_teacher_forcing_probability(141, config) == 0.1


@pytest.mark.parametrize(
    'config',
    [
        schedule_config(ae_teacher_forcing_p_final=-0.1),
        schedule_config(ae_teacher_forcing_p_final=1.1),
        schedule_config(ae_teacher_forcing_ramp_start_epoch=0),
        schedule_config(ae_teacher_forcing_ramp_end_epoch=80),
    ],
)
def test_teacher_forcing_schedule_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        util.validate_ae_teacher_forcing_schedule(config)


def test_decoder_accepts_per_sample_scheduled_sampling_probabilities():
    torch.manual_seed(3)
    decoder = AutoregressiveSectionDecoder(
        grassdata.SEQUENCE_TYPE_FUSELAGE, feature_size=7, rnn_type='rnn'
    )
    features = torch.randn(2, 7)
    sections = torch.randn(2, util.SECTION_COUNT_RANGE[1], util.FUSELAGE_SECTION_SIZE)

    teacher_forced, _ = decoder(features, sections, torch.tensor([1.0, 1.0]))
    free_running, _ = decoder(features, sections, torch.tensor([0.0, 0.0]))

    assert teacher_forced.shape == sections.shape
    assert free_running.shape == sections.shape
    assert torch.allclose(teacher_forced[:, 0], free_running[:, 0])
    assert not torch.allclose(teacher_forced[:, 1:], free_running[:, 1:])


def test_probability_one_preserves_teacher_forced_section_outputs():
    torch.manual_seed(4)
    decoder = AutoregressiveSectionDecoder(
        grassdata.SEQUENCE_TYPE_FUSELAGE, feature_size=7, rnn_type='rnn'
    )
    features = torch.randn(2, 7)
    sections = torch.randn(2, util.SECTION_COUNT_RANGE[1], util.FUSELAGE_SECTION_SIZE)

    hidden = torch.tanh(decoder.initial_hidden(features))
    previous = decoder.bos.unsqueeze(0).expand(features.size(0), -1)
    raw_sections = []
    for index in range(util.SECTION_COUNT_RANGE[1]):
        hidden = decoder.rnn(previous, hidden)
        raw_sections.append(decoder.output(hidden))
        previous = sections[:, index, :]
    expected = decoder._constrain_sections(torch.stack(raw_sections, dim=1))
    actual, _ = decoder(features, sections, torch.tensor([1.0, 1.0]))

    assert torch.allclose(actual, expected)
