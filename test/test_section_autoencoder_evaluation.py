import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grassdata
import section_autoencoder
import section_autoencoder_evaluation
import section_parameter_codec


def _config():
    return SimpleNamespace(feature_size=10, hidden_size=10, ae_rnn_type='rnn')


def _statistics(sequence_type):
    size = grassdata.sequence_section_size(sequence_type)
    return {
        'schema': section_parameter_codec.SECTION_PARAMETER_CODEC_SCHEMA,
        'sequence_type': sequence_type,
        'mean': torch.zeros(size),
        'std': torch.ones(size),
        'constant_mask': torch.zeros(size, dtype=torch.bool),
    }


def _samples(sequence_type):
    count = torch.tensor([2, 4])
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        sections = torch.randn(2, 8, 29)
        sections[..., 22:24] = torch.rand(2, 8, 2) + 0.1
        sections[..., 27] = torch.rand(2, 8) + 0.2
    else:
        sections = torch.randn(2, 8, 5)
        sections[..., 0] = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        sections[:, 0, 0] = 0.0
        sections[..., 3:5] = torch.rand(2, 8, 2) + 0.2
    for index, section_count in enumerate(count.tolist()):
        sections[index, section_count:] = 0.0
    return [
        {'sections': sections[index], 'section_count': count[index]}
        for index in range(sections.size(0))
    ]


@pytest.mark.parametrize('sequence_type', [
    grassdata.SEQUENCE_TYPE_WING,
    grassdata.SEQUENCE_TYPE_FUSELAGE,
])
@pytest.mark.parametrize('mode', ['teacher_forced', 'free_running'])
def test_evaluate_loader_reports_physical_geometry_and_count_metrics(sequence_type, mode):
    model = section_autoencoder.SectionAutoencoder(
        sequence_type, _config(), _statistics(sequence_type)
    )
    result = section_autoencoder_evaluation.evaluate_loader(
        model,
        DataLoader(_samples(sequence_type), batch_size=2, shuffle=False),
        torch.device('cpu'),
        mode,
    )

    assert result['sequence_type'] == sequence_type
    assert result['mode'] == mode
    assert result['leaf_count'] == 2
    assert result['normalized_training_loss']['total']['mean'] >= 0.0
    assert result['geometry_error']['section_mean_rms_m']['mean'] >= 0.0
    assert len(result['sample_rows']) == 2
    assert result['section_count']['labels'] == list(range(2, 9))
    assert sum(map(sum, result['section_count']['confusion_matrix'])) == 2
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        assert 'cst_n1_n2' in result['physical_parameter_error']
        assert 'chord_m_relative_percent_mean' in result['sample_rows'][0]
    else:
        assert 'length_m' in result['physical_parameter_error']
        assert 'length_m_mae' in result['sample_rows'][0]


def test_distribution_summary_rejects_empty_values():
    with pytest.raises(ValueError, match='empty'):
        section_autoencoder_evaluation.distribution_summary([])
