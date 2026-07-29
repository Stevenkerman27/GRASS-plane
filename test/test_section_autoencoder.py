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
import train_section_autoencoder
import util


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


def parameter_statistics(sequence_type):
    section_size = grassdata.sequence_section_size(sequence_type)
    return {
        'schema': section_parameter_codec.SECTION_PARAMETER_CODEC_SCHEMA,
        'sequence_type': sequence_type,
        'mean': torch.zeros(section_size),
        'std': torch.ones(section_size),
        'constant_mask': torch.zeros(section_size, dtype=torch.bool),
    }


def all_parameter_statistics():
    return {
        sequence_type: parameter_statistics(sequence_type)
        for sequence_type in (
            grassdata.SEQUENCE_TYPE_WING,
            grassdata.SEQUENCE_TYPE_FUSELAGE,
        )
    }


def valid_sections(sequence_type, batch_size=2):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        sections = torch.randn(batch_size, 8, 29)
        sections[..., 22:24] = torch.rand(batch_size, 8, 2) + 0.1
        sections[..., 27:28] = torch.rand(batch_size, 8, 1) + 0.1
        return sections
    sections = torch.randn(batch_size, 8, 5)
    sections[..., 0] = torch.arange(8, dtype=sections.dtype).unsqueeze(0) + 1.0
    sections[:, 0, 0] = 0.0
    sections[..., 3:5] = torch.rand(batch_size, 8, 2) + 0.1
    return sections


@pytest.mark.parametrize('sequence_type', [
    grassdata.SEQUENCE_TYPE_WING,
    grassdata.SEQUENCE_TYPE_FUSELAGE,
])
def test_fitted_parameter_codec_round_trips_valid_physical_sections(sequence_type):
    sections = valid_sections(sequence_type)
    counts = torch.tensor([3, 8])
    dataset = [
        {
            'sections': sections[index],
            'section_count': counts[index],
        }
        for index in range(sections.size(0))
    ]
    statistics = section_parameter_codec.fit_section_parameter_statistics(
        dataset, sequence_type
    )
    codec = section_parameter_codec.SectionParameterCodec(sequence_type, statistics)

    normalized = codec.normalize(sections, counts)
    reconstructed = codec.denormalize(normalized)
    valid_mask = grassdata.section_mask(counts, sequence_type=sequence_type)

    assert torch.allclose(reconstructed[valid_mask], sections[valid_mask], atol=1e-5, rtol=1e-5)
    assert torch.equal(normalized[~valid_mask], torch.zeros_like(normalized[~valid_mask]))
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        assert torch.equal(reconstructed[:, 0, 0], torch.zeros(sections.size(0)))


@pytest.mark.parametrize(
    ('sequence_type', 'section_size'),
    [
        (grassdata.SEQUENCE_TYPE_WING, 29),
        (grassdata.SEQUENCE_TYPE_FUSELAGE, 5),
    ],
)
def test_section_autoencoder_forward_and_backward(sequence_type, section_size):
    config = model_config()
    model = section_autoencoder.SectionAutoencoder(
        sequence_type, config, parameter_statistics(sequence_type)
    )
    sections = valid_sections(sequence_type)
    count = torch.tensor([2, 8])

    reconstructed, count_logits = model(sections, count, torch.tensor([1.0, 1.0]))
    losses = section_autoencoder.reconstruction_losses(
        model, reconstructed, count_logits, sections, count
    )
    losses['total'].mean().backward()

    assert reconstructed.shape == sections.shape
    assert count_logits.shape == (2, 7)
    assert torch.isfinite(losses['total']).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pretrained_section_checkpoints_load_into_full_autoencoder():
    config = model_config()
    checkpoints = {}
    statistics = all_parameter_statistics()
    for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE):
        model = section_autoencoder.SectionAutoencoder(
            sequence_type, config, statistics[sequence_type]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoints[sequence_type] = section_autoencoder.build_checkpoint(
            model, 1, optimizer, scheduler, {'total': 0.0}, config
        )

    full_encoder = GRASSEncoder(config, statistics)
    full_decoder = GRASSDecoder(config, statistics)
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
            GRASSEncoder(config, statistics),
            GRASSDecoder(config, statistics),
            checkpoints,
            config,
        )

    checkpoints[grassdata.SEQUENCE_TYPE_WING]['feature_size'] = config.feature_size
    checkpoints[grassdata.SEQUENCE_TYPE_WING]['parameter_statistics']['mean'][0] += 1.0
    with pytest.raises(ValueError, match='statistics'):
        section_autoencoder.apply_pretrained_section_autoencoders(
            GRASSEncoder(config, statistics),
            GRASSDecoder(config, statistics),
            checkpoints,
            config,
        )


@pytest.mark.parametrize('sequence_type', [
    grassdata.SEQUENCE_TYPE_WING,
    grassdata.SEQUENCE_TYPE_FUSELAGE,
])
def test_section_autoencoder_checkpoint_loads_for_evaluation(monkeypatch, sequence_type):
    config = model_config()
    model = section_autoencoder.SectionAutoencoder(
        sequence_type, config, parameter_statistics(sequence_type)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    checkpoint = section_autoencoder.build_checkpoint(
        model, 1, optimizer, scheduler, {'total': 0.0}, config
    )
    path = section_autoencoder.final_checkpoint_path('virtual_checkpoints', sequence_type)
    monkeypatch.setattr(section_autoencoder.Path, 'is_file', lambda _path: True)
    monkeypatch.setattr(section_autoencoder.torch, 'load', lambda *_args, **_kwargs: checkpoint)

    loaded, loaded_checkpoint = section_autoencoder.load_section_autoencoder(
        path, sequence_type, torch.device('cpu')
    )

    assert loaded.sequence_type == sequence_type
    assert loaded.training is False
    assert loaded_checkpoint['epoch'] == 1


def test_final_checkpoint_path_uses_last_checkpoint_name():
    assert section_autoencoder.final_checkpoint_path(
        'models/section_autoencoder', grassdata.SEQUENCE_TYPE_WING
    ) == Path('models/section_autoencoder/last_wing.pt')


def test_autoencoder_optimizer_uses_configured_weight_decay():
    parameter = torch.nn.Parameter(torch.zeros(()))
    config = SimpleNamespace(ae_lr=1e-3, ae_weight_decay=1e-5)

    optimizer = train_autoencoder.make_autoencoder_optimizer([parameter], config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(1e-5)


def test_overfit_split_is_a_deterministic_two_aircraft_shared_subset(monkeypatch):
    config = SimpleNamespace(
        legacy_data=False,
        ae_validation_fraction=0.1,
        structured_data_paths=('first.pt', 'second.pt'),
        ae_seed=17,
        overfit=True,
        section_ae_checkpoint_dir='models/section_autoencoder',
    )
    datasets = {
        'first.pt': [0, 1, 2],
        'second.pt': [3, 4, 5],
    }
    monkeypatch.setattr(
        train_section_autoencoder, 'StructuredGRASSDataset', lambda path: datasets[path]
    )

    training, evaluation = train_section_autoencoder.make_aircraft_splits(config)
    repeated_training, repeated_evaluation = train_section_autoencoder.make_aircraft_splits(config)

    assert training is evaluation
    assert len(training) == util.SECTION_AE_OVERFIT_AIRCRAFT_COUNT
    assert training.indices == repeated_training.indices
    assert repeated_training is repeated_evaluation
    assert train_section_autoencoder.section_ae_checkpoint_dir(config) == Path(
        'models/section_autoencoder/overfit'
    )
