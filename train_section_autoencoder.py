"""Pretrain independent wing and fuselage section autoencoders."""

from math import ceil
from pathlib import Path

import matplotlib
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import grassdata
import section_autoencoder
import section_parameter_codec
import util
from grassdata import StructuredGRASSDataset
from train_autoencoder import (
    choose_device,
    current_learning_rate,
    make_autoencoder_optimizer,
    make_learning_rate_scheduler,
    set_seed,
)


def make_aircraft_splits(config):
    if config.legacy_data:
        raise ValueError('train_section_autoencoder.py requires structured sequence data; omit --legacy_data.')
    if not 0.0 < config.ae_validation_fraction < 1.0:
        raise ValueError('--ae_validation_fraction must be strictly between 0 and 1.')
    if not config.structured_data_paths:
        raise ValueError('--structured_data_paths must contain at least one dataset path.')

    dataset = ConcatDataset([StructuredGRASSDataset(path) for path in config.structured_data_paths])
    if config.overfit:
        if len(dataset) < util.SECTION_AE_OVERFIT_AIRCRAFT_COUNT:
            raise ValueError(
                'dataset must contain at least '
                f'{util.SECTION_AE_OVERFIT_AIRCRAFT_COUNT} aircraft for --overfit.'
            )
        indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(config.ae_seed)
        )[:util.SECTION_AE_OVERFIT_AIRCRAFT_COUNT].tolist()
        aircraft = Subset(dataset, indices)
        return aircraft, aircraft

    validation_size = max(1, int(round(len(dataset) * config.ae_validation_fraction)))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError('dataset must retain at least one training aircraft after validation split.')
    return random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config.ae_seed),
    )


def section_ae_checkpoint_dir(config):
    directory = Path(config.section_ae_checkpoint_dir)
    if config.overfit:
        return directory / 'overfit'
    return directory


def make_leaf_loaders(training_aircraft, validation_aircraft, sequence_type, config):
    training_leaves = section_autoencoder.extract_section_leaves(training_aircraft, sequence_type)
    validation_leaves = section_autoencoder.extract_section_leaves(validation_aircraft, sequence_type)
    loader_generator = torch.Generator().manual_seed(config.ae_seed)
    return (
        DataLoader(
            training_leaves, batch_size=config.ae_batch_size, shuffle=True,
            generator=loader_generator,
        ),
        DataLoader(validation_leaves, batch_size=config.ae_batch_size, shuffle=False),
    )


def mean_metrics(metric_sums, batch_count):
    if batch_count == 0:
        raise RuntimeError('Leaf dataloader produced no batches.')
    return {name: value / batch_count for name, value in metric_sums.items()}


def run_epoch(model, loader, optimizer, device, training, gradient_clip):
    model.train(training)
    metric_sums = None
    for batch in loader:
        global_parameters = batch['z_global'].to(device)
        sections = batch['sections'].to(device)
        with torch.set_grad_enabled(training):
            predicted_global_parameters, predicted_sections = model(global_parameters, sections)
            loss_vectors = section_autoencoder.reconstruction_losses(
                model, predicted_global_parameters, predicted_sections, global_parameters, sections
            )
            losses = {name: value.mean() for name, value in loss_vectors.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        if metric_sums is None:
            metric_sums = {name: 0.0 for name in losses}
        for name, loss in losses.items():
            metric_sums[name] += loss.detach().item()

    metrics = mean_metrics(metric_sums, len(loader))
    return metrics


def save_loss_curves(history, learning_rate_history, path):
    metric_names = tuple(history['train'])
    plot_count = len(metric_names) + 1
    columns = 2
    figure, axes = plt.subplots(ceil(plot_count / columns), columns, figsize=(12, 3.5 * ceil(plot_count / columns)))
    axes = list(axes.flat)
    epochs = range(1, len(history['train']['total']) + 1)
    for axis, metric_name in zip(axes[:len(metric_names)], metric_names):
        axis.plot(epochs, history['train'][metric_name], label='train')
        axis.plot(epochs, history['validation'][metric_name], label='validation')
        axis.set_title(metric_name)
        axis.set_yscale('log')
        axis.grid(True, alpha=0.3)
        axis.legend()
    axis = axes[len(metric_names)]
    axis.plot(epochs, learning_rate_history)
    axis.set_title('learning_rate')
    axis.set_yscale('log')
    axis.grid(True, alpha=0.3)
    for axis in axes[plot_count:]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def section_ae_final_epoch(sequence_type):
    if sequence_type == grassdata.SEQUENCE_TYPE_WING:
        return util.SECTION_AE_WING_FINAL_EPOCH
    if sequence_type == grassdata.SEQUENCE_TYPE_FUSELAGE:
        return util.SECTION_AE_FUSELAGE_FINAL_EPOCH
    raise ValueError(f'Unsupported section sequence type: {sequence_type}')


def train_sequence_type(sequence_type, training_aircraft, validation_aircraft, config, device):
    train_loader, validation_loader = make_leaf_loaders(
        training_aircraft, validation_aircraft, sequence_type, config
    )
    parameter_statistics = section_parameter_codec.fit_section_parameter_statistics(
        train_loader.dataset, sequence_type
    )
    model = section_autoencoder.SectionAutoencoder(
        sequence_type, config, parameter_statistics
    ).to(device)
    optimizer = make_autoencoder_optimizer(model.parameters(), config)
    scheduler = make_learning_rate_scheduler(optimizer, config)
    checkpoint_dir = section_ae_checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_epoch = section_ae_final_epoch(sequence_type)
    if final_epoch < 1:
        raise ValueError(f'final epoch must be at least 1, got {final_epoch}.')
    print(
        f'{sequence_type}: train leaves={len(train_loader.dataset)}; '
        f'validation leaves={len(validation_loader.dataset)}; '
        f'final_epoch={final_epoch}'
    )

    history = {'train': None, 'validation': None}
    learning_rate_history = []
    for epoch in range(1, final_epoch + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, True, config.ae_gradient_clip
        )
        validation_metrics = run_epoch(
            model, validation_loader, optimizer, device, False, config.ae_gradient_clip
        )
        if history['train'] is None:
            history = {
                'train': {name: [] for name in train_metrics},
                'validation': {name: [] for name in validation_metrics},
            }
        for name in history['train']:
            history['train'][name].append(train_metrics[name])
            history['validation'][name].append(validation_metrics[name])
        scheduler.step(validation_metrics['total'])
        learning_rate = current_learning_rate(optimizer)
        learning_rate_history.append(learning_rate)
        if epoch % config.ae_log_every == 0:
            print(
                f'{sequence_type} epoch={epoch}/{final_epoch} '
                f'train_total={train_metrics["total"]:.6f} '
                f'validation_total={validation_metrics["total"]:.6f} '
                f'learning_rate={learning_rate:.8g}'
            )
        if epoch == final_epoch:
            checkpoint = section_autoencoder.build_checkpoint(
                model, epoch, optimizer, scheduler, validation_metrics, config
            )
            torch.save(
                checkpoint,
                section_autoencoder.final_checkpoint_path(checkpoint_dir, sequence_type),
            )

    save_loss_curves(
        history,
        learning_rate_history,
        checkpoint_dir / f'loss_curves_{sequence_type}.png',
    )
    with (checkpoint_dir / f'training_metrics_{sequence_type}.yaml').open(
            'w', encoding='utf-8') as stream:
        yaml.safe_dump(
            {
                'schema': section_autoencoder.SECTION_AUTOENCODER_CHECKPOINT_SCHEMA,
                'sequence_type': sequence_type,
                'loss': history,
                'learning_rate': learning_rate_history,
            },
            stream,
            sort_keys=False,
        )


def main():
    config = util.get_args()
    util.validate_ae_weight_decay(config.ae_weight_decay)
    device = choose_device(config)
    set_seed(config.ae_seed, device.type == 'cuda')
    training_aircraft, validation_aircraft = make_aircraft_splits(config)
    print(
        f'Using {device}; train aircraft={len(training_aircraft)}; '
        f'validation aircraft={len(validation_aircraft)}; feature_size={config.feature_size}; '
        f'section_hidden_size={util.SECTION_CODEC_HIDDEN_SIZE}; overfit={config.overfit}; '
        f'weight_decay={config.ae_weight_decay}; '
        f'checkpoint_dir={section_ae_checkpoint_dir(config)}'
    )
    for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE):
        train_sequence_type(sequence_type, training_aircraft, validation_aircraft, config, device)


if __name__ == '__main__':
    main()
