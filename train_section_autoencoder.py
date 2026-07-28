"""Pretrain independent wing and fuselage section autoencoders."""

from math import ceil
from pathlib import Path

import matplotlib
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, random_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import grassdata
import section_autoencoder
import util
from grassdata import StructuredGRASSDataset
from train_autoencoder import choose_device, current_learning_rate, make_learning_rate_scheduler, set_seed


def make_aircraft_splits(config):
    if config.legacy_data:
        raise ValueError('train_section_autoencoder.py requires structured sequence data; omit --legacy_data.')
    if not 0.0 < config.ae_validation_fraction < 1.0:
        raise ValueError('--ae_validation_fraction must be strictly between 0 and 1.')
    if not config.structured_data_paths:
        raise ValueError('--structured_data_paths must contain at least one dataset path.')

    dataset = ConcatDataset([StructuredGRASSDataset(path) for path in config.structured_data_paths])
    validation_size = max(1, int(round(len(dataset) * config.ae_validation_fraction)))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError('dataset must retain at least one training aircraft after validation split.')
    return random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config.ae_seed),
    )


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


def run_epoch(model, loader, optimizer, device, training, teacher_forcing_probability, gradient_clip):
    model.train(training)
    metric_sums = None
    free_metric_sums = None if training else {}
    for batch in loader:
        sections = batch['sections'].to(device)
        section_count = batch['section_count'].to(device)
        probability = torch.full(
            (sections.size(0),), teacher_forcing_probability, dtype=sections.dtype, device=device
        )
        with torch.set_grad_enabled(training):
            predicted_sections, count_logits = model(sections, section_count, probability)
            loss_vectors = section_autoencoder.reconstruction_losses(
                model.sequence_type, predicted_sections, count_logits, sections, section_count
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

        if not training:
            with torch.no_grad():
                free_probability = torch.zeros(sections.size(0), dtype=sections.dtype, device=device)
                free_sections, free_count_logits = model(sections, section_count, free_probability)
                free_vectors = section_autoencoder.reconstruction_losses(
                    model.sequence_type, free_sections, free_count_logits, sections, section_count
                )
            if not free_metric_sums:
                free_metric_sums = {name: 0.0 for name in free_vectors}
            for name, value in free_vectors.items():
                free_metric_sums[name] += value.mean().item()
    metrics = mean_metrics(metric_sums, len(loader))
    if training:
        return metrics, None
    return metrics, mean_metrics(free_metric_sums, len(loader))


def save_loss_curves(history, free_history, learning_rate_history, probability_history, path):
    metric_names = tuple(history['train'])
    plot_count = len(metric_names) + 3
    columns = 2
    figure, axes = plt.subplots(ceil(plot_count / columns), columns, figsize=(12, 3.5 * ceil(plot_count / columns)))
    axes = axes.flat
    epochs = range(1, len(history['train']['total']) + 1)
    for axis, metric_name in zip(axes, metric_names):
        axis.plot(epochs, history['train'][metric_name], label='train')
        axis.plot(epochs, history['validation'][metric_name], label='validation_teacher')
        axis.set_title(metric_name)
        axis.set_yscale('log')
        axis.grid(True, alpha=0.3)
        axis.legend()
    axis = next(axes)
    axis.plot(epochs, free_history['total'], label='validation_free')
    axis.set_title('free_total')
    axis.set_yscale('log')
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis = next(axes)
    axis.plot(epochs, learning_rate_history)
    axis.set_title('learning_rate')
    axis.set_yscale('log')
    axis.grid(True, alpha=0.3)
    axis = next(axes)
    axis.plot(epochs, probability_history)
    axis.set_title('teacher_forcing_probability')
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.3)
    for axis in axes:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_sequence_type(sequence_type, training_aircraft, validation_aircraft, config, device):
    train_loader, validation_loader = make_leaf_loaders(
        training_aircraft, validation_aircraft, sequence_type, config
    )
    model = section_autoencoder.SectionAutoencoder(sequence_type, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ae_lr)
    scheduler = make_learning_rate_scheduler(optimizer, config)
    checkpoint_dir = Path(config.section_ae_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f'{sequence_type}: train leaves={len(train_loader.dataset)}; '
        f'validation leaves={len(validation_loader.dataset)}'
    )

    history = {'train': None, 'validation': None}
    free_history = None
    learning_rate_history = []
    probability_history = []
    best_validation_total = float('inf')
    for epoch in range(1, config.ae_epochs + 1):
        probability = util.ae_teacher_forcing_probability(epoch, config)
        train_metrics, _ = run_epoch(
            model, train_loader, optimizer, device, True, probability, config.ae_gradient_clip
        )
        validation_metrics, free_validation_metrics = run_epoch(
            model, validation_loader, optimizer, device, False, 1.0, config.ae_gradient_clip
        )
        if history['train'] is None:
            history = {
                'train': {name: [] for name in train_metrics},
                'validation': {name: [] for name in validation_metrics},
            }
            free_history = {name: [] for name in free_validation_metrics}
        for name in history['train']:
            history['train'][name].append(train_metrics[name])
            history['validation'][name].append(validation_metrics[name])
            free_history[name].append(free_validation_metrics[name])
        scheduler.step(validation_metrics['total'])
        learning_rate = current_learning_rate(optimizer)
        learning_rate_history.append(learning_rate)
        probability_history.append(probability)
        if epoch % config.ae_log_every == 0:
            print(
                f'{sequence_type} epoch={epoch}/{config.ae_epochs} '
                f'train_total={train_metrics["total"]:.6f} '
                f'validation_total={validation_metrics["total"]:.6f} '
                f'free_total={free_validation_metrics["total"]:.6f} '
                f'p_teacher={probability:.6f} learning_rate={learning_rate:.8g}'
            )
        checkpoint = section_autoencoder.build_checkpoint(
            model, epoch, optimizer, scheduler, validation_metrics, config
        )
        torch.save(checkpoint, checkpoint_dir / f'last_{sequence_type}.pt')
        if validation_metrics['total'] < best_validation_total:
            best_validation_total = validation_metrics['total']
            torch.save(checkpoint, checkpoint_dir / f'best_{sequence_type}.pt')

    save_loss_curves(
        history,
        free_history,
        learning_rate_history,
        probability_history,
        checkpoint_dir / f'loss_curves_{sequence_type}_{config.ae_rnn_type}.png',
    )
    with (checkpoint_dir / f'training_metrics_{sequence_type}_{config.ae_rnn_type}.yaml').open(
            'w', encoding='utf-8') as stream:
        yaml.safe_dump(
            {
                'schema': section_autoencoder.SECTION_AUTOENCODER_CHECKPOINT_SCHEMA,
                'sequence_type': sequence_type,
                'loss': history,
                'free_loss': free_history,
                'learning_rate': learning_rate_history,
                'teacher_forcing_probability': probability_history,
            },
            stream,
            sort_keys=False,
        )


def main():
    config = util.get_args()
    config.ae_rnn_type = util.validate_ae_rnn_type(config.ae_rnn_type)
    util.validate_ae_teacher_forcing_schedule(config)
    device = choose_device(config)
    set_seed(config.ae_seed, device.type == 'cuda')
    training_aircraft, validation_aircraft = make_aircraft_splits(config)
    print(
        f'Using {device}; train aircraft={len(training_aircraft)}; '
        f'validation aircraft={len(validation_aircraft)}; '
        f'recurrent_cell={config.ae_rnn_type}; feature_size={config.feature_size}; '
        f'hidden_size={config.hidden_size}'
    )
    for sequence_type in (grassdata.SEQUENCE_TYPE_WING, grassdata.SEQUENCE_TYPE_FUSELAGE):
        train_sequence_type(sequence_type, training_aircraft, validation_aircraft, config, device)


if __name__ == '__main__':
    main()
