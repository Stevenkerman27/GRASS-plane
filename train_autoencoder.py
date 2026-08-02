"""Train the deterministic tree autoencoder for structured aircraft data."""

import random
from pathlib import Path

import matplotlib
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchfoldext import FoldExt

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import grassmodel
import grassdata
import section_autoencoder
import section_parameter_codec
import util
from grassdata import StructuredGRASSDataset
from grassmodel import GRASSDecoder, GRASSEncoder


def collate_trees(batch):
    return batch


def choose_device(config):
    if config.no_cuda:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; rerun with --no_cuda.')
    if config.gpu < 0:
        raise ValueError('--gpu must be non-negative when CUDA is enabled.')
    torch.cuda.set_device(config.gpu)
    return torch.device(f'cuda:{config.gpu}')


def set_seed(seed, cuda_enabled):
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_enabled:
        torch.cuda.manual_seed_all(seed)


def make_loaders(config):
    if config.legacy_data:
        raise ValueError('train_autoencoder.py requires structured sequence data; omit --legacy_data.')
    if not 0.0 < config.ae_validation_fraction < 1.0:
        raise ValueError('--ae_validation_fraction must be strictly between 0 and 1.')

    if not config.structured_data_paths:
        raise ValueError('--structured_data_paths must contain at least one dataset path.')
    datasets = [StructuredGRASSDataset(path) for path in config.structured_data_paths]
    dataset = ConcatDataset(datasets)
    validation_size = int(round(len(dataset) * config.ae_validation_fraction))
    validation_size = max(1, validation_size)
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError('dataset must retain at least one training sample after validation split.')

    split_generator = torch.Generator().manual_seed(config.ae_seed)
    training_set, validation_set = random_split(
        dataset, [training_size, validation_size], generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(config.ae_seed)
    train_loader = DataLoader(
        training_set,
        batch_size=config.ae_batch_size,
        shuffle=True,
        collate_fn=collate_trees,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.ae_batch_size,
        shuffle=False,
        collate_fn=collate_trees,
    )
    section_statistics = {
        sequence_type: section_parameter_codec.fit_section_parameter_statistics(
            section_autoencoder.extract_section_leaves(training_set, sequence_type),
            sequence_type,
        )
        for sequence_type in (
            grassdata.SEQUENCE_TYPE_WING,
            grassdata.SEQUENCE_TYPE_FUSELAGE,
        )
    }
    return train_loader, validation_loader, section_statistics


def aggregate_losses(apply_results, has_boxes, has_symmetry, has_node_types, batch_size, device):
    result_index = 0
    zero = torch.zeros((), device=device)
    geometry_loss = zero
    component_loss = zero
    symmetry_loss = zero
    node_type_loss = zero

    if has_boxes:
        box_losses = apply_results[result_index].sum(dim=0) / batch_size
        geometry_loss = box_losses[:6].sum()
        component_loss = box_losses[6]
        result_index += 1
    if has_symmetry:
        symmetry_loss = apply_results[result_index].sum() / batch_size
        result_index += 1
    if has_node_types:
        node_type_loss = apply_results[result_index].sum() / batch_size

    total_loss = (
        util.AE_LOSS_WEIGHTS['geometry'] * geometry_loss
        + util.AE_LOSS_WEIGHTS['component'] * component_loss
        + util.AE_LOSS_WEIGHTS['symmetry'] * symmetry_loss
        + util.AE_LOSS_WEIGHTS['node_type'] * node_type_loss
    )
    return {
        'geometry': geometry_loss,
        'component': component_loss,
        'symmetry': symmetry_loss,
        'node_type': node_type_loss,
        'total': total_loss,
    }


def reconstruction_losses(encoder, decoder, batch, cuda_enabled, device):
    encoder_fold = FoldExt(cuda=cuda_enabled)
    encoded_nodes = [
        grassmodel.encode_structure_fold(encoder_fold, tree, use_sampler=False)
        for tree in batch
    ]
    encoded_features = encoder_fold.apply(encoder, [encoded_nodes])[0]
    encoded_features = torch.split(encoded_features, 1, dim=0)

    decoder_fold = FoldExt(cuda=cuda_enabled)
    box_nodes = []
    symmetry_nodes = []
    node_type_nodes = []
    for tree, feature in zip(batch, encoded_features):
        boxes, symmetries, node_types = grassmodel.decode_structure_fold(
            decoder_fold,
            feature,
            tree,
            use_sample_decoder=False,
        )
        box_nodes.extend(boxes)
        symmetry_nodes.extend(symmetries)
        node_type_nodes.extend(node_types)

    apply_lists = []
    if box_nodes:
        apply_lists.append(box_nodes)
    if symmetry_nodes:
        apply_lists.append(symmetry_nodes)
    if node_type_nodes:
        apply_lists.append(node_type_nodes)
    if not apply_lists:
        raise RuntimeError('decoded batch has no reconstruction loss nodes.')
    apply_results = decoder_fold.apply(decoder, apply_lists)
    return aggregate_losses(
        apply_results,
        bool(box_nodes),
        bool(symmetry_nodes),
        bool(node_type_nodes),
        len(batch),
        device,
    )


def free_generation_metrics(encoder, decoder, batch, cuda_enabled):
    encoder_fold = FoldExt(cuda=cuda_enabled)
    encoded_nodes = [
        grassmodel.encode_structure_fold(encoder_fold, tree, use_sampler=False)
        for tree in batch
    ]
    root_features = encoder_fold.apply(encoder, [encoded_nodes])[0]
    generated = decoder.decode_free(root_features)
    sample_count = len(generated)
    if sample_count == 0:
        raise RuntimeError('free decoder received an empty batch')
    return {
        'tree_valid_fraction': sum(item['tree_valid'] for item in generated) / sample_count,
        'leaf_count': sum(item['leaf_count'] for item in generated) / sample_count,
        'tree_depth': sum(item['tree_depth'] for item in generated) / sample_count,
        'forced_by_limit_count': sum(item['forced_by_limit_count'] for item in generated) / sample_count,
    }


def mean_metrics(metric_sums, batch_count):
    if batch_count == 0:
        raise RuntimeError('dataloader produced no batches.')
    return {name: value / batch_count for name, value in metric_sums.items()}


def run_epoch(
        encoder, decoder, loader, optimizer, config, cuda_enabled, device, training):
    if training:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    metric_sums = {name: 0.0 for name in util.AE_LOSS_WEIGHTS}
    metric_sums['total'] = 0.0
    free_metric_sums = {
        'tree_valid_fraction': 0.0,
        'leaf_count': 0.0,
        'tree_depth': 0.0,
        'forced_by_limit_count': 0.0,
    } if not training else None
    for batch in loader:
        with torch.set_grad_enabled(training):
            losses = reconstruction_losses(
                encoder, decoder, batch, cuda_enabled, device
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    config.ae_gradient_clip,
                )
                optimizer.step()
        for name, loss in losses.items():
            metric_sums[name] += loss.detach().item()
        if not training:
            with torch.no_grad():
                free_metrics = free_generation_metrics(encoder, decoder, batch, cuda_enabled)
            for name, value in free_metrics.items():
                free_metric_sums[name] += value
    metrics = mean_metrics(metric_sums, len(loader))
    if training:
        return metrics, None
    free_metrics = mean_metrics(free_metric_sums, len(loader))
    return metrics, free_metrics


def save_checkpoint(
        path, epoch, encoder, decoder, optimizer, scheduler, validation_metrics,
        free_validation_metrics, config):
    section_statistics = {
        grassdata.SEQUENCE_TYPE_WING:
            encoder.wing_section_encoder.parameter_codec.export_statistics(),
        grassdata.SEQUENCE_TYPE_FUSELAGE:
            encoder.fuselage_section_encoder.parameter_codec.export_statistics(),
    }
    torch.save(
        {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'validation_metrics': validation_metrics,
            'free_validation_metrics': free_validation_metrics,
            'feature_size': config.feature_size,
            'hidden_size': config.hidden_size,
            'section_hidden_size': util.SECTION_CODEC_HIDDEN_SIZE,
            'section_statistics': section_statistics,
        },
        path,
    )


def format_metrics(metrics):
    return ' '.join(f'{name}={value:.6f}' for name, value in metrics.items())


def make_learning_rate_scheduler(optimizer, config):
    if not 0.0 < config.ae_lr_decay_factor < 1.0:
        raise ValueError('--ae_lr_decay_factor must be strictly between 0 and 1.')
    if config.ae_lr_decay_patience < 0:
        raise ValueError('--ae_lr_decay_patience must be non-negative.')
    if config.ae_lr_min <= 0.0:
        raise ValueError('--ae_lr_min must be positive.')
    if config.ae_lr_min > config.ae_lr:
        raise ValueError('--ae_lr_min must not exceed --ae_lr.')
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.ae_lr_decay_factor,
        patience=config.ae_lr_decay_patience,
        min_lr=config.ae_lr_min,
    )


def make_autoencoder_optimizer(parameters, config):
    util.validate_ae_weight_decay(config.ae_weight_decay)
    return torch.optim.AdamW(
        parameters,
        lr=config.ae_lr,
        weight_decay=config.ae_weight_decay,
    )


def current_learning_rate(optimizer):
    learning_rates = {group['lr'] for group in optimizer.param_groups}
    if len(learning_rates) != 1:
        raise RuntimeError('AE optimizer parameter groups have inconsistent learning rates.')
    return learning_rates.pop()


def save_loss_curves(history, learning_rate_history, checkpoint_dir):
    epochs = range(1, len(history['train']['total']) + 1)
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for axis, metric_name in zip(axes.flat[:-1], history['train']):
        axis.plot(epochs, history['train'][metric_name], label='train')
        axis.plot(epochs, history['validation'][metric_name], label='validation')
        axis.set_title(metric_name)
        axis.set_ylabel('loss')
        axis.set_yscale('log')
        axis.grid(True, alpha=0.3)
        axis.legend()
    learning_rate_axis = axes.flat[5]
    learning_rate_axis.plot(epochs, learning_rate_history, label='learning rate')
    learning_rate_axis.set_title('learning_rate')
    learning_rate_axis.set_ylabel('learning rate')
    learning_rate_axis.set_yscale('log')
    learning_rate_axis.grid(True, alpha=0.3)
    learning_rate_axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel('epoch')
    figure.tight_layout()
    figure.savefig(checkpoint_dir / 'loss_curves.png', dpi=160)
    plt.close(figure)


def save_free_generation_metrics(history, checkpoint_dir):
    epochs = range(1, len(history['tree_valid_fraction']) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for axis, metric_name in zip(axes.flat, history):
        axis.plot(epochs, history[metric_name])
        axis.set_title(metric_name)
        axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel('epoch')
    figure.tight_layout()
    figure.savefig(checkpoint_dir / 'free_generation_metrics.png', dpi=160)
    plt.close(figure)


def save_metrics_report(history, free_history, learning_rate_history, checkpoint_dir):
    report = {
        'loss': history,
        'free_generation': free_history,
        'learning_rate': learning_rate_history,
    }
    report_path = checkpoint_dir / 'training_metrics.yaml'
    with report_path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(report, stream, sort_keys=False)


def main():
    config = util.get_args()
    util.validate_ae_weight_decay(config.ae_weight_decay)
    device = choose_device(config)
    cuda_enabled = device.type == 'cuda'
    set_seed(config.ae_seed, cuda_enabled)
    train_loader, validation_loader, section_statistics = make_loaders(config)

    encoder = GRASSEncoder(config, section_statistics).to(device)
    decoder = GRASSDecoder(config, section_statistics).to(device)
    if config.ae_section_pretrained_checkpoint_dir:
        section_autoencoder.load_pretrained_section_autoencoders(
            encoder,
            decoder,
            config.ae_section_pretrained_checkpoint_dir,
            config,
            device,
        )
        print(
            'Loaded section-AE initialization from '
            f'{config.ae_section_pretrained_checkpoint_dir}; all section parameters remain trainable.'
        )
    optimizer = make_autoencoder_optimizer(
        list(encoder.parameters()) + list(decoder.parameters()), config
    )
    scheduler = make_learning_rate_scheduler(optimizer, config)
    checkpoint_dir = Path(config.ae_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'Using {device}; train samples={len(train_loader.dataset)}; '
        f'validation samples={len(validation_loader.dataset)}; '
        f'section_hidden_size={util.SECTION_CODEC_HIDDEN_SIZE}; '
        f'weight_decay={config.ae_weight_decay}'
    )
    best_validation_total = float('inf')
    history = {
        split: {metric_name: [] for metric_name in (*util.AE_LOSS_WEIGHTS, 'total')}
        for split in ('train', 'validation')
    }
    free_history = {
        'tree_valid_fraction': [],
        'leaf_count': [],
        'tree_depth': [],
        'forced_by_limit_count': [],
    }
    learning_rate_history = []
    for epoch in range(1, config.ae_epochs + 1):
        train_metrics, _ = run_epoch(
            encoder,
            decoder,
            train_loader,
            optimizer,
            config,
            cuda_enabled,
            device,
            training=True,
        )
        validation_metrics, free_validation_metrics = run_epoch(
            encoder,
            decoder,
            validation_loader,
            optimizer,
            config,
            cuda_enabled,
            device,
            training=False,
        )
        for metric_name in history['train']:
            history['train'][metric_name].append(train_metrics[metric_name])
            history['validation'][metric_name].append(validation_metrics[metric_name])
        for metric_name in free_history:
            free_history[metric_name].append(free_validation_metrics[metric_name])
        scheduler.step(validation_metrics['total'])
        learning_rate = current_learning_rate(optimizer)
        learning_rate_history.append(learning_rate)
        if epoch % config.ae_log_every == 0:
            print(
                f'epoch={epoch}/{config.ae_epochs} '
                f'train[{format_metrics(train_metrics)}] '
                f'validation[{format_metrics(validation_metrics)}] '
                f'free_validation[{format_metrics(free_validation_metrics)}] '
                f'learning_rate={learning_rate:.8g}'
            )
        if epoch % util.AE_CHECKPOINT_EVERY == 0 or epoch == config.ae_epochs:
            save_checkpoint(
                checkpoint_dir / 'last.pt', epoch, encoder, decoder, optimizer, scheduler,
                validation_metrics, free_validation_metrics, config
            )
            if validation_metrics['total'] < best_validation_total:
                best_validation_total = validation_metrics['total']
                save_checkpoint(
                    checkpoint_dir / 'best.pt', epoch, encoder, decoder, optimizer, scheduler,
                    validation_metrics, free_validation_metrics, config
                )
    save_loss_curves(
        history,
        learning_rate_history,
        checkpoint_dir,
    )
    save_free_generation_metrics(free_history, checkpoint_dir)
    save_metrics_report(
        history,
        free_history,
        learning_rate_history,
        checkpoint_dir,
    )


if __name__ == '__main__':
    main()
