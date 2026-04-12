import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='grass_pytorch')
    parser.add_argument('--box_code_size', type=int, default=13)
    parser.add_argument('--feature_size', type=int, default=80)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--symmetry_size', type=int, default=8)
    parser.add_argument('--max_box_num', type=int, default=30)
    parser.add_argument('--max_sym_num', type=int, default=10)

    # GAN Hyperparameters
    parser.add_argument('--alpha1', type=float, default=0.001, help='Weight for reconstruction loss in GAN training')
    parser.add_argument('--alpha2', type=float, default=0.1, help='Weight for KL divergence loss in GAN training')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Weight for gradient penalty in WGAN-GP')
    parser.add_argument('--gan_lr', type=float, default=1e-4, help='Learning rate for GAN optimizers')
    parser.add_argument('--n_critic', type=int, default=1, help='Number of generator updates per discriminator update')
    parser.add_argument('--gan_epochs', type=int, default=10, help='Number of epochs for GAN training')
    parser.add_argument('--gan_batch_size', type=int, default=12, help='Batch size for GAN training')

    # VAE parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--kl_weight_target', type=float, default=0.03)
    parser.add_argument('--kl_anneal_epochs', type=int, default=20)
    parser.add_argument('--kl_tolerance', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--show_log_every', type=int, default=3)
    parser.add_argument('--save_log', action='store_true', default=False)
    parser.add_argument('--save_log_every', type=int, default=3)
    parser.add_argument('--save_snapshot', action='store_true', default=False)
    parser.add_argument('--save_snapshot_every', type=int, default=5)
    parser.add_argument('--no_plot', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_by', type=float, default=1)
    parser.add_argument('--lr_decay_every', type=float, default=1)

    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args