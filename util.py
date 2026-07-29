import math
from argparse import ArgumentParser

COMPONENT_FUSELAGE = 0
COMPONENT_WING = 1
COMPONENT_ENGINE = 2

COMPONENT_NAMES = {
    COMPONENT_FUSELAGE: 'fuselage',
    COMPONENT_WING: 'wing',
    COMPONENT_ENGINE: 'engine',
}

FUSELAGE_SECTION_SIZE = 5
OBB_GEOMETRY_SIZE = 10
AIRFOIL_DEFAULT_OUTPUT_POINTS = 100
AIRFOIL_DEFAULT_POINT_DENSITY_BETA = 1.3
AIRFOIL_TRAILING_EDGE_X = 1.0
AIRFOIL_LEADING_EDGE_X = 0.0
COMPONENT_CLASS_SIZE = 3

# Fixed-leading-edge CST: two shape coefficient vectors, two trailing-edge
# ordinates, and two independently fitted class-function exponents.
CST_MIN_CLASS_FUNCTION_EXPONENT = 0.05
CST_SURFACE_SHAPE_COEFFICIENTS = 10
CST_BOUNDARY_CODE_SIZE = 4
CST_AIRFOIL_CODE_SIZE = 2 * CST_SURFACE_SHAPE_COEFFICIENTS + CST_BOUNDARY_CODE_SIZE
CST_FIT_CONFIG = {
    'iterations': 700,
    'lr': 0.005,
    'loss_scale': 100.0,
    'coefficient_reg': 0.0,
    'leading_edge_window': 18,
    'leading_edge_weight_amplitude': 4.8,
    'scheduler_patience': 500,
    'scheduler_factor': 0.5,
    'log_interval': 100,
    'surface_shape_coefficients': CST_SURFACE_SHAPE_COEFFICIENTS,
    'initial_n1': 0.5,
    'initial_n2': 1.0,
    'minimum_class_function_exponent': CST_MIN_CLASS_FUNCTION_EXPONENT,
}
SECTION_COUNT_RANGE = (2, 8)
AUXILIARY_WING_SECTION_COUNT_RANGE = (2, 4)
# Default recurrent cell for autoencoder training; set to 'rnn' or 'gru'.
AE_RNN_TYPE = 'rnn'
AE_CHECKPOINT_EVERY = 10
SECTION_AE_OVERFIT_AIRCRAFT_COUNT = 2
SECTION_AE_WING_FINAL_EPOCH = 250
SECTION_AE_FUSELAGE_FINAL_EPOCH = 200
AE_TEACHER_FORCING_P_FINAL = 0.1
AE_TEACHER_FORCING_RAMP_START_EPOCH = 60
AE_TEACHER_FORCING_RAMP_END_EPOCH = 80
AE_WEIGHT_DECAY = 1e-5
WING_LOSS_WEIGHTS = {
    'position': 5.0,
    'chord': 1.0,
    'twist': 1.0,
    'cst_code': 1.0,
    'section_count': 3.0,
}
FUSELAGE_LOSS_WEIGHTS = {
    'position': 1.0,
    'size': 1.0,
    'section_count': 2.0,
}
AE_LOSS_WEIGHTS = {
    'geometry': 5.0,
    'component': 0.1,
    'symmetry': 0.1,
    'node_type': 0.2,
}
FREE_DECODE_MAX_LEAF_NODES = 3
FREE_DECODE_MAX_TREE_DEPTH = 4
FREE_DECODE_ERROR_AIRFOIL_POINTS = 60
FREE_DECODE_ERROR_ELLIPSE_POINTS = 32
COMPONENT_GEOMETRY_SIZES = {
    # Conventional-data compatibility; flying-wing fuselages are sequence payloads.
    COMPONENT_FUSELAGE: OBB_GEOMETRY_SIZE,
    COMPONENT_ENGINE: OBB_GEOMETRY_SIZE,
}


def component_name(component):
    if component not in COMPONENT_NAMES:
        raise ValueError(f"Unknown component type: {component}")
    return COMPONENT_NAMES[component]


def component_one_hot(component):
    component_name(component)
    return [float(component == index) for index in range(COMPONENT_CLASS_SIZE)]


def validate_ae_rnn_type(rnn_type):
    if rnn_type not in ('rnn', 'gru'):
        raise ValueError(
            f"--ae_rnn_type must be 'rnn' or 'gru', got {rnn_type!r}."
        )
    return rnn_type


def validate_ae_teacher_forcing_schedule(config):
    if not 0.0 <= config.ae_teacher_forcing_p_final <= 1.0:
        raise ValueError('--ae_teacher_forcing_p_final must be in [0, 1].')
    if config.ae_teacher_forcing_ramp_start_epoch < 1:
        raise ValueError('--ae_teacher_forcing_ramp_start_epoch must be at least 1.')
    if config.ae_teacher_forcing_ramp_end_epoch <= config.ae_teacher_forcing_ramp_start_epoch:
        raise ValueError(
            '--ae_teacher_forcing_ramp_end_epoch must exceed '
            '--ae_teacher_forcing_ramp_start_epoch.'
        )


def validate_ae_weight_decay(weight_decay):
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError('--ae_weight_decay must be finite and non-negative.')


def ae_teacher_forcing_probability(epoch, config):
    if epoch < 1:
        raise ValueError(f'epoch must be at least 1, got {epoch}.')
    validate_ae_teacher_forcing_schedule(config)
    ramp_start = config.ae_teacher_forcing_ramp_start_epoch
    ramp_end = config.ae_teacher_forcing_ramp_end_epoch
    if epoch < ramp_start:
        return 1.0
    if epoch >= ramp_end:
        return config.ae_teacher_forcing_p_final
    progress = (epoch - ramp_start) / (ramp_end - ramp_start)
    return 1.0 + progress * (config.ae_teacher_forcing_p_final - 1.0)


def get_args():
    parser = ArgumentParser(description='grass_pytorch')
    parser.add_argument('--box_code_size', type=int, default=13)
    parser.add_argument('--feature_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--symmetry_size', type=int, default=8)
    parser.add_argument('--max_box_num', type=int, default=30)
    parser.add_argument('--max_sym_num', type=int, default=10)

    # VAE parameters
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--vae_lambda_geom', type=float, default=0.4, help='Weight for box geometry MSE loss')
    parser.add_argument('--vae_lambda_cls', type=float, default=1.0, help='Weight for box category cross-entropy loss')
    parser.add_argument('--vae_lambda_sym', type=float, default=0.5, help='Weight for symmetry parameter MSE loss')
    parser.add_argument('--vae_lambda_cat', type=float, default=0.2, help='Weight for node type classification cross-entropy loss')
    parser.add_argument('--kl_weight_target', type=float, default=0.03)
    parser.add_argument('--kl_anneal_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--show_log_every', type=int, default=10)
    parser.add_argument('--save_log', action='store_true', default=False)
    parser.add_argument('--save_log_every', type=int, default=3)
    parser.add_argument('--save_snapshot', action='store_true', default=False)
    parser.add_argument('--save_snapshot_every', type=int, default=5)
    parser.add_argument('--no_plot', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=.001)

    # Deterministic autoencoder parameters.  The AE does not use the VAE
    # sampler, KL divergence, or sample decoder.
    parser.add_argument('--ae_epochs', type=int, default=250)
    parser.add_argument('--ae_batch_size', type=int, default=16)
    parser.add_argument('--ae_lr', type=float, default=8e-4)
    parser.add_argument('--ae_weight_decay', type=float, default=AE_WEIGHT_DECAY)
    parser.add_argument('--ae_lr_decay_factor', type=float, default=0.6)
    parser.add_argument('--ae_lr_decay_patience', type=int, default=8)
    parser.add_argument('--ae_lr_min', type=float, default=1e-6)
    parser.add_argument('--ae_validation_fraction', type=float, default=0.1)
    parser.add_argument('--ae_seed', type=int, default=0)
    parser.add_argument(
        '--overfit', action='store_true', default=False,
        help='Train the section AE on one deterministic, two-aircraft diagnostic subset.',
    )
    parser.add_argument('--ae_gradient_clip', type=float, default=1.0)
    parser.add_argument('--ae_log_every', type=int, default=1)
    parser.add_argument('--ae_checkpoint_dir', type=str, default='models/autoencoder')
    parser.add_argument(
        '--ae_teacher_forcing_p_final', type=float,
        default=AE_TEACHER_FORCING_P_FINAL,
        help='Final probability of feeding the target previous section during scheduled sampling.',
    )
    parser.add_argument(
        '--ae_teacher_forcing_ramp_start_epoch', type=int,
        default=AE_TEACHER_FORCING_RAMP_START_EPOCH,
        help='First 1-based epoch of the linear teacher-forcing probability ramp.',
    )
    parser.add_argument(
        '--ae_teacher_forcing_ramp_end_epoch', type=int,
        default=AE_TEACHER_FORCING_RAMP_END_EPOCH,
        help='Last 1-based epoch of the linear teacher-forcing probability ramp.',
    )
    parser.add_argument(
        '--ae_rnn_type', choices=('rnn', 'gru'), default=AE_RNN_TYPE,
        help='Recurrent cell used by both sequence encoders and autoregressive decoders.',
    )
    parser.add_argument(
        '--section_ae_checkpoint_dir', type=str, default='models/section_autoencoder',
        help='Directory for independent wing and fuselage section-AE checkpoints.',
    )
    parser.add_argument(
        '--ae_section_pretrained_checkpoint_dir', type=str, default='',
        help='Optional directory containing final last_wing.pt and last_fuselage.pt for joint-AE initialization.',
    )

    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='data')
    from project_paths import AIRCRAFT_DATASET_SPECS
    parser.add_argument(
        '--structured_data_paths',
        nargs='+',
        default=[
            str(AIRCRAFT_DATASET_SPECS[layout]['dataset'])
            for layout in ('flying_wing', 'conventional_canard')
        ],
    )
    parser.add_argument('--legacy_data', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
