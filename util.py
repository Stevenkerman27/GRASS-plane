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
ENGINE_GEOMETRY_SIZE = OBB_GEOMETRY_SIZE
AIRFOIL_DEFAULT_OUTPUT_POINTS = 100
AIRFOIL_DEFAULT_POINT_DENSITY_BETA = 1.3
AIRFOIL_TRAILING_EDGE_X = 1.0
AIRFOIL_LEADING_EDGE_X = 0.0
COMPONENT_CLASS_SIZE = 3
STANDARD_OBB_BOX_SIZE = OBB_GEOMETRY_SIZE + COMPONENT_CLASS_SIZE
AIRFOIL_UPPER_SURFACE = 'upper'
AIRFOIL_LOWER_SURFACE = 'lower'

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
MIN_SECTION_COUNT, MAX_SECTION_COUNT = SECTION_COUNT_RANGE
SEQUENCE_RNN_LAYERS = 1
COMPONENT_SECTION_SIZES = {
    COMPONENT_FUSELAGE: FUSELAGE_SECTION_SIZE,
    COMPONENT_WING: CST_AIRFOIL_CODE_SIZE + 5,
}
WING_LOSS_WEIGHTS = {
    'position': 1.0,
    'chord': 1.0,
    'twist': 1.0,
    'cst_code': 1.0,
    'decoded_curve': 1.0,
    'section_count': 1.0,
}
FUSELAGE_LOSS_WEIGHTS = {
    'position': 1.0,
    'size': 1.0,
    'section_count': 1.0,
}

COMPONENT_GEOMETRY_SIZES = {
    # Conventional-data compatibility; flying-wing fuselages are sequence payloads.
    COMPONENT_FUSELAGE: OBB_GEOMETRY_SIZE,
    COMPONENT_ENGINE: ENGINE_GEOMETRY_SIZE,
}


def component_name(component):
    if component not in COMPONENT_NAMES:
        raise ValueError(f"Unknown component type: {component}")
    return COMPONENT_NAMES[component]


def component_one_hot(component):
    component_name(component)
    return [float(component == index) for index in range(COMPONENT_CLASS_SIZE)]


def section_mask(section_count, device=None):
    """Derive the only valid padding mask from a tensor of section counts."""
    import torch

    count = torch.as_tensor(section_count, device=device, dtype=torch.long).reshape(-1)
    if torch.any(count < MIN_SECTION_COUNT) or torch.any(count > MAX_SECTION_COUNT):
        raise ValueError(
            f'section_count must be in [{MIN_SECTION_COUNT}, {MAX_SECTION_COUNT}], '
            f'got {count.tolist()}'
        )
    return torch.arange(MAX_SECTION_COUNT, device=count.device).unsqueeze(0) < count.unsqueeze(1)


def get_args():
    parser = ArgumentParser(description='grass_pytorch')
    parser.add_argument('--box_code_size', type=int, default=13)
    parser.add_argument('--feature_size', type=int, default=80)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--symmetry_size', type=int, default=8)
    parser.add_argument('--max_box_num', type=int, default=30)
    parser.add_argument('--max_sym_num', type=int, default=10)

    # GAN Hyperparameters
    parser.add_argument('--alpha1', type=float, default=0.05, help='Weight for VAE reconstruction loss in GAN training')
    parser.add_argument('--alpha2', type=float, default=0.05, help='Weight for VAE KL divergence loss in GAN training')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Weight for gradient penalty in WGAN-GP')
    parser.add_argument('--gan_lambda_geom', type=float, default=0.4, help='Weight for box geometry MSE loss')
    parser.add_argument('--gan_lambda_cls', type=float, default=1.0, help='Weight for box category cross-entropy loss')
    parser.add_argument('--gan_lambda_sym', type=float, default=0.5, help='Weight for symmetry parameter MSE loss')
    parser.add_argument('--gan_lambda_cat', type=float, default=0.2, help='Weight for node type classification cross-entropy loss')
    parser.add_argument('--gan_lr', type=float, default=1e-4, help='Learning rate for GAN optimizers')
    parser.add_argument('--gan_beta1', type=float, default=0.0, help='Beta1 for GAN Adam optimizer')
    parser.add_argument('--gan_beta2', type=float, default=0.9, help='Beta2 for GAN Adam optimizer')
    parser.add_argument('--n_critic', type=int, default=1, help='Number of discriminator updates per generator update')
    parser.add_argument('--gan_k_candidates', type=int, default=5, help='Number of structure candidates for each noise vector in G step')
    parser.add_argument('--gan_epochs', type=int, default=100, help='Number of epochs for GAN training')
    parser.add_argument('--gan_batch_size', type=int, default=10, help='Batch size for GAN training')
    parser.add_argument('--gan_temperature', type=float, default=1.0, help='Temperature for categorical sampling of GAN candidate structures')

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

    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--structured_data_path', type=str, default='data/flying_wing_dataset/flying_wing_dataset.pt')
    parser.add_argument('--legacy_data', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
