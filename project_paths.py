"""Authoritative filesystem locations shared by project compatibility shims."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_AIRCRAFT_TOOLS_ROOT = PROJECT_ROOT.parent / "aircraft-tools"
DATA_DIR = PROJECT_ROOT / 'data'
FLYING_WING_DATASET_DIR = DATA_DIR / 'flying_wing_dataset'
CONVENTIONAL_CANARD_DATASET_DIR = DATA_DIR / 'conventional_canard_dataset'
AIRCRAFT_DATASET_SPECS = {
    'flying_wing': {
        'label': 'Flying-Wing',
        'directory': FLYING_WING_DATASET_DIR,
        'dataset': FLYING_WING_DATASET_DIR / 'flying_wing_dataset.pt',
    },
    'conventional_canard': {
        'label': 'Conventional/Canard',
        'directory': CONVENTIONAL_CANARD_DATASET_DIR,
        'dataset': CONVENTIONAL_CANARD_DATASET_DIR / 'conventional_canard_dataset.pt',
    },
}
