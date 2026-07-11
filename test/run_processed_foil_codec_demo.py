import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import airfoil_codec
import util


DEFAULT_AIRFOIL_PATH = (
    airfoil_codec.DEFAULT_NN_ROOT / "foildata" / "processed_foil" / "_falcon.dat"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "airfoil_codec_demo"


def plot_fit(raw_target_points, decoded_curve, control_points, save_path):
    target_np = raw_target_points.detach().cpu().numpy()
    curve_np = decoded_curve.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(target_np[:, 0], target_np[:, 1], 'k.', label='Original .dat', markersize=3)
    plt.plot(curve_np[:, 0], curve_np[:, 1], 'r-', label='Decoded Bezier', linewidth=2)

    colors = {
        util.AIRFOIL_UPPER_SURFACE: 'tab:green',
        util.AIRFOIL_LOWER_SURFACE: 'tab:blue',
    }
    for surface_name in (util.AIRFOIL_UPPER_SURFACE, util.AIRFOIL_LOWER_SURFACE):
        cp = control_points[surface_name].detach().cpu().numpy()
        plt.plot(
            cp[:, 0],
            cp[:, 1],
            'x--',
            color=colors[surface_name],
            label=f'{surface_name.title()} control points',
            alpha=0.7,
            markersize=5,
        )

    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Processed Airfoil Bezier Encode/Decode')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Encode/decode one processed_foil airfoil with GRASS airfoil_codec")
    parser.add_argument("--airfoil", type=Path, default=DEFAULT_AIRFOIL_PATH)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    result = airfoil_codec.encode_processed_airfoil_dat(
        args.airfoil,
        device=args.device,
        verbose=args.verbose,
    )
    encode_config = result['fit_config']
    print(f"Encoding airfoil: {result['source_airfoil']}")
    print(f"Using coord_norm: {result['coord_norm_path']}")
    print(f"Device: {result['device']}")
    print(f"Iterations: {encode_config['iterations']}")

    decoded_norm_curve = airfoil_codec.decode_airfoil_code(
        result['code'],
        num_output_points=result['raw_target_points'].shape[0],
        point_density_beta=encode_config['point_density_beta'],
    ).squeeze(0)
    decoded_curve = airfoil_codec.denormalize_points(decoded_norm_curve, result['coord_norm'])
    control_points = {
        surface_name: airfoil_codec.denormalize_points(points.squeeze(0), result['coord_norm'])
        for surface_name, points in result['control_points'].items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result['source_airfoil']).stem
    encoded_path = args.output_dir / f"{stem}_encoded_bezier.pt"
    plot_path = args.output_dir / f"{stem}_bezier_fit.png"

    torch.save({
        'source_airfoil': result['source_airfoil'],
        'coord_norm_path': result['coord_norm_path'],
        'fit_config': encode_config,
        'code': result['code'],
        'control_points': result['control_points'],
        'weights': result['weights'],
        'metrics': result['metrics'],
    }, encoded_path)

    plot_fit(
        result['raw_target_points'].squeeze(0),
        decoded_curve,
        control_points,
        plot_path,
    )

    print(f"Code shape: {tuple(result['code'].shape)}")
    print(
        f"Final normalized MAE: {result['metrics']['mae']:.8f}, "
        f"MSE: {result['metrics']['mse']:.8f}, "
        f"Leading-edge MAE: {result['metrics']['leading_edge_mae']:.8f}, "
        f"Max point error: {result['metrics']['max_point_error']:.8f}"
    )
    print(f"Saved encoded parameters to: {encoded_path}")
    print(f"Saved visualization to: {plot_path}")


if __name__ == "__main__":
    main()
