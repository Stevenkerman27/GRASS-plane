"""Fit every processed airfoil once using CPU worker processes and persist the codes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cst_airfoil_codec
if __package__:
    from data import cst_airfoil_code_cache
else:
    import cst_airfoil_code_cache


DEFAULT_AIRFOIL_DIR = REPO_ROOT / "foildata" / "processed_foil"
DEFAULT_WORKERS = 16
DEFAULT_THREADS_PER_WORKER = 1
DEFAULT_TOP_ERROR_COUNT = 10
DEFAULT_VISUALIZATION_DIR = DATA_DIR / 'airfoil_fit_visualizations'
DEFAULT_REPORT_PATH = DATA_DIR / 'cst_fit_report.json'
FIT_REPORT_SCHEMA = 'cst_fit_report_v1'


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CPU CST codes for all processed airfoils.")
    parser.add_argument("--airfoil-dir", type=Path, default=DEFAULT_AIRFOIL_DIR)
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cst_airfoil_code_cache.pt")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threads-per-worker", type=int, default=DEFAULT_THREADS_PER_WORKER)
    parser.add_argument("--top-error-count", type=int, default=DEFAULT_TOP_ERROR_COUNT)
    parser.add_argument("--visualization-dir", type=Path, default=DEFAULT_VISUALIZATION_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def load_airfoil_paths(airfoil_dir):
    if not airfoil_dir.is_dir():
        raise NotADirectoryError(f"Airfoil directory does not exist: {airfoil_dir}")
    paths = sorted(airfoil_dir.glob("*.dat"))
    if not paths:
        raise ValueError(f"Airfoil directory contains no .dat files: {airfoil_dir}")
    return paths


def fit_missing_airfoils(cache, missing_paths, workers, threads_per_worker, cache_path):
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if threads_per_worker <= 0:
        raise ValueError(f"threads_per_worker must be positive, got {threads_per_worker}")
    if not missing_paths:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(cst_airfoil_code_cache.fit_airfoil_code_cpu, path, threads_per_worker): path
            for path in missing_paths
        }
        completed_count = 0
        try:
            for future in as_completed(futures):
                source_path = futures[future]
                fitted_entry = future.result()
                cst_airfoil_code_cache.store_fitted_entry(cache, fitted_entry)
                completed_count += 1
                cst_airfoil_code_cache.save_cache(cache, cache_path)
                print(
                    f"[{completed_count}/{len(missing_paths)}] fitted {source_path.name}",
                    flush=True,
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise


def cached_entries_for_paths(cache, paths):
    entries = []
    for path in paths:
        entry = cst_airfoil_code_cache.lookup_entry(cache, path)
        if entry is None:
            raise RuntimeError(f'No cached CST entry for {path}')
        entries.append((path, entry))
    return entries


def top_error_entries(entries, count):
    if count <= 0:
        raise ValueError(f'top-error-count must be positive, got {count}')
    return sorted(
        entries,
        key=lambda item: (-item[1]['metrics']['max_point_error'], item[0].name),
    )[:count]


def print_top_errors(entries):
    print(f'Largest CST max-point errors: {len(entries)}', flush=True)
    for rank, (path, entry) in enumerate(entries, start=1):
        metrics = entry['metrics']
        print(
            f'  {rank:02d}. {path.name}: '
            f'max_point_error={metrics["max_point_error"]:.8f}, '
            f'mae={metrics["mae"]:.8f}',
            flush=True,
        )


def save_fit_visualizations(entries, visualization_dir):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(visualization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, (path, entry) in enumerate(entries, start=1):
        target_points = cst_airfoil_codec.load_dat(path)
        decoded_points = cst_airfoil_codec.decode_cst_airfoil_code(
            torch.tensor(entry['code'], dtype=torch.float32)
        ).squeeze(0)
        target = target_points.numpy()
        decoded = decoded_points.numpy()
        metrics = entry['metrics']
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(target[:, 0], target[:, 1], color='black', linewidth=1.5, label='Target')
        axis.plot(decoded[:, 0], decoded[:, 1], color='tab:red', linewidth=1.0, label='CST decoded')
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.grid(True, linewidth=0.4)
        axis.legend()
        axis.set_title(
            f'{path.name} | max point error {metrics["max_point_error"]:.6f} | '
            f'MAE {metrics["mae"]:.6f}'
        )
        figure.tight_layout()
        output_path = output_dir / f'{rank:02d}_{path.stem}.png'
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        print(f'  saved {output_path}', flush=True)


def metric_summary(entries):
    if not entries:
        raise ValueError('Cannot summarize metrics for an empty entry list')
    return {
        metric_name: {
            'min': min(entry['metrics'][metric_name] for _, entry in entries),
            'mean': sum(entry['metrics'][metric_name] for _, entry in entries) / len(entries),
            'max': max(entry['metrics'][metric_name] for _, entry in entries),
        }
        for metric_name in cst_airfoil_codec.CST_FIT_METRIC_KEYS
    }


def build_fit_report(entries, error_entries):
    return {
        'schema': FIT_REPORT_SCHEMA,
        'sample_count': len(entries),
        'metric_summary': metric_summary(entries),
        'largest_max_point_errors': [
            {
                'rank': rank,
                'source_airfoil': str(path.resolve()),
                'metrics': entry['metrics'],
            }
            for rank, (path, entry) in enumerate(error_entries, start=1)
        ],
        'samples': [
            {
                'source_airfoil': str(path.resolve()),
                'metrics': entry['metrics'],
            }
            for path, entry in entries
        ],
    }


def save_fit_report(report, report_path):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    temporary_path.replace(path)


def main():
    args = parse_args()
    paths = load_airfoil_paths(args.airfoil_dir)
    cache = cst_airfoil_code_cache.load_cache(args.cache)
    missing_paths = cst_airfoil_code_cache.missing_airfoil_paths(cache, paths)
    print("Python executable:", sys.executable, flush=True)
    print(f"Airfoil files: {len(paths)}; cached: {len(paths) - len(missing_paths)}; missing: {len(missing_paths)}", flush=True)
    fit_missing_airfoils(cache, missing_paths, args.workers, args.threads_per_worker, args.cache)
    entries = cached_entries_for_paths(cache, paths)
    errors = top_error_entries(entries, args.top_error_count)
    print_top_errors(errors)
    save_fit_visualizations(errors, args.visualization_dir)
    save_fit_report(build_fit_report(entries, errors), args.report)
    print(f"CST fit report: {args.report}", flush=True)
    print(f"Airfoil cache ready: {args.cache}", flush=True)


if __name__ == "__main__":
    main()
