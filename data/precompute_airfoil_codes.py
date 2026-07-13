"""Fit every processed airfoil once using CPU worker processes and persist the codes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import airfoil_code_cache


DEFAULT_AIRFOIL_DIR = Path(r"D:\3D\Projects\ML\NN\foildata\processed_foil")
DEFAULT_WORKERS = 4
DEFAULT_THREADS_PER_WORKER = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CPU Bezier codes for all processed airfoils.")
    parser.add_argument("--airfoil-dir", type=Path, default=DEFAULT_AIRFOIL_DIR)
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "airfoil_code_cache.pt")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threads-per-worker", type=int, default=DEFAULT_THREADS_PER_WORKER)
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
            executor.submit(airfoil_code_cache.fit_airfoil_code_cpu, path, threads_per_worker): path
            for path in missing_paths
        }
        completed_count = 0
        try:
            for future in as_completed(futures):
                source_path = futures[future]
                fitted_entry = future.result()
                airfoil_code_cache.store_fitted_entry(cache, fitted_entry)
                completed_count += 1
                airfoil_code_cache.save_cache(cache, cache_path)
                print(
                    f"[{completed_count}/{len(missing_paths)}] fitted {source_path.name}",
                    flush=True,
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise


def main():
    args = parse_args()
    paths = load_airfoil_paths(args.airfoil_dir)
    cache = airfoil_code_cache.load_cache(args.cache)
    missing_paths = airfoil_code_cache.missing_airfoil_paths(cache, paths)
    print("Python executable:", sys.executable, flush=True)
    print(f"Airfoil files: {len(paths)}; cached: {len(paths) - len(missing_paths)}; missing: {len(missing_paths)}", flush=True)
    fit_missing_airfoils(cache, missing_paths, args.workers, args.threads_per_worker, args.cache)
    print(f"Airfoil cache ready: {args.cache}", flush=True)


if __name__ == "__main__":
    main()
