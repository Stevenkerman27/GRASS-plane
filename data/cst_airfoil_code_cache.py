"""Persistent, content-addressed CPU cache for processed-airfoil CST codes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch

import cst_airfoil_codec
import util


CACHE_SCHEMA = 'processed_airfoil_cst_cache_v3'
CACHE_REPLACE_RETRY_COUNT = 12
CACHE_REPLACE_RETRY_SECONDS = 0.25


def sha256_file(path):
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f'File does not exist: {source_path}')
    digest = hashlib.sha256()
    with source_path.open('rb') as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def airfoil_identity(airfoil_path, shape_coefficient_count=None):
    source_path = Path(airfoil_path).resolve()
    coefficient_count = cst_airfoil_codec.resolve_shape_coefficient_count(
        shape_coefficient_count
    )
    identity = {
        'encoding_version': cst_airfoil_codec.PROCESSED_AIRFOIL_ENCODING_VERSION,
        'source_sha256': sha256_file(source_path),
        'fit_config': dict(util.CST_FIT_CONFIG),
    }
    if coefficient_count != util.CST_SURFACE_SHAPE_COEFFICIENTS:
        identity['shape_coefficient_count'] = coefficient_count
    return identity


def cache_key(identity):
    serialized = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def entry_shape_coefficient_count(entry):
    identity = entry['identity']
    if 'shape_coefficient_count' not in identity:
        return util.CST_SURFACE_SHAPE_COEFFICIENTS
    return cst_airfoil_codec.resolve_shape_coefficient_count(
        identity['shape_coefficient_count']
    )


def empty_cache():
    return {'schema': CACHE_SCHEMA, 'entries': {}}


def validate_cache(cache):
    if not isinstance(cache, dict):
        raise TypeError(f'Airfoil cache must be dict, got {type(cache).__name__}')
    if cache.get('schema') != CACHE_SCHEMA:
        raise ValueError(f'Airfoil cache schema must be {CACHE_SCHEMA!r}')
    entries = cache.get('entries')
    if not isinstance(entries, dict):
        raise TypeError('Airfoil cache entries must be a dict')
    for key, entry in entries.items():
        if not isinstance(key, str):
            raise TypeError('Airfoil cache keys must be strings')
        for required_key in ('identity', 'source_airfoil', 'code', 'metrics'):
            if required_key not in entry:
                raise KeyError(f'Airfoil cache entry {key} missing {required_key}')
        if cache_key(entry['identity']) != key:
            raise ValueError(f'Airfoil cache entry {key} has an inconsistent identity')
        coefficient_count = entry_shape_coefficient_count(entry)
        expected_code_size = cst_airfoil_codec.cst_airfoil_code_size(coefficient_count)
        if len(entry['code']) != expected_code_size:
            raise ValueError(
                f"Airfoil cache entry {key} has code length {len(entry['code'])}, "
                f'expected {expected_code_size}'
            )
        metrics = entry['metrics']
        if not isinstance(metrics, dict):
            raise TypeError(f'Airfoil cache entry {key} metrics must be a dict')
        for metric_name in cst_airfoil_codec.CST_FIT_METRIC_KEYS:
            if metric_name not in metrics:
                raise KeyError(f'Airfoil cache entry {key} missing metric {metric_name!r}')
            metric_value = metrics[metric_name]
            if not isinstance(metric_value, (int, float)) or not math.isfinite(metric_value):
                raise ValueError(
                    f'Airfoil cache entry {key} metric {metric_name!r} must be finite'
                )


def load_cache(cache_path):
    path = Path(cache_path)
    if not path.exists():
        return empty_cache()
    cache = torch.load(path, map_location='cpu', weights_only=True)
    validate_cache(cache)
    return cache


def save_cache(cache, cache_path):
    validate_cache(cache)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(cache, temporary_path)
    for attempt in range(CACHE_REPLACE_RETRY_COUNT):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == CACHE_REPLACE_RETRY_COUNT - 1:
                raise
            time.sleep(CACHE_REPLACE_RETRY_SECONDS)


def lookup_entry(cache, airfoil_path, shape_coefficient_count=None):
    identity = airfoil_identity(airfoil_path, shape_coefficient_count)
    return cache['entries'].get(cache_key(identity))


def lookup_code(cache, airfoil_path, shape_coefficient_count=None):
    entry = lookup_entry(cache, airfoil_path, shape_coefficient_count)
    return None if entry is None else list(entry['code'])


def fit_airfoil_code_cpu(airfoil_path, threads_per_worker, shape_coefficient_count=None):
    if threads_per_worker <= 0:
        raise ValueError(f'threads_per_worker must be positive, got {threads_per_worker}')
    torch.set_num_threads(threads_per_worker)
    source_path = Path(airfoil_path).resolve()
    coefficient_count = cst_airfoil_codec.resolve_shape_coefficient_count(
        shape_coefficient_count
    )
    identity = airfoil_identity(source_path, coefficient_count)
    fitted = cst_airfoil_codec.encode_processed_airfoil_dat(
        source_path, device='cpu', shape_coefficient_count=coefficient_count
    )
    code = fitted['code'].detach().to(device='cpu', dtype=torch.float32).tolist()
    expected_code_size = cst_airfoil_codec.cst_airfoil_code_size(coefficient_count)
    if len(code) != expected_code_size:
        raise ValueError(
            f'Airfoil code for {source_path} has {len(code)} values, '
            f'expected {expected_code_size}'
        )
    return {
        'key': cache_key(identity),
        'entry': {
            'identity': identity,
            'source_airfoil': str(source_path),
            'code': code,
            'metrics': {
                metric_name: float(fitted['metrics'][metric_name])
                for metric_name in cst_airfoil_codec.CST_FIT_METRIC_KEYS
            },
        },
    }


def store_fitted_entry(cache, fitted_entry):
    key = fitted_entry['key']
    entry = fitted_entry['entry']
    if cache_key(entry['identity']) != key:
        raise ValueError('Fitted airfoil cache entry has an inconsistent identity')
    cache['entries'][key] = entry


def missing_airfoil_paths(cache, airfoil_paths, shape_coefficient_count=None):
    missing = []
    for airfoil_path in airfoil_paths:
        if lookup_code(cache, airfoil_path, shape_coefficient_count) is None:
            missing.append(Path(airfoil_path).resolve())
    return missing
