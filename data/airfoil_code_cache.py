"""Persistent, content-addressed CPU cache for processed-airfoil Bezier codes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

import airfoil_codec
import util


CACHE_SCHEMA = "processed_airfoil_code_cache_v1"
CACHE_REPLACE_RETRY_COUNT = 12
CACHE_REPLACE_RETRY_SECONDS = 0.25


def sha256_file(path):
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"File does not exist: {source_path}")
    digest = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def airfoil_identity(airfoil_path):
    source_path = Path(airfoil_path).resolve()
    coord_norm_path = airfoil_codec.DEFAULT_COORD_NORM_PATH.resolve()
    return {
        "encoding_version": airfoil_codec.PROCESSED_AIRFOIL_ENCODING_VERSION,
        "source_sha256": sha256_file(source_path),
        "coord_norm_sha256": sha256_file(coord_norm_path),
        "fit_config": airfoil_codec.optimized_split_surface_fit_config(),
    }


def cache_key(identity):
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def empty_cache():
    return {"schema": CACHE_SCHEMA, "entries": {}}


def validate_cache(cache):
    if not isinstance(cache, dict):
        raise TypeError(f"Airfoil cache must be dict, got {type(cache).__name__}")
    if cache.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"Airfoil cache schema must be {CACHE_SCHEMA!r}")
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("Airfoil cache entries must be a dict")
    for key, entry in entries.items():
        if not isinstance(key, str):
            raise TypeError("Airfoil cache keys must be strings")
        for required_key in ("identity", "source_airfoil", "code"):
            if required_key not in entry:
                raise KeyError(f"Airfoil cache entry {key} missing {required_key}")
        if cache_key(entry["identity"]) != key:
            raise ValueError(f"Airfoil cache entry {key} has an inconsistent identity")
        if len(entry["code"]) != util.AIRFOIL_BEZIER_CODE_SIZE:
            raise ValueError(
                f"Airfoil cache entry {key} has code length {len(entry['code'])}, "
                f"expected {util.AIRFOIL_BEZIER_CODE_SIZE}"
            )


def load_cache(cache_path):
    path = Path(cache_path)
    if not path.exists():
        return empty_cache()
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_cache(cache)
    return cache


def save_cache(cache, cache_path):
    validate_cache(cache)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(cache, temporary_path)
    for attempt in range(CACHE_REPLACE_RETRY_COUNT):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == CACHE_REPLACE_RETRY_COUNT - 1:
                raise
            time.sleep(CACHE_REPLACE_RETRY_SECONDS)


def lookup_code(cache, airfoil_path):
    identity = airfoil_identity(airfoil_path)
    entry = cache["entries"].get(cache_key(identity))
    if entry is None:
        return None
    return list(entry["code"])


def fit_airfoil_code_cpu(airfoil_path, threads_per_worker):
    if threads_per_worker <= 0:
        raise ValueError(f"threads_per_worker must be positive, got {threads_per_worker}")
    torch.set_num_threads(threads_per_worker)
    source_path = Path(airfoil_path).resolve()
    identity = airfoil_identity(source_path)
    fitted = airfoil_codec.encode_processed_airfoil_dat(source_path, device="cpu")
    code = fitted["code"].detach().to(device="cpu", dtype=torch.float32).tolist()
    if len(code) != util.AIRFOIL_BEZIER_CODE_SIZE:
        raise ValueError(
            f"Airfoil code for {source_path} has {len(code)} values, "
            f"expected {util.AIRFOIL_BEZIER_CODE_SIZE}"
        )
    return {
        "key": cache_key(identity),
        "entry": {
            "identity": identity,
            "source_airfoil": str(source_path),
            "code": code,
        },
    }


def store_fitted_entry(cache, fitted_entry):
    key = fitted_entry["key"]
    entry = fitted_entry["entry"]
    if cache_key(entry["identity"]) != key:
        raise ValueError("Fitted airfoil cache entry has an inconsistent identity")
    cache["entries"][key] = entry


def missing_airfoil_paths(cache, airfoil_paths):
    missing = []
    for airfoil_path in airfoil_paths:
        if lookup_code(cache, airfoil_path) is None:
            missing.append(Path(airfoil_path).resolve())
    return missing
