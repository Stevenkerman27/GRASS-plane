# CST Airfoil Pipeline

The airfoil pipeline has two stages:

1. `foildata/manage_foildata.py` converts raw Selig `.dat` files to processed,
   normalized coordinates.
2. `data/precompute_airfoil_codes.py` fits and caches one CST code per
   processed airfoil.

`util.py` and `cst_airfoil_codec.py` are the authoritative configuration and
codec implementations.

## Processed coordinates

The source directory is `foildata/coord_seligFmt/`; the output directory is
`foildata/processed_foil/`. Processing filters the configured exclusion list,
resamples to `AIRFOIL_DEFAULT_OUTPUT_POINTS` (`100`) using
`AIRFOIL_DEFAULT_POINT_DENSITY_BETA` (`1.3`), normalizes the leading edge to
`(0, 0)`, and places the trailing-edge midpoint at `x=1`.

The point order is upper-surface trailing edge to leading edge, followed by
lower-surface leading edge to trailing edge. Re-running the manager removes
stale processed `.dat` files that were not regenerated.

## CST code

The default code has 19 values:

```text
[upper_shape(8), lower_shape(8), trailing_edge_thickness, N1, N2]
```

For `xi` in `[0, 1]`:

```text
upper =  xi * t_TE / 2 + xi^N1 * (1-xi)^N2 * S_upper(xi)
lower = -xi * t_TE / 2 + xi^N1 * (1-xi)^N2 * S_lower(xi)
```

Both surfaces share the fixed leading edge `(0, 0)`. `N1` and `N2` are fit
per airfoil and must be strictly greater than
`util.CST_MIN_CLASS_FUNCTION_EXPONENT` (`0.05`). The initial values in
`CST_FIT_CONFIG` are optimization starting points, not forced output values.

`cst_airfoil_codec.py` owns pack, unpack, decode, and fit behavior. Callers
should derive dimensions from `util.CST_AIRFOIL_CODE_SIZE` rather than repeat
the 19D layout.

## Cache and diagnostics

Run the cache builder in the `myml` environment:

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\precompute_airfoil_codes.py
```

Default artifacts stay under `data/`:

```text
data/cst_airfoil_code_cache.pt
data/cst_fit_report.json
data/airfoil_fit_visualizations/
```

The cache records the source path, CST configuration, code, and fit metrics.
Non-default coefficient counts receive a `_cN` suffix on all three artifact
names. The dataset converter fails fast when a required code is missing.
