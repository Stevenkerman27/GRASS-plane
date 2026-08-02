# Aircraft Schema and Codec

This is the current contract for generated structured datasets and the
deterministic aircraft autoencoder. The implementation is split between
`util.py`, `grassdata.py`, `section_parameter_codec.py`, `grassmodel.py`, and
`section_autoencoder.py`.

## Fixed component payloads

Only `fuselage` and `wing` are sequence components. Their dimensions and
station counts come from `util.py`:

| sequence type | `z_global` | `z_section` | stations |
| --- | ---: | ---: | ---: |
| `wing` | 5 | 24 | 6 |
| `fuselage` | 6 | 6 | 10 |

The typed `.pt` leaf box stores:

```text
component
sequence_type
z_global
z_section
```

`z_section` is fixed-length. It is not a variable-length sequence and does
not use padding, masks, EOS, or a serialized `section_count` field. The
validator may accept a redundant `section_count` only when it equals the
configured fixed count; it is discarded after validation. Sequence leaves
must not define the old `geometry` or `sections` input keys.

### Wing fields

```text
z_global  = [x_LE, y_LE, z_LE, root_chord, half_span]
z_section = [span_fraction, chord_fraction, twist_deg,
             dihedral_deg, sweep_deg, CST_code(19)] * 6
```

`z_global` uses metres. Span and chord are normalized to the root; angles are
stored in degrees. Station zero is the root. Dihedral and sweep are local
segment angles derived from consecutive leading-edge positions. The 19D CST
code is `[upper_shape(8), lower_shape(8), trailing_edge_thickness, N1, N2]`.

### Fuselage fields

```text
z_global  = [x_nose, y_center, z_center, length, width, height]
z_section = [x_fraction, width_fraction, height_fraction,
             Super_MaxWidthLoc, Super_M, Super_N] * 10
```

Station `x_fraction` is exactly `i / 9`. Width and height fractions are
positive. Super-ellipse ranges and adjacent-change limits are defined in
`data/aircraft_dataset_common.py`.

## Tree representation

`boxes`, `ops`, and `syms` describe a postorder GRASS tree:

| op | meaning |
| ---: | --- |
| `0` | `BOX` leaf |
| `1` | `ADJ`, combine two child nodes |
| `2` | `SYM`, mirror one generator node |

The current datasets use reflection across the XZ plane:

```text
[1, 0, 1, 0, 0, 0, 0, 0]
```

The flying-wing tree is fuselage adjacent to a mirrored main wing. The
conventional/canard tree adds a mirrored auxiliary wing. Layout roles do not
create new component or decoder types: every wing uses the same `wing` codec.

## Fixed section autoencoder

`SectionEncoder` applies two `Conv1d` layers with 64 channels to the fixed
section matrix, concatenates the normalized global vector, then uses a 200D
hidden layer and a `tanh` feature output. `SectionDecoder` maps the feature
through a 200D hidden layer and reconstructs the fixed global and section
payload in one pass. The independent section AE and full tree AE use the same
modules.

`SectionParameterCodec` is the only model-space transform. It fits means and
standard deviations from the training split and stores them in checkpoints.
Positive values are log-transformed before normalization:

- wing global `chord/span`, section chord fraction, and CST `N1/N2`;
- fuselage global `length/width/height` and section width/height fractions.

The CST exponents must be strictly greater than
`CST_MIN_CLASS_FUNCTION_EXPONENT` (`0.05`). Constant dimensions use standard
deviation `1`. Missing or mismatched checkpoint statistics are errors.

The fixed section losses are:

```text
wing      = global MSE + section geometry MSE + CST code MSE
fuselage  = global MSE + section shape MSE
```

Tree-level losses additionally cover component class, node type, and symmetry
parameters. Their weights are defined only in `util.py`.

## Training compatibility

`train_section_autoencoder.py` trains the two component codecs after splitting
whole aircraft with `ae_seed`; `train_autoencoder.py` trains the recursive
tree AE on the combined typed datasets. `--overfit` writes under the
`overfit/` subdirectory. The final independent checkpoints are
`last_wing.pt` and `last_fuselage.pt`.

The current fixed-section datasets and checkpoints are incompatible with the
old variable-length RNN/GRU, padding/EOS, legacy OBB, VAE, and GAN contracts.
Do not repair those formats by truncating, padding, or renaming fields.
