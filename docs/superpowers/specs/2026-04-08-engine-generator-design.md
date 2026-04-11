# Engine Generator Design for Aircraft Dataset (10D OBB)

## Overview
Update the procedural dataset generator (`data/generate_dataset.py`) to include wing-mounted engines. Aircraft will have either 2 or 4 engines (1 or 2 per wing). Engines are represented as 10D OBBs aligned with the free-stream direction (X-axis).

## Geometry (10D OBB)
*   **Alignment:** Engines face forward. The centerline (P1 to P2) is parallel to the Y-axis (spanwise), or X-axis (longitudinal), but based on 10D parsing rules:
    *   To make the engine parallel to the X-axis (free stream), the cross product in `draw3dobb.py` dictates its orientation.
    *   For a cylinder-like engine aligned along X, we can set P1 and P2 along the X-axis: `x1 < x2`, `y1 = y2`, `z1 = z2`. Then chord `L` and thickness `H` define the cross-section (Y and Z dimensions).
    *   *Correction for 10D standard*: Actually, in our 10D format, P1 to P2 is the "span" of the box. If an engine's main length is along X, P1 and P2 should be `[x_front, y, z]` and `[x_back, y, z]`. The chord `L` would then be width (Y), and `H` would be height (Z). Both faces will have the same dimensions: `L1 = L2 = engine_width`, `H1 = H2 = engine_height`.
*   **Dimensions:**
    *   Length (X-axis): `random.uniform(0.1, 0.25)`
    *   Diameter/Width (Y and Z): `random.uniform(0.04, 0.08)`
    *   The engine is placed under the wing, slightly forward of the wing's leading edge at that specific Y-span location.

## Tree Structure (GRASS Syntax)
We construct the right wing and its engine(s) first, combine them, and then mirror the entire assembly.

**Node Sequence (1 Engine per wing):**
1.  `BOX` (Wing)
2.  `BOX` (Engine 1)
3.  `ADJ` (Connect Engine 1 to Wing)
4.  `SYM` (Mirror the [Wing + Engine] assembly across XZ plane)
5.  `ADJ` (Connect the mirrored assembly to the Fuselage)

**Node Sequence (2 Engines per wing):**
1.  `BOX` (Wing)
2.  `BOX` (Engine 1 - Inboard)
3.  `ADJ` (Connect Engine 1 to Wing)
4.  `BOX` (Engine 2 - Outboard)
5.  `ADJ` (Connect Engine 2 to [Wing + Engine 1])
6.  `SYM` (Mirror the [Wing + Engines] assembly across XZ plane)
7.  `ADJ` (Connect the mirrored assembly to the Fuselage)

## Placement Rules
*   **Count:** Randomly choose 1 or 2 engines per side.
*   **Y-Position:**
    *   If 1 engine: Placed between 30% and 50% of the half-span.
    *   If 2 engines: Inboard at 25%-40%, outboard at 55%-75% of the half-span.
*   **X/Z-Position:**
    *   X: Interpolated based on wing sweep at the chosen Y-position, shifted slightly forward.
    *   Z: Interpolated based on wing dihedral, shifted down by the engine's radius + clearance.

## Data Padding
Because the number of nodes per aircraft is now variable (depending on the engine count), the `max_ops_len`, `max_boxes_len`, and `max_syms_len` logic in `generate_dataset.py` will naturally pad the smaller trees to the size of the largest tree in the batch using zero vectors, which is correct for GRASS `MAT` file formatting.
