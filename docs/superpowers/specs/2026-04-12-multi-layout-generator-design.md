# Multi-Layout OBB Generator Design for Aircraft Dataset

## Overview
This design document specifies the architecture and logic for refactoring `data/generate_dataset.py` to support multiple mainstream aircraft layouts (Conventional, Canard, Flying Wing), along with diverse engine placements (nose, wing, tail) and tail types (Conventional, T-tail, V-tail). The generator will assemble these components based on predefined core templates and dynamic strategies, adhering strictly to the GRASS physical hierarchy (ADJ/SYM) to maintain topological correctness for the VAE/GAN networks.

## 1. Core Generator Architecture

The linear generation script will be refactored into a `Component-based Builder` pattern.

### 1.1 Tree Assembler (`TreeAssembler`)
A utility class to abstract away the complexity of pushing `BOX` nodes and correctly inserting `ADJ` and `SYM` operations. This ensures the output `ops` list adheres to the reverse-polish notation (post-order traversal) required by `grassdata.py`.
*   `push_box(box_params)`: Appends `BOX` and returns a reference.
*   `apply_adj()`: Appends an `ADJ` node (implicitly merging the last two active nodes).
*   `apply_sym(sym_params)`: Appends a `SYM` node.

### 1.2 Component Factories
Standalone functions or class methods that generate the 13D parameter arrays `[x1,y1,z1, x2,y2,z2, L1,H1, L2,H2, C_fuse, C_wing, C_eng]` for individual components:
*   `build_fuselage(...)`
*   `build_wing(...)`
*   `build_tail(...)`: Handles regular, T-tail, V-tail geometries.
*   `build_canard(...)`
*   `build_engine(...)`: Handled as cylindrical OBBs.

## 2. Core Layout Templates

The builder will randomly select one of three main templates for each sample. Each template manages its own valid sub-configurations.

### 2.1 Conventional Layout (`build_conventional`)
*   **Main Wing:** Located mid/aft on the fuselage.
*   **Tail:**
    *   *Regular* (Single V-stab + split H-stab)
    *   *T-tail* (H-stab mounted on top of V-stab)
    *   *V-tail* (Two angled stabilizers, no H-stab)
*   **Engines:**
    *   *Wing-mounted* (1 or 2 per side)
    *   *Tail-mounted* (1 center, or 2 side-mounted)
    *   *Nose-mounted* (1 center)
*   **Constraints:** If tail engine is center-mounted, tail cannot be a central Regular/T-tail (must be twin-tail or V-tail to avoid collision).

### 2.2 Canard Layout (`build_canard_layout`)
*   **Main Wing:** Located far aft.
*   **Canard:** Small forward wing near the nose.
*   **Tail:** Vertical stabilizer only (single or twin). No horizontal stabilizer.
*   **Engines:** Wing-mounted or Tail-mounted.

### 2.3 Flying Wing Layout (`build_flying_wing`)
*   **Fuselage/Wing Integration:** Extremely short/wide fuselage blended seamlessly into massive swept wings.
*   **Tail:** None (or very small winglet-style vertical stabilizers at the wingtips).
*   **Engines:** Embedded (approximated as attached flush to the trailing edge) or Wing-mounted.

## 3. Physical Hierarchy (ADJ/SYM Logic)

The topological structure must reflect physical attachment. The `TreeAssembler` will execute sequences such as:

*   **Wing + Wing-Engines:**
    ```
    push(Wing) -> push(Engine) -> apply_adj() -> apply_sym(XZ-plane) -> push(Fuselage) -> apply_adj()
    ```
*   **T-Tail:**
    ```
    push(H-Tail) -> apply_sym() -> push(V-Tail) -> apply_adj() -> [Attach to Fuselage] -> apply_adj()
    ```
*   **V-Tail:**
    ```
    push(V-Tail_half) -> apply_sym(XZ-plane) -> [Attach to Fuselage] -> apply_adj()
    ```

## 4. Normalization and Data Export

The global isotropic scaling logic (scaling all continuous coordinates/dimensions by the maximum absolute value found in the instance) will be preserved from the `2026-04-12-dataset-scaling-design.md` spec to ensure all values remain in the `[-1, 1]` range suitable for the GAN/VAE.

## 5. Implementation Steps

1.  **Refactor Structure:** Create the `TreeAssembler` class in `generate_dataset.py`.
2.  **Component Methods:** Implement parametric builders for fuselage, wing, canard, tail (with variants), and engines.
3.  **Template Logic:** Implement `build_conventional`, `build_canard_layout`, and `build_flying_wing` functions.
4.  **Main Loop:** Update `generate_aircraft()` to randomly dispatch to one of the templates.
5.  **Scaling & Padding:** Maintain the existing instance-level max-scaling and matrix zero-padding for MATLAB export.
6.  **Validation:** Run generation script and verify visual output using `visualize_dataset.py` to ensure ADJ/SYM hierarchies are correctly decoded by `grassmodel.py`.