# Multi-Layout OBB Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the dataset generator to support Conventional, Canard, and Flying Wing layouts with diverse engine and tail configurations using a component-based builder and OBB Tree Assembler.

**Architecture:** A `TreeAssembler` will manage the push and adjacency/symmetry operations of OBBs. Specific `build_*` functions will generate 13D OBBs for individual parts. `build_conventional()`, `build_canard_layout()`, and `build_flying_wing()` will act as layout templates that orchestrate the assembly using the builder.

**Tech Stack:** Python, Numpy, Scipy

---

### Task 1: Create TreeAssembler and Component Builders

**Files:**
- Modify: `data/generate_dataset.py`

- [ ] **Step 1: Define TreeAssembler class**
Add the `TreeAssembler` class at the top of the file after the constants.

```python
class TreeAssembler:
    def __init__(self):
        self.boxes = []
        self.ops = []
        self.syms = []

    def push_box(self, box):
        self.boxes.append(box)
        self.ops.append(BOX_OP)
        return len(self.boxes) - 1

    def apply_adj(self):
        self.ops.append(ADJ_OP)

    def apply_sym(self, sym_vector):
        self.syms.append(sym_vector)
        self.ops.append(SYM_OP)
```

- [ ] **Step 2: Implement build_fuselage and build_engine**
Add base component generators.

```python
def build_fuselage():
    L = random.uniform(0.8, 1.5)
    H = random.uniform(0.05, 0.15)
    W = random.uniform(0.05, 0.15)
    return [0, 0, 0, L, 0, 0, W, H, W, H, 1, 0, 0], L, H, W

def build_engine(x, y, z):
    length = random.uniform(0.15, 0.25)
    width = random.uniform(0.04, 0.08)
    return [x - length/2, y, z, x + length/2, y, z, width, width, width, width, 0, 0, 1]
```

- [ ] **Step 3: Implement build_wing**

```python
def build_wing(x_root, y_root, z_root, span_half, root_chord, taper, dihedral, sweep):
    tip_chord = root_chord * taper
    thick = random.uniform(0.02, 0.05)
    tip_thick = thick * random.uniform(0.3, 0.9)
    x2 = x_root + sweep
    y2 = y_root + span_half
    z2 = z_root + dihedral
    return [x_root, y_root, z_root, x2, y2, z2, root_chord, thick, tip_chord, tip_thick, 0, 1, 0]
```

### Task 2: Implement Layout Templates

**Files:**
- Modify: `data/generate_dataset.py`

- [ ] **Step 1: Implement build_canard_layout and build_flying_wing**

```python
def build_canard_layout(assembler):
    fuse_box, L_fuse, H_fuse, W_fuse = build_fuselage()
    assembler.push_box(fuse_box)
    
    # Canard
    canard_span = random.uniform(0.1, 0.25)
    canard_box = build_wing(0.1 * L_fuse, W_fuse/2, 0, canard_span, 0.1, 0.5, 0, 0.05)
    assembler.push_box(canard_box)
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj()
    
    # Main Wing (Aft)
    wing_span = random.uniform(0.4, 0.8)
    wing_box = build_wing(0.6 * L_fuse, W_fuse/2, 0, wing_span, 0.3, 0.4, 0, 0.3)
    assembler.push_box(wing_box)
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj()
    
    # V-Tail only
    vtail_box = build_wing(0.9 * L_fuse, 0, H_fuse/2, random.uniform(0.15, 0.3), 0.15, 0.5, 0, 0.1)
    # V-Tail orientation trick: for vertical, it spans along Z
    vtail_box[1] = 0 # y1
    vtail_box[2] = H_fuse/2 # z1
    vtail_box[4] = 0 # y2
    vtail_box[5] = H_fuse/2 + random.uniform(0.15, 0.3) # z2
    assembler.push_box(vtail_box)
    assembler.apply_adj()

def build_flying_wing(assembler):
    # Short central body
    L = random.uniform(0.3, 0.6)
    W = random.uniform(0.2, 0.4)
    fuse_box = [0, 0, 0, L, 0, 0, W, 0.1, W, 0.1, 1, 0, 0]
    assembler.push_box(fuse_box)
    
    # Massive wings
    wing_box = build_wing(0.1 * L, W/2, 0, random.uniform(0.6, 1.2), L*0.8, 0.2, 0.05, 0.6)
    assembler.push_box(wing_box)
    
    # Wing-mounted engine
    eng_box = build_engine(0.2 * L + 0.3, W/2 + 0.2, -0.05)
    assembler.push_box(eng_box)
    assembler.apply_adj()
    
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj()
```

- [ ] **Step 2: Implement build_conventional layout**

```python
def build_conventional(assembler):
    fuse_box, L_fuse, H_fuse, W_fuse = build_fuselage()
    assembler.push_box(fuse_box)
    
    # Main Wing
    wing_span = random.uniform(0.4, 0.8)
    wing_box = build_wing(0.3 * L_fuse, W_fuse/2, 0, wing_span, 0.25, 0.5, 0.05, 0.2)
    assembler.push_box(wing_box)
    
    # Wing-mounted engine
    eng_box = build_engine(0.3 * L_fuse + 0.1, W_fuse/2 + wing_span*0.4, -0.05)
    assembler.push_box(eng_box)
    assembler.apply_adj()
    
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj()
    
    # T-Tail configuration
    # H-Tail
    htail_box = build_wing(0.9 * L_fuse, 0, H_fuse/2 + 0.25, random.uniform(0.15, 0.3), 0.1, 0.6, 0, 0.05)
    assembler.push_box(htail_box)
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # V-Tail portion of T-Tail
    vtail_box = build_wing(0.9 * L_fuse, 0, H_fuse/2, 0.25, 0.15, 0.6, 0, 0.1)
    vtail_box[1] = 0; vtail_box[2] = H_fuse/2; vtail_box[4] = 0; vtail_box[5] = H_fuse/2 + 0.25
    assembler.push_box(vtail_box)
    assembler.apply_adj() # Connect H-Tail pair to V-Tail
    
    assembler.apply_adj() # Connect full T-Tail to Fuselage
```

### Task 3: Refactor Main Generation and Scaling

**Files:**
- Modify: `data/generate_dataset.py`

- [ ] **Step 1: Replace generate_aircraft logic**
Replace the body of `generate_aircraft` to use the templates. Retain the scaling code at the bottom.

```python
def generate_aircraft():
    assembler = TreeAssembler()
    layout_type = random.choice(['conventional', 'canard', 'flying_wing'])
    
    if layout_type == 'conventional':
        build_conventional(assembler)
    elif layout_type == 'canard':
        build_canard_layout(assembler)
    else:
        build_flying_wing(assembler)
        
    boxes, ops, syms = assembler.boxes, assembler.ops, assembler.syms

    # --- Isotropic Scaling per instance ---
    # [Keep the existing scaling loop exactly as it was]
    max_val = 0.0
    for box in boxes:
        for val in box[:10]:
            if abs(val) > max_val:
                max_val = abs(val)

    if max_val > 0:
        for i in range(len(boxes)):
            for j in range(10):
                boxes[i][j] = (boxes[i][j] / max_val) * 2.0 - 1.0
            for j in range(10, 13):
                boxes[i][j] = boxes[i][j] * 2.0 - 1.0
        
        for i in range(len(syms)):
            sym_type = round(syms[i][0])
            if sym_type == 0:
                for j in range(1, 4):
                    syms[i][j] = (syms[i][j] / max_val)
            for j in range(4, 7):
                syms[i][j] = (syms[i][j] / max_val) * 2.0 - 1.0

    return boxes, ops, syms
```

- [ ] **Step 2: Verification Run**
Generate a few samples to ensure shape stability.

Run: `python data/generate_dataset.py`
Expected: `Generated dataset with 300 aircraft.`

Run: `python data/visualize_dataset.py`
Expected: A popup window rendering the different layouts successfully.
