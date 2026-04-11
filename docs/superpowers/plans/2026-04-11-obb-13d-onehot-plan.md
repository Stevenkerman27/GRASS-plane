# OBB 13D One-Hot Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the OBB parameterization from 10D to 13D by appending a 3D one-hot vector representing the part category (Fuselage, Wing/Tail, Engine) and update the loss functions to handle mixed (geometric + categorical) outputs.

**Architecture:** We will increase the global `box_code_size` to 13. The dataset generator will append one-hot vectors to boxes. The neural network's `BoxDecoder` will apply `Tanh` only to the first 10 dimensions (geometry) and output raw logits for the last 3 dimensions (categorical). The `boxLossEstimator` will combine MSE loss for geometry and CrossEntropy loss for the categories. `draw3dobb.py` will use these categories for semantic coloring.

**Tech Stack:** Python, PyTorch, SciPy

---

### Task 1: Update Global Configuration

**Files:**
- Modify: `util.py`

- [ ] **Step 1: Change `box_code_size` default from 10 to 13**
Modify `util.py`:
```python
    parser.add_argument('--box_code_size', type=int, default=13)
```

- [ ] **Step 2: Commit**
```bash
git add util.py
git commit -m "chore: update box_code_size to 13 for semantic one-hot encoding"
```

---

### Task 2: Update Dataset Generation

**Files:**
- Modify: `data/generate_dataset.py`

- [ ] **Step 1: Append one-hot encoding to each box in `generate_aircraft`**
Modify `generate_aircraft` to append the appropriate `[1, 0, 0]`, `[0, 1, 0]`, or `[0, 0, 1]` to each box format list.

```python
    # ... In Fuselage section ...
    fuse_box = [0, 0, 0, L_fuse, 0, 0, W_fuse, H_fuse, W_fuse, H_fuse, 1, 0, 0]
    
    # ... In Main Wing section ...
    wing_box = [x1_w, y1_w, z1_w, x2_w, y2_w, z2_w, 
                wing_root_chord, wing_root_thick, wing_tip_chord, wing_tip_thick, 0, 1, 0]
                
    # ... In Engines section ...
    eng1_box = [x_e1_center - eng_length/2, y_e1, z_e1,
                x_e1_center + eng_length/2, y_e1, z_e1,
                eng_width, eng_height, eng_width, eng_height, 0, 0, 1]
    eng2_box = [x_e2_center - eng_length/2, y_e2, z_e2,
                x_e2_center + eng_length/2, y_e2, z_e2,
                eng_width, eng_height, eng_width, eng_height, 0, 0, 1]
                
    # ... In Horizontal Stabilizer section ...
    htail_box = [x1_h, y1_h, z1_h, x2_h, y2_h, z2_h,
                 htail_root_chord, htail_thick, htail_tip_chord, htail_thick, 0, 1, 0]
                 
    # ... In Vertical Stabilizer section ...
    vtail_box = [x1_v, y1_v, z1_v, x2_v, y2_v, z2_v,
                 vtail_root_chord, vtail_thick, vtail_tip_chord, vtail_thick, 0, 1, 0]
```

- [ ] **Step 2: Update tensor padding matrix size**
In `main()` of `generate_dataset.py`, change `b_mat = np.zeros((10, MAX_BOXES))` to `b_mat = np.zeros((13, MAX_BOXES))`.

```python
        # pad boxes
        b_mat = np.zeros((13, MAX_BOXES))
```

- [ ] **Step 3: Run the generator to verify and create updated `.mat` files**
Run: `python data/generate_dataset.py`
Expected: "Generated dataset with 300 aircraft."

- [ ] **Step 4: Commit**
```bash
git add data/generate_dataset.py data/box_data.mat data/op_data.mat data/sym_data.mat
git commit -m "feat: append semantic one-hot encoding to OBB dataset"
```

---

### Task 3: Update Neural Network Architecture and Loss

**Files:**
- Modify: `grassmodel.py`

- [ ] **Step 1: Update `BoxDecoder` to split Tanh and Logits**
In `grassmodel.py`, modify the `BoxDecoder.forward` method to apply `Tanh` only to the geometric features (first 10 elements). The last 3 elements remain as raw logits.

```python
    def forward(self, parent_feature):
        import torch
        vector = self.mlp(parent_feature)
        vector_geom = self.tanh(vector[:, :10])
        vector_cls = vector[:, 10:]
        return torch.cat([vector_geom, vector_cls], dim=1)
```

- [ ] **Step 2: Update `boxLossEstimator` to use Mixed Loss**
In `GRASSDecoder`, replace the `boxLossEstimator` implementation:

```python
    def boxLossEstimator(self, box_feature, gt_box_feature, lambda_cls=1.0):
        import torch
        losses = []
        for b, gt in zip(box_feature, gt_box_feature):
            geom_l = self.mseLoss(b[:10], gt[:10]).mul(0.4)
            
            # creLoss expects input (1, C) and target (1)
            pred_logits = b[10:].unsqueeze(0)
            target_class = torch.argmax(gt[10:]).unsqueeze(0)
            cls_l = self.creLoss(pred_logits, target_class)
            
            losses.append(geom_l + lambda_cls * cls_l)
            
        return torch.stack(losses, 0)
```

- [ ] **Step 3: Verify the models still compile by running tests**
Run: `python test_gan_components.py` (if available, or a simple test script).
Expected: The models should successfully instantiate and compute loss without dimensionality mismatches.

- [ ] **Step 4: Commit**
```bash
git add grassmodel.py
git commit -m "feat: implement mixed loss and logit outputs for 13D OBBs in grassmodel"
```

---

### Task 4: Update Visualization with Semantic Coloring

**Files:**
- Modify: `draw3dobb.py`

- [ ] **Step 1: Extract category and assign colors in `draw`**
In `draw3dobb.py`, modify the beginning of `draw(ax, p, color)` to determine the color based on the predicted class (argmax of the one-hot encoding). If the length is 13, override the provided `color`.

```python
def draw(ax, p, color):
    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt

    # Determine semantic color if 13D
    if len(p) >= 13:
        cls_idx = np.argmax(p[10:13])
        if cls_idx == 0:
            color = 'gray' # Fuselage
        elif cls_idx == 1:
            color = 'blue' # Wings / Stabilizers
        elif cls_idx == 2:
            color = 'red'  # Engines

    # 10D geometry format: [x1, y1, z1, x2, y2, z2, L1, H1, L2, H2]
    c1 = np.array(p[0:3])
```

- [ ] **Step 2: Verify Visualization**
Run: `python data/visualize_dataset.py`
Expected: A 3D plot should appear displaying the aircraft components with correct semantic coloring (Gray for fuselage, Blue for wings/tails, Red for engines).

- [ ] **Step 3: Commit**
```bash
git add draw3dobb.py
git commit -m "feat: add semantic coloring based on one-hot encoding in visualization"
```
