# Dataset Isotropic Scaling & GAN Feature Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement per-instance isotropic scaling in dataset generation and remove artificial `SampleDecoder` alignment for real features in GAN training.

**Architecture:** We will modify `generate_dataset.py` to scale all aircraft coordinates and sizes by the absolute maximum dimension for each generated aircraft, ensuring a natural `[-1, 1]` bounding box. Then, we will update `train_GAN.py` to pass the raw `root_code` from the VAE directly as `real_features`, rather than wrapping it with the Tanh-activated `sampleDecoder`, as the VAE output should now naturally match the space of the fake features generated via `sampleDecoder`.

**Tech Stack:** Python, PyTorch, NumPy

---

### Task 1: Update Dataset Generation (Per-Instance Isotropic Scaling)

**Files:**
- Modify: `data/generate_dataset.py`

- [ ] **Step 1: Modify `generate_aircraft` to perform isotropic scaling**

Modify `data/generate_dataset.py`. In `generate_aircraft()`, before returning `boxes, ops, syms`, find the maximum absolute coordinate/dimension across all boxes and scale them.

```python
# In data/generate_dataset.py, at the end of generate_aircraft() before the return statement:
    # --- Isotropic Scaling per instance ---
    max_val = 0.0
    for box in boxes:
        # Check all continuous values (indices 0 to 9)
        # x1, y1, z1, x2, y2, z2, L1, H1, L2, H2
        for val in box[:10]:
            if abs(val) > max_val:
                max_val = abs(val)
                
    if max_val > 0:
        for i in range(len(boxes)):
            for j in range(10):
                boxes[i][j] = boxes[i][j] / max_val
                
    return boxes, ops, syms
```

- [ ] **Step 2: Generate the new dataset**

Run the dataset generation script to produce the updated `.mat` files.

Run: `python data/generate_dataset.py`
Expected: Output showing "Generated dataset with 300 aircraft." (or similar).

- [ ] **Step 3: Commit the changes to the dataset generator**

```bash
git add data/generate_dataset.py data/*.mat
git commit -m "feat(data): implement per-instance isotropic scaling for aircraft dataset"
```

---

### Task 2: Remove SampleDecoder for Real Features in GAN Training

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Update `train_discriminator_step`**

In `train_GAN.py`, locate the `train_discriminator_step` function. Currently, it appends the decoded root code:
`real_features_list.append(decoder.sampleDecoder(root_code))`
Change it to append `root_code` directly.

```python
# In train_GAN.py, inside train_discriminator_step:
        real_features_list = []
        for fnode in enc_fold_nodes:
            root_code, _ = torch.chunk(fnode, 2, 1)
            # Remove sampleDecoder to use raw latent space
            real_features_list.append(root_code)
        real_features = torch.cat(real_features_list, dim=0)
```

- [ ] **Step 2: Verify script syntax and execution start**

Run a quick syntax check on `train_GAN.py` by running it briefly (you can stop it if it starts training).

Run: `python train_GAN.py --gan_epochs 1 --no_plot`
Expected: Script parses correctly, models/data load, and the LR Range test starts without crashing. (You can abort the LR range test input).

- [ ] **Step 3: Commit the changes to GAN training**

```bash
git add train_GAN.py
git commit -m "fix(gan): use raw root_code for real features in discriminator"
```
