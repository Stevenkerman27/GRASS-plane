# GRASS OBB 13D One-Hot Encoding and Mixed Loss Design Spec

## Overview
This design document specifies the changes required to expand the existing 10D Oriented Bounding Box (OBB) parameterization into a 13D vector to include a one-hot encoding for the part category (Fuselage, Wing/Tail, Engine). It also details the necessary updates to the neural network loss function to handle both continuous geometric parameters and discrete categorical variables.

## 1. Data Structure Expansion (13D Encoding)
The `box_code_size` will be updated from 10 to 13 globally.

The new 13D vector is defined as: `[x1, y1, z1, x2, y2, z2, L1, H1, L2, H2, C_fuse, C_wing, C_eng]`

*   **Indices 0-9 (10D):** Continuous geometric features (centers and dimensions).
*   **Indices 10-12 (3D):** One-hot encoded part category:
    *   `[1, 0, 0]`: Fuselage
    *   `[0, 1, 0]`: Wing and Stabilizers (Main Wing, Horizontal Tail, Vertical Tail)
    *   `[0, 0, 1]`: Engine

## 2. Dataset Generation Updates
The dataset generator script (`data/generate_dataset.py`) will be modified to append the appropriate 3D one-hot vector to the end of each generated OBB.
*   Fuselage box: append `[1, 0, 0]`
*   Main wings, horizontal tails, vertical tails: append `[0, 1, 0]`
*   Engines: append `[0, 0, 1]`

## 3. Neural Network Architecture and Loss Function
### 3.1 Architecture
*   The final fully connected layers that output the `box_code` will now output a 13D vector (raw logits for the categorical part).

### 3.2 Loss Computation (Mixed Loss)
The box reconstruction loss in the autoencoder/GAN training loop will be split into two components:
1.  **Geometric Loss (MSE/L1):** 
    *   `pred_geom = pred[:, :10]`
    *   `target_geom = target[:, :10]`
    *   Calculated using existing `MSELoss` (or `L1Loss`).
2.  **Classification Loss (Cross Entropy):**
    *   `pred_cls_logits = pred[:, 10:]`
    *   `target_cls = argmax(target[:, 10:], dim=1)` (Convert one-hot back to class indices 0, 1, or 2).
    *   Calculated using `nn.CrossEntropyLoss`.
3.  **Total Loss:**
    *   `Total_Box_Loss = Geom_Loss + lambda_cls * Cls_Loss`
    *   `lambda_cls` acts as a balancing weight (defaulting to 1.0 initially, subject to tuning).

### 3.3 Inference / Generation
When the network generates a new box (e.g., in the decoder or GAN generator), the 13D output will be post-processed:
*   The geometric part (first 10D) is taken as-is (or passed through sigmoid/tanh depending on normalization).
*   The categorical part (last 3D) is processed using `argmax` to determine the predicted class, and then converted back into a hard one-hot encoding for saving and downstream visualization.

## 4. Visualization Updates (`draw3dobb.py`)
The visualization script will be updated to parse the new 13D format.
*   It will extract the categorical one-hot encoding (indices 10-12) to determine the semantic type of the box.
*   Colors will be assigned dynamically based on the category (e.g., Fuselage = Gray/White, Wings = Blue, Engines = Red) to improve interpretability of the generated shapes.

## 5. Implementation Steps
1.  **Configuration:** Update `box_code_size = 13` in `util.py`.
2.  **Data Generation:** Update `generate_dataset.py` to append the 3D one-hot vector and regenerate the `.mat` data files.
3.  **Loss Function:** Implement the split loss logic in the training scripts (`train.py`, `train_GAN.py` or within `grassmodel.py`'s `boxLoss` function depending on architecture).
4.  **Inference:** Update the box prediction logic in `grassmodel.py` to output raw logits for the class and apply argmax during the generation phase.
5.  **Visualization:** Update `draw3dobb.py` and `visualize_dataset.py` to handle the 13D format and apply semantic coloring.
