import numpy as np
from scipy.io import savemat
import random
import os

NUM_SAMPLES = 200

# Constants matching Tree.NodeType from grassdata.py
BOX_OP = 0
ADJ_OP = 1
SYM_OP = 2

def generate_aircraft():
    """
    Generates a single conventional fixed-wing aircraft.
    Returns lists of: boxes (10D), ops (int), syms (8D)
    """
    boxes = []
    ops = []
    syms = []
    
    # --- 1. FUSELAGE ---
    L_fuse = random.uniform(0.8, 1.2)
    H_fuse = random.uniform(0.08, 0.15)
    W_fuse = random.uniform(0.08, 0.15)
    # Nose to Tail along X
    # Format: [x1, y1, z1, x2, y2, z2, L1, H1, L2, H2]
    fuse_box = [0, 0, 0, L_fuse, 0, 0, W_fuse, H_fuse, W_fuse, H_fuse]
    
    boxes.append(fuse_box)
    ops.append(BOX_OP)
    
    # --- 2. MAIN WING PAIR ---
    # Right wing parameters
    wing_pos_x = random.uniform(0.25 * L_fuse, 0.45 * L_fuse)
    wing_span_half = random.uniform(0.4, 0.8) # y distance from root to tip
    wing_root_chord = random.uniform(0.15, 0.3)
    taper_ratio = random.uniform(0.3, 0.7)
    wing_tip_chord = wing_root_chord * taper_ratio
    wing_root_thick = random.uniform(0.02, 0.05)
    wing_tip_thick = wing_root_thick * random.uniform(0.5, 0.9)
    sweep_dist = random.uniform(0.0, 0.2) # x shift backward at tip
    dihedral_dist = random.uniform(0.0, 0.1) # z shift upward at tip
    
    x1_w, y1_w, z1_w = wing_pos_x, W_fuse / 2.0, 0
    x2_w, y2_w, z2_w = wing_pos_x + sweep_dist, y1_w + wing_span_half, dihedral_dist
    
    wing_box = [x1_w, y1_w, z1_w, x2_w, y2_w, z2_w, 
                wing_root_chord, wing_root_thick, wing_tip_chord, wing_tip_thick]
    
    boxes.append(wing_box)
    ops.append(BOX_OP)
    
    # Symmetry for main wings
    # Reflective symmetry across XZ plane: s = [1 (for reflective), ref_normal_x, ref_normal_y, ref_normal_z, ref_point_x, ref_point_y, ref_point_z, 0]
    # In visualize_dataset.py, l3 = abs(s[0] - 1), so s[0] must be 1 for reflection.
    # Normal is [0, 1, 0] (Y-axis), ref_point is [0, 0, 0]
    sym_wing = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    syms.append(sym_wing)
    ops.append(SYM_OP)
    
    # Attach Wings to Fuselage
    ops.append(ADJ_OP)
    
    # --- 3. HORIZONTAL STABILIZER PAIR ---
    # Right tail parameters
    htail_pos_x = random.uniform(0.85 * L_fuse, 0.95 * L_fuse)
    htail_span_half = random.uniform(0.15, 0.3)
    htail_root_chord = random.uniform(0.08, 0.15)
    htail_taper = random.uniform(0.4, 0.8)
    htail_tip_chord = htail_root_chord * htail_taper
    htail_thick = random.uniform(0.01, 0.02)
    htail_sweep = random.uniform(0.0, 0.1)
    
    x1_h, y1_h, z1_h = htail_pos_x, W_fuse / 2.0, 0
    x2_h, y2_h, z2_h = htail_pos_x + htail_sweep, y1_h + htail_span_half, 0 # no dihedral usually
    
    htail_box = [x1_h, y1_h, z1_h, x2_h, y2_h, z2_h,
                 htail_root_chord, htail_thick, htail_tip_chord, htail_thick]
                 
    boxes.append(htail_box)
    ops.append(BOX_OP)
    
    # Symmetry for horizontal tail
    sym_htail = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    syms.append(sym_htail)
    ops.append(SYM_OP)
    
    # Attach Horizontal Tail
    ops.append(ADJ_OP)
    
    # --- 4. VERTICAL STABILIZER ---
    vtail_pos_x = htail_pos_x # Often aligned
    vtail_span = random.uniform(0.15, 0.3) # z distance
    vtail_root_chord = random.uniform(0.1, 0.2)
    vtail_taper = random.uniform(0.3, 0.6)
    vtail_tip_chord = vtail_root_chord * vtail_taper
    vtail_thick = random.uniform(0.01, 0.02)
    vtail_sweep = random.uniform(0.05, 0.15)
    
    x1_v, y1_v, z1_v = vtail_pos_x, 0, H_fuse / 2.0
    x2_v, y2_v, z2_v = vtail_pos_x + vtail_sweep, 0, z1_v + vtail_span
    
    vtail_box = [x1_v, y1_v, z1_v, x2_v, y2_v, z2_v,
                 vtail_root_chord, vtail_thick, vtail_tip_chord, vtail_thick]
                 
    boxes.append(vtail_box)
    ops.append(BOX_OP)
    
    # Attach Vertical Tail
    ops.append(ADJ_OP)
    
    return boxes, ops, syms

def main():
    all_boxes = []
    all_ops = []
    all_syms = []
    
    max_ops_len = 0
    max_boxes_len = 0
    max_syms_len = 0
    
    samples = []
    for _ in range(NUM_SAMPLES):
        b, o, s = generate_aircraft()
        samples.append((b, o, s))
        max_boxes_len = max(max_boxes_len, len(b))
        max_ops_len = max(max_ops_len, len(o))
        max_syms_len = max(max_syms_len, len(s))
        
    MAX_BOXES = max_boxes_len # 4
    MAX_OPS = max_ops_len     # 9
    MAX_SYMS = max_syms_len   # 2
    
    padded_boxes = []
    padded_ops = []
    padded_syms = []
    
    for b, o, s in samples:
        # pad boxes
        b_mat = np.zeros((10, MAX_BOXES))
        for i, box in enumerate(b):
            b_mat[:, i] = box
        padded_boxes.append(b_mat)
        
        o_mat = np.zeros((MAX_OPS, 1), dtype=np.int32)
        for i, op in enumerate(o):
            o_mat[i, 0] = op
        padded_ops.append(o_mat)
        
        s_mat = np.zeros((8, MAX_SYMS))
        for i, sym in enumerate(s):
            s_mat[:, i] = sym
        padded_syms.append(s_mat)
        
    # Concatenate along axis 1
    final_boxes = np.concatenate(padded_boxes, axis=1) # [10, NUM_SAMPLES * MAX_BOXES]
    final_ops = np.concatenate(padded_ops, axis=1)     # [MAX_OPS, NUM_SAMPLES]
    final_syms = np.concatenate(padded_syms, axis=1)   # [8, NUM_SAMPLES * MAX_SYMS]
    
    savemat('data/box_data.mat', {'boxes': final_boxes})
    savemat('data/op_data.mat', {'ops': final_ops})
    savemat('data/sym_data.mat', {'syms': final_syms})
    
    print(f"Generated dataset with {NUM_SAMPLES} aircraft.")

if __name__ == "__main__":
    main()