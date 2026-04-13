import numpy as np
from scipy.io import savemat
import random
import os

NUM_SAMPLES = 200

# Constants matching Tree.NodeType from grassdata.py
BOX_OP = 0
ADJ_OP = 1
SYM_OP = 2

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

def build_fuselage():
    L = random.uniform(0.8, 1.5)
    H = random.uniform(0.05, 0.15)
    W = random.uniform(0.05, 0.15)
    return [0, 0, 0, L, 0, 0, W, H, W, H, 1, 0, 0], L, H, W

def get_engine_dims():
    length = random.uniform(0.15, 0.25)
    width = random.uniform(0.04, 0.08)
    return length, width

def build_engine(x, y, z, length, width):
    return [x - length/2, y, z, x + length/2, y, z, width, width, width, width, 0, 0, 1]

def build_wing(x_root, y_root, z_root, span_half, type):
    if type == "lifting":
        AR = random.uniform(0.9, 8)
    else:
        AR = random.uniform(0.8, 2.5)
    taper = random.uniform(0.3, 1.0)
    root_chord = span_half/AR*2/(1+taper)
    dihedral = random.uniform(-span_half*0.1, span_half*0.1)
    sweep = random.uniform(0, root_chord * 1)
    tip_chord = root_chord * taper
    thick = random.uniform(0.02, 0.05)
    tip_thick = thick * random.uniform(0.3, 0.9)
    x2 = x_root + sweep
    y2 = y_root + span_half
    z2 = z_root + dihedral
    return [x_root, y_root, z_root, x2, y2, z2, root_chord, thick, tip_chord, tip_thick, 0, 1, 0]

def emit_engine_group(assembler, config, x, y, z, y2=None):
    """
    config: 'single', 'twin', 'quad'
    (x, y, z): Primary engine position
    y2: External engine y-offset for quad
    """
    length, width = get_engine_dims()
    if config == 'single':
        assembler.push_box(build_engine(x, 0, z, length, width))
    elif config == 'twin':
        assembler.push_box(build_engine(x, y, z, length, width))
        assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Mirror across YZ
    elif config == 'quad':
        assembler.push_box(build_engine(x, y, z, length, width))   # Inner
        assembler.push_box(build_engine(x, y2, z, length, width))  # Outer
        assembler.apply_adj()
        assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def emit_wing_engines(assembler, config, x, y, z, y2=None):
    """
    Specifically for wing-mounted engines on ONE side. 
    Assumes a wing box is already at the top of the stack.
    """
    length, width = get_engine_dims()
    if config == 'twin': # One engine per wing
        assembler.push_box(build_engine(x, y, z, length, width))
        assembler.apply_adj() # Join engine 1 to wing
    elif config == 'quad': # Two engines per wing
        assembler.push_box(build_engine(x, y, z, length, width)) # Inner
        assembler.apply_adj() # Join inner engine to wing
        assembler.push_box(build_engine(x, y2, z, length, width)) # Outer
        assembler.apply_adj() # Join outer engine to (wing + inner engine)

def build_canard_layout(assembler):
    fuse_box, L_fuse, H_fuse, W_fuse = build_fuselage()
    assembler.push_box(fuse_box)
    
    # Canard (Forewing)
    canard_span = random.uniform(0.15, 0.3)
    canard_box = build_wing(0.1 * L_fuse, W_fuse/2, 0, canard_span, "stab")
    assembler.push_box(canard_box)
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj() # Connect Canard to Fuselage
    
    # Main Wing (Aft)
    wing_span = random.uniform(0.5, 0.9)
    wing_pos_x = random.uniform(0.6, 0.8) * L_fuse
    wing_box = build_wing(wing_pos_x, W_fuse/2, 0, wing_span, "lifting")
    assembler.push_box(wing_box)
    
    # Engine logic for Canard
    eng_loc = random.choice(['wing', 'rear'])
    if eng_loc == 'wing':
        config = random.choice(['twin', 'quad'])
        y_inner = W_fuse/2 + wing_span * 0.3
        y_outer = W_fuse/2 + wing_span * 0.7
        emit_wing_engines(assembler, config, wing_pos_x + 0.1, y_inner, -0.05, y_outer)
    
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj() # Connect Wing (and its engines) to Fuselage
    
    if eng_loc == 'rear':
        config = random.choice(['single', 'twin'])
        emit_engine_group(assembler, config, 0.95 * L_fuse, W_fuse/2 + 0.05, 0)
        assembler.apply_adj() # Connect Rear Engines to Fuselage

    # 垂尾
    v_span = random.uniform(0.15, 0.3)
    vtail_box = build_wing(0.95 * L_fuse, 0, H_fuse/2, v_span, "stab")
    vtail_box[1] = 0; vtail_box[2] = H_fuse/2; vtail_box[4] = 0; vtail_box[5] = H_fuse/2 + v_span
    assembler.push_box(vtail_box)
    assembler.apply_adj() # Connect 垂尾 to Fuselage

def build_flying_wing(assembler):
    L = random.uniform(0.3, 0.8)
    thinness = random.uniform(0.1, 0.8)
    H = L*thinness
    width = random.uniform(0.8, 1.2)
    W = H*width
    fuse_box = [0, 0, 0, L, 0, 0, W, H, W, H, 1, 0, 0]
    assembler.push_box(fuse_box)
    
    wing_span = random.uniform(0.8, 1.4)
    wing_box = build_wing(random.uniform(0.3, 0.8) * L, W/2, 0, wing_span, "lifting")
    assembler.push_box(wing_box)
    
    eng_loc = random.choice(['wing', 'rear'])
    if eng_loc == 'wing':
        config = random.choice(['twin', 'quad'])
        y_inner = W/2 + wing_span * 0.25
        y_outer = W/2 + wing_span * 0.55
        emit_wing_engines(assembler, config, 0.3 * L + 0.2, y_inner, -0.05, y_outer)
    
    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj()
    
    if eng_loc == 'rear':
        # Single central engine for flying wing
        emit_engine_group(assembler, 'single', L*0.9, 0, 0.05)
        assembler.apply_adj()

def build_conventional(assembler):
    fuse_box, L_fuse, H_fuse, W_fuse = build_fuselage()
    assembler.push_box(fuse_box)

    # Main Wing
    wing_span = random.uniform(0.6, 1.2)
    wing_pos_x = random.uniform(0.25, 0.45) * L_fuse
    wing_box = build_wing(wing_pos_x, W_fuse/2, 0, wing_span, "lifting")
    assembler.push_box(wing_box)

    eng_loc = random.choice(['nose', 'wing', 'rear'])

    if eng_loc == 'wing':
        config = random.choice(['twin', 'quad'])
        y_inner = W_fuse/2 + wing_span * 0.35
        y_outer = W_fuse/2 + wing_span * 0.7
        emit_wing_engines(assembler, config, wing_pos_x + 0.15, y_inner, -0.08, y_outer)

    assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembler.apply_adj() # Connect Wing (and its engines) to Fuselage
    
    if eng_loc == 'nose':
        emit_engine_group(assembler, 'single', -0.1, 0, 0)
        assembler.apply_adj()
    elif eng_loc == 'rear':
        config = random.choice(['single', 'twin'])
        # If single, put it higher (base of tail) or right at the end
        z_off = 0.05 if config == 'single' else 0.0
        emit_engine_group(assembler, config, 0.85 * L_fuse, W_fuse/2 + 0.08, z_off)
        assembler.apply_adj()
    
    # Tail
    tail_type = random.choice(['T', 'conventional'])
    v_span = random.uniform(0.25, 0.4)
    if tail_type == 'T':
        vt_pos =  random.uniform(0.9, 1.0)*L_fuse
        ht_span = random.uniform(0.2, 0.45)
        htail_box = build_wing(vt_pos+ht_span*random.uniform(0, 0.3), 0, H_fuse/2 + v_span, ht_span, "stab")
        assembler.push_box(htail_box)
        assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vtail_box = build_wing(vt_pos, 0, H_fuse/2, v_span,  "stab")
        vtail_box[1] = 0; vtail_box[2] = H_fuse/2; vtail_box[4] = 0; vtail_box[5] = H_fuse/2 + v_span
        assembler.push_box(vtail_box)
        assembler.apply_adj()
    else:
        vtail_box = build_wing(0.92 * L_fuse, 0, H_fuse/2, v_span, "stab")
        vtail_box[1] = 0; vtail_box[2] = H_fuse/2; vtail_box[4] = 0; vtail_box[5] = H_fuse/2 + v_span
        assembler.push_box(vtail_box)
        assembler.apply_adj()
        htail_box = build_wing(0.88 * L_fuse, W_fuse/2, 0, random.uniform(0.2, 0.4), "stab")
        assembler.push_box(htail_box)
        assembler.apply_sym([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    assembler.apply_adj()

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
    max_val = 0.0
    for box in boxes:
        for val in box[:10]:
            if abs(val) > max_val:
                max_val = abs(val)

    if max_val > 0:
        for i in range(len(boxes)):
            # Normalize continuous values (0-9) to [-1, 1]
            for j in range(10):
                boxes[i][j] = (boxes[i][j] / max_val) * 2.0 - 1.0
            # Map one-hot categories (10-12) from {0, 1} to {-1, 1}
            for j in range(10, 13):
                boxes[i][j] = boxes[i][j] * 2.0 - 1.0
        
        for i in range(len(syms)):
            # s[0] is the type flag (-1: Rot, 0: Trans, 1: Refl)
            sym_type = round(syms[i][0])
            
            # 1-3: Normals, Axes, or Translation Vector
            if sym_type == 0: # Translational: it's a delta vector
                for j in range(1, 4):
                    syms[i][j] = (syms[i][j] / max_val) # Scale only, NO shift
            else: # Reflective or Rotational: it's a direction/unit vector
                pass # Normals/Axes should NOT be scaled or shifted
            
            # 4-6: Reference Points, Centers, or End Points
            for j in range(4, 7):
                syms[i][j] = (syms[i][j] / max_val) * 2.0 - 1.0

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
        
    MAX_BOXES = max_boxes_len
    MAX_OPS = max_ops_len
    MAX_SYMS = max_syms_len
    
    padded_boxes = []
    padded_ops = []
    padded_syms = []
    
    for b, o, s in samples:
        # pad boxes
        b_mat = np.zeros((13, MAX_BOXES))
        for i, box in enumerate(b):
            b_mat[:, i] = box
        padded_boxes.append(b_mat)
        
        o_mat = np.full((MAX_OPS, 1), -1, dtype=np.int32)
        for i, op in enumerate(o):
            o_mat[i, 0] = op
        padded_ops.append(o_mat)
        
        s_mat = np.zeros((8, MAX_SYMS))
        for i, sym in enumerate(s):
            s_mat[:, i] = sym
        padded_syms.append(s_mat)
        
    # Concatenate along axis 1
    final_boxes = np.concatenate(padded_boxes, axis=1) # [13, NUM_SAMPLES * MAX_BOXES]
    final_ops = np.concatenate(padded_ops, axis=1)     # [MAX_OPS, NUM_SAMPLES]
    final_syms = np.concatenate(padded_syms, axis=1)   # [8, NUM_SAMPLES * MAX_SYMS]
    
    savemat('data/box_data.mat', {'boxes': final_boxes})
    savemat('data/op_data.mat', {'ops': final_ops})
    savemat('data/sym_data.mat', {'syms': final_syms})
    
    print(f"Generated dataset with {NUM_SAMPLES} aircraft.")

if __name__ == "__main__":
    main()
