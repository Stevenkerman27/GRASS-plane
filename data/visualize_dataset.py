import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
import math
from grassdata import GRASSDataset
from draw3dobb import showGenshape
from grassmodel import vrrotvec2mat

def get_s_boxes(reBox, s):
    """
    Generate multiple boxes from a single box and symmetry parameters.
    Adapted from grassmodel.py's decode_structure logic.
    """
    # Ensure tensors are on the same device (CUDA as per project convention)
    s = s.cuda()
    reBox = reBox.cuda()
    
    if s.dim() == 2: s = s.squeeze(0)
    if reBox.dim() == 2: reBox = reBox.squeeze(0)
    
    reBoxes = [reBox.unsqueeze(0)]
    
    l1 = abs(s[0] + 1) # Rotational
    l2 = abs(s[0])     # Translational
    l3 = abs(s[0] - 1) # Reflective

    if l1 < 0.15:
        sList = torch.split(s, 1, 0)
        bList = torch.split(reBox, 1, 0)
        f1 = torch.cat([sList[1], sList[2], sList[3]])
        f1 = f1/torch.norm(f1)
        f2 = torch.cat([sList[4], sList[5], sList[6]])
        folds = round((1/s[7]).item())
        for i in range(folds-1):
            rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
            rotm = vrrotvec2mat(rotvector)
            c1 = torch.cat([bList[0], bList[1], bList[2]])
            c2 = torch.cat([bList[3], bList[4], bList[5]])
            dims = torch.cat([bList[6], bList[7], bList[8], bList[9]])
            
            newc1 = rotm.matmul(c1.add(-f2)).add(f2)
            newc2 = rotm.matmul(c2.add(-f2)).add(f2)
            
            if len(bList) >= 13:
                cls = torch.cat([bList[10], bList[11], bList[12]])
                newbox = torch.cat([newc1, newc2, dims, cls])
            else:
                newbox = torch.cat([newc1, newc2, dims])
            reBoxes.append(newbox.unsqueeze(0))

    if l2 < 0.15:
        sList = torch.split(s, 1, 0)
        bList = torch.split(reBox, 1, 0)
        trans = torch.cat([sList[1], sList[2], sList[3]])
        trans_end = torch.cat([sList[4], sList[5], sList[6]])
        c1 = torch.cat([bList[0], bList[1], bList[2]])
        c2 = torch.cat([bList[3], bList[4], bList[5]])
        dims = torch.cat([bList[6], bList[7], bList[8], bList[9]])
        
        trans_length = math.sqrt(torch.sum(trans**2).item())
        # Use c1 as reference point for translational total length
        trans_total = math.sqrt(torch.sum(trans_end.add(-c1)**2).item())
        folds = round(trans_total/max(trans_length, 1e-6))
        for i in range(folds):
            c1 = torch.cat([bList[0], bList[1], bList[2]])
            c2 = torch.cat([bList[3], bList[4], bList[5]])
            newc1 = c1.add(trans.mul(i+1))
            newc2 = c2.add(trans.mul(i+1))
            if len(bList) >= 13:
                cls = torch.cat([bList[10], bList[11], bList[12]])
                newbox = torch.cat([newc1, newc2, dims, cls])
            else:
                newbox = torch.cat([newc1, newc2, dims])
            reBoxes.append(newbox.unsqueeze(0))

    if l3 < 0.15:
        sList = torch.split(s, 1, 0)
        bList = torch.split(reBox, 1, 0)
        ref_normal = torch.cat([sList[1], sList[2], sList[3]])
        ref_normal = ref_normal/torch.norm(ref_normal)
        ref_point = torch.cat([sList[4], sList[5], sList[6]])
        
        c1 = torch.cat([bList[0], bList[1], bList[2]])
        c2 = torch.cat([bList[3], bList[4], bList[5]])
        dims = torch.cat([bList[6], bList[7], bList[8], bList[9]])
        
        # Reflection logic for points c1 and c2
        v1 = c1.add(-ref_point)
        dist1 = torch.sum(v1 * ref_normal)
        newc1 = c1.add(ref_normal.mul(-2 * dist1))
        
        v2 = c2.add(-ref_point)
        dist2 = torch.sum(v2 * ref_normal)
        newc2 = c2.add(ref_normal.mul(-2 * dist2))
        
        if len(bList) >= 13:
            cls = torch.cat([bList[10], bList[11], bList[12]])
            newbox = torch.cat([newc1, newc2, dims, cls])
        else:
            newbox = torch.cat([newc1, newc2, dims])
        reBoxes.append(newbox.unsqueeze(0))

    return reBoxes

def extract_boxes(node, boxes_list, labels_list, next_id=[1]):
    """
    Recursively extract OBB boxes and their Part IDs from a Tree.Node.
    Matches the post-order sequence of print_assembly_steps.
    """
    if node.is_leaf():
        curr_id = next_id[0]
        next_id[0] += 1
        # Add a single box for the leaf
        boxes_list.append(node.box.detach().cpu().numpy().squeeze())
        labels_list.append(curr_id)
        return curr_id, len(boxes_list) - 1, len(boxes_list)
    
    elif node.is_sym():
        # 1. Recurse into child
        child_id, s_idx, e_idx = extract_boxes(node.left, boxes_list, labels_list, next_id)
        
        # 2. Apply symmetry to all existing boxes in the subtree and add new ones
        sym_param = node.sym.cuda() if node.sym.is_cuda else node.sym
        new_boxes = []
        for i in range(s_idx, e_idx):
            box_tensor = torch.from_numpy(boxes_list[i]).float().cuda()
            # get_s_boxes returns [original, copy1, copy2, ...]
            # We already have 'original' in the list, so we only need the copies
            copies = get_s_boxes(box_tensor, sym_param)[1:]
            for cp in copies:
                new_boxes.append(cp.detach().cpu().numpy().squeeze())
        
        # Add new boxes to the list with the same label as the subtree result
        for nb in new_boxes:
            boxes_list.append(nb)
            labels_list.append(child_id)
            
        return child_id, s_idx, len(boxes_list)
        
    elif node.is_adj():
        left_id, sl, el = extract_boxes(node.left, boxes_list, labels_list, next_id)
        right_id, sr, er = extract_boxes(node.right, boxes_list, labels_list, next_id)
        
        # Combine IDs for the group (e.g., "1, 2")
        curr_id = f"{left_id}, {right_id}"
        return curr_id, sl, len(boxes_list)

def get_box_label(box):
    """Identify component type from 13D box vector."""
    if box.dim() == 2: box = box.squeeze(0)
    # Indices 10, 11, 12 are one-hot categories scaled to [-1, 1]
    # Value > 0 means the category is active (originally 1.0)
    if box[10] > 0: return "Fuselage"
    if box[11] > 0: return "Wing/Tail"
    if box[12] > 0: return "Engine"
    return "Unknown"

def print_assembly_steps(node, step=[1], next_id=[1]):
    """
    Post-order traversal to print assembly steps with specific Part IDs.
    Returns (Part ID, Description).
    """
    if node.is_leaf():
        label = get_box_label(node.box)
        curr_id = next_id[0]
        next_id[0] += 1
        print(f"Step {step[0]}: Create Part {curr_id} ({label})")
        step[0] += 1
        return curr_id, label
    
    elif node.is_sym():
        child_id, child_desc = print_assembly_steps(node.left, step, next_id)
        s_type = "Reflective" if abs(node.sym[0][0] - 1) < 0.1 else \
                 "Translational" if abs(node.sym[0][0]) < 0.1 else \
                 "Rotational" if abs(node.sym[0][0] + 1) < 0.1 else "Unknown"
        
        print(f"Step {step[0]}: Apply {s_type} symmetry to Part(s) {child_id}")
        step[0] += 1
        return child_id, f"Symmetric {child_desc}"
    
    elif node.is_adj():
        left_id, left_desc = print_assembly_steps(node.left, step, next_id)
        right_id, right_desc = print_assembly_steps(node.right, step, next_id)
        
        print(f"Step {step[0]}: Join Part(s) {left_id} and Part(s) {right_id}")
        step[0] += 1
        return f"{left_id}, {right_id}", f"Group({left_desc}+{right_desc})"

def print_tree_structure(node, indent=""):
    """Recursive function to print the tree hierarchy."""
    if node.is_leaf():
        label = get_box_label(node.box)
        print(f"{indent}└── [BOX] {label}")
    elif node.is_sym():
        s_type = "Reflective" if abs(node.sym[0][0] - 1) < 0.1 else \
                 "Translational" if abs(node.sym[0][0]) < 0.1 else \
                 "Rotational" if abs(node.sym[0][0] + 1) < 0.1 else "Unknown"
        print(f"{indent}└── [SYM] {s_type}")
        print_tree_structure(node.left, indent + "    ")
    elif node.is_adj():
        print(f"{indent}└── [ADJ] Join")
        print_tree_structure(node.left, indent + "    ├── ")
        print_tree_structure(node.right, indent + "    └── ")

def main():
    dataset_path = './data'
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        dataset = GRASSDataset(dir=dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples: {total_samples}")

    if total_samples == 0:
        print("No samples found in dataset.")
        return

    num_to_show = min(5, total_samples)
    indices = random.sample(range(total_samples), num_to_show)
    
    print(f"Randomly selected {num_to_show} samples for visualization.")

    for i, idx in enumerate(indices):
        print("\n" + "="*65)
        print(f"[{i+1}/{num_to_show}] Visualizing sample index: {idx}")
        tree = dataset[idx]
        
        print("\n--- Tree Hierarchy ---")
        print_tree_structure(tree.root)
        
        print("\n--- Assembly Sequence (Post-order) ---")
        # Initialize step and part counters for each sample
        print_assembly_steps(tree.root, step=[1], next_id=[1])
        print("-" * 45)

        boxes = []
        labels = []
        # extraction matches assembly sequence IDs
        extract_boxes(tree.root, boxes, labels, next_id=[1])
        
        if len(boxes) > 0:
            # Denormalize from [-1, 1] back to [0, 1] for visualization
            boxes_np = np.array(boxes)
            boxes_denorm = (boxes_np + 1.0) / 2.0
            showGenshape(boxes_denorm, labels=labels)
        else:
            print(f"Warning: No boxes found for sample {idx}")

if __name__ == "__main__":
    main()
