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
        
        newbox = torch.cat([newc1, newc2, dims])
        reBoxes.append(newbox.unsqueeze(0))

    return reBoxes

def extract_boxes(node, s, boxes_list):
    """
    Recursively extract OBB boxes from a Tree.Node, handling symmetry propagation.
    """
    if node.is_leaf():
        reBoxes = get_s_boxes(node.box, s)
        for rb in reBoxes:
            boxes_list.append(rb.detach().cpu().numpy().squeeze())
    elif node.is_adj():
        extract_boxes(node.left, s, boxes_list)
        extract_boxes(node.right, s, boxes_list)
    elif node.is_sym():
        # SYM node uses its own sym parameter for its child
        extract_boxes(node.left, node.sym, boxes_list)

def main():
    dataset_path = './data'
    print(f"Loading dataset from {dataset_path}...")
    
    # Default identity symmetry (far from -1, 0, 1)
    identity_s = torch.ones(8).mul(10)
    
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

    num_to_show = min(10, total_samples)
    indices = random.sample(range(total_samples), num_to_show)
    
    print(f"Randomly selected {num_to_show} samples for visualization.")

    for i, idx in enumerate(indices):
        print(f"[{i+1}/{num_to_show}] Visualizing sample index: {idx}")
        tree = dataset[idx]
        
        boxes = []
        # Start with identity symmetry
        extract_boxes(tree.root, identity_s, boxes)
        
        if len(boxes) > 0:
            showGenshape(boxes)
        else:
            print(f"Warning: No boxes found for sample {idx}")

if __name__ == "__main__":
    main()
