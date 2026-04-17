import sys
import os
import torch
import random
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grassdata import GRASSDataset
from torchfoldext import FoldExt
import util
from draw3dobb import showGenshape

# Import visualization helpers from data.visualize_dataset
from data.visualize_dataset import print_assembly_steps, extract_boxes

# We need the GANDiscriminator and GANWrapper classes to be available in the scope 
# so that torch.load can properly deserialize the Discriminator object.
from train_GAN import GANDiscriminator, GANWrapper

def main():
    config = util.get_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not config.no_cuda else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join(config.save_path, 'gan_discriminator_model.pkl')
    print(f"Loading discriminator from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please ensure you have run train_GAN.py and it has saved the discriminator.")
        return

    # Load discriminator
    discriminator = torch.load(model_path, map_location=device, weights_only=False)
    discriminator.to(device)
    discriminator.eval()

    print(f"Loading dataset from {config.data_path}...")
    try:
        dataset = GRASSDataset(dir=config.data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples: {total_samples}")

    if total_samples == 0:
        return

    num_to_show = min(10, total_samples)
    indices = random.sample(range(total_samples), num_to_show)
    
    print(f"Randomly selected {num_to_show} samples for scoring and visualization.")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            print("\n" + "="*65)
            print(f"[{i+1}/{num_to_show}] Visualizing sample index: {idx}")
            tree = dataset[idx]
            
            # 1. Score the tree using Discriminator
            fold = FoldExt(cuda=(device.type == 'cuda'))
            try:
                # get_root_features expects a fold object and a batch (list of trees)
                features = discriminator.get_root_features(fold, [tree])
                score = discriminator(features).item()
                print(f"\n*** Discriminator Score (Higher is more 'Real'): {score:.4f} ***")
            except Exception as e:
                print(f"\nError scoring tree: {e}")

            print("\n--- Assembly Sequence (Post-order) ---")
            print_assembly_steps(tree.root, step=[1], next_id=[1])
            print("-" * 45)

            # 2. Visualize the tree
            boxes = []
            labels = []
            extract_boxes(tree.root, boxes, labels, next_id=[1])
            
            if len(boxes) > 0:
                boxes_np = np.array(boxes)
                # Denormalize from [-1, 1] back to [0, 1] for visualization
                boxes_denorm = (boxes_np + 1.0) / 2.0
                showGenshape(boxes_denorm, labels=labels)
            else:
                print(f"Warning: No boxes found for sample {idx}")

if __name__ == "__main__":
    main()