import torch
from torch import nn
import grassmodel
from draw3dobb import showGenshape
import os
import util

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config for feature_size and save_path
    config = util.get_args()
    model_path = os.path.join(config.save_path, 'gan_decoder_model.pkl')
    
    print(f"Loading decoder from {model_path}...")
    # Load decoder
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    decoder = torch.load(model_path, map_location=device, weights_only=False)
    decoder.to(device)
    decoder.eval()

    # Use a fixed seed for reproducibility during debugging if desired
    # torch.manual_seed(42)

    with torch.no_grad():
        for i in range(10):
            print(f"\nGenerating shape {i+1}/10...")
            # Sample from latent space. 
            # Note: Sometimes GANs perform better with a smaller scale (e.g., 0.8) 
            # if the training data distribution was tight.
            noise_scale = 1.0 
            root_code = torch.randn(1, config.feature_size, device=device) * noise_scale
            
            # Decode structure
            boxes = grassmodel.decode_structure(decoder, root_code)
            
            # Statistics for debugging
            if boxes:
                num_boxes = len(boxes)
                print(f"  Successfully generated {num_boxes} boxes.")
                
                # Check for degenerate boxes (e.g., all zeros or NaNs)
                box_data = torch.cat(boxes, 0)
                if torch.isnan(box_data).any():
                    print("  Warning: Generated boxes contain NaNs!")
                
                # Visualize
                showGenshape(box_data.detach().cpu().numpy())
            else:
                print("  Warning: No boxes generated for this sample. The tree might have collapsed.")

if __name__ == "__main__":
    main()
