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

    with torch.no_grad():
        for i in range(10):
            print(f"Generating shape {i+1}/10...")
            # Sample from latent space
            root_code = torch.randn(1, config.feature_size, device=device)
            # Decode structure
            boxes = grassmodel.decode_structure(decoder, root_code)
            # Visualize
            if boxes:
                showGenshape(torch.cat(boxes, 0).detach().cpu().numpy())
            else:
                print("Warning: No boxes generated for this sample.")

if __name__ == "__main__":
    main()
