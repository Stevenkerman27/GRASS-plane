import torch
from torch import nn
import grassmodel
from draw3dobb import showGenshape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model to the correct device
decoder = torch.load('./models/vae_decoder_model.pkl', map_location=device, weights_only=False)
decoder.to(device)
decoder.eval()

for i in range(10):
    # Generate root code on the same device as the model
    root_code = torch.randn(1, 80).to(device)

    with torch.no_grad():
        boxes = grassmodel.decode_structure(decoder, root_code)

    if not boxes:
        print(f"Iteration {i}: No boxes generated.")
        continue

    # Denormalize from [-1, 1] back to [0, 1] for visualization
    boxes_cat = torch.cat(boxes, 0)
    boxes_denorm = (boxes_cat + 1.0) / 2.0

    print(f"Displaying shape {i+1}/10 with {len(boxes)} boxes...")
    showGenshape(boxes_denorm.detach().cpu().numpy())