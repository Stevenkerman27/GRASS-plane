import torch
from torch import nn
import grassmodel
from draw3dobb import showGenshape


decoder = torch.load('./models/vae_decoder_model.pkl', weights_only=False)


for i in range(10):
    root_code = torch.randn(1,80).cuda()
    boxes = grassmodel.decode_structure(decoder, root_code)
    # Denormalize from [-1, 1] back to [0, 1] for visualization
    boxes_cat = torch.cat(boxes, 0)
    boxes_denorm = (boxes_cat + 1.0) / 2.0
    showGenshape(boxes_denorm.detach().cpu().numpy())