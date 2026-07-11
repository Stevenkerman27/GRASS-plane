# GRASS in Pytorch
This is a Pytorch implementation of the paper "[GRASS: Generative Recursive Autoencoders for Shape Structures](http://kevinkaixu.net/projects/grass.html)". The paper is about learning a generative model for 3D shape structures by structural encoding and decoding with Recursive Neural Networks. This code was originally written by [Chenyang Zhu](http://www.sfu.ca/~cza68/) and is being improved and maintained here in this repository.

The repository provides the legacy flat-box VAE training and generation path, plus an experimental WGAN-GP path in `train_GAN.py`. The typed aircraft-box path has encoder and teacher-forced reconstruction support, but is not yet wired into the formal training, GAN, or free-generation entry points. See `docs/aircraft_layout_vae_definitions.md` and `docs/typed_box_airfoil_encoding.md` for the current boundary.

## Usage
**Dependencies**

grass_pytorch should be run with Python 3.x and PyTorch 2.9.

grass_pytorch depends on torchfold which is a pytorch tool developed by [Illia Polosukhin](https://github.com/ilblackdragon). It is used for dynamic batching the computations in a dynamic computation graph. The computations across all nodes of all trees are batched based on their module names and dispatched to GPU for parallelization. 

**Training**
```
python train.py --data_path data --save_path models
```
This entry point trains the legacy flat 13D `.mat` dataset. Run `python train.py --help` to read the current arguments and defaults from their source of truth in `util.py`.

**Generation (legacy flat-box path)**
```
python VAE_gen.py
```
This samples a random root code, decodes it into a legacy 13D box tree, and displays it with `draw3dobb.py`. The OpenVSP and typed-OBB example uses separate environments; see `docs/openvsp_obb_json_visualization.md`.



## Acknowledgement
This code uses the 'torchfold' in pytorch-tools developed by [Illia Polosukhin](https://github.com/ilblackdragon).
