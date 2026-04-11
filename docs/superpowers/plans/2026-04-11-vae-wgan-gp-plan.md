# VAE-WGAN-GP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the second stage of the GRASS framework by training a VAE-WGAN-GP using the pre-trained autoencoder components.

**Architecture:** A new script `train_GAN.py` will be created. It defines a `GANDiscriminator` that wraps the pre-trained `GRASSEncoder` with an additional linear layer. The script will load pre-trained autoencoder weights, initialize separate optimizers for the generator (decoder) and discriminator, and run a WGAN-GP training loop with gradient penalty.

**Tech Stack:** PyTorch, Python 3

---

### Task 1: Implement Discriminator Wrapper

**Files:**
- Create: `train_GAN.py`
- Create: `test_gan_components.py`

- [ ] **Step 1: Write the failing test for GANDiscriminator**

Create `test_gan_components.py`:
```python
import torch
import util
from grassmodel import GRASSEncoder
from train_GAN import GANDiscriminator

def test_discriminator():
    config = util.get_args()
    config.cuda = False
    encoder = GRASSEncoder(config)
    discriminator = GANDiscriminator(encoder, config)
    
    # Dummy feature vector (output of encoder)
    dummy_feature = torch.randn(1, config.feature_size)
    
    score = discriminator.fc(dummy_feature)
    assert score.shape == (1, 1)
    print("test_discriminator passed")

if __name__ == "__main__":
    test_discriminator()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_gan_components.py`
Expected: FAIL with "ImportError: cannot import name 'GANDiscriminator'"

- [ ] **Step 3: Write minimal implementation in train_GAN.py**

Create `train_GAN.py`:
```python
import torch
from torch import nn

class GANDiscriminator(nn.Module):
    def __init__(self, encoder, config):
        super(GANDiscriminator, self).__init__()
        self.encoder = encoder
        # Map from feature_size to hidden_size, then to 1 (linear output for WGAN-GP)
        self.fc = nn.Sequential(
            nn.Linear(config.feature_size, config.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, feature):
        return self.fc(feature)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_gan_components.py`
Expected: PASS and prints "test_discriminator passed"

- [ ] **Step 5: Commit**

```bash
git add test_gan_components.py train_GAN.py
git commit -m "feat: implement GANDiscriminator wrapper"
```

---

### Task 2: Training Script Initialization & Data Loading

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Add setup and data loading code to train_GAN.py**

Append to `train_GAN.py`:
```python
import time
import os
from time import gmtime, strftime
from datetime import datetime
import torch.utils.data
from torchfoldext import FoldExt
import util
from dynamicplot import DynamicPlot

from grassdata import GRASSDataset
from grassmodel import GRASSDecoder

def my_collate(batch):
    return batch

def setup_training():
    config = util.get_args()
    config.cuda = not config.no_cuda
    if config.gpu < 0 and config.cuda:
        config.gpu = 0
    torch.cuda.set_device(config.gpu)
    
    print("Loading pre-trained models...")
    # Using cpu to load first to avoid device mismatch, then move to cuda if needed
    encoder = torch.load(os.path.join(config.save_path, 'vae_encoder_model.pkl'), map_location='cpu')
    decoder = torch.load(os.path.join(config.save_path, 'vae_decoder_model.pkl'), map_location='cpu')
    
    discriminator = GANDiscriminator(encoder, config)
    
    if config.cuda:
        discriminator.cuda()
        decoder.cuda()
        
    print("Loading data ...")
    grass_data = GRASSDataset(config.data_path)
    train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
    
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    g_opt = torch.optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    return config, discriminator, decoder, train_iter, d_opt, g_opt
```

- [ ] **Step 2: Commit**

```bash
git add train_GAN.py
git commit -m "feat: add initialization and data loading for GAN training"
```

---

### Task 3: Discriminator Training Step (WGAN-GP)

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Add gradient penalty function**

Append to `train_GAN.py`:
```python
def compute_gradient_penalty(discriminator, real_features, fake_features, config):
    alpha = torch.rand(real_features.size(0), 1)
    if config.cuda:
        alpha = alpha.cuda()
        
    interpolates = (alpha * real_features + ((1 - alpha) * fake_features)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones(real_features.size(0), 1)
    if config.cuda:
        fake = fake.cuda()
        
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

- [ ] **Step 2: Add discriminator train step**

Append to `train_GAN.py`:
```python
import grassmodel

def train_discriminator_step(batch, discriminator, decoder, d_opt, config):
    d_opt.zero_grad()
    
    # 1. Real Features
    enc_fold = FoldExt(cuda=config.cuda)
    enc_fold_nodes = []
    for example in batch:
        enc_fold_nodes.append(grassmodel.encode_structure_fold(enc_fold, example))
    enc_fold_nodes = enc_fold.apply(discriminator.encoder, [enc_fold_nodes])
    enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
    
    real_features = []
    for fnode in enc_fold_nodes:
        root_code, kl_div = torch.chunk(fnode, 2, 1)
        real_features.append(root_code)
    real_features = torch.cat(real_features, dim=0)
    
    d_real = discriminator(real_features).mean()
    
    # 2. Fake Features
    z_p = torch.randn(len(batch), config.feature_size)
    if config.cuda:
        z_p = z_p.cuda()
    
    # We must decode then encode to get fake features, but here we can just use decoder's SampleDecoder
    # to get feature representation if we treat the generated feature as the "fake feature".
    # Wait, the Generator outputs a tree, but Discriminator inputs a tree and encodes it.
    # To be fully differentiable through tree generation is hard because of discrete tree structure.
    # However, GRASSDecoder generates features recursively. 
    # For GAN, we can generate a root feature from noise, and treat that as the fake feature directly,
    # because the autoencoder latent space (root feature) is what we want to match!
    # Let's match the latent space: real_features vs fake_features (z_p mapped through SampleDecoder).
    
    fake_features = decoder.sampleDecoder(z_p)
    d_fake = discriminator(fake_features).mean()
    
    # 3. Gradient Penalty
    gp = compute_gradient_penalty(discriminator, real_features.data, fake_features.data, config)
    
    # 4. Total D Loss
    d_loss = d_fake - d_real + 10.0 * gp
    d_loss.backward()
    d_opt.step()
    
    return d_loss.item(), d_real.item(), d_fake.item(), gp.item()
```

- [ ] **Step 3: Commit**

```bash
git add train_GAN.py
git commit -m "feat: implement discriminator WGAN-GP step"
```

---

### Task 4: Generator Training Step & Main Loop

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Add generator train step**

Append to `train_GAN.py`:
```python
def train_generator_step(batch, discriminator, decoder, g_opt, config):
    g_opt.zero_grad()
    
    # 1. Adversarial Loss
    z_p = torch.randn(len(batch), config.feature_size)
    if config.cuda:
        z_p = z_p.cuda()
    
    fake_features = decoder.sampleDecoder(z_p)
    g_adv_loss = -discriminator(fake_features).mean()
    
    # 2. Reconstruction & KL Loss (on real batch)
    enc_fold = FoldExt(cuda=config.cuda)
    enc_fold_nodes = []
    for example in batch:
        enc_fold_nodes.append(grassmodel.encode_structure_fold(enc_fold, example))
    enc_fold_nodes = enc_fold.apply(discriminator.encoder, [enc_fold_nodes])
    enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
    
    dec_fold = FoldExt(cuda=config.cuda)
    dec_fold_nodes = []
    kld_fold_nodes = []
    for example, fnode in zip(batch, enc_fold_nodes):
        root_code, kl_div = torch.chunk(fnode, 2, 1)
        dec_fold_nodes.append(grassmodel.decode_structure_fold(dec_fold, root_code, example))
        kld_fold_nodes.append(kl_div)
        
    total_loss = dec_fold.apply(decoder, [dec_fold_nodes, kld_fold_nodes])
    recon_loss = total_loss[0].sum() / len(batch)
    kldiv_loss = total_loss[1].sum().mul(-0.05) / len(batch)
    
    g_loss = g_adv_loss + 0.01 * recon_loss + 10.0 * kldiv_loss
    g_loss.backward()
    g_opt.step()
    
    return g_loss.item(), g_adv_loss.item(), recon_loss.item(), kldiv_loss.item()
```

- [ ] **Step 2: Add main loop**

Append to `train_GAN.py`:
```python
def main():
    config, discriminator, decoder, train_iter, d_opt, g_opt = setup_training()
    
    print("Start GAN training ...")
    start = time.time()
    
    if config.save_snapshot:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        snapshot_folder = os.path.join(config.save_path, 'gan_snapshots_'+strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)
            
    header = '     Time    Epoch     Iteration    Progress(%)  D_Loss  G_Loss  ReconLoss'
    log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>8.2f},{:>8.2f},{:>10.2f}'.split(','))
    print(header)
    
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_iter):
            # Train Discriminator
            d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, d_opt, config)
            
            # Train Generator (1 update per D update, or standard n_critic updates. Default 1 for now)
            g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, decoder, g_opt, config)
            
            if batch_idx % config.show_log_every == 0:
                print(log_template.format(strftime("%H:%M:%S", time.gmtime(time.time()-start)),
                    epoch, config.epochs, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                    d_loss, g_loss, recon))
                    
        if config.save_snapshot and (epoch+1) % config.save_snapshot_every == 0:
            print("Saving snapshots ...")
            torch.save(discriminator.encoder, snapshot_folder+f'//gan_encoder_epoch_{epoch+1}.pkl')
            torch.save(decoder, snapshot_folder+f'//gan_decoder_epoch_{epoch+1}.pkl')
            
    print("Saving final models ...")
    torch.save(discriminator.encoder, config.save_path+'//gan_encoder_model.pkl')
    torch.save(decoder, config.save_path+'//gan_decoder_model.pkl')
    print("DONE")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add train_GAN.py
git commit -m "feat: implement generator step and main training loop"
```
