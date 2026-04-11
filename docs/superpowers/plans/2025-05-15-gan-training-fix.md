# GAN Training Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize GAN training by aligning feature spaces and freezing the pre-trained encoder.

**Architecture:** 
- Freeze the VAE Encoder in `setup_training`.
- Align the Discriminator's real feature path by passing encoder outputs through `SampleDecoder`.
- Correct the WGAN-GP training loop order to update the Critic `n_critic` times before updating the Generator.

**Tech Stack:** PyTorch, TorchFoldExt, Matplotlib.

---

### Task 1: Freeze Encoder and Scope Optimizers

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Update `setup_training` to freeze encoder and scope `d_opt`**

```python
def setup_training():
    # ... existing loading code ...
    discriminator = GANDiscriminator(encoder, config)
    
    # Freeze encoder
    for param in discriminator.encoder.parameters():
        param.requires_grad = False
    discriminator.encoder.eval()
    
    if config.cuda:
        discriminator.cuda()
        decoder.cuda()
    
    # ... loading data ...
    
    # Scope d_opt to only discriminator.fc
    d_opt = torch.optim.Adam(discriminator.fc.parameters(), lr=config.gan_lr, betas=(0.5, 0.9))
    g_opt = torch.optim.Adam(decoder.parameters(), lr=config.gan_lr, betas=(0.5, 0.9))
    
    return config, discriminator, decoder, train_iter, d_opt, g_opt
```

- [ ] **Step 2: Commit changes**

```bash
git add train_GAN.py
git commit -m "fix: freeze encoder and scope discriminator optimizer"
```

---

### Task 2: Align Feature Space in Discriminator Step

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Update `train_discriminator_step` to use `sampleDecoder` for real features**

```python
def train_discriminator_step(batch, discriminator, decoder, d_opt, config):
    d_opt.zero_grad()
    
    # 1. Real Features (Aligned to [-1, 1] space)
    enc_fold = FoldExt(cuda=config.cuda)
    enc_fold_nodes = []
    for example in batch:
        enc_fold_nodes.append(grassmodel.encode_structure_fold(enc_fold, example))
    
    with torch.no_grad():
        enc_fold_nodes = enc_fold.apply(discriminator.encoder, [enc_fold_nodes])
        enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
        
        real_features_list = []
        for fnode in enc_fold_nodes:
            root_code, _ = torch.chunk(fnode, 2, 1)
            # Pass through sampleDecoder to align space
            real_features_list.append(decoder.sampleDecoder(root_code))
        real_features = torch.cat(real_features_list, dim=0)
    
    # 2. Fake Features (Already in [-1, 1] space)
    z_p = torch.randn(len(batch), config.feature_size)
    if config.cuda:
        z_p = z_p.cuda()
    
    # Use current sampleDecoder for fake features (will be detached for D update)
    fake_features = decoder.sampleDecoder(z_p)
    
    # 3. Discriminator Outputs
    d_real = discriminator(real_features).mean()
    d_fake = discriminator(fake_features).mean()
    
    # 4. Gradient Penalty (Using .data to avoid tracking through sampleDecoder here)
    gp = compute_gradient_penalty(discriminator, real_features.data, fake_features.data, config)
    
    # 5. Total D Loss
    d_loss = d_fake - d_real + config.lambda_gp * gp
    d_loss.backward()
    d_opt.step()
    
    return d_loss.item(), d_real.item(), d_fake.item(), gp.item()
```

- [ ] **Step 2: Commit changes**

```bash
git add train_GAN.py
git commit -m "fix: align real features to [-1, 1] space in discriminator step"
```

---

### Task 3: Standardize Training Loop Order

**Files:**
- Modify: `train_GAN.py`

- [ ] **Step 1: Update `main` and `run_lr_range_test` to use standard WGAN-GP update order**

```python
# In run_lr_range_test
for i, batch in enumerate(dataloader):
    # ...
    # Standard WGAN-GP: Update D n_critic times first
    for _ in range(config.n_critic):
        d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, optimizer_D, config)
    
    # Then update G
    g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, decoder, optimizer_G, config)
    # ...
```

```python
# In main
for epoch in range(config.gan_epochs):
    for batch_idx, batch in enumerate(train_iter):
        # 1. Update Discriminator n_critic times
        for _ in range(config.n_critic):
            d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, d_opt, config)
        
        # 2. Update Generator once
        g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, decoder, g_opt, config)
        # ...
```

- [ ] **Step 2: Fix logging labels in `main`**

```python
# Header
header = '     Time    Epoch     Iteration    Progress(%)  D_Loss  G_Loss  ReconLoss'
# Template (ensure order matches)
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>8.2f},{:>8.2f},{:>10.2f}'.split(','))
# ...
print(log_template.format(strftime("%H:%M:%S", time.gmtime(time.time()-start)),
    epoch, config.gan_epochs, 1+batch_idx, len(train_iter),
    100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.gan_epochs),
    d_loss, g_loss, recon))
```

- [ ] **Step 3: Commit changes**

```bash
git add train_GAN.py
git commit -m "fix: standardize training loop order and correct logging"
```

---

### Task 4: Final Verification

- [ ] **Step 1: Run LR Range Test**

Run: `python train_GAN.py --no_plot` (Wait for prompt, then examine `models/lr_range_test.png`)
Expected: Smoothed D/G losses should stay relatively stable/bounded for low LRs, with G loss not showing immediate exponential growth.

- [ ] **Step 2: Monitor first few epochs of training**

Run: `python train_GAN.py`
Expected: `G_Loss` should start at a reasonable value and not show immediate exponential growth. `ReconLoss` should remain stable.
