import time
import os
from time import gmtime, strftime
from datetime import datetime
import math
import copy
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.data
from torchfoldext import FoldExt
import util
from dynamicplot import DynamicPlot

from grassdata import GRASSDataset
from grassmodel import GRASSDecoder
import grassmodel

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

def my_collate(batch):
    return batch

def setup_training():
    config = util.get_args()
    config.cuda = not config.no_cuda
    if config.gpu < 0 and config.cuda:
        config.gpu = 0
    torch.cuda.set_device(config.gpu)

    if config.no_plot:
        import matplotlib
        matplotlib.use('Agg')
    
    print("Loading pre-trained models...")
    # Using cpu to load first to avoid device mismatch, then move to cuda if needed
    encoder = torch.load(os.path.join(config.save_path, 'vae_encoder_model.pkl'), map_location='cpu', weights_only=False)
    decoder = torch.load(os.path.join(config.save_path, 'vae_decoder_model.pkl'), map_location='cpu', weights_only=False)
    
    discriminator = GANDiscriminator(encoder, config)
    
    # Freeze encoder
    for param in discriminator.encoder.parameters():
        param.requires_grad = False
    discriminator.encoder.eval()
    
    if config.cuda:
        discriminator.cuda()
        decoder.cuda()
        
    print("Loading data ...")
    grass_data = GRASSDataset(config.data_path)
    train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.gan_batch_size, shuffle=True, collate_fn=my_collate)
    
    # Scope d_opt to only discriminator.fc
    d_opt = torch.optim.Adam(discriminator.fc.parameters(), lr=config.gan_lr, betas=(0.5, 0.9))
    g_opt = torch.optim.Adam(decoder.parameters(), lr=config.gan_lr, betas=(0.5, 0.9))
    
    return config, discriminator, decoder, train_iter, d_opt, g_opt

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
    
    # 4. Gradient Penalty
    gp = compute_gradient_penalty(discriminator, real_features.data, fake_features.data, config)
    
    # 5. Total D Loss
    d_loss = d_fake - d_real + config.lambda_gp * gp
    d_loss.backward()
    d_opt.step()
    
    return d_loss.item(), d_real.item(), d_fake.item(), gp.item()

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
    
    g_loss = g_adv_loss + config.alpha1 * recon_loss + config.alpha2 * kldiv_loss
    g_loss.backward()
    g_opt.step()
    
    return g_loss.item(), g_adv_loss.item(), recon_loss.item(), kldiv_loss.item()

def run_lr_range_test(config, dataloader, discriminator, decoder):
    print("--- Starting LR Range Test ---")
    
    # Save initial state
    d_state = copy.deepcopy(discriminator.state_dict())
    g_state = copy.deepcopy(decoder.state_dict())
    
    lr_start = 1e-7
    lr_end = 1.0  # WGAN 通常测到 1.0 足够发现崩溃点
    
    optimizer_G = torch.optim.Adam(decoder.parameters(), lr=lr_start, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_start, betas=(0.5, 0.9))
    
    total_steps = len(dataloader)
    if total_steps <= 1:
        total_steps = 2
        
    lr_mult = (lr_end / lr_start) ** (1 / total_steps)
    
    lrs = []
    d_losses_record = []
    g_losses_record = []
    
    beta = 0.2
    avg_d_loss = 0.0
    avg_g_loss = 0.0
    best_d_loss = float('inf')
    initial_d_loss = None
    
    for i, batch in enumerate(dataloader):
        current_lr = optimizer_D.param_groups[0]['lr']
        
        # Standard WGAN-GP: Update D n_critic times first
        for _ in range(config.n_critic):
            d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, optimizer_D, config)
        
        avg_d_loss = beta * avg_d_loss + (1 - beta) * d_loss
        smoothed_d_loss = avg_d_loss / (1 - beta ** (i + 1))
        
        if i == 0:
            initial_d_loss = smoothed_d_loss
            
        if i > 0 and (abs(smoothed_d_loss) > abs(initial_d_loss) * 2 or math.isnan(smoothed_d_loss)):
            print(f"Loss diverged at step {i}, stopping LR test early.")
            break
            
        if smoothed_d_loss < best_d_loss:
            best_d_loss = smoothed_d_loss
            
        # Then update G
        g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, decoder, optimizer_G, config)
        current_g_loss_val = g_loss

        avg_g_loss = beta * avg_g_loss + (1 - beta) * current_g_loss_val
        smoothed_g_loss = avg_g_loss / (1 - beta ** (i + 1))
        
        lrs.append(current_lr)
        d_losses_record.append(smoothed_d_loss)
        g_losses_record.append(smoothed_g_loss)
        
        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= lr_mult
        for param_group in optimizer_D.param_groups:
            param_group['lr'] *= lr_mult
            
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, d_losses_record, label='Smoothed D Loss')
    plt.plot(lrs, g_losses_record, label='Smoothed G Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    os.makedirs(config.save_path, exist_ok=True)
    img_path = os.path.join(config.save_path, 'lr_range_test.png')
    plt.savefig(img_path)
    plt.close()
    
    discriminator.load_state_dict(d_state)
    decoder.load_state_dict(g_state)
    
    while True:
        try:
            user_lr = input(f"Please examine '{img_path}' and enter the selected learning rate: ")
            final_lr = float(user_lr.strip())
            if final_lr > 0:
                break
        except ValueError:
            pass
            
    return final_lr

def main():
    config, discriminator, decoder, train_iter, d_opt, g_opt = setup_training()
    
    final_lr = run_lr_range_test(config, train_iter, discriminator, decoder)
    config.gan_lr = final_lr
    for param_group in d_opt.param_groups:
        param_group['lr'] = final_lr
    for param_group in g_opt.param_groups:
        param_group['lr'] = final_lr
        
    print("Start GAN training ...")
    start = time.time()
    
    total_iter = config.gan_epochs * len(train_iter)
    if not config.no_plot:
        plot_x = [x for x in range(total_iter)]
        plot_d_loss = [None for x in range(total_iter)]
        plot_g_loss = [None for x in range(total_iter)]
        plot_recon_loss = [None for x in range(total_iter)]
        dyn_plot = DynamicPlot(title='GAN Training loss over iterations (GRASS)', xdata=plot_x, ydata={'D_Loss':plot_d_loss, 'G_Loss':plot_g_loss, 'Reconstruction_Loss':plot_recon_loss})
        iter_id = 0
        max_loss = 0
        min_loss = 0
    
    if config.save_snapshot:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        snapshot_folder = os.path.join(config.save_path, 'gan_snapshots_'+strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)
            
    header = '     Time    Epoch     Iteration    Progress(%)  D_Loss  G_Loss  ReconLoss'
    log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>8.2f},{:>8.2f},{:>10.2f}'.split(','))
    print(header)
    
    for epoch in range(config.gan_epochs):
        for batch_idx, batch in enumerate(train_iter):
            # 1. Update Discriminator n_critic times
            for _ in range(config.n_critic):
                d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, d_opt, config)
            
            # 2. Update Generator once
            g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, decoder, g_opt, config)
            
            if batch_idx % config.show_log_every == 0:
                print(log_template.format(strftime("%H:%M:%S", time.gmtime(time.time()-start)),
                    epoch, config.gan_epochs, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.gan_epochs),
                    d_loss, g_loss, recon))
            
            if not config.no_plot:
                plot_d_loss[iter_id] = d_loss
                plot_g_loss[iter_id] = g_loss
                plot_recon_loss[iter_id] = recon
                max_loss = max(max_loss, d_loss, g_loss, recon)
                min_loss = min(min_loss, d_loss, g_loss, recon)
                dyn_plot.setxlim(0., (iter_id+1)*1.05)
                dyn_plot.setylim(min_loss - 0.1 * abs(min_loss), max_loss*1.05)
                dyn_plot.update_plots(ydata={'D_Loss':plot_d_loss, 'G_Loss':plot_g_loss, 'Reconstruction_Loss':plot_recon_loss})
                iter_id += 1
                    
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
