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
    def __init__(self, config):
        super(GANDiscriminator, self).__init__()
        # Discriminator has its own RvNN encoder
        self.encoder = grassmodel.GRASSEncoder(config)
        # 3-layer scorer (2 hidden layers) matching MATLAB's Wdc1, Wdc2, Wscore
        self.fc = nn.Sequential(
            nn.Linear(config.feature_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, features):
        return self.fc(features)

    def get_root_features(self, fold, batch):
        """Encodes a batch of structures using the discriminator's encoder and extracts root features."""
        nodes = []
        for example in batch:
            # Match MATLAB: encode directly to root feature without Sampler
            nodes.append(grassmodel.encode_structure_fold(fold, example, use_sampler=False))
        
        # Apply the fold to get encoded features
        encoded = fold.apply(self.encoder, [nodes])
        return encoded[0]

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
    vae_encoder = torch.load(os.path.join(config.save_path, 'vae_encoder_model.pkl'), map_location='cpu', weights_only=False)
    decoder = torch.load(os.path.join(config.save_path, 'vae_decoder_model.pkl'), map_location='cpu', weights_only=False)
    
    # Initialize Discriminator with its own encoder (can be initialized with VAE encoder weights or random)
    discriminator = GANDiscriminator(config)
    # Following MATLAB's philosophy of using the same architecture but independent params for D
    # We can copy the pre-trained weights to speed up convergence
    discriminator.encoder.load_state_dict(vae_encoder.state_dict())
    
    if config.cuda:
        discriminator.cuda()
        vae_encoder.cuda()
        decoder.cuda()
        
    print("Loading data ...")
    grass_data = GRASSDataset(config.data_path)
    train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.gan_batch_size, shuffle=True, collate_fn=my_collate)
    
    # D updates its entire structure (Encoder + FC)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=config.gan_lr, betas=(config.gan_beta1, config.gan_beta2))
    # G (Generator) in VAE-GAN context often includes the VAE Encoder as well to maintain latent space consistency
    g_opt = torch.optim.Adam(list(decoder.parameters()) + list(vae_encoder.parameters()), lr=config.gan_lr, betas=(config.gan_beta1, config.gan_beta2))
    
    return config, discriminator, vae_encoder, decoder, train_iter, d_opt, g_opt

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

class GANWrapper(nn.Module):
    """Wrapper to allow TorchFold to call methods from both G and D.Encoder in one pass.
    If detach_gen is True, generator outputs are detached to prevent G updates during D step,
    while still allowing D's encoder to receive gradients.
    """
    def __init__(self, generator, discriminator_encoder, detach_gen=False):
        super(GANWrapper, self).__init__()
        self.generator = generator
        self.discriminator_encoder = discriminator_encoder
        self.detach_gen = detach_gen
        
    def boxDecoder(self, f): 
        res = self.generator.boxDecoder(f)
        return res.detach() if self.detach_gen else res
        
    def adjDecoder(self, f): 
        res = self.generator.adjDecoder(f)
        if self.detach_gen:
            return res[0].detach(), res[1].detach()
        return res
        
    def symDecoder(self, f): 
        res = self.generator.symDecoder(f)
        if self.detach_gen:
            return res[0].detach(), res[1].detach()
        return res
        
    def sampleDecoder(self, f): 
        res = self.generator.sampleDecoder(f)
        return res.detach() if self.detach_gen else res
    
    def nodeClassifier(self, f): 
        res = self.generator.nodeClassifier(f)
        return res.detach() if self.detach_gen else res
        
    def classifyLossEstimator(self, label_vector, gt_label_vector):
        return self.generator.classifyLossEstimator(label_vector, gt_label_vector)
    
    def boxEncoder(self, b): return self.discriminator_encoder.boxEncoder(b)
    def adjEncoder(self, l, r): return self.discriminator_encoder.adjEncoder(l, r)
    def symEncoder(self, f, s): return self.discriminator_encoder.symEncoder(f, s)
    def sampleEncoder(self, f): return self.discriminator_encoder.sampleEncoder(f)

def generate_and_encode_fake(fold, wrapper, batch, z_p):
    """Recursively generates a structure using G and then encodes it using D.Encoder.
    Also collects classification losses for structural integrity.
    """
    cat_losses = []
    def recurse(node, feature):
        # Ensure the generator maintains structural knowledge even in GAN path
        label_logits = fold.add('nodeClassifier', feature)
        cat_losses.append(fold.add('classifyLossEstimator', label_logits, node.label))
        
        if node.is_leaf():
            # G: Gen Box -> (Optional Detach) -> D: Encode Box
            gen_box = fold.add('boxDecoder', feature)
            return fold.add('boxEncoder', gen_box)
        elif node.is_adj():
            # G: Split -> (Optional Detach) -> Recurse -> D: Merge
            l_f, r_f = fold.add('adjDecoder', feature).split(2)
            return fold.add('adjEncoder', recurse(node.left, l_f), recurse(node.right, r_f))
        elif node.is_sym():
            # G: Split Sym -> (Optional Detach) -> Recurse -> D: Merge Sym
            child_f, sym_p = fold.add('symDecoder', feature).split(2)
            return fold.add('symEncoder', recurse(node.left, child_f), sym_p)

    encoded_root_nodes = []
    for i, example in enumerate(batch):
        # G: noise to root feature
        root_f = fold.add('sampleDecoder', z_p[i:i+1])
        # Structural recursive path
        res = recurse(example.root, root_f)
        # D: Final encoding stage
        encoded_root_nodes.append(res)
        
    results = fold.apply(wrapper, [encoded_root_nodes, cat_losses])
    return results[0], results[1]

def train_discriminator_step(batch, discriminator, decoder, d_opt, grass_data, config):
    d_opt.zero_grad()
    
    # 1. Real path: Real structure -> D.Encoder -> D.FC
    real_fold = FoldExt(cuda=config.cuda)
    real_features = discriminator.get_root_features(real_fold, batch)
    
    # 2. Fake path: Noise z -> G (decoder) -> Object -> D.Encoder -> D.FC
    K = config.gan_k_candidates
    batch_size = len(batch)
    z_p = torch.randn(batch_size, config.feature_size)
    if config.cuda:
        z_p = z_p.cuda()
    
    all_candidate_trees = []
    for i in range(batch_size):
        random_indices = torch.randint(0, len(grass_data), (K,))
        for idx in random_indices:
            all_candidate_trees.append(grass_data[idx])
            
    z_p_expanded = z_p.repeat_interleave(K, dim=0)
    
    # Use wrapper with detach_gen=True to block gradients to G but allow them to D.Encoder
    wrapper = GANWrapper(decoder, discriminator.encoder, detach_gen=True)
    fake_fold = FoldExt(cuda=config.cuda)
    
    # Note: We don't need cat_losses in D step as we don't update G
    fake_features_all, _ = generate_and_encode_fake(fake_fold, wrapper, all_candidate_trees, z_p_expanded)
    
    # Score all candidates to find the best ones
    with torch.no_grad():
        scores_all = discriminator(fake_features_all)
        scores_reshaped = scores_all.view(batch_size, K)
        best_indices = torch.argmax(scores_reshaped, dim=1)
        
    gather_indices = best_indices + torch.arange(0, batch_size, device=scores_all.device) * K
    fake_features = fake_features_all[gather_indices]
        
    d_real_out = discriminator(real_features)
    d_fake_out = discriminator(fake_features)
    
    d_real = d_real_out.mean()
    d_fake = d_fake_out.mean()
    
    # GP calculation (detach features to only compute GP on Scorer/FC part)
    gp = compute_gradient_penalty(discriminator, real_features.detach(), fake_features.detach(), config)
    
    d_loss = d_fake - d_real + config.lambda_gp * gp
    d_loss.backward()
    d_opt.step()
    
    return d_loss.item(), d_real.item(), d_fake.item(), gp.item()

def train_generator_step(batch, discriminator, vae_encoder, decoder, g_opt, grass_data, config):
    g_opt.zero_grad()
    
    # Adversarial Loss: No detach, we want gradients for G
    K = config.gan_k_candidates
    batch_size = len(batch)
    z_p = torch.randn(batch_size, config.feature_size)
    if config.cuda:
        z_p = z_p.cuda()
        
    all_candidate_trees = []
    for i in range(batch_size):
        random_indices = torch.randint(0, len(grass_data), (K,))
        for idx in random_indices:
            all_candidate_trees.append(grass_data[idx])
    
    z_p_expanded = z_p.repeat_interleave(K, dim=0)
    # G step: detach_gen=False, gradients flow through everything
    wrapper = GANWrapper(decoder, discriminator.encoder, detach_gen=False)
    adv_fold = FoldExt(cuda=config.cuda)
    
    fake_features_all, cat_losses_all = generate_and_encode_fake(adv_fold, wrapper, all_candidate_trees, z_p_expanded)
    
    with torch.no_grad():
        scores_all = discriminator(fake_features_all)
        scores_reshaped = scores_all.view(batch_size, K)
        best_indices = torch.argmax(scores_reshaped, dim=1)
        
    gather_indices = best_indices + torch.arange(0, batch_size, device=scores_all.device) * K
    best_fake_features = fake_features_all[gather_indices]
    
    # Select corresponding cat losses for the best candidates
    # We must be careful here: cat_losses_all is a long tensor of all losses
    # The number of cat_losses per tree can vary, but generate_and_encode_fake processed them in sequence
    # To simplify and ensure stability, we take the mean of all cat_losses_all or just reconstruction cat_loss
    # For now, let's include cat_loss from the adversarial path to ensure nodeClassifier is updated
    g_cat_adv_loss = cat_losses_all.mean()
    
    g_adv_loss = -discriminator(best_fake_features).mean()
    
    # 2. VAE Loss (Reconstruction path using vae_encoder)
    enc_fold = FoldExt(cuda=config.cuda)
    enc_fold_nodes = []
    for example in batch:
        enc_fold_nodes.append(grassmodel.encode_structure_fold(enc_fold, example))
    enc_fold_nodes = enc_fold.apply(vae_encoder, [enc_fold_nodes])
    enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
    
    dec_fold = FoldExt(cuda=config.cuda)
    box_nodes, sym_nodes, cat_nodes, kld_fold_nodes = [], [], [], []
    for example, fnode in zip(batch, enc_fold_nodes):
        root_code, kl_div = torch.chunk(fnode, 2, 1)
        b_nodes, s_nodes, c_nodes = grassmodel.decode_structure_fold(dec_fold, root_code, example)
        box_nodes.extend(b_nodes)
        sym_nodes.extend(s_nodes)
        cat_nodes.extend(c_nodes)
        kld_fold_nodes.append(kl_div)
        
    apply_lists = [l for l in [box_nodes, sym_nodes, cat_nodes, kld_fold_nodes] if l]
    apply_res = dec_fold.apply(decoder, apply_lists)
    
    res_idx = 0
    device = torch.cuda.current_device() if config.cuda else 'cpu'
    
    geom_loss = apply_res[res_idx].sum(dim=0)[0] / len(batch) if box_nodes else torch.tensor(0.0, device=device)
    cls_loss = apply_res[res_idx].sum(dim=0)[1] / len(batch) if box_nodes else torch.tensor(0.0, device=device)
    if box_nodes: res_idx += 1
        
    sym_loss = apply_res[res_idx].sum() / len(batch) if sym_nodes else torch.tensor(0.0, device=device)
    if sym_nodes: res_idx += 1
        
    cat_loss = apply_res[res_idx].sum() / len(batch) if cat_nodes else torch.tensor(0.0, device=device)
    if cat_nodes: res_idx += 1
        
    kldiv_loss = apply_res[res_idx].sum().mul(-0.5) / len(batch) if kld_fold_nodes else torch.tensor(0.0, device=device)
        
    recon_loss = (config.lambda_geom * geom_loss + 
                  config.lambda_cls * cls_loss + 
                  config.lambda_sym * sym_loss + 
                  config.lambda_cat * cat_loss)
                  
    g_loss = g_adv_loss + config.alpha1 * recon_loss + config.alpha2 * kldiv_loss
    g_loss.backward()
    g_opt.step()
    
    return g_loss.item(), g_adv_loss.item(), recon_loss.item(), kldiv_loss.item()

def run_lr_range_test(config, dataloader, discriminator, vae_encoder, decoder):
    print("--- Starting LR Range Test ---")
    
    # Save initial state
    d_state = copy.deepcopy(discriminator.state_dict())
    ve_state = copy.deepcopy(vae_encoder.state_dict())
    vd_state = copy.deepcopy(decoder.state_dict())
    
    lr_start = 1e-7
    lr_end = 0.1
    
    optimizer_G = torch.optim.Adam(list(decoder.parameters()) + list(vae_encoder.parameters()), lr=lr_start, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_start, betas=(0.5, 0.9))
    
    total_steps = len(dataloader)
    if total_steps <= 1:
        total_steps = 2
        
    lr_mult = (lr_end / lr_start) ** (1 / total_steps)
    
    lrs = []
    
    metrics = {
        'd_loss': [], 'g_loss': [], 
        'd_real': [], 'd_fake': [], 'w_dist': [], 'gp': [],
        'g_adv': [], 'recon': [], 'kld': []
    }
    
    beta = 0.2
    avgs = {k: 0.0 for k in metrics.keys()}
    
    for i, batch in enumerate(dataloader):
        current_lr = optimizer_D.param_groups[0]['lr']
        
        for _ in range(config.n_critic):
            d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, optimizer_D, dataloader.dataset, config)
        
        g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, vae_encoder, decoder, optimizer_G, dataloader.dataset, config)

        w_dist = d_real - d_fake
        
        current_vals = {
            'd_loss': d_loss, 'g_loss': g_loss,
            'd_real': d_real, 'd_fake': d_fake, 'w_dist': w_dist, 'gp': gp,
            'g_adv': g_adv, 'recon': recon, 'kld': kld
        }
        
        lrs.append(current_lr)
        for k, v in current_vals.items():
            avgs[k] = beta * avgs[k] + (1 - beta) * v
            smoothed = avgs[k] / (1 - beta ** (i + 1))
            metrics[k].append(smoothed)
        
        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= lr_mult
        for param_group in optimizer_D.param_groups:
            param_group['lr'] *= lr_mult
            
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Total Losses
    axs[0, 0].plot(lrs, metrics['d_loss'], label='D Total Loss')
    axs[0, 0].plot(lrs, metrics['g_loss'], label='G Total Loss')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title('Total Losses')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.5)
    
    # 2. Critic Outputs
    axs[0, 1].plot(lrs, metrics['d_real'], label='D Real')
    axs[0, 1].plot(lrs, metrics['d_fake'], label='D Fake')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title('Critic Outputs')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", ls="-", alpha=0.5)
    
    # 3. W-Distance & GP
    axs[1, 0].plot(lrs, metrics['w_dist'], label='W-Distance (Real - Fake)')
    axs[1, 0].plot(lrs, metrics['gp'], label='Gradient Penalty')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_title('W-Distance and GP')
    axs[1, 0].legend()
    axs[1, 0].grid(True, which="both", ls="-", alpha=0.5)
    
    # 4. Generator Losses
    axs[1, 1].plot(lrs, metrics['g_adv'], label='G Adversarial')
    axs[1, 1].plot(lrs, metrics['recon'], label='G Reconstruction')
    axs[1, 1].plot(lrs, metrics['kld'], label='G KL Divergence')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_title('Generator Decoupled Losses')
    axs[1, 1].legend()
    axs[1, 1].grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(config.save_path, exist_ok=True)
    img_path = os.path.join(config.save_path, 'lr_range_test.png')
    plt.savefig(img_path)
    plt.close()
    
    discriminator.load_state_dict(d_state)
    vae_encoder.load_state_dict(ve_state)
    decoder.load_state_dict(vd_state)
    
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
    config, discriminator, vae_encoder, decoder, train_iter, d_opt, g_opt = setup_training()
    grass_data = train_iter.dataset
    
    final_lr = run_lr_range_test(config, train_iter, discriminator, vae_encoder, decoder)
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
        plot_w_dist = [None for x in range(total_iter)]
        plot_recon_loss = [None for x in range(total_iter)]
        dyn_plot = DynamicPlot(title='GAN Training metrics over iterations', xdata=plot_x, ydata={'D_Loss':plot_d_loss, 'G_Loss':plot_g_loss, 'W_Dist':plot_w_dist, 'Recon_Loss':plot_recon_loss})
        iter_id = 0
        max_loss = 0
        min_loss = 0
    
    if config.save_snapshot:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        snapshot_folder = os.path.join(config.save_path, 'gan_snapshots_'+strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)
            
    header = '     Time    Epoch     Iteration    Progress(%)  D_Loss  G_Loss  W-Dist   GP   ReconLoss'
    log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>8.2f},{:>8.2f},{:>8.2f},{:>6.2f},{:>10.2f}'.split(','))
    print(header)
    
    for epoch in range(config.gan_epochs):
        for batch_idx, batch in enumerate(train_iter):
            # 1. Update Discriminator n_critic times
            for _ in range(config.n_critic):
                d_loss, d_real, d_fake, gp = train_discriminator_step(batch, discriminator, decoder, d_opt, grass_data, config)
            
            # 2. Update Generator once
            g_loss, g_adv, recon, kld = train_generator_step(batch, discriminator, vae_encoder, decoder, g_opt, grass_data, config)
            
            w_dist = d_real - d_fake
            
            if batch_idx % config.show_log_every == 0:
                print(log_template.format(strftime("%H:%M:%S", time.gmtime(time.time()-start)),
                    epoch, config.gan_epochs, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.gan_epochs),
                    d_loss, g_loss, w_dist, gp, recon))
            
            if not config.no_plot:
                plot_d_loss[iter_id] = d_loss
                plot_g_loss[iter_id] = g_loss
                plot_w_dist[iter_id] = w_dist
                plot_recon_loss[iter_id] = recon
                max_loss = max(max_loss, d_loss, g_loss, recon, w_dist)
                min_loss = min(min_loss, d_loss, g_loss, recon, w_dist)
                dyn_plot.setxlim(0., (iter_id+1)*1.05)
                dyn_plot.setylim(min_loss - 0.1 * abs(min_loss), max_loss*1.05)
                dyn_plot.update_plots(ydata={'D_Loss':plot_d_loss, 'G_Loss':plot_g_loss, 'W_Dist':plot_w_dist, 'Recon_Loss':plot_recon_loss})
                iter_id += 1
                    
        if config.save_snapshot and (epoch+1) % config.save_snapshot_every == 0:
            print("Saving snapshots ...")
            torch.save(vae_encoder, snapshot_folder+f'//gan_encoder_epoch_{epoch+1}.pkl')
            torch.save(decoder, snapshot_folder+f'//gan_decoder_epoch_{epoch+1}.pkl')
            
    print("Saving final models ...")
    torch.save(vae_encoder, config.save_path+'//gan_encoder_model.pkl')
    torch.save(decoder, config.save_path+'//gan_decoder_model.pkl')
    print("DONE")

if __name__ == "__main__":
    main()
