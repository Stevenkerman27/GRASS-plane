在训练开始前添加LR Range test. 在一个 Epoch 内，让学习率从一个极小值(如10^{-7})增长到一个较大值(如0.1), 绘制出学习率对应的 Loss图并保存。随后要求用户在终端输入生成器和判别器的学习率并使用输入的学习率正式开始训练。示例代码如下：

def run_lr_range_test(config, dataloader, device):
    print("--- Starting LR Range Test ---")
    
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    
    lr_start = 1e-7
    lr_end = 1.0  # WGAN 通常测到 1.0 足够发现崩溃点
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_start, betas=(0.0, 0.9))
    
    total_steps = len(dataloader)
    if total_steps <= 1:
        total_steps = 2
        
    lambda_gp = config.get('lambda_gp')
    n_critic = config.get('n_critic')
    
    # 采用指数级（按倍数）增长
    lr_mult = (lr_end / lr_start) ** (1 / total_steps)
    
    lrs = []
    d_losses_record = []
    g_losses_record = []
    
    # 引入 EMA 平滑变量
    beta = 0.2
    avg_d_loss = 0.0
    avg_g_loss = 0.0
    best_d_loss = float('inf')
    initial_d_loss = None
    
    for i, (foils, conds) in enumerate(dataloader):
        foils = foils.to(device)
        conds = conds.to(device)
        batch_size = foils.size(0)
        
        # 获取当前学习率用于记录
        current_lr = optimizer_D.param_groups[0]['lr']
        
        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, config.get('noise_dimension')).to(device)
        fake_foils = generator(z, conds)
        
        real_validity = discriminator(foils, conds)
        fake_validity = discriminator(fake_foils.detach(), conds)
        
        gradient_penalty, _ = compute_gradient_penalty(
            discriminator, foils, fake_foils.detach(), conds, device
        )
        
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        
        # 计算 EMA 平滑 D Loss
        avg_d_loss = beta * avg_d_loss + (1 - beta) * d_loss.item()
        smoothed_d_loss = avg_d_loss / (1 - beta ** (i + 1))

        if i == 0:
            initial_d_loss = smoothed_d_loss
        
        # 防爆机制。如果发散，立即停止，保护图表比例尺
        if i > 0 and (abs(smoothed_d_loss) > abs(initial_d_loss) * 2 or math.isnan(smoothed_d_loss)):
            print(f"Loss diverged at step {i}, stopping LR test early.")
            break
            
        if smoothed_d_loss < best_d_loss:
            best_d_loss = smoothed_d_loss
            
        # --- Train Generator ---
        current_g_loss_val = 0.0
        # 遵循 n_critic 设定
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z_gen = torch.randn(batch_size, config.get('noise_dimension')).to(device)
            fake_foil_gen = generator(z_gen, conds)
            fake_validity_gen = discriminator(fake_foil_gen, conds)
            g_loss = -torch.mean(fake_validity_gen)
            g_loss.backward()
            optimizer_G.step()
            current_g_loss_val = g_loss.item()
        else:
            current_g_loss_val = g_losses_record[-1] if len(g_losses_record) > 0 else 0.0

        # 平滑 G loss
        avg_g_loss = beta * avg_g_loss + (1 - beta) * current_g_loss_val
        smoothed_g_loss = avg_g_loss / (1 - beta ** (i + 1))
        
        lrs.append(current_lr)
        d_losses_record.append(smoothed_d_loss)
        g_losses_record.append(smoothed_g_loss)
        
        # 更新学习率 (乘以常数)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= lr_mult
        for param_group in optimizer_D.param_groups:
            param_group['lr'] *= lr_mult
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, d_losses_record, label='Smoothed D Loss')
    plt.plot(lrs, g_losses_record, label='Smoothed G Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('model/lr_range_test.png')
    plt.close()
    
    # 交互输入逻辑保留你的原样即可
    while True:
        try:
            user_lr = input("Please examine 'model/lr_range_test.png' and enter the selected learning rate: ")
            final_lr = float(user_lr.strip())
            if final_lr > 0:
                break
        except ValueError:
            pass
            
    return final_lr