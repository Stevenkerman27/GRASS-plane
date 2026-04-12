import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import grassdata
import grassmodel
from torchfoldext import FoldExt
import numpy as np

"""
测试项目: 潜在空间分布分析 (Latent Space Distribution)
测试目的: 
    1. 验证编码器产生的 Latent Code 是否符合标准正态分布 N(0, 1)。
    2. 如果分布偏离严重（如均值不为0或方差极小），说明 KL 散度正则化失效，这是导致生成结果单一的常见原因。
"""

def analyze_latent_space():
    try:
        encoder = torch.load('./models/vae_encoder_model.pkl', map_location='cpu', weights_only=False)
        encoder.eval()
        dataset = grassdata.GRASSDataset('data')
    except (FileNotFoundError, Exception) as e:
        print(f"Error initializing test: {e}")
        return

    all_means = []
    all_stds = []
    num_samples = min(50, len(dataset))

    print(f"Analyzing latent codes for {num_samples} dataset samples...")
    for i in range(num_samples):
        fold = FoldExt(cuda=False)
        node = grassmodel.encode_structure_fold(fold, dataset[i])
        # 将 node 包装在列表中，符合 torchfold 接口
        fnode = fold.apply(encoder, [[node]])[0]
        
        # VAE 编码器输出的前 80 维是 root_code (mean/reparameterization)
        root_code, _ = torch.chunk(fnode, 2, 1)
        all_means.append(root_code.mean().item())
        all_stds.append(root_code.std().item())

    avg_mean = np.mean(all_means)
    avg_std = np.mean(all_stds)

    print(f"\n--- Latent Code 统计数据 (期望值: Mean ~0.0, Std ~1.0) ---")
    print(f"  实际平均均值: {avg_mean:.4f}")
    print(f"  实际平均标准差: {avg_std:.4f}")

    if abs(avg_mean) > 0.5 or avg_std < 0.2:
        print("\n[RESULT] 警告: Latent 空间分布严重偏离！KL 散度正则化可能没有正确生效。")
    else:
        print("\n[RESULT] 通过: Latent 空间分布基本符合预期。")

if __name__ == "__main__":
    analyze_latent_space()
