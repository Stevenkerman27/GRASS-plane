import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import grassmodel

"""
测试项目: VAE 生成样本的多样性验证 (Generation Variance Check)
测试目的: 
    1. 验证对于不同的随机噪声输入 (root_code)，解码器生成的 3D 盒子数量和位置是否具有统计学上的差异。
    2. 诊断模型是否发生了严重的模式坍塌 (Mode Collapse)，即无论输入如何都产生几乎相同的输出。
"""

def verify_diversity():
    try:
        decoder = torch.load('./models/vae_decoder_model.pkl', map_location='cpu', weights_only=False)
        decoder.eval()
    except FileNotFoundError:
        print("Error: VAE model not found in ./models/")
        return

    all_box_counts = []
    all_box_means = []
    num_samples = 10

    print(f"Testing {num_samples} random samples...")
    for i in range(num_samples):
        # 使用标准正态分布噪声
        root_code = torch.randn(1, 80)
        boxes = grassmodel.decode_structure(decoder, root_code)
        all_box_counts.append(len(boxes))
        
        if len(boxes) > 0:
            # 计算所有生成盒子的坐标/维度的平均值作为“指纹”
            stacked_boxes = torch.cat(boxes, 0).detach().numpy()
            all_box_means.append(np.mean(stacked_boxes))
        else:
            all_box_means.append(0.0)

    print(f"Box counts: {all_box_counts}")
    print(f"Box means:  {[f'{m:.6f}' for m in all_box_means]}")

    # 判断是否完全一致
    if len(set(all_box_counts)) == 1 and np.var(all_box_means) < 1e-12:
        print("\n[RESULT] 警告: 检测到严重的模式坍塌！所有样本的结构和数值几乎完全一致。")
    elif np.var(all_box_means) < 1e-6:
        print("\n[RESULT] 提示: 样本之间存在极微小差异，但多样性非常低，模型可能处于欠训练或收敛到平均形状。")
    else:
        print("\n[RESULT] 通过: 样本之间存在明显的数值或结构差异。")

if __name__ == "__main__":
    verify_diversity()
