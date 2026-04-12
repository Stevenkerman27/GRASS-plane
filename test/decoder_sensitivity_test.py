import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import grassmodel
import numpy as np

"""
测试项目: 解码器对 Latent Code 的敏感度测试 (Decoder Sensitivity)
测试目的: 
    1. 通过手动放大随机噪声的尺度（例如 x5, x10），观察解码器的输出是否随之变化。
    2. 如果在标准噪声下输出几乎一致，但在放大噪声后输出发生显著变化，说明解码器只是在原点附近过于“钝感”。
    3. 辅助判断是因为权重坍塌还是仅仅因为欠训练导致的“平坦”潜在空间。
"""

def test_sensitivity():
    try:
        decoder = torch.load('./models/vae_decoder_model.pkl', map_location='cpu', weights_only=False)
        decoder.eval()
    except FileNotFoundError:
        print("Error: VAE model not found in ./models/")
        return

    print("Testing sensitivity for different scales of random noise input:")
    for scale in [1.0, 5.0, 10.0]:
        all_means = []
        for i in range(10):
            root_code = torch.randn(1, 80) * scale
            boxes = grassmodel.decode_structure(decoder, root_code)
            if len(boxes) > 0:
                m = torch.cat(boxes, 0).mean().item()
                all_means.append(m)
        
        # 计算输出的变化程度
        variance = np.var(all_means) if all_means else 0.0
        print(f"  噪声 Scale {scale:4.1f}: 输出均值方差 = {variance:.10f}")

    print("\n--- 诊断结论建议 ---")
    print("1. 如果 Scale 1.0 方差接近 0 但 Scale 5.0/10.0 方差很大：说明模型欠训练，尚未学会在 N(0,1) 范围内的区分度。")
    print("2. 如果所有 Scale 方差都接近 0：说明解码器已死，权重发生了某种形式的严重坍塌。")

if __name__ == "__main__":
    test_sensitivity()
