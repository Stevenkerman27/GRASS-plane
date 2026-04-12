import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import grassmodel

"""
测试项目: VAE 权重分布检测 (Weight Distribution Inspector)
测试目的: 
    1. 检查 SampleDecoder 和 NodeClassifier 的权重是否消失 (Vanishing) 或过大。
    2. 判断网络是否处于“死掉”状态，即权重过小导致无法传递有效信号。
"""

def inspect_weights():
    try:
        decoder = torch.load('./models/vae_decoder_model.pkl', map_location='cpu', weights_only=False)
        decoder.eval()
    except FileNotFoundError:
        print("Error: VAE model not found in ./models/")
        return

    print("--- SampleDecoder 权重分布 (Noise to Latent Feature) ---")
    for name, param in decoder.sample_decoder.named_parameters():
        print(f"  {name:12s}: mean={param.mean().item():.4f}, std={param.std().item():.4f}, max={param.abs().max().item():.4f}")

    print("\n--- NodeClassifier 权重分布 (Node Type Classification) ---")
    for name, param in decoder.node_classifier.named_parameters():
        print(f"  {name:12s}: mean={param.mean().item():.4f}, std={param.std().item():.4f}, max={param.abs().max().item():.4f}")

if __name__ == "__main__":
    inspect_weights()
