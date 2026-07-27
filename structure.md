# 旧扁平 GRASS VAE-GAN 架构概览

本文是旧扁平 13D box VAE-GAN 路径的高层概览，`train_GAN.py` 为实验性实现。模型、训练与 typed 路径的实际接入状态以 `docs/aircraft_layout_vae_definitions.md` 为准；typed schema 与翼型编码以 `docs/typed_box_airfoil_encoding.md` 为准。

## 1. 核心组件架构

系统中包含五个关键模块，分布在编码、采样、解码和判别四个阶段：

### A. VAE RvNN 编码器 (`vae_encoder`)
- **类型**: 递归神经网络 (Recursive Neural Network)。
- **输入**: 层次化树结构物体（来自数据集）。
- **逻辑**: 从叶节点（Box）开始，根据拓扑关系递归地合并特征，最终在根节点生成全局特征向量。
- **输出**: 形状的压缩表示。

### B. Sampler (采样层)
- **输入**: `vae_encoder` 的根节点特征。
- **逻辑**: 计算均值 ($\mu$) 和方差 ($\log\sigma^2$)。使用重参数化技巧 $z = \mu + \epsilon \cdot \sigma$ 采样隐变量。
- **输出**: n-D 隐空间编码 (n-D code)。

### C. Sample Decoder (生成器首层)
- **输入**: 隐变量 $z$（来自 Sampler 或随机噪声）。
- **逻辑**: 一个全连接层，将隐空间编码映射回 RvNN 能够理解的根节点隐藏特征。
- **输出**: 递归解码的起始特征。

### D. RvNN 解码器 (`decoder` / Generator)
- **类型**: 递归神经网络。
- **输入**: 根节点特征。
- **逻辑**: 根据节点类型（Leaf, Adj, Sym）递归地拆分特征。
    - **Leaf**: 此文档的自由生成路径输出 13 维扁平盒子参数（10 几何 + 3 类别）。typed 路径按部件使用 payload，翼面为 `[8,29] + section_count` CST 截面序列；其 teacher-forced 重构由 `train.py` 默认路径使用。
    - **Adj/Sym**: 输出子节点的特征向量。
- **输出**: 重构或生成的完整树结构物体。

### E. 判别器 RvNN 编码器 (`discriminator.encoder`)
- **类型**: 独立的递归神经网络（结构与 `vae_encoder` 相同，但权重独立）。
- **输入**: 真实物体 或 生成物体。
- **逻辑**: 重新对物体进行结构编码，提取用于分类的深层特征。
- **输出**: 用于 WGAN 评分的根节点特征。

---

## 2. 训练与更新逻辑

系统采用 **WGAN-GP** 策略进行联合训练，每个 Batch 包含两个主要的更新步骤：

### 第一步：判别器更新 (Discriminator Step)
**目标**：提升判别器识别“生成伪造结构”的能力。
- **操作**:
    1. 使用 `discriminator.encoder` 编码真实物体，计算得分 $D(real)$。
    2. 使用 `decoder` 生成伪造物体，并用 `discriminator.encoder` 编码，计算得分 $D(fake)$。
    3. 计算 **梯度惩罚 (Gradient Penalty)** 以满足 Lipschitz 约束。
- **更新对象**: `discriminator.encoder` + `discriminator.fc` (线性层)。
- **损失函数**: $L_D = E[D(fake)] - E[D(real)] + \lambda_{gp} \cdot GP$。

### 第二步：生成器更新 (Generator/VAE Step)
**目标**：同时提升重构精度、隐空间分布规范性以及骗过判别器的能力。
- **操作**:
    1. **对抗路径**: 将随机噪声通过 `decoder` 生成物体，经 `discriminator` 评分，计算对抗损失 $L_{adv} = -E[D(fake)]$。
    2. **重构路径**: 将真实物体通过 `vae_encoder` -> `Sampler` -> `decoder`，计算重构损失 $L_{recon}$（MSE + CrossEntropy）。
    3. **正则路径**: 计算 Sampler 的 KL 散度。
- **更新对象**: `vae_encoder` + `decoder` (含 Sample Decoder)。
- **损失函数**: $L_G = L_{adv} + \alpha_1 \cdot L_{recon} + \alpha_2 \cdot L_{KL}$。

---

## 3. 核心差异总结

| 特性 | VAE 编码器 | 判别器编码器 |
| :--- | :--- | :--- |
| **训练目的** | 学习如何“理解”并压缩真实结构 | 学习如何“挑刺”并识别非自然结构 |
| **梯度来源** | 重构损失 + KL 散度 | 对抗损失 (WGAN) |
| **更新频率** | 随生成器一起更新 | 每步更新 $n_{critic}$ 次 |
| **权重初始化** | 预训练 VAE 权重 | 由 VAE 权重初始化，随后独立演化 |

---
*注：本架构遵循 MATLAB 原始实现，但在 PyTorch 中引入了 WGAN-GP 和动态折叠 (TorchFold) 技术以提升训练稳定性和效率。*
