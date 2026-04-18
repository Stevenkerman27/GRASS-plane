# 项目架构与开发约束准则 (System Architecture & Development Constraints)

## 1. 全局架构与设计规约 (Global Architecture & Design Conventions)

- **单一来源配置**: 所有模型的超参数、GAN的权重参数等魔术变量必须统一集中维护在 `util.py` 中，禁止在业务逻辑代码中硬编码。
- **组件化生成与拓扑组装 (Component-based Generation)**: 对于复杂的树形拓扑数据（如 OBB 树）生成，必须采用组件化工厂（Component Factories）与统一的装配器（如 `TreeAssembler`）架构，以隔离物理组件参数与后根遍历（Post-order traversal，`BOX`, `ADJ`, `SYM`）的栈操作复杂性，保证多布局（鸭翼、飞翼等）和变体组合拓扑的正确性。
- **VAE-WGAN-GP 双阶段训练架构**:
  - 第一阶段 (Autoencoder): 通过 `grassmodel.py` 训练重构与 KL 散度损失。
  - 第二阶段 (GAN): 通过 `train_GAN.py` 复用第一阶段预训练权重，解码器直接作为 GAN 生成器，编码器由 `GANDiscriminator` 包装以输出线性标量作为 Critic 判别器。
  - 拓扑候选采样 (Scouting Phase): 为防止严重的拓扑模式坍塌 (Topological Mode Collapse)，生成器与判别器在训练时必须基于随机噪声与真实数据潜变量 (mu) 的 L2 距离来获取 Top-K 候选树，再通过判别器打分与 Temperature Scaling 进行 Categorical 采样。严禁在全数据集中进行完全随机的盲抽。
- **动态批处理框架**: 核心逻辑依赖 `torchfold` 进行结构递归神经网络的动态批处理。由于原有的 `pytorch_tools` 包不可用，应统一使用独立安装的 `torchfold` 包并继承其 `Fold` 类。
- **设备一致性**: 全局根据 `config.cuda` 的布尔值决定设备挂载，通常使用 `if config.cuda: tensor = tensor.cuda()` 模式。避免在未检查配置的情况下直接硬编码 `.cuda()`，以兼容 CPU 环境。

## 2. 状态管理与核心防线 (State Management & Core Defenses)

- **梯度回传稳定性 (WGAN-GP)**: 在应用梯度惩罚 (Gradient Penalty) 时，需使用 `torch.autograd.grad` 并确保对插值向量执行 `requires_grad_(True)` 以正确计算关于输入的导数。
- **标量值获取**: 严禁使用已废弃的 `.data[0]`。必须统一使用 `.item()` 从 0 维张量中获取标量值，以兼容现代 PyTorch (>=1.0)。特别是当 0 维张量需要作为 Python 内置函数（如 `round()`, `math.sqrt()` 等）的参数时，必须显式调用 `.item()` 转换为 Python 原生标量，否则将引发 `TypeError: type Tensor doesn't define __X__ method`。
- **损失函数输入规范**: 
  - `nn.CrossEntropyLoss` 的 target 在 batch_size 为 1 时必须确保维度匹配（通常需要 `target.unsqueeze(0)`）。
  - 在手动聚合来自 Fold 的节点损失时，优先使用 `torch.stack` 而非 `torch.cat` 以处理 0 维损失张量。
- **变量封装**: 停止显式使用 `torch.autograd.Variable`。在 PyTorch 0.4+ 中，Tensor 已合并 Variable 功能，需统一改为 `.detach()`。
- **特征空间对齐 (Feature Alignment)**: 判别器必须在同一数值量级比较特征。通过在生成数据集时执行各向同性的几何缩放 (Isotropic Scaling)，确保所有数据样本天然限制在 `[-1, 1]` 空间内。这使得 VAE 编码器输出的 latent code 与 GAN 生成器（经过 Tanh 激活）输出的 fake feature 自动处于同一量级，无需再对真实样本额外通过 `SampleDecoder` 转换，避免了引入人工约束导致的模式崩溃。
- **WGAN-GP 训练时序**: 严格执行 `n_critic` 次判别器 (Critic) 更新后执行 1 次生成器更新的标准时序，确保判别器能准确估计真假分布间的 EM 距离。

## 3. 环境陷阱与第三方 API 怪癖 (Environment & API Peculiarities)

- **torchfold 兼容性**: 在现代环境中，`torchfold.Fold` 的构造函数签名为 `(self, volatile=False, cuda=False)`。扩展类 `FoldExt` 必须严格遵循此签名。
- **中文路径与编码**: 工作区涉及中文字符路径时，需注意 Python 环境的 IO 编码处理（见 `GEMINI.md` 规则）。

## 4. 短期路线图与技术债 (Short-term Roadmap & Tech Debt)

### 4.1 待办核心功能 (Backlog)

### 4.2 技术债清理
- [done] 优化 `Sampler` 与损失函数的向量化实现。
