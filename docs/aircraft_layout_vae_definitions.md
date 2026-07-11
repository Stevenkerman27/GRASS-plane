# 飞行器布局编码与 GRASS VAE 定义

本文档总结当前代码中已经实际实现的飞行器布局编码、VAE 训练、递归 GRASS 模型与 torchfold 适配细节。它区分仍由正式训练和自由生成入口使用的旧扁平 13D 路径，以及已实现 encoder/teacher-forced decoder smoke test、但尚未接入正式训练或自由生成的 typed 路径。typed schema 与实施状态的权威定义位于 `docs/typed_box_airfoil_encoding.md`。`train_GAN.py`、`GAN_gen.py`、判别器和 WGAN-GP 训练不作为本文定义来源。

## 1. 权威代码位置

- 数据树定义与 `.mat` 数据加载: `grassdata.py`
- 飞行器数据生成: `data/generate_dataset.py`
- VAE 编码器、解码器、损失估计与生成递归: `grassmodel.py`
- VAE 训练入口: `train.py`
- 模型超参数与损失权重参数: `util.py`
- torchfold 兼容扩展: `torchfoldext.py`
- 传统 13 维 box 的生成结果可视化: `draw3dobb.py`；typed 翼面 OBB 使用 `test/plot_obb.py`。
- MATLAB 原始格式参考: `Grass-matlab/data/Data Format.txt`

## 2. 飞行器布局树编码

### 2.1 节点类型

当前树结构由 `grassdata.Tree.NodeType` 定义三类节点:

| 节点 | 数值 | 含义 |
| --- | ---: | --- |
| `BOX` | 0 | 叶节点，保存一个部件 box 编码 |
| `ADJ` | 1 | 二元装配节点，连接两个子结构 |
| `SYM` | 2 | 一元对称节点，对一个 generator 子结构施加对称 |

每个 `Tree.Node` 保存:

- `box`: 仅 `BOX` 节点使用。
- `sym`: 仅 `SYM` 节点使用。
- `left`: `ADJ` 的左子节点，或 `SYM` 的 generator 子节点。
- `right`: 仅 `ADJ` 使用。
- `label`: `torch.LongTensor([node_type.value])`，用于节点类型分类损失。

### 2.2 后序栈式操作序列

数据中的 `ops` 是后序栈式表达式，而不是显式 parent-child 索引。

构树逻辑位于 `Tree.__init__(boxes, ops, syms)`:

1. 遍历 `ops`。
2. 遇到 `BOX(0)`: 从 `box_list` 取一个 box，压入栈。
3. 遇到 `ADJ(1)`: 从栈顶弹出两个节点，分别作为 `left_node` 与 `right_node`，构造一个 `ADJ` 节点后压回栈。
4. 遇到 `SYM(2)`: 从栈顶弹出一个节点作为 generator，从 `sym_param` 取一个 symmetry 参数，构造一个 `SYM` 节点后压回栈。
5. 遍历结束后栈中必须只剩一个根节点，否则 `assert len(queue) == 1` 失败。

注意: 当前 `ADJ` 的两个子节点顺序由 `grassdata.py` 中 `left_node = queue.pop(); right_node = queue.pop()` 决定。文档和后续代码应以这个实现为准，不自行改写左右子树语义。

### 2.3 旧扁平 13 维 box 编码与 typed 路径

旧的扁平训练路径使用 `util.py` 中的 `box_code_size = 13`。新的 typed box 路径按部件保存 payload：机身和发动机使用 10 维 geometry，翼面使用 8 维 geometry 加 30 维 Bezier 翼型 code；其带类别 one-hot 的扁平导出长度分别为 13D 与 41D。详见 `docs/typed_box_airfoil_encoding.md`。

13 维 box 向量格式为:

```text
[x1, y1, z1, x2, y2, z2, L1, H1, L2, H2, cls_fuselage, cls_wing_tail, cls_engine]
```

几何部分为前 10 维:

- `c1 = [x1, y1, z1]`: 第一个截面中心。
- `c2 = [x2, y2, z2]`: 第二个截面中心。
- `L1, H1`: 第一个截面的 chord/length 与 thickness/height。
- `L2, H2`: 第二个截面的 chord/length 与 thickness/height。

语义类别为后 3 维:

- `[1, 0, 0]`: fuselage。
- `[0, 1, 0]`: wing 或 stabilizer。
- `[0, 0, 1]`: engine。

数据生成时类别 one-hot 会从 `{0, 1}` 映射到 `{-1, 1}`。训练时 `boxLossEstimator` 对预测 box 的后 3 维直接使用 logits，与 ground truth 后 3 维的 `argmax` 作为类别标签计算交叉熵。

### 2.4 8 维 symmetry 编码

当前 `symmetry_size = 8`，格式继承自 GRASS 原始约定:

```text
[type, p1, p2, p3, q1, q2, q3, r]
```

`type` 的含义:

- `-1`: rotational symmetry。
- `0`: translational symmetry。
- `1`: reflectional symmetry。

各类型解释:

| 类型 | `p1:p3` | `q1:q3` | `r` |
| --- | --- | --- | --- |
| rotation | 旋转轴方向 | 旋转轴上一点 | `1 / repetitions` |
| translation | 位移向量 | generator 最后一次平移后的中心参考点 | 未使用 |
| reflection | 镜像平面法向 | 镜像平面参考点 | 未使用 |

生成阶段 `decode_structure` 使用 `abs(s[0] + 1) < 0.15`、`abs(s[0]) < 0.15`、`abs(s[0] - 1) < 0.15` 判断三类 symmetry。

### 2.5 当前飞行器数据生成范围

`data/generate_dataset.py` 当前生成 3 类布局:

- `conventional`: 常规布局。
- `canard`: 鸭翼布局。
- `flying_wing`: 飞翼布局。

基础部件工厂:

- `build_fuselage()`: 机身，类别 `[1, 0, 0]`。
- `build_wing(...)`: 主翼、鸭翼、平尾、垂尾，类别 `[0, 1, 0]`。
- `build_engine(...)`: 发动机，类别 `[0, 0, 1]`。

拓扑装配通过 `TreeAssembler` 完成:

- `push_box(box)`: 添加 box，并向 `ops` 写入 `BOX_OP`。
- `apply_adj()`: 向 `ops` 写入 `ADJ_OP`。
- `apply_sym(sym_vector)`: 保存 symmetry 参数，并向 `ops` 写入 `SYM_OP`。

当前数据集中常用镜像 symmetry:

```text
[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

该参数表示 reflectional symmetry，法向为 `[0, 1, 0]`，参考点为原点，即关于 `XZ` 平面对称，用于生成左右机翼、左右发动机等。

### 2.6 当前归一化逻辑

`generate_aircraft()` 对每个样本计算所有 box 前 10 个连续几何值的最大绝对值 `max_val`，再进行实例级缩放。

代码当前对 box 前 10 维执行:

```text
boxes[i][j] = (boxes[i][j] / max_val) * 2.0 - 1.0
```

对类别后 3 维执行:

```text
boxes[i][j] = boxes[i][j] * 2.0 - 1.0
```

对 symmetry:

- translational symmetry 的位移向量 `s[1:4]` 只除以 `max_val`，不平移。
- rotation/reflection 的方向向量不缩放、不平移。
- 参考点或终点 `s[4:7]` 执行 `(value / max_val) * 2.0 - 1.0`。

注意: 这里记录的是当前代码事实。若后续发现几何归一化公式不合理，应通过单独任务修改数据生成逻辑，并同步更新本文档。

## 3. VAE 模型定义

### 3.1 全局尺寸参数

旧扁平路径当前使用的 `util.py` 默认参数:

| 参数 | 默认值 | 用途 |
| --- | ---: | --- |
| `box_code_size` | 13 | box 输入/输出维度 |
| `feature_size` | 80 | 树节点 latent feature 维度 |
| `hidden_size` | 200 | 多数 MLP 中间层维度 |
| `symmetry_size` | 8 | symmetry 参数维度 |
| `max_box_num` | 30 | 数据/生成相关上限参数，VAE 主训练未直接使用 |
| `max_sym_num` | 10 | 数据/生成相关上限参数，VAE 主训练未直接使用 |

### 3.2 编码器

`GRASSEncoder` 同时保留旧扁平 head，并实现 typed leaf head:

- `BoxEncoder`: `Linear(box_code_size, feature_size)` + `Tanh`。
- `fuselageBoxEncoder`、`wingBoxEncoder`、`engineBoxEncoder`: 分别接收机身 10D、翼面 `8D + 30D` 和发动机 10D payload。
- `AdjEncoder`: 左右子 feature 分别线性映射到 hidden，再相加，经 `Tanh`、`Linear(hidden_size, feature_size)`、`Tanh`。
- `SymEncoder`: 子 feature 与 symmetry 参数分别线性映射到 hidden，再相加，经 `Tanh`、`Linear(hidden_size, feature_size)`、`Tanh`。
- `Sampler`: VAE 采样层。

编码递归入口为 `encode_structure_fold(fold, tree, use_sampler=True)`:

- 旧扁平 `BOX`: 调用 `boxEncoder(box)`。
- typed `BOX`: 根据 `component` 分派至三个 typed box encoder；翼面额外传入 `airfoil` code。
- `ADJ`: 递归编码左右子树，再调用 `adjEncoder(left, right)`。
- `SYM`: 递归编码 generator 子树，再调用 `symEncoder(feature, sym)`。
- 根节点 feature 默认继续调用 `sampleEncoder(feature)`。

当 `use_sampler=False` 时，函数返回未经 VAE sampler 的原始根 feature。该开关主要用于需要纯结构编码的路径。

### 3.3 Sampler 与 KL 定义

`Sampler.forward(input)` 的步骤:

1. `encode = tanh(mlp1(input))`
2. `mu = mlp2mu(encode)`
3. `logvar = mlp2var(encode)`
4. `std = exp(0.5 * logvar)`
5. `eps = torch.randn_like(std)`
6. `z = mu + eps * std`
7. `KLD_element = -mu^2 - exp(logvar) + 1 + logvar`
8. 返回 `torch.cat([z, KLD_element], 1)`

训练阶段在 `train.py` 中拆分:

```text
root_code, kl_div = torch.chunk(fnode, 2, 1)
kldiv_total = apply_res[res_idx].sum().mul(-0.5)
kldiv_loss = kldiv_total * kl_weight / batch_size
```

因此 KL 标量等价于:

```text
KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

`kl_weight` 使用线性 annealing:

```text
kl_weight = kl_weight_target * min(1.0, epoch / kl_anneal_epochs)
```

默认 `kl_weight_target = 0.03`，`kl_anneal_epochs = 20`。

### 3.4 解码器

`GRASSDecoder` 同时保留旧扁平 decoder，并实现 typed teacher-forced decoder:

- `SampleDecoder`: 将 VAE latent `z` 映射回递归解码器根 feature。
- `AdjDecoder`: 父 feature 解码为左右两个子 feature。
- `SymDecoder`: 父 feature 解码为 generator 子 feature 与 symmetry 参数。
- `BoxDecoder`: 父 feature 解码为 13 维 box。
- `fuselageBoxDecoder`、`wingBoxDecoder`、`engineBoxDecoder`: 分别输出对应 typed payload。
- `componentClassifier`: 输出 fuselage/wing/engine 三类 logits；不与 `NodeClassifier` 的 `BOX/ADJ/SYM` 分类混用。
- `NodeClassifier`: 父 feature 分类为 `BOX/ADJ/SYM` 三类 logits。

`BoxDecoder` 的输出规则:

- 前 10 维几何经过 `Tanh`，约束到 `[-1, 1]`。
- 后 3 维类别 logits 不经过 `Tanh`，供 `CrossEntropyLoss` 使用。

训练递归入口为 `decode_structure_fold(fold, feature, tree)`:

1. 先调用 `sampleDecoder(feature)`。
2. 按 ground truth tree 拓扑递归解码。
3. 每个节点都计算 node type 分类损失。
4. 旧扁平 `BOX` 节点计算 13D box 的几何损失和类别损失；typed `BOX` 使用 ground truth component 选择 decoder head，计算 payload 与组件分类损失。
5. `SYM` 节点计算 symmetry 参数损失，并继续解码 generator 子树。
6. `ADJ` 节点解码左右子 feature，并继续解码左右子树。

训练时解码拓扑由 ground truth tree 决定；typed 路径目前只实现此 teacher-forced 重构。自由生成时由 `NodeClassifier` 预测拓扑的实现目前只支持旧扁平 13D box。

## 4. VAE 损失函数定义

### 4.1 旧扁平 13D box loss

`boxLossEstimator(box_feature, gt_box_feature)` 对 batch 内每个 box 分别计算:

```text
geom_l = MSE(pred_box[:10], gt_box[:10])
cls_l = CrossEntropyLoss(pred_box[10:].unsqueeze(0), argmax(gt_box[10:]).unsqueeze(0))
```

返回形状为 `[num_boxes, 2]` 的损失张量，第二维分别为几何损失与类别损失。

训练聚合:

```text
box_loss_raw = sum(all_box_losses, dim=0) / batch_size
geom_loss = box_loss_raw[0]
cls_loss = box_loss_raw[1]
```

typed 路径同样返回两列 `[payload_loss, component_cls_loss]` 以复用该聚合形状；翼面 payload loss 为 8D 几何 MSE 与 30D Bezier code MSE 之和。详细字段、loss 与完成状态以 `docs/typed_box_airfoil_encoding.md` 为准。

### 4.2 symmetry loss

`symLossEstimator(sym_param, gt_sym_param)` 对每个 symmetry 参数向量计算 MSE:

```text
sym_loss = MSE(pred_sym, gt_sym)
```

训练聚合:

```text
sym_loss = sum(all_sym_losses) / batch_size
```

### 4.3 node classification loss

`classifyLossEstimator(label_vector, gt_label_vector)` 使用:

```text
CrossEntropyLoss(pred_node_logits.unsqueeze(0), gt_node_label.unsqueeze(0))
```

其中 `gt_node_label` 是 `BOX=0`、`ADJ=1`、`SYM=2`。

训练聚合:

```text
cat_loss = sum(all_node_classification_losses) / batch_size
```

### 4.4 total loss

当前 `train.py` 中 VAE 总损失:

```text
total_loss =
    vae_lambda_geom * geom_loss
  + vae_lambda_cls  * cls_loss
  + vae_lambda_sym  * sym_loss
  + vae_lambda_cat  * cat_loss
  + kldiv_loss
```

默认权重:

| 参数 | 默认值 |
| --- | ---: |
| `vae_lambda_geom` | 0.4 |
| `vae_lambda_cls` | 1.0 |
| `vae_lambda_sym` | 0.5 |
| `vae_lambda_cat` | 0.2 |
| `kl_weight_target` | 0.03 |

优化器当前在 `train.py` 中直接写为:

```text
Adam(encoder.parameters(), lr=1e-3)
Adam(decoder.parameters(), lr=1e-3)
```

注意: 虽然 `util.py` 有 `--lr` 参数，当前 VAE 训练代码没有使用该参数，而是硬编码 `1e-3`。

## 5. VAE 训练流程

当前 `train.py` 的旧扁平训练路径中，每个 batch 的步骤为:

1. `DataLoader` 通过自定义 `my_collate` 直接返回 `Tree` 对象列表，避免 PyTorch 默认 collate 破坏树结构。
2. 创建 `enc_fold = FoldExt(cuda=config.cuda)`。
3. 对 batch 中每棵树调用 `encode_structure_fold(enc_fold, example)`，只记录动态计算图。
4. 调用 `enc_fold.apply(encoder, [enc_fold_nodes])` 批量执行同名模块操作。
5. 将每个样本的 sampler 输出拆成 `root_code` 与 `kl_div`。
6. 创建 `dec_fold = FoldExt(cuda=config.cuda)`。
7. 对每个样本调用 `decode_structure_fold(dec_fold, root_code, example)` 收集 box/sym/category loss 节点。
8. 将非空 loss node list 传入 `dec_fold.apply(decoder, apply_lists)`。
9. 聚合各类损失，计算 total loss。
10. `encoder_opt.zero_grad()`、`decoder_opt.zero_grad()`。
11. `total_loss.backward()`。
12. `encoder_opt.step()`、`decoder_opt.step()`。

模型保存:

- 最终编码器: `models/vae_encoder_model.pkl`
- 最终解码器: `models/vae_decoder_model.pkl`
- 可选 snapshot: `models/snapshots_*/vae_*_epoch_*.pkl`

## 6. 旧扁平路径的自由生成流程

`VAE_gen.py` 当前自由生成步骤:

1. 加载 `models/vae_decoder_model.pkl`。
2. 采样 `root_code = torch.randn(1, 80)`。
3. 调用 `grassmodel.decode_structure(decoder, root_code)`。
4. 将生成 box 拼接后执行 `(boxes + 1.0) / 2.0` 反归一化用于显示。
5. 调用 `showGenshape(...)` 可视化。

`decode_structure(model, root_code)` 的内部逻辑:

1. `sampleDecoder(root_code)` 得到根 feature。
2. 使用栈保存待展开 feature。
3. 每次弹出一个 feature，通过 `nodeClassifier` 预测节点类型。
4. 若预测 `ADJ`: 调用 `adjDecoder` 得到左右子 feature，并压栈。
5. 若预测 `SYM`: 调用 `symDecoder` 得到 generator feature 与 symmetry 参数，并压栈。
6. 若预测 `BOX`: 调用 `boxDecoder` 得到基础 box，再根据当前继承的 symmetry 参数生成复制 box。
7. 返回所有 box 列表。

自由生成阶段没有 ground truth tree，因此拓扑完全由 `NodeClassifier` 的逐节点预测决定。

## 7. torchfold 与 GRASS 特殊实现

### 7.1 为什么需要 torchfold

GRASS 是递归神经网络，每个样本的树拓扑和节点数可能不同。普通 batch 不能直接把整棵树堆成一个固定张量。

torchfold 的作用是:

- 递归遍历时先记录每个模块调用，而不是立即执行。
- 按模块名和 step 将不同树中的同类操作批量化。
- 调用 `fold.apply(model, lists)` 时统一执行，从而提高 GPU 并行效率。

### 7.2 FoldExt 的当前适配

`FoldExt` 继承 `torchfold.Fold`，构造函数保持现代 torchfold 签名:

```text
FoldExt(volatile=False, cuda=False)
```

当前主要改动:

- `add(op, *args)` 接受 `Fold.Node`、`int`、`torch.Tensor`、`torch.FloatTensor`、`torch.LongTensor`。
- `_batch_args(...)` 对普通 tensor 参数使用 `torch.cat(arg, 0)` 组成 batch。
- 若 `self._cuda` 为真，将拼接后的 tensor 调用 `.cuda()`。

这使得 `node.box`、`node.sym`、`node.label` 等张量可以作为 fold 参数参与动态批处理。

### 7.3 模块名约束

`fold.add('boxEncoder', ...)` 这类字符串必须对应模型对象上的同名方法:

- Encoder:
  - `boxEncoder`
  - `fuselageBoxEncoder`
  - `wingBoxEncoder`
  - `engineBoxEncoder`
  - `adjEncoder`
  - `symEncoder`
  - `sampleEncoder`
- Decoder:
  - `sampleDecoder`
  - `adjDecoder`
  - `symDecoder`
  - `boxDecoder`
  - `fuselageBoxDecoder`
  - `wingBoxDecoder`
  - `engineBoxDecoder`
  - `componentClassifier`
  - `nodeClassifier`
  - `boxLossEstimator`
  - `fuselageBoxLossEstimator`
  - `wingBoxLossEstimator`
  - `engineBoxLossEstimator`
  - `symLossEstimator`
  - `classifyLossEstimator`

这些字符串是 torchfold 动态派发的接口名，属于代码中的隐式 API。后续重命名模块方法时必须同步修改所有 `fold.add(...)` 调用。

## 8. 与 MATLAB 原版的关系

MATLAB 原始实现使用显式 tree structure、manual forward/backward 和 12 维 OBB 格式。当前 PyTorch 工程已经为飞行器布局做了以下实际改动:

- 旧扁平 box 从原始 12 维 OBB 改为 13 维飞机部件编码；typed box 路径使用部件特定 payload。
- 树拓扑从 MATLAB `treekids` 形式转为当前 `.mat` 中的后序 `ops` 栈式编码。
- 反向传播由 PyTorch autograd 与 torchfold 动态批处理负责。
- 旧扁平 VAE 的 KL、重构、节点分类损失在 `train.py` 中统一聚合；typed loss 尚未接入该正式训练入口。

`Grass-matlab` 是参考实现，按项目约束禁止修改。

## 9. 当前代码事实与后续注意点

- `util.py` 是超参数的集中入口，但当前 `train.py` 仍硬编码 Adam 学习率 `1e-3`，与 `--lr` 参数不一致。
- 当前 box 几何归一化公式记录为代码事实；若后续需要严格归一到 `[-1, 1]`，应专门审查 `data/generate_dataset.py`。
- `draw3dobb.py` 的可视化几何不是传统完整 OBB 12 维格式，而是根据旧扁平 10 维几何与 3 维类别绘制飞机部件；typed 翼面 OBB 使用 8D geometry 与 30D Bezier code。
- typed 结构化数据集、encoder 和 teacher-forced decoder 已有单元测试；正式训练、GAN 和自由生成尚未支持 typed component 输出。
- 所有后续实现应保持 `BOX/ADJ/SYM`、部件特定 typed box schema、8 维 symmetry、`fold.add` 模块名的一致性，避免在多个位置手动复制不同定义。
