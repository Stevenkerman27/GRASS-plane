# 分层飞行器联合 AE 编码定义

## 状态与范围

本文定义飞翼、常规和鸭式样本的联合确定性 autoencoder（AE）目标 schema。它取代此前按固定飞翼或固定常规/鸭翼树展开的 typed 自由解码定义。实施前生成的辅助翼 `[4,29]` `.pt` 文件不符合本规范，不能混入联合训练。

引入 model-space 标准化和逐步 latent 条件之前生成的 section-AE v1 与全机 AE checkpoint 不包含训练统计量，且 decoder 输入层尺寸不同，不能加载到本模型；必须重新训练。标准化后的 loss 与旧物理参数 MSE 不同量纲，数值不能横向比较。

权威配置和实现位置：

- `util.py`：组件类别、截面维度、截面数范围、损失权重和自由展开上限。
- `grassdata.py`：typed `.pt` 样本校验和后序 `ops` 构树。
- `grassmodel.py`：递归编码器、递归 decoder、序列 RNN 和自由展开。
- `section_parameter_codec.py`：物理截面与标准化 model-space 截面之间的唯一可逆变换，以及训练集统计量拟合。
- `section_autoencoder.py`：独立机翼/机身截面 AE 的模型、共享重建损失和预训练 checkpoint 读取。
- `train_section_autoencoder.py`：独立叶节点预训练。
- `train_autoencoder.py`：两个数据集的联合加载、teacher-forced 训练和验证。
- `data/generate_aircraft_datasets.ps1`：飞翼与常规/鸭翼数据集的唯一生成入口。

发动机和垂尾不属于本 schema。组件类别只有 `fuselage` 与 `wing`；主翼、平尾和鸭翼不是模型类别。

## 树表示

`grassdata.Tree.NodeType` 只有三种节点：

| 节点 | 数值 | 含义 |
| --- | ---: | --- |
| `BOX` | 0 | 部件序列叶节点 |
| `ADJ` | 1 | 二元装配节点 |
| `SYM` | 2 | 对 generator 子树施加对称的一元节点 |

`ops` 是后序栈表达式，不保存显式 parent-child 索引。读入时：`BOX` 压栈；`ADJ` 弹出两个节点并按 `left = queue.pop()`、`right = queue.pop()` 组装；`SYM` 弹出一个 generator 并配对一个 `syms` 向量。遍历结束时栈必须只剩一个根节点。

`symmetry_size=8`，参数格式为：

```text
[type, p1, p2, p3, q1, q2, q3, r]
```

当前数据使用关于 XZ 平面的反射：`[1, 0, 1, 0, 0, 0, 0, 0]`。其中 `type=1` 为反射，`p1:p3` 是镜像平面法向，`q1:q3` 是平面参考点。

## 统一序列 Payload

所有 typed `.pt` 翼叶节点都使用相同 schema，不存储 `main_wing` 或 `auxiliary_wing` 角色：

```text
fuselage.sections[8, 5] = [x, y, z, width, height]
wing.sections[8, 29]    = [CST24, leading_edge_x, leading_edge_y,
                           leading_edge_z, chord, twist]
```

`section_count` 是唯一的有效长度与 padding 权威字段。`sections` 在有效前缀之后必须补零，不能额外保存 mask 或 EOS。

- 机身和翼的模型允许长度均为 `2..8`。
- 数据生成时主翼实际采样 `2..8`；辅助翼实际采样 `2..4`，但仍补零存为 `[8,29]`。
- 机身截面按机头到机尾排序，`x` 严格递增，`width`、`height` 为正。
- 翼截面按右半翼 `+Y` 的 root-to-tip 排序，`chord` 为正。

“辅助翼最多四段”是数据采样分布约束，不是组件类型、decoder 类型或自由生成语法。自由生成的任一 `wing` 叶节点均可预测 `2..8` 段。

## 编码器与 Teacher-Forced 解码器

`fuselage` 与统一的 `wing` 各使用一层序列 encoder：

```text
valid_sections[0:N, D] -> {RNN | GRU} -> final_hidden[feature_size]
```

`ADJ` 与 `SYM` encoder 将子 feature（以及 symmetry 参数）组合为父 feature，最终根 feature 即整架飞机的确定性 AE 编码。AE 不使用 `Sampler`、KL divergence 或 `SampleDecoder`。

decoder 的结构递归训练仍使用真实树：每个节点计算 `BOX/ADJ/SYM` 的 node-classification 损失；真实叶组件选择机身或统一翼 RNN decoder；`ADJ` 解码两个子 feature，`SYM` 解码一个 generator feature 和 symmetry 参数。

机身和翼序列 decoder 都是自回归循环单元。RNN 内部只处理标准化后的 model-space 截面：

1. 父 feature 映射为初始 hidden state。
2. 第一步输入该 decoder 可学习的 BOS 向量。
3. 每一步 RNN 输入为 `[previous_model_section, parent_feature, t / 7]`；父 feature 不只用于初始化 hidden。
4. 训练时第 `t+1` 步的 `previous_model_section` 按 scheduled sampling 选择真实或预测的第 `t` 截面。
5. 最多输出 8 个截面，并独立输出 `section_count` logits。

训练使用 scheduled sampling 缩小 teacher forcing 与自由自回归之间的输入分布差异。设 `p_teacher(epoch)` 为每个样本、每个解码步输入真实前一截面的概率；否则输入模型刚预测的截面，且该预测保持计算图连接。配置定义在 `util.py`：

```text
p_final = 0.1
ramp_start_epoch = 60
ramp_end_epoch = 80

p_teacher(epoch) = 1.0                         epoch < ramp_start_epoch
                 = linear(1.0 -> p_final)      ramp_start_epoch <= epoch <= ramp_end_epoch
                 = p_final                      epoch > ramp_end_epoch
```

训练 epoch 使用该 `p_teacher`；验证始终使用 `p_teacher=1.0`，从而保持 teacher-forced validation loss、`ReduceLROnPlateau` 和不同训练运行之间可比。自由树与自由序列解码始终只反馈预测截面。

训练配置 `--ae_rnn_type {rnn,gru}` 是编码器与解码器的共同循环单元选择，默认 `gru`。`rnn` 使用 `torch.nn.RNN` 与 `torch.nn.RNNCell` （`tanh`）；`gru` 使用 `torch.nn.GRU` 与 `torch.nn.GRUCell`。其他模型、序列约束、损失与训练流程不变。checkpoint 保存 `ae_rnn_type`，以便比较实验及校验权重架构。

### 截面 model-space 与标准化

数据集 schema 始终保存物理参数；转换只发生在模型边界。统计量只能从训练 split 的有效截面拟合，验证集不得参与，并必须随 checkpoint 保存。encoder、decoder 和重建损失使用同一组统计量；缺失或不一致必须报错，不能回退到未标准化路径。

正值参数先变换再标准化：

```text
wing:     log(chord), log(N1 - CST_MIN_CLASS_FUNCTION_EXPONENT),
          log(N2 - CST_MIN_CLASS_FUNCTION_EXPONENT)
fuselage: log(width), log(height), log(x[t] - x[t-1]) for t > 0
```

其余 CST 参数、位置和 twist 保持线性。每个 model-space 维度使用训练集的 `mean/std` 标准化；零方差维度显式使用 `std=1`。机身第一站 `x=0` 是固定模型约束，其 model-space 增量为零且不参与增量统计。decoder 输出标准化 model-space 参数，正值通过反标准化后的 `exp` 还原，不再使用 `softplus`。

## 叶节点预训练

全机 AE 训练前可先训练不含 `ADJ`、`SYM`、node classifier 或 component classifier 的独立截面 AE。数据必须先按整机样本以同一 `ae_seed` 划分训练/验证集，然后才分别抽取 `wing` 与 `fuselage` 叶节点；同一飞机的叶节点不得跨 split。

```text
wing.sections[8,29], count -> SectionEncoder -> feature_size -> WingSectionDecoder
fuselage.sections[8,5], count -> SectionEncoder -> feature_size -> FuselageSectionDecoder
```

两个 AE 使用与全机模型相同的 `SectionEncoder`、`AutoregressiveSectionDecoder`、截面 model-space codec、有效截面 mask、scheduled sampling 和参数/count 损失。每个训练运行只选择最终 epoch 的 `last_wing.pt` 与 `last_fuselage.pt`；训练不会按验证损失另存最佳 epoch。checkpoint 必须包含：`schema=grass_section_autoencoder_v2`、`parameter_statistics.schema=grass_section_parameter_codec_v1` 与统计量、`sequence_type`、`feature_size`、`hidden_size`、`ae_rnn_type`、对应 encoder/decoder state dict、优化器与调度器 state dict、teacher-forcing 配置和验证指标。

`train_autoencoder.py --ae_section_pretrained_checkpoint_dir <directory>` 必须读取这两个最终 checkpoint，严格校验 schema、sequence type、feature/hidden size 与循环单元类型，再写入全机 `wing/fuselage_section_encoder` 和两个 section decoder。加载后所有参数继续参与联合微调，不冻结；缺失或不匹配的 checkpoint 必须报错。

## 自由树与自由序列解码

自由生成没有飞翼、常规或鸭翼的固定树模板，也没有“主翼/辅助翼”角色分类或专用 decoder。根 feature 进入待展开栈后，`NodeClassifier` 在每一个待展开 feature 上预测：

- `BOX`：`componentClassifier` 选择 `fuselage` 或 `wing`，随后调用对应的自回归 RNN decoder。
- `ADJ`：`AdjDecoder` 产生两个子 feature，二者继续由 `NodeClassifier` 决定。
- `SYM`：`SymDecoder` 产生 generator 子 feature 和 symmetry 参数，generator 继续由 `NodeClassifier` 决定。

因此，树拓扑由逐节点 node-classifier 预测决定；常规、鸭翼或异常的几何布局由生成翼根站位等连续参数决定，而不是布局标签。模型可能生成训练分布外的拓扑或几何，训练数据只改变其概率，不保证几何有效性。

自由展开仅施加以下运行时上限，不施加固定语法：

```text
maximum leaf nodes = 3
maximum tree depth = 4
maximum sequence sections = 8
```

达到叶节点或深度上限时，展开器必须屏蔽会超过上限的结构选择，并以可行的 `BOX` 结束该分支。截面数由该叶 decoder 的 `section_count` 预测决定，达到 8 段时停止；不使用 EOS。树深度和截面序列长度是两个独立概念。

## 训练与验证

联合训练集由飞翼与常规/鸭翼两个 typed `.pt` 样本列表组成，并在合并后以固定随机种子划分训练/验证集。不能依赖目录或 batch 顺序暗含布局标签。

优化目标为 teacher-forced 重建损失：

- 有效截面 payload：在标准化 model-space 中计算翼的 position、chord、twist、CST code 和 count，以及机身的 position、size 和 count。
- 叶组件分类：`fuselage / wing`。
- 节点类型分类：`BOX / ADJ / SYM`。
- symmetry 参数重建。

所有参数项只在真实 `section_count` 导出的 mask 上计算。CST 曲线解码和三维截面误差不参与反向传播，只作为独立验证与可视化指标；精确重建 CST code 即定义为精确重建对应翼型。损失权重集中定义于 `util.AE_LOSS_WEIGHTS`、`util.WING_LOSS_WEIGHTS` 与 `util.FUSELAGE_LOSS_WEIGHTS`。

自由验证不接受真实树作为展开模板：它报告自由树是否在上限内有效、节点和组件预测、预测叶数、树深度、截面数、受上限强制结束的节点数，以及生成 payload。全机 AE 的 teacher-forced 验证总重建损失仍是选择最佳 checkpoint 的标准；section-AE 则只使用最终 epoch checkpoint。

优化器为 `AdamW`，默认 `--ae_weight_decay=1e-5`；该衰减独立于学习率调度，用于抑制过大的参数范数，不改变 position/chord/twist/CST 的重建损失权重。每个 epoch 的 teacher-forced validation total loss 传给 PyTorch `ReduceLROnPlateau`：连续 `8` 个 epoch 未改善时，学习率乘 `0.5`，不低于 `1e-6`。调度器在 checkpoint 中保存 state dict；`loss_curves.png` 同时记录 train/validation 损失和实际学习率，`free_generation_metrics.png` 记录 free-running 指标。

checkpoint 还保存 scheduled-sampling 的 `p_final`、ramp 起止 epoch；训练 metrics 和 loss 曲线记录每个 epoch 实际使用的 `p_teacher`。

### Independent section-AE free reconstruction visualization

`data/visualize_section_autoencoder.py` evaluates only the validation-aircraft split produced with the same `structured_data_paths`, `ae_seed`, and `ae_validation_fraction` as `train_section_autoencoder.py`. It loads the final-epoch `last_wing.pt` and `last_fuselage.pt`, encodes each sampled leaf, and invokes each decoder's free-running `generate` path without target-section feedback.

For every sampled leaf, target geometry is a gray dashed reference. Generated wing sections are reconstructed from CST airfoil curves, leading-edge positions, chord, and twist; generated fuselage sections are reconstructed as ellipses from `[x, y, z, width, height]`. Shared valid section indices are colored by their target-extent-normalized 3D point-cloud RMS. Extra generated sections are purple, while missing target sections remain visible in the gray reference. The diagnostic opens an interactive 3D Matplotlib window for each sample and does not alter loss, checkpoint selection, model weights, or files.

## Torchfold

树拓扑和节点数可变，训练路径通过 `torchfoldext.FoldExt` 记录递归模块调用，再按模块名批量执行。`fold.add(...)` 中的字符串是模型方法的隐式 API；重命名 encoder、decoder 或 loss 方法时必须同步更新调用点。

`Grass-matlab` 仅作原始 GRASS 格式和算法参考，禁止修改。旧 13D OBB/VAE/GAN 路径与本 typed 联合 AE schema 不兼容，不得通过补零或截断混入训练数据。

## 自由解码误差可视化

自由解码树不以输入样本的树作为展开模板，因此它不是固定拓扑的逐节点重建。误差可视化必须先展开真值与生成树的完整几何，再以组件类型分别匹配 `fuselage` 和 `wing`；匹配代价由归一化组件位置与包围范围构成，并在小规模组件集合中取最小总成本匹配。不同类型组件、缺失真值组件和额外生成组件均不得强行建立截面对应。

对每对已匹配组件，生成截面按自身的归一化序列位置与真值组件插值得到的对应截面比较：机身使用机头至机尾位置和椭圆轮廓的 3D RMS；机翼使用根至尖累计翼展位置，将 CST 翼型、leading edge、chord 与 twist 插值并还原为 3D 翼型轮廓，再计算 3D RMS。该值同时包含位置、尺寸、扭转和翼型误差，并以真值整机特征长度归一化。

可视化中，已匹配的自由解码截面以绿到红表示低到高的归一化几何 RMS，真值作为浅灰线框参照；未匹配生成组件为紫色，未匹配真值组件为灰色虚线。未匹配状态表示拓扑或组件数量误差，不得以零或任意截面误差替代。该可视化仅用于诊断自由解码质量，不参与训练损失或 checkpoint 选择。

## 极小样本过拟合诊断

`train_section_autoencoder.py --overfit` 使用 `ae_seed` 从合并后的整机数据集中无放回选取固定 2 架飞机。该同一子集同时用于训练和评估，不创建验证划分；因此 teacher-forced 和 free-running 指标只用于判断模型能否记住训练几何，不能表示泛化能力。机翼和机身仍独立训练，实际叶节点数由这 2 架飞机的树决定。

为避免覆盖正常训练权重，`--overfit` 的 checkpoint、损失曲线和训练指标统一写到 `<section_ae_checkpoint_dir>/overfit/`。`data/visualize_section_autoencoder.py --overfit` 读取同一目录，并只显示该过拟合集中的自由重建。

## Section AE 误差评估

`data/evaluate_section_autoencoder.py` 是独立 section AE 的批量只读评估工具。它必须从最终 epoch 的 `last_wing.pt` 和 `last_fuselage.pt` 读取各自的 model-space 统计量，并用与 `train_section_autoencoder.py` 相同的 `structured_data_paths`、`ae_seed`、`ae_validation_fraction` 划分验证集；`--overfit` 则使用同一个两架飞机子集。评估不能重新拟合统计量，不能更新参数，也不能影响训练。

每个 sequence type 都分别报告 teacher-forced 与 free-running 的结果。训练同定义的 `position`、`chord`、`twist`、`cst_code`、`size` 和 `section_count` 项保持在标准化 model-space 中，便于与训练日志对应。物理空间诊断仅统计真值 `section_count` 导出的有效截面：机翼报告 CST（含 N1/N2）、leading-edge `xyz`、chord、twist；机身报告 `x/y/z`、width、height 和总长度。截面数还报告准确率、平均绝对误差和预测/真值混淆矩阵。

几何指标不参与反向传播或训练损失。机翼将 CST code、leading edge、chord 和 twist 还原为固定采样点数的三维翼型轮廓，机身还原为椭圆截面轮廓；逐截面 RMS 以米报告，并以对应真值部件的最大几何尺度归一化为比例。评估工具把汇总写入 `<section_ae_checkpoint_dir>/evaluation/<mode>_<sequence_type>_summary.json`，逐样本记录写入 CSV，便于比较 RNN、GRU 和不同 checkpoint；同名结果默认覆盖，传入 `--no-overwrite` 才禁止覆盖。
