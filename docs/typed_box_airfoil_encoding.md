# Typed Box 与翼型 Bezier 编码设计

本文档定义后续将翼型编码并入 GRASS 飞行器布局生成的实现方案。该方案不继续兼容 `.mat` box 矩阵格式，不使用最大联合 schema，也不要求机身、发动机保存无意义的翼型占位字段。

设计目标是在尽量不改动现有 GRASS 树结构的前提下，让不同部件使用不同的 box payload 和 encoder/decoder head:

```text
GRASS topology: BOX / ADJ / SYM
BOX payload:
  fuselage -> fuselage geometry
  wing     -> wing geometry + root/tip airfoil Bezier codes
  engine   -> engine geometry
```

当前几何维度定义为：机身和发动机各使用 10 维几何参数；翼面使用 8 维几何参数
`[x1, y1, z1, x2, y2, z2, root_chord, tip_chord]`，其原有的两项相对厚度由根、尖各一组
30 维 Bezier 翼型编码取代。若导出带类别 one-hot 的扁平 OBB 向量，则机身/发动机为 13 维，
翼面为 71 维。

## 1. 设计原则

1. `BOX/ADJ/SYM` 拓扑不变。
2. `AdjEncoder`、`SymEncoder`、`Sampler`、`AdjDecoder`、`SymDecoder` 尽量不变。
3. 改造集中在 leaf `BOX` 的数据表示、box encoder、box decoder 和 box loss。
4. 不再为了 `.mat` 存储把所有部件强行 padding 到同一语义向量。
5. schema、字段名、类别 id、维度必须集中定义，禁止散落硬编码。
6. 使用 fail-fast: 遇到未知部件类型、缺失字段、维度不匹配时直接报错。

## 2. 权威 schema

当前项目已经使用 `util.py` 作为参数与定义入口。为避免不必要地增加代码结构，typed box 的 schema 常量第一版应优先集中写入 `util.py`，作为 box payload 的单一来源，集中定义:

- 部件类别 id。
- 数据字段名。
- 各部件几何维度。
- 翼型 Bezier 编码维度。
- 训练 loss 中各项名称。

只有当后续 schema 内容明显膨胀，使 `util.py` 难以阅读或产生循环依赖时，才考虑拆出 `box_schema.py`。拆分前需要给出明确依据，并同步更新本文档。

建议类别:

```python
COMPONENT_FUSELAGE = 0
COMPONENT_WING = 1
COMPONENT_ENGINE = 2
```

注意区分两类 label:

- `Tree.NodeType`: `BOX/ADJ/SYM`，用于树拓扑分类。
- `ComponentType`: `fuselage/wing/engine`，用于 leaf box 内部的部件类型分类。

## 3. 数据存储格式

新数据集建议保存为结构化 `.pt` 文件，例如:

```python
[
    {
        "boxes": [
            {
                "component": COMPONENT_FUSELAGE,
                "geometry": torch.FloatTensor([...]),
            },
            {
                "component": COMPONENT_WING,
                "geometry": torch.FloatTensor([...]),
                "airfoil": torch.FloatTensor([...]),
            },
        ],
        "ops": torch.LongTensor([...]),
        "syms": torch.FloatTensor([...]),
    },
]
```

这里的 `.pt` 是 typed 训练数据集的结构化载体。当前固定常规数据集先在 `vsppytools` 中为每架飞机写出 JSON 与 `.vsp3`，再由 `data/json_to_typed_obb_dataset.py` 在 `myml` 中拟合翼型并写入 `data/conventional_dataset/conventional_dataset.pt`。`data/visualize_dataset.py` 直接读取该 `.pt`，不在绘图时重新拟合。详见 `docs/conventional_dataset_generation.md`。

使用 plain `dict/list/tensor/int`，避免在 `torch.save` 中序列化自定义 dataclass。这样更容易配合 `torch.load(..., weights_only=True)`，也减少跨文件反序列化耦合。

`boxes` 的顺序仍与后序 `ops` 中的 `BOX` 出现顺序一致。`ops` 仍为后序栈式表达式，`syms` 仍按 `SYM` 出现顺序保存。这样可以复用当前 `Tree` 构造思想，只替换 box payload 的来源。

## 4. Tree 与 Dataset 改造

尽量保留 `grassdata.Tree.Node` 的结构:

```text
Node.box
Node.sym
Node.left
Node.right
Node.node_type
Node.label
```

变化只在于 `Node.box` 从单个固定维度 tensor 变为结构化 dict。

“保留现有 `.mat` 构造函数，另加结构化构造入口”的意思是: 不要求新方案继续使用 `.mat` 数据，但也不在第一步删除或重写当前 `GRASSDataset` 和 `Tree.__init__(boxes, ops, syms)`。旧入口可以暂时留给已有实验、调试脚本和文档中记录的当前实现使用；新数据路径通过额外入口构造 `Tree`，例如 `Tree.from_structured_sample(sample)`。

这样做不是为了兼容 `.mat`，而是为了避免一次性重构破坏现有代码。等结构化 `.pt` 数据集和 typed box 训练稳定后，再单独决定是否清理旧 `.mat` 路径。

建议新增 dataset 类:

```text
StructuredGRASSDataset
```

职责:

1. 从 `.pt` 读取 sample list。
2. 对每个 sample 做 schema 校验。
3. 构造 `Tree`。
4. 不在 dataset 中静默补默认字段。

当前 `GRASSDataset` 可暂时保留给旧实验使用，但新训练入口应显式选择结构化数据集，避免新旧数据格式混用。

## 5. Encoder 设计

`GRASSEncoder` 保留递归接口，但 leaf box 改为按部件类型分派:

```text
fuselageBoxEncoder(geometry) -> feature_size
wingBoxEncoder(geometry, airfoil) -> feature_size
engineBoxEncoder(geometry) -> feature_size
```

`encode_structure_fold` 中只在 Python 递归记录阶段读取 `node.box["component"]`，然后调用对应的 `fold.add(...)`。传入 fold 的仍然是 tensor，因此不要求 `torchfoldext` 支持 dict batch。

示意:

```python
if node.is_leaf():
    box = node.box
    component = box["component"]
    if component == COMPONENT_WING:
        return fold.add("wingBoxEncoder", box["geometry"], box["airfoil"])
    if component == COMPONENT_FUSELAGE:
        return fold.add("fuselageBoxEncoder", box["geometry"])
    if component == COMPONENT_ENGINE:
        return fold.add("engineBoxEncoder", box["geometry"])
    raise ValueError(f"Unknown component type: {component}")
```

这样 `AdjEncoder` 和 `SymEncoder` 仍接收统一的 `feature_size`，树级网络无需改结构。

## 6. Decoder 设计

Decoder 需要同时解决两个问题:

1. 预测 leaf 属于哪类部件。
2. 按部件类型输出对应 payload。

推荐新增:

```text
componentClassifier(feature) -> logits over fuselage/wing/engine
fuselageBoxDecoder(feature) -> fuselage geometry
wingBoxDecoder(feature) -> wing geometry + airfoil Bezier code
engineBoxDecoder(feature) -> engine geometry
```

训练时使用 ground truth component 选择对应 decoder head，避免早期训练中 component 预测错误导致无法计算稳定重构损失。自由生成时使用 `componentClassifier` 的预测结果选择 decoder head。

现有 `NodeClassifier` 继续只预测 `BOX/ADJ/SYM`，不要把部件类型混入节点类型。

## 7. Loss 设计

BOX leaf loss 拆分为:

```text
component_cls_loss = CrossEntropy(component_logits, gt_component)
payload_loss =
  fuselage: MSE(pred_fuselage_geometry, gt_fuselage_geometry)
  wing:     MSE(pred_wing_geometry, gt_wing_geometry)
          + airfoil_loss(pred_airfoil, gt_airfoil)
  engine:   MSE(pred_engine_geometry, gt_engine_geometry)
```

`airfoil_loss` 第一版可使用 Bezier code 的 MSE。后续如发现相同翼型存在多组等价控制点，可增加 decoded curve loss:

```text
curve_loss = MSE(decode_bezier(pred_airfoil), decode_bezier(gt_airfoil))
```

训练聚合时建议保持现有 VAE total loss 结构，只把原来的 box `geom_loss + cls_loss` 替换为:

```text
box_payload_loss + component_cls_loss
```

具体权重应进入 `util.py` 参数，而不是在训练代码中硬编码。

## 8. 翼型 Bezier 编码

外部实验位于:

```text
D:\3D\Projects\ML\NN\encode_dat.py
```

当前较适合迁入 GRASS 的是 split-surface rational Bezier 编码:

```text
upper control points + lower control points + upper weights + lower weights
```

按当前外部配置:

```text
surface_control_points = 5
```

单个截面的完整编码维度为:

```text
upper cp:    5 * 2 = 10
lower cp:    5 * 2 = 10
upper weight: 5
lower weight: 5
total: 30
```

每个翼面按 `[root_code, tip_code]` 拼接，故训练 payload 使用 60D。单截面仍保存完整 30 维，不压缩固定尾缘或重复前缘。垂尾的两个单截面 code 从各自上表面严格构造下表面：控制点反序后围绕物理 `y=0` 弦线镜像，权重反序；这避免了把方向相反的上下表面控制数组直接设为相等而导致退化轮廓。

## 9. 外部代码与神经网络调用

GRASS 不在主训练代码中散落 `sys.path.insert` 到 `D:\3D\Projects\ML\NN` 后直接 import 脚本。原因:

- `encode_dat.py` 依赖同项目 `model.py`、`train_surrogate.py`、`config.yaml` 和相对路径。
- 外部项目脚本包含训练、可视化、文件输出等副作用。
- 跨项目直接 import 会让 GRASS 的训练环境和路径假设变脆。

当前做法:

1. GRASS 保留 `airfoil_codec.py`，作为 30 维 split-surface rational Bezier code 的拟合、pack/unpack 和 decode 单一来源。
2. `airfoil_codec.py` 中集中保存外部工程已优化的 split-surface 拟合参数，并实现相同的学习率调度器。
3. `airfoil_codec.encode_processed_airfoil_dat(...)` 直接读取外部 `coord_norm.pt` 归一化资产，但不导入外部 Python 模块或 YAML。
4. 所有外部路径必须在 `airfoil_codec.py` 中集中定义并检查存在性。

当前 fail-fast 约束:

- 优化配置的 `surface_control_points` 必须等于 `util.AIRFOIL_SURFACE_CONTROL_POINTS`。
- 优化配置推导的 code 维度必须等于 `util.AIRFOIL_BEZIER_CODE_SIZE`。
- `airfoil_codec.DEFAULT_COORD_NORM_PATH` 必须存在并包含完整坐标范围。

可接受的 API 形态:

```text
encode_airfoil_dat(dat_path, codec_config) -> airfoil_code
decode_airfoil_code(airfoil_code, codec_config) -> sampled_curve
load_airfoil_surrogate(config, checkpoint_path, norm_path) -> model_bundle
evaluate_airfoil_surrogate(curve, conditions, model_bundle) -> aero_values
```

气动 surrogate 第一阶段不进入 GRASS VAE 主损失。建议先作为生成后的评价或筛选工具。等 typed box + Bezier 重构稳定后，再考虑把 surrogate loss 接入训练。

## 10. 最小改造顺序

推荐分阶段实现，避免一次性重写整个工程。

### 阶段 1: 结构化数据与 typed box encoder

目标:

- 在 `util.py` 中集中新增 typed box schema 常量。
- 新增结构化 `.pt` 数据读取路径。
- 新增 `StructuredGRASSDataset`。
- 新增 `fuselageBoxEncoder`、`wingBoxEncoder`、`engineBoxEncoder`。
- 修改 `encode_structure_fold` 的 leaf 分派。

此阶段只保证 typed box 的 encoder 路径和小批量 forward smoke test。主训练入口仍使用旧 `GRASSDataset`，不要在 typed decoder/loss 完成前把 `train.py` 或 `train_GAN.py` 切到 structured 数据。

当前已完成:

- `util.py` 中新增 component id、payload key、几何维度和 `AIRFOIL_BEZIER_CODE_SIZE`。
- `grassdata.py` 中新增 `Tree.from_structured_sample(...)`、`StructuredGRASSDataset` 和结构化样本校验。
- `grassmodel.py` 中新增 typed box encoder head，并在 `encode_structure_fold` 中对 dict box 进行分派。
- `test/test_structured_typed_box.py` 覆盖结构化 `.pt` 读取、typed encoder 前向、wing 缺失 airfoil 和非 wing 携带 airfoil 的 fail-fast。

尚未完成:

- 根据真实翼型来源生成 structured `.pt` 数据。
- typed box decoder、component classifier 和 typed box loss。
- structured 数据的正式训练入口。

### 阶段 2: typed box decoder 与 teacher-forced 重构

目标:

- 新增 `componentClassifier`。
- 新增三个 box decoder head。
- 修改 `decode_structure_fold` 中 leaf 的 loss 收集。
- 新增 component classification loss 和 payload loss。
- 保持 ground truth tree 拓扑 teacher-forcing 不变。

此阶段完成后，应能在小数据集上重构训练。

当前已完成:

- `grassmodel.py` 中新增 `componentClassifier`。
- `grassmodel.py` 中新增 `fuselageBoxDecoder`、`wingBoxDecoder`、`engineBoxDecoder`。
- `decode_structure_fold` 对 structured dict box 使用 ground truth component 选择 decoder head。
- typed box loss 返回 `[payload_loss, component_cls_loss]`，可复用现有训练循环中 box loss 的两列聚合形状。
- `test/test_structured_typed_box.py` 已覆盖 teacher-forced typed decoder loss 的 backward。

尚未完成:

- 正式把 structured 数据接入 `train.py` / `train_GAN.py`。
- 自由生成阶段 `decode_structure` 的 typed component 输出。
- 真实翼型 Bezier code 的数据生成或预处理。

### 阶段 3: 翼型 Bezier code 接入 wing payload

目标:

- 新增 `airfoil_codec.py`。
- 在数据生成或预处理阶段给 wing/tail 写入 `airfoil` 字段。
- 增加 wing airfoil loss。
- 增加 Bezier decode 可视化验证。

此阶段不接入气动 surrogate。

当前已完成:

- 新增 `airfoil_codec.py`，在本项目内提供 split-surface rational Bezier 的 `pack/unpack/decode`。
- 新增 `encode_airfoil_dat(...)` 和 `fit_airfoil_points(...)`，作为 GRASS 本地拟合实现和单元测试覆盖。
- `airfoil_codec.py` 内置外部工程优化后的 fit 参数和学习率调度器，并直接输出 GRASS wing payload 所需的 30 维 `airfoil` code。
- 支持 `coord_norm.pt` 的坐标归一化/反归一化，但不依赖外部 Python 模块。
- `test/test_airfoil_codec.py` 覆盖 code 打包、解包、decode、`.dat` 短迭代拟合、优化配置和缺失配置报错。
- `test/run_processed_foil_codec_demo.py` 默认直接读取 `foildata/processed_foil` 中的单个翼型，保存编码结果和拟合对比图。

示例:

```powershell
D:\Software\anaconda\envs\myml\python.exe test\run_processed_foil_codec_demo.py --device cpu
```

默认输出:

```text
outputs\airfoil_codec_demo\*_encoded_bezier.pt
outputs\airfoil_codec_demo\*_bezier_fit.png
```

尚未完成:

- 将真实翼型文件选择策略接入 `data/generate_dataset.py` 或新的 structured 数据预处理脚本。
- 批量生成 structured `.pt` 数据集。
- Bezier decode 可视化脚本。

### 阶段 4: 自由生成与后处理

目标:

- 修改 `decode_structure` 的 leaf 生成逻辑。
- 自由生成时用 `componentClassifier` 选择 box decoder head。
- 生成 wing 时输出 wing geometry + airfoil code。
- 增加可视化或导出时的翼型 decode。

### 阶段 5: surrogate 评价或约束

目标:

- 加载外部 surrogate checkpoint。
- 对生成翼型进行气动评价。
- 先用于筛选，再评估是否进入训练 loss。

## 11. 需要避免的实现方式

- 不要把所有部件 padding 成最大联合 schema。
- 不要给 fuselage/engine 保存无意义的 airfoil 全零字段。
- 不要让 `NodeClassifier` 同时预测树节点类型和部件类型。
- 不要在多个文件手写相同的字段切片、类别 id 或 Bezier 维度。
- 不要直接修改 `Grass-matlab`。
- 不要在训练代码中用 `config.get(..., default)` 掩盖缺失配置。

## 12. 验证清单

每个阶段至少验证:

- 单个 structured sample 能构造成 `Tree`。
- `ops` 后序栈构造后的根节点唯一。
- 各 leaf 的 component 与字段完整性校验通过。
- `encode_structure_fold` 能对 mixed component batch 前向。
- wing payload 缺少 `airfoil` 时 fail-fast。
- fuselage/engine payload 出现 `airfoil` 字段时按 schema 规则处理。第一版建议直接报错。
- typed decoder 的 teacher-forced loss 能 backward。
- Bezier code decode 后曲线维度和数值范围符合 codec 配置。
