# Aircraft Dataset Generation

本文档统一定义常规/鸭式布局和飞翼数据集的生成流程、序列 schema 及转换约束。
`data/generate_aircraft_datasets.ps1` 是唯一的生成入口。脚本只清空两个已注册的输出目录：

- `data/flying_wing_dataset`：200 个飞翼样本；
- `data/conventional_canard_dataset`：200 个样本，常规布局与鸭式布局各占 50%。

数据集路径唯一地定义在 `project_paths.AIRCRAFT_DATASET_SPECS`，生成器和
`data/visualize_dataset.py` 共用该配置。脚本先在 `vsppytools` 环境生成 OpenVSP/JSON，
再在 `myml` 环境转换为 `.pt`。OpenVSP JSON generation writes its fixed sequence
type literals directly and never imports `grassdata.py`.

可从任意工作目录运行；脚本会先切换到仓库根目录，再读取 `project_paths.py` 中的权威数据集路径。推荐从仓库根目录运行：

```powershell
.\data\generate_aircraft_datasets.ps1
```

`data/visualize_dataset.py` 默认随机选择布局和样本；使用 `--layout flying_wing` 或
`--layout conventional_canard`，并配合 `--index N` 可固定选择。

## 常规/鸭式布局

该 schema 生成不带发动机和垂尾的常规布局与鸭式布局。两种布局使用相同的 GRASS 树和组件
类别，唯一布局差异是辅助翼相对于主翼的纵向位置。后序操作及 JSON `box_order` 固定为：

```text
fuselage, main_wing_right, SYM(main_wing_right), ADJ,
auxiliary_wing_right, SYM(auxiliary_wing_right), ADJ
```

`SYM` 是穿过 XZ 平面的反射，仅复制对应的右半翼。组件分类只有 `fuselage` 和 `wing`，
主翼与辅助翼由固定树位置区分：

```text
fuselage.sections[8, 5]              = [x, y, z, width, height], count in [2, 8]
main_wing_right.sections[8, 29]      = [CST24, leading_edge_xyz3, chord, twist], count in [2, 8]
auxiliary_wing_right.sections[8, 29] = [CST24, leading_edge_xyz3, chord, twist], count in [2, 4]
```

`section_count` 是每个序列 padding 的唯一权威字段；辅助翼有效截面限制为 `2..4`，但使用与
主翼相同的 `[8, 29]` schema。联合 AE 中所有翼叶共享 RNN encoder/decoder，并在 `2..8`
内预测 count。

辅助翼的翼型、弦长、扭转和二次平面形参数范围与主翼相同，全翼展为主翼全翼展的
`0.20..0.80`。布局间距由所有截面的前缘及按扭转计算的后缘形成的纵向包络确定：常规布局要求
`auxiliary_x_min - main_x_max`，鸭式布局要求 `main_x_min - auxiliary_x_max`，实际边缘间距
均随机为 `0.10..0.45 * fuselage_length`。辅助翼根弦的纵向范围还必须完全位于机身
`[0, fuselage_length]` 内；无法同时满足根部连接和间隙的构型整体重采样。

常规/鸭式数据集由统一脚本生成 200 个 OpenVSP/JSON 样本，再转换为
`data/conventional_canard_dataset/conventional_canard_dataset.pt`。训练入口
`train_autoencoder.py` 默认读取该文件。

## 飞翼布局

飞翼 JSON schema 为 `flying_wing_variable_layout_v2`，拓扑为
`symmetric_flying_wing_fuselage_sequence_v1`。`data/aircraft_dataset_common.py` 是 JSON、
GRASS 组件树和 OpenVSP 构造的唯一实现；采样边界与相邻截面约束均在该模块中定义。

每个样本包含中心机身序列叶和右侧主翼 generator，`SYM` 节点在 XZ 平面镜像生成左翼。不生成
平尾、垂尾或发动机。`box_order` 固定为 `fuselage`、`main_wing_right`。机身和主翼均为
`2..8` 个截面的可变长度序列；OpenVSP 创建机身后，必须对整个截面栈逐站调用
`ResetXSecSkinParms`，使相邻站位线性过渡。

主翼严格按 `+Y` 方向 root-to-tip 排列，每站翼型独立从处理后的翼型库随机选择。有效 JSON
截面为：

```text
{airfoil_source: str, leading_edge_xyz: [x, y, z], chord: float, twist: float}
```

配置中的扭转范围和相邻变化上限使用角度（degree），生成后的 JSON `twist` 使用 rad；旋转中心
始终是该站前缘点。OpenVSP 每个翼段的 `Sweep_Location` 和 `Twist_Location` 均为 `0.0`，
段间前缘 z 坐标通过 `Dihedral` 写入，正扭转表示正迎角。

相邻有效截面必须满足外侧弦长相对内侧不超过 `20%`、前缘 x 坐标变化不超过 `0.06 m`，以及
相邻扭转变化不超过 `2.3°`（内部校验时转换为弧度）。两截面样本的翼尖弦长比使用
`0.80..1.00`；三至八截面样本沿用历史上界，并按 `0.8^(N-1)` 收紧可达下界。转换为 `.pt`
后，主翼每站编码为 `[CST24, leading_edge_xyz3, chord, twist]`，机身每站编码为
`[x, y, z, width, height]`，均零填充到最多 8 站。

飞翼生成器默认写入 200 个同索引 `.vsp3` 和 `.json` 文件，样本种子为 `root_seed + sample_index`，
可按索引重建。`.pt` 转换依赖 `data/cst_airfoil_code_cache.pt`，缓存需先在 `myml` 环境生成：

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\precompute_airfoil_codes.py --workers 4
.\data\generate_aircraft_datasets.ps1
```

## 转换与验证

生成器写入前验证 JSON schema；转换器保存后重新加载，并用
`grassdata.validate_structured_sample` 验证每个结构化样本。既有常规布局数据不符合飞翼 schema，
必须删除并完整重建，不能混用。
