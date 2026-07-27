# 飞翼数据集生成

## 定义

批量生成的 JSON schema 为 `flying_wing_variable_layout_v2`，拓扑为
`symmetric_flying_wing_fuselage_sequence_v1`。`data/aircraft_dataset_common.py` 是 JSON、
GRASS 组件树和 OpenVSP 构造的唯一实现；采样边界与相邻截面约束固定写在该模块的
生成和验证逻辑中，不作为可修改的配置。

每个样本包含一个中心机身序列叶和右侧主翼 generator，`SYM` 节点在 `XZ` 平面镜像生成左翼。
不生成平尾、垂尾、发动机或发动机对应的 `SYM`。机身与主翼均为可变长度截面序列；组件树的
box 顺序固定为 `fuselage`、`main_wing_right`，且必须由 JSON 的 `box_order` 明确记录。
OpenVSP 创建机身后，必须对其整个截面栈逐站调用 `ResetXSecSkinParms`，等价于界面的
`Clear Skinning for the Entire Stack`。因此相邻机身站位之间采用线性过渡，和机身截面序列一致。

## 主翼截面

主翼按 `+Y` 严格 root-to-tip 排列，并随机生成 `2..8` 个截面。每站翼型独立地从处理后的
翼型库随机选择。JSON 的有效截面为：

```text
{airfoil_source: str, leading_edge_xyz: [x, y, z], chord: float, twist: float}
```

配置中的翼面扭转范围和相邻变化上限使用角度（degree）；生成后 JSON 的 `twist` 单位为 rad，
旋转中心始终为该站前缘点 `leading_edge_xyz`。OpenVSP 的每个翼段
将 `Sweep_Location` 和 `Twist_Location` 均设为 `0.0`，使 OpenVSP 几何与 JSON 和 `.pt`
编码使用同一个前缘参考点；段间前缘 z 坐标通过 OpenVSP 的 `Dihedral` 写入。正扭转表示正迎角。

总体翼根弦长、尖弦比、根部前缘 x 位置、翼尖后掠、上反角和扭转范围沿用原主翼范围。
相邻有效截面必须满足：

- 外侧弦长相对内侧弦长变化不超过 `20%`；
- 前缘 x 坐标变化不超过 `0.06 m`；
- 配置的相邻扭转变化不超过 `2.3°`（内部校验转换为弧度）。

由于两截面主翼只有一个相邻区间，无法同时满足历史的 `0.40..0.75` 翼尖弦长比和
`20%` 弦长变化上限。经确认，只有两截面样本使用 `0.80..1.00` 翼尖弦长比；三至八截面
样本仍以历史范围为上界，并按 `0.8^(N-1)` 收紧可达下界。

转换到 `.pt` 后，主翼每站编码为 `[CST24, leading_edge_xyz3, chord, twist]`，机身每站编码为
`[x, y, z, width, height]`；两者均零填充到最多 8 站。`section_count` 是有效截面数的唯一
权威字段，padding mask 只能由 `util.section_mask(section_count)` 导出。

## 生成与转换

默认输出目录为 `data/flying_wing_dataset`。生成器默认写入 `200` 个同索引 `.vsp3` 和
`.json` 文件；样本随机种子为 `root_seed + sample_index`，可按索引重建。完整流程入口为
仓库根目录的 `generate_flying_wing_dataset.ps1`：它先在 `vsppytools` 环境执行
`data/generate_flying_wing_dataset.py`，再在 `myml` 环境执行
`data/json_to_typed_obb_dataset.py`。脚本固定生成 `200` 个样本并默认覆盖已有同索引文件。

`.pt` 转换依赖 `data/cst_airfoil_code_cache.pt`，该缓存必须事先在 `myml` 环境生成：

```powershell
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\precompute_airfoil_codes.py --workers 4
.\generate_flying_wing_dataset.ps1
```

生成器写入前验证 JSON schema；转换器保存后重新加载，并用
`grassdata.validate_structured_sample` 验证每个结构化样本。既有常规布局数据不符合本
schema，必须删除并完整重建，不能混用。
