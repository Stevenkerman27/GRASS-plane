# OpenVSP 与 OBB 可视化流程

本文档记录单机 OpenVSP 验证脚本与当前批量 OBB 数据集可视化的职责边界。`vsppytools` 环境没有 PyTorch 和 matplotlib 3D，因此 OpenVSP 生成与结构化 `.pt` 读取必须在两个 Python 环境中运行。

## 1. OpenVSP 生成

`test/create_obb_vsp.py` 只负责：

- 构造 conventional aircraft 的参考几何，其中机身以 `fuselage_nose`、`fuselage_center`、`fuselage_tail` 三个连续叶节点表示；机翼、机身和短舱都传入同一个 `tess_int=0.025`，以保持 OpenVSP 网格尺度一致。
- 在 `vsppytools` 环境中生成 OpenVSP 半模 `.vsp3`。
- 使用本项目 `foildata/processed_foil/_falcon.dat` 作为主翼、垂尾和平尾的文件翼型。
- 写出不含 CST code 的中间几何 JSON。

默认输出：

```text
outputs/conventional_openvsp_obb/conventional_half.vsp3
outputs/conventional_openvsp_obb/conventional_geometry.json
```

中间 JSON 使用 `conventional_geometry_v1` schema，关键字段：

- `half_components`：与 GRASS 后序 `BOX` 节点顺序一致的半机部件。机身和发动机为 10D geometry；翼面为 root-to-tip `sections`，每项保存翼型来源、四分之一弦点、弦长和 twist。
- `full_draw_components`：补齐左右对称部件的完整飞机几何。
- `airfoil_source`：OpenVSP 与 CST 拟合共用的 `.dat` 文件。
- `box_order`、`full_draw_box_order`、`ops`、`syms`：部件顺序和树拓扑。

运行：

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe test\create_obb_vsp.py
```

该脚本不导入 PyTorch 或 matplotlib，不会尝试拟合 CST code。

批量数据集使用 `data/aircraft_dataset_common.py` 的 `conventional_variable_layout_v3` schema：主翼、垂尾、平尾各保存 `2..8` 个有序截面；发动机总数可为 `0/2/4`，非零短舱仅保留右侧 generator 并由 `SYM` 镜像。垂尾在 OpenVSP 写入前从来源上表面构造严格镜像的对称翼型，对应 OBB CST code 也应用同一约束。单机旧验证脚本仍使用自己的 `conventional_geometry_v1` 中间 JSON，不能与批量 schema 混用。

## 2. 数据集 OBB 绘图

`data/visualize_dataset.py` 是当前的交互式 OBB 可视化入口，必须在 `myml` 环境运行。它直接读取 `data/flying_wing_dataset/flying_wing_dataset.pt`，随机选择一个结构化样本并由 `SYM` 节点展开完整飞机：

- 机身和发动机绘制为椭圆截面；翼面绘制为梯形平面轮廓。
- 支持可变常规构型中的镜像 `SYM` 节点；遇到其他对称类型会直接报错。
- 不重新读取 `.dat` 或拟合 CST code，因此弹出窗口前不需要进行翼型优化。

运行：

```powershell
D:\Software\anaconda\envs\myml\python.exe data\visualize_dataset.py
```

可选参数：

```powershell
D:\Software\anaconda\envs\myml\python.exe data\visualize_dataset.py --index 17
D:\Software\anaconda\envs\myml\python.exe data\visualize_dataset.py --seed 20260711
```

批量生成、JSON-to-PT 转换和数据文件位置详见 `docs/flying_wing_dataset_generation.md`。单机 JSON 仍是 `test/create_obb_vsp.py` 的中间验证产物，不再有独立的 JSON 绘图入口。

## 3. OpenVSP 机身和短舱

示例机身与 OpenVSP 使用同一组四个椭圆截面站位和尺寸。首尾截面均为正尺寸，分别形成机头和机尾的收缩过渡。机身和发动机短舱都将 `CapUMinOption`、`CapUMaxOption` 设置为 OpenVSP `ROUND_END_CAP`；该设置只影响 `.vsp3` 曲面，不改变 OBB 数据。
