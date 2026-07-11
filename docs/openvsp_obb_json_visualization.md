# OpenVSP 与 OBB 可视化流程

本文档记录 `test/create_obb_vsp.py` 和 `test/plot_obb.py` 的职责边界。`vsppytools` 环境没有 PyTorch 和 matplotlib 3D，因此 OpenVSP 生成与 Bezier 拟合/绘图必须拆成两个 Python 进程运行。

## 1. OpenVSP 生成

`test/create_obb_vsp.py` 只负责：

- 构造 conventional aircraft 的参考几何，其中机身以 `fuselage_nose`、`fuselage_center`、`fuselage_tail` 三个连续叶节点表示；机翼、机身和短舱都传入同一个 `tess_int=0.025`，以保持 OpenVSP 网格尺度一致。
- 在 `vsppytools` 环境中生成 OpenVSP 半模 `.vsp3`。
- 使用 `D:\3D\Projects\ML\NN\foildata\processed_foil\_falcon.dat` 作为主翼、垂尾和平尾的文件翼型。
- 写出不含 Bezier code 的中间几何 JSON。

默认输出：

```text
outputs/conventional_openvsp_obb/conventional_half.vsp3
outputs/conventional_openvsp_obb/conventional_geometry.json
```

中间 JSON 使用 `conventional_geometry_v1` schema，关键字段：

- `half_components`：与 GRASS 后序 `BOX` 节点顺序一致的半机部件几何。机身和发动机为 10D；翼面为 8D `[x1, y1, z1, x2, y2, z2, root_chord, tip_chord]`。
- `full_draw_components`：补齐左右对称部件的完整飞机几何。
- `airfoil_source`：OpenVSP 与 Bezier 拟合共用的 `.dat` 文件。
- `box_order`、`full_draw_box_order`、`ops`、`syms`：部件顺序和树拓扑。

运行：

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe test\create_obb_vsp.py
```

该脚本不导入 PyTorch 或 matplotlib，不会尝试拟合 Bezier code。

## 2. OBB 绘图

`test/plot_obb.py` 必须在 `myml` 环境运行。它读取中间 JSON，并直接调用项目内的 `airfoil_codec.encode_processed_airfoil_dat(...)`：

- 读取 `airfoil_source` 的 `.dat` 坐标。
- 使用 `airfoil_codec.py` 中集中定义的优化拟合参数、坐标归一化和学习率调度器。
- 在内存中构造 typed OBB：机身/发动机为 13D `[geometry10, class3]`，翼面为 41D `[geometry8, bezier30, class3]`。
- 绘制椭圆机身和发动机短舱、平面翼面轮廓。当前不从 30D code 重建翼型曲线。

该步骤不写最终 OBB JSON 或 `.pt` 文件；每次绘图都会重新拟合源翼型，避免缓存的 code 与当前 `.vsp3` 或 `.dat` 文件不一致。

运行：

```powershell
D:\Software\anaconda\envs\myml\python.exe test\plot_obb.py --device cpu
```

可选参数：

```powershell
D:\Software\anaconda\envs\myml\python.exe test\plot_obb.py --json outputs\conventional_openvsp_obb\conventional_geometry.json --device cuda
```

绘图脚本会对中间 JSON schema、部件几何维度、Bezier code 长度和组件 one-hot 长度进行 fail-fast 校验。

## 3. OpenVSP 机身和短舱

示例机身与 OpenVSP 使用同一组四个椭圆截面站位和尺寸。首尾截面均为正尺寸，分别形成机头和机尾的收缩过渡。机身和发动机短舱都将 `CapUMinOption`、`CapUMaxOption` 设置为 OpenVSP `ROUND_END_CAP`；该设置只影响 `.vsp3` 曲面，不改变 OBB 数据。
