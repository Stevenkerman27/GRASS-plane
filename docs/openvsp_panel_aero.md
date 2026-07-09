# OpenVSP Panel Aero Runner

本文记录 `infrastructure.py` 中 VSPAERO 面元法运行约定。

## OpenVSP 版本和环境

- OpenVSP Python API 位于 `vsppytools` 环境。
- 当前验证到的 OpenVSP 版本路径为 `D:\3D\Projects\SUAVE\OpenVSP-3.50.5-win64\python\openvsp`。

## VSPAERO set 约定

OpenVSP 3.50.5 对 `VSPAEROComputeGeometry` 和 `VSPAEROSweep` 的输入说明为:

- `GeomSet`: Thick surface geometry Set for analysis.
- `ThinGeomSet`: Thin surface geometry Set for analysis.

因此代码中采用单一来源常量:

- `VSPAERO_THICK_GEOM_SET = vsp.SET_SHOWN`
- `VSPAERO_THIN_GEOM_SET = vsp.SET_SHOWN`
- `VSPAERO_NO_GEOM_SET = vsp.SET_NONE`

VLM 跑法使用:

```text
GeomSet = SET_NONE
ThinGeomSet = SET_SHOWN
```

Panel 跑法使用:

```text
GeomSet = SET_SHOWN
ThinGeomSet = SET_NONE
```

这对应“厚几何设为 shown，薄面/VLM 设为 none”的操作。

## 函数边界

`runaero_panel(...)` 用于不带作动盘的面元法 alpha sweep。它复用 VLM 跑法的参考量、收敛参数和 `.polar` 解析逻辑，但强制隐藏所有 PROP 几何，并且不配置 actuator disk。

返回值保持与 `runaero(...)` 一致:

```text
drag, alpha, lift, net_drag, power, Cl_list, CMy
```

其中 panel 跑法没有作动盘，`power` 为 0，`net_drag` 等于气动阻力。

## 简单机翼加机身测试

运行:

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe test\run_simple_wing_panel.py
```

脚本会在以下目录生成文件:

```text
outputs\simple_wing_panel\
```

主要检查文件:

- `simple_wing_panel.vsp3`
- `simple_wing_panel.polar`

脚本会创建一个椭圆多段机身和一个简单梯形翼，然后用 panel 法运行 VSPAERO alpha sweep。参考面积、展长和弦长仍按机翼定义，机身作为厚面几何参与 panel 几何计算。

运行完成后，打开 `simple_wing_panel.vsp3`，检查 OpenVSP 中简单梯形翼、椭圆机身和二者相对位置是否正常，并确认 VSPAERO panel 网格没有异常破面。

## 梯形翼面积约定

`think_trapwing(...)` 的 `span` 表示半展长，但返回的 `wing_S` 按左右对称后的全翼面积计算。因此面积累加使用:

```text
span_i * (c_i + c_{i+1})
```

这里不再乘 `0.5`。当 `fuse_w == 0` 时，函数返回单段梯形翼:

```text
spanlist = [span]
chordlist = [root, tip]
twistlist = [twist]
```

当 `fuse_w > 0` 时，函数返回根部等弦段加外翼梯形段:

```text
spanlist = [fuse_w, span - fuse_w]
chordlist = [root, root, tip]
twistlist = [0, twist]
```
