# OpenVSP Fuselage Creation

## Geometry Set

`create_fuselage(...)` assigns its `FUSELAGE` geometry to the user geometry set
named `fuselage`. `runaero_panel(...)` uses this set as its thick `GeomSet`;
the separately assigned `wing` set is passed as the thin `ThinGeomSet`.

本文记录共享包 `aircraft_tools.openvsp_infrastructure` 中简单多段式机身的创建约定。

## 函数接口

`create_fuselage(pos, xsec_list, tess_int)` 在当前 OpenVSP model 中添加一个 `FUSELAGE` 几何，并返回该几何的 `geom_id`。

`pos` 沿用现有 `create_wing(...)` 的位置字典风格:

```python
{"name": "fuselage", "x": 0.0, "y": 0.0, "z": 0.0, "yr": 0.0}
```

`xsec_list` 使用绝对 `x` 站位，单位与当前 OpenVSP model 一致:

```python
[
    {"x": 0.00, "width": 0.05, "height": 0.06},
    {"x": 0.18, "width": 0.13, "height": 0.14},
    {"x": 0.90, "width": 0.13, "height": 0.14},
    {"x": 1.20, "width": 0.06, "height": 0.07},
]
```

函数内部会把绝对 `x` 站位转换为 OpenVSP FUSELAGE 的 `XLocPercent`，并将 `Design.Length` 设为 `xsec_list[-1]["x"] - xsec_list[0]["x"]`。机身几何的 `X_Rel_Location` 会使用 `pos["x"] + xsec_list[0]["x"]`，因此 `xsec_list` 可以从非零绝对站位开始。

`tess_int` 是唯一的目标网格尺寸，必须为正数，单位与机身尺寸一致。每段的 `SectTess_U` 复用机翼的 `next_tess_value(...)` 取整规则，并由相邻截面站位距离推导；整条机身的 `Tess_W` 则由所有正尺寸椭圆截面中最大周长（Ramanujan 近似）推导。当前 `vsppytools` 环境的 OpenVSP 3.50.5 对 FUSELAGE 周向离散接受 `9, 17, 25, ...` 等合法档位，因此实现会选择与目标周向数量最接近的档位。这样同一 `tess_int` 下，机翼、机身和短舱的网格边长保持同一量级；零尺寸端站不会参与周向尺度计算。

## 截面约定

- 每个正尺寸截面使用椭圆 `XSec`，通过 `Ellipse_Width` 和 `Ellipse_Height` 控制尺寸；首尾截面可据此形成非零尺寸的机头和机尾。
- 为兼容旧模型，首尾截面也允许同时使用 `width=0` 和 `height=0` 的默认点截面；中间截面不允许零尺寸。
- `x` 必须严格递增。
- 每个截面的宽高必须同时为零，或同时为正数；禁止一项为零、另一项为正数，以及负尺寸。

这些输入检查采用 fail-fast 方式；不合理输入会直接抛出 `ValueError`，避免生成看似成功但几何异常的 `.vsp3`。

## OBB 线性过渡

完成所有截面形状与站位设置后，`create_fuselage(...)` 会对 `XSecSurf` 中每一个截面调用
`vsp.ResetXSecSkinParms(xsec_id)`。这等价于 OpenVSP 界面的 `Clear Skinning for the Entire Stack`：
相邻站位的表面过渡不再使用默认的 skinning 参数，而保持线性，确保三段机身曲面与其 OBB
表示一致。

## 验证脚本

GRASS 项目中提供了包含多段椭圆机身和梯形翼的 panel 验证脚本:

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe test\run_simple_wing_panel.py
```

输出文件位于:

```text
D:\3D\Projects\GRASS\GRASS-plane\outputs\simple_wing_panel\simple_wing_panel.vsp3
```

运行完成后，用 OpenVSP 打开该文件，检查椭圆多段机身、机翼相对位置和厚面几何是否正常。若只需验证 conventional OBB 示例的非零端站、圆端盖和统一网格尺度，运行 `test/create_obb_vsp.py`；其完整流程见 `docs/openvsp_obb_json_visualization.md`。
