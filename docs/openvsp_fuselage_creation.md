# OpenVSP Fuselage Creation

本文记录共享包 `aircraft_tools.openvsp_infrastructure` 中简单多段式机身的创建约定。

## 函数接口

`create_fuselage(pos, xsec_list, tess_u=12, tess_w=17)` 在当前 OpenVSP model 中添加一个 `FUSELAGE` 几何，并返回该几何的 `geom_id`。

`pos` 沿用现有 `create_wing(...)` 的位置字典风格:

```python
{"name": "fuselage", "x": 0.0, "y": 0.0, "z": 0.0, "yr": 0.0}
```

`xsec_list` 使用绝对 `x` 站位，单位与当前 OpenVSP model 一致:

```python
[
    {"x": 0.00, "width": 0.00, "height": 0.00},
    {"x": 0.12, "width": 0.10, "height": 0.09},
    {"x": 0.45, "width": 0.14, "height": 0.12},
    {"x": 0.85, "width": 0.12, "height": 0.11},
    {"x": 1.05, "width": 0.00, "height": 0.00},
]
```

函数内部会把绝对 `x` 站位转换为 OpenVSP FUSELAGE 的 `XLocPercent`，并将 `Design.Length` 设为 `xsec_list[-1]["x"] - xsec_list[0]["x"]`。机身几何的 `X_Rel_Location` 会使用 `pos["x"] + xsec_list[0]["x"]`，因此 `xsec_list` 可以从非零绝对站位开始。

## 截面约定

- 首尾截面为 OpenVSP 默认点截面，`width` 和 `height` 必须为 `0`。
- 中间截面保持 OpenVSP 默认椭圆 `XSec`，通过 `Ellipse_Width` 和 `Ellipse_Height` 控制尺寸。
- `x` 必须严格递增。
- 中间截面的 `width` 和 `height` 必须为正数。

这些输入检查采用 fail-fast 方式；不合理输入会直接抛出 `ValueError`，避免生成看似成功但几何异常的 `.vsp3`。

## 验证脚本

GRASS 项目中提供了一个最小几何验证脚本:

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe test\run_simple_fuselage.py
```

输出文件位于:

```text
D:\3D\Projects\GRASS\GRASS-plane\outputs\simple_fuselage\simple_fuselage.vsp3
```

运行完成后，用 OpenVSP 打开该文件，检查椭圆多段机身、机翼相对位置和厚面几何是否正常。
