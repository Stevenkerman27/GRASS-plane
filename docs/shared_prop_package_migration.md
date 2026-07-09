# Shared Prop Package Migration

本文记录 `prop.py` 收编进共享包后的迁移方案。

## 单一来源

螺旋桨数据读取和等推力求解的权威实现现在位于:

- `D:\3D\Projects\aircraft-tools\aircraft_tools\prop_model.py`

该模块与 `aircraft_tools.openvsp_infrastructure` 同属一个共享包，避免 `prop.py` 和 `infrastructure.py` 在不同项目中继续分叉。

## 兼容策略

为了不一次性修改所有旧脚本，当前项目保留薄兼容入口:

- [prop.py](/D:/3D/Projects/GRASS/GRASS-plane/prop.py)

它只负责:

1. 定位共享包根目录 `D:\3D\Projects\aircraft-tools`
2. 将该目录加入 `sys.path`
3. 从 `aircraft_tools.prop_model` re-export 现有符号

因此现有 `import prop` 代码可以继续工作。

## 共享包内部依赖

共享版 `openvsp_infrastructure` 不再依赖项目根目录下的 `prop.py`，而是显式通过:

```python
from aircraft_tools import prop_model
```

加载包内权威实现。

这样即使某个项目没有本地 `prop.py` 文件，共享包内部逻辑也仍然可解析。

## 后续建议

`SUAVE\opt` 中直接 `import prop` 的脚本也建议统一保留为薄兼容层，或者逐步改成:

```python
from aircraft_tools import prop_model as prop
```

等两个项目都稳定迁移后，可以再考虑移除兼容层。
