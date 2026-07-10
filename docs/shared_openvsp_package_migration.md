# Shared OpenVSP Package Migration

本文记录 `GRASS-plane` 与 `D:\3D\Projects\SUAVE\opt` 共用 `infrastructure.py` 时的单一来源方案。

## 目标

过去两个项目各自保存一份 `infrastructure.py`，改动会逐渐分叉，无法保证同步。

现在将公共实现提升为独立 Python 包:

- 共享包根目录: `D:\3D\Projects\aircraft-tools`
- 包名: `aircraft_tools`
- 当前权威模块: `aircraft_tools.openvsp_infrastructure`

这样公共 OpenVSP 几何和气动逻辑只保留一份源码，满足 DRY 和单一来源要求。

## 当前迁移策略

第一阶段采用兼容迁移，避免一次性改动两个项目的大量导入代码。

`GRASS-plane` 中保留一个薄兼容入口:

- [infrastructure.py](/D:/3D/Projects/GRASS/GRASS-plane/infrastructure.py)

它只负责:

1. 定位共享包目录 `D:\3D\Projects\aircraft-tools`
2. 将共享包目录加入 `sys.path`
3. 将 `infrastructure` 模块名指向 `aircraft_tools.openvsp_infrastructure` 模块对象

因此本项目原有 `import infrastructure as infra` 调用方式可以先不改。注意这里必须使用模块对象别名，而不是 `from ... import *` 复制符号；否则 `infra.file_name = ...` 和 `infra.case_name = ...` 只会修改 shim 自己的变量，真正执行函数的共享包模块仍然会使用默认的 `test.vsp3` 和 `test`。

## 推荐的后续迁移

后续建议把 `SUAVE\opt\infrastructure.py` 也改成同样的薄兼容入口，或者直接把业务代码改成:

```python
from aircraft_tools.openvsp_infrastructure import runaero, runaero_panel
```

当两个项目都稳定改用共享包后，可以视情况删除各自项目中的兼容入口。

## 安装建议

若希望不依赖手动注入 `sys.path`，建议分别在两个环境中做 editable install:

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe -m pip install -e D:\3D\Projects\aircraft-tools
D:\Software\anaconda\envs\myml\python.exe -m pip install -e D:\3D\Projects\aircraft-tools
```

这样两个项目都能直接导入 `aircraft_tools`，并共享同一份实现。
