# 翼型数据管理

`foildata/manage_foildata.py` 将 Selig `.dat` 翼型筛选、重采样后写入
`foildata/processed_foil`。`util.py` 是该流程和翼型 codec 采样形状的唯一配置来源，不读取或导入
NN 工程。

| `util.py` 常量 | 当前值 | 用途 |
| --- | ---: | --- |
| `AIRFOIL_DEFAULT_OUTPUT_POINTS` | 100 | 每个处理后翼型的坐标数。|
| `AIRFOIL_DEFAULT_POINT_DENSITY_BETA` | 1.3 | 前缘加密的参数指数。|

重采样顺序为上表面尾缘到前缘、再下表面前缘到尾缘。对于 100 点，
上表面取 51 点，下表面取 50 点，拼接时去除重复前缘点，最终保持 100 点。
重采样后以共享前缘和两个尾缘点的中点定义弦线，将翼型平移、旋转、等比缩放到
前缘 `(0, 0)`、尾缘中点 `(1, 0)` 的局部坐标系。最后两个尾缘点的 `x` 直接写为 `1`。

`foildata/manage_foildata.py` 与 `cst_airfoil_codec.py` 均从 `util.py` 读取这些常量。修改后必须重新处理
翼型并重新生成依赖其编码的数据工件。
