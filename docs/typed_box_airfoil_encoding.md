# Typed Box 与 CST 翼型编码

typed wing payload 使用 8D 翼面几何和根、尖各一条 24D CST code；因此翼型 payload 为 48D，带 3D
部件类别的扁平 wing OBB 为 59D。机身与发动机保持 10D 几何和 13D 扁平 OBB。

`util.py` 是组件 schema、翼型维度、采样布局和 `CST_FIT_CONFIG` 的唯一来源。`cst_airfoil_codec.py`
是 CST 的 pack/unpack/decode/fit 单一来源，使用物理坐标，尾缘 `x=1`、共享前缘固定为 `(0,0)`。

单截面 CST code 的布局为：

```text
[upper_shape[10], lower_shape[10], upper_te_y, lower_te_y, N1, N2]
```

`N1`、`N2` 是每条 code 独立拟合出的正值；`CST_FIT_CONFIG` 中仅保存优化初值和下界。处理后翼型必须有
`AIRFOIL_DEFAULT_OUTPUT_POINTS` 个点，并按上表面尾缘到前缘、下表面前缘到尾缘排序。

运行 `data/precompute_airfoil_codes.py` 生成 `data/cst_airfoil_code_cache.pt`，随后运行
`data/json_to_typed_obb_dataset.py` 生成 structured 数据集。缓存缺失翼型时转换器立即报错。
