# Typed Box 与 CST 翼型编码

该文件保留 CST 24D 的定义；翼面叶 payload 已迁移为 `sections[max_sections=8, 29] + section_count`，不再使用 8D 翼面几何和根/尖 48D 扁平 code。29D 与 mask、twist 约定以 `docs/hierarchical_aircraft_encoding.md` 和 `util.py` 为准。

typed wing payload 使用零填充的 `[8,29]` 截面张量和 `section_count`。每个有效截面为 `[CST24, leading_edge_xyz3, chord, twist]`；机身使用零填充的 `[8,5]` 截面张量，每站为 `[x,y,z,width,height]`，发动机暂时保持 10D OBB geometry。

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
