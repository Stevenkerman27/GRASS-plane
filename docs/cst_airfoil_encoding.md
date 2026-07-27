# CST 翼型编码

`cst_airfoil_codec.py` 是项目唯一的翼型编解码与拟合实现。CST 24D code 用于每个翼面截面，并由`data/cst_airfoil_code_cache.py` 缓存。

## 编码

处理后翼型使用物理坐标，两个尾缘的 `x` 固定为 `1`，共享前缘固定为 `(0, 0)`。令：

```text
xi = x
y = xi y_TE + xi^N1 (1 - xi)^N2 S(xi)
```

其中 `S` 是 10 项 Bernstein 多项式。`N1`、`N2` 是每个翼型独立拟合并写入 code 的正值参数，初始值分别为`0.5`、`1.0`，且解码和拟合均要求它们大于 `util.CST_MIN_CLASS_FUNCTION_EXPONENT`。24D code 依次保存：上表面10 项、下表面 10 项、上尾缘 `y`、下尾缘 `y`、`N1`、`N2`。所有常量及拟合参数的唯一来源是 `util.py`。

`CST_FIT_CONFIG` 中的 `initial_n1`、`initial_n2` 仅是优化初值，不会固定输出 code 中的指数；每次拟合结果会保存实际优化后的 `N1/N2`。

## 预计算诊断

`data/precompute_airfoil_codes.py` 将每个翼型的 24D code 及 `mae`、`mse`、`max_point_error`、前缘局部误差写入 CST 缓存。预计算完成后按 `max_point_error` 降序打印前 10 个翼型，并从缓存 code解码生成 10 张单独的对比图到 `data/airfoil_fit_visualizations`。可用 `--top-error-count` 和`--visualization-dir` 修改数量与输出目录。默认还会写入 `data/cst_fit_report.json`：其中包含前 10 个最大误差样本、每个样本的全部拟合指标，以及所有样本上每项指标的最小值、均值、最大值。可用 `--report`修改报告路径。
