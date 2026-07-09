# OpenVSP 参数化气动样本与曲线代理模型定义

本文档定义一个独立于 GRASS 框架的 OpenVSP 参数化飞行器样本生成与气动代理模型数据流程。该流程的目标不是生成 GRASS 树结构，而是自动创建气动上合理的常规、鸭翼、飞翼布局 OpenVSP 样本，并生成用于训练代理模型的极曲线和俯仰力矩曲线标签。

## 1. 目标与边界

### 1.1 目标

数据集目标:

- 参数化生成多样且气动可行的飞行器布局。
- 自动创建 OpenVSP 几何。
- 使用 VSPAERO 涡格/面元气动结果获得升力、诱导阻力、俯仰力矩等曲线。
- 使用 OpenVSP 寄生阻力工具或独立寄生阻力估算获得 `CD0`。
- 将 `CD0` 与 VSPAERO 的诱导阻力组合成总阻力极曲线。
- 训练代理模型预测:
  - 升力曲线 `CL(alpha)`。
  - 诱导阻力曲线 `CDi(alpha)` 或极曲线 `CDi(CL)`。
  - 总阻力极曲线 `CD(CL)`。
  - 俯仰力矩曲线 `Cm(alpha)`。

### 1.2 边界

- 本流程不依赖 GRASS、VAE、GAN 或树结构编码。
- OpenVSP 几何生成器是独立的数据源。
- 代理模型的输入可以是高层布局参数、派生几何参数、OpenVSP 几何特征，或后续另行定义的 latent code。
- VSPAERO 不负责机身黏性阻力和完整寄生阻力，因此总阻力不能只使用 VSPAERO 输出的 `CDi`。
- 本流程主要面向低速到中低速、预失速范围内的布局级气动估算。涡格法不能可靠预测失速、分离、跨声速激波和复杂黏性干扰。

## 2. 总体数据流水线

推荐流水线:

```text
layout_family
  -> high_level_design_variables
  -> derived_geometry_parameters
  -> OpenVSP geometry generation
  -> geometry validity checks
  -> VSPAERO alpha sweep
  -> parasite drag estimation
  -> aerodynamic curve assembly
  -> curve coefficient fitting
  -> sample validity filtering
  -> surrogate training dataset
```

每个样本必须保存:

- `sample_id`
- `layout_family`
- `high_level_params`
- `derived_params`
- `openvsp_model_path`
- `vspaero_case_config`
- `parasite_drag_config`
- `raw_vspaero_results`
- `raw_parasite_drag_results`
- `curve_fit_coefficients`
- `validity_flags`
- `failure_reason`

## 3. 布局族定义

### 3.1 常规布局

必选部件:

- 机身
- 主翼
- 平尾
- 垂尾

可选部件:

- 发动机短舱
- 翼尖小翼
- 简化起落架整流件

主要设计变量:

- 主翼面积 `S_ref`
- 主翼展长 `b`
- 主翼展弦比 `AR`
- 主翼梢根比 `taper`
- 主翼后掠角 `sweep`
- 主翼上反角 `dihedral`
- 主翼安装位置 `x_wing`
- 主翼安装角 `incidence_wing`
- 平尾尾容量系数 `Vh`
- 垂尾尾容量系数 `Vv`
- 平尾力臂 `l_h`
- 垂尾力臂 `l_v`
- 重心位置 `x_cg`

建议派生:

```text
S_ref = b^2 / AR
c_bar = S_ref / b
S_h = Vh * S_ref * c_bar / l_h
S_v = Vv * S_ref * b / l_v
```

### 3.2 鸭翼布局

必选部件:

- 机身
- 鸭翼
- 主翼
- 垂尾

可选部件:

- 发动机短舱
- 腹鳍或双垂尾

主要设计变量:

- 主翼参数: `S_ref, AR, taper, sweep, dihedral, x_wing`
- 鸭翼面积比 `S_canard / S_ref`
- 鸭翼力臂 `l_c`
- 鸭翼安装角 `incidence_canard`
- 鸭翼高度位置 `z_canard`
- 主翼和鸭翼的纵向间距
- 重心位置 `x_cg`

建议约束:

- 鸭翼必须位于主翼前方。
- 鸭翼不能与机身或主翼严重穿插。
- 鸭翼面积比不应完全随机，应由配平能力和布局尺度限制。
- 鸭翼布局的纵向稳定性可能依赖具体升力分配，不应默认等同于常规尾翼布局。

### 3.3 飞翼布局

必选部件:

- 中心体或翼身融合主体
- 主翼分段
- 控制面或后缘等效面

可选部件:

- 翼尖小翼
- 小垂尾
- 发动机进气/短舱简化体

主要设计变量:

- 展长 `b`
- 参考面积 `S_ref`
- 展弦比 `AR`
- 后掠角 `sweep`
- 梢根比 `taper`
- 扭转分布 `twist_root, twist_tip`
- 外翼安装角
- 中心体尺度
- 重心位置 `x_cg`

建议约束:

- 飞翼必须显式控制 `Cm(alpha)`，不能只看 `CL/CD`。
- 后掠、扭转、重心位置和控制面等效偏角应共同约束。
- 若不建模 reflex airfoil 或控制面偏转，飞翼样本很可能难以配平。

## 4. 参数化采样策略

### 4.1 不直接随机 OpenVSP 控制点

OpenVSP 几何参数应由高层设计变量派生，而不是直接随机每个截面和控制点。

推荐结构:

```text
高层设计变量 -> 派生尺寸 -> 部件参数 -> OpenVSP 几何
```

这样可以保证:

- 参数有明确物理含义。
- 样本更容易气动可行。
- 后续代理模型输入更稳定。
- 失败样本更容易追踪原因。

### 4.2 初始采样方法

推荐初始 DOE:

- 每个布局族单独定义参数范围。
- 使用 Latin Hypercube 或 Sobol 序列采样。
- 按布局族分层采样，避免常规布局数量压倒鸭翼和飞翼。
- 对关键派生参数做硬约束过滤。

不建议:

- 所有参数独立均匀随机。
- 直接对所有 OpenVSP 几何字段做随机扰动。
- 不区分布局族地混合采样。

### 4.3 后续主动采样

初始数据集建立后，可以使用主动学习补样:

- 在代理模型误差大的区域补样。
- 在曲线拟合残差大的区域补样。
- 在可行/不可行边界附近补样。
- 对低覆盖的布局族和性能区间补样。

## 5. 几何可行性过滤

在运行 VSPAERO 前必须做廉价几何检查。

### 5.1 通用硬约束

- 参考面积、展长、弦长、厚度均为正。
- `AR`、`taper`、`sweep`、`dihedral` 在布局族允许范围内。
- 左右翼面对称正确。
- 主翼、尾翼、鸭翼与机身不存在严重自交。
- 尾翼和鸭翼相对主翼的位置满足布局族定义。
- 重心位置位于合理机体长度范围。
- OpenVSP 几何能成功保存并生成气动网格。

### 5.2 气动先验约束

建议在 VSPAERO 前计算低成本派生指标:

- `S_ref`
- `MAC`
- `x_ac_wing` 估算
- `x_cg / MAC`
- `tail_volume`
- `canard_volume`
- `vertical_tail_volume`
- 翼载或尺度代理量

常规布局应优先保证:

- 平尾有足够力臂和面积。
- `x_cg` 不明显落在中性点之后。
- 垂尾面积和力臂不过小。

鸭翼布局应优先保证:

- 鸭翼位于主翼前方。
- 鸭翼面积和力臂可提供合理配平力矩。
- 鸭翼安装角不导致极端前翼升力需求。

飞翼布局应优先保证:

- 后掠、扭转、重心位置组合能产生合理 `Cm`。
- 外翼扭转不过大。
- 不生成过小根弦或极端尖锐翼尖。

## 6. VSPAERO 气动计算定义

### 6.1 VSPAERO 输出用途

VSPAERO 涡格法/面元法结果用于提供:

- `CL(alpha)`
- `CDi(alpha)`
- `Cm(alpha)`
- 可选: `CY(beta)`、`Cl(beta)`、`Cn(beta)` 等稳定性导数

核心限制:

- `CDi` 可用于诱导阻力。
- 不应把 VSPAERO 的阻力输出直接当作完整总阻力。
- 机身、短舱、干扰、表面摩擦等寄生阻力需另行估算。

### 6.2 alpha sweep

建议对每个样本运行统一的迎角扫描:

```text
alpha_grid = [-4, -2, 0, 2, 4, 6, 8, 10, 12]
```

如果后续需要更宽范围，可以扩展到:

```text
alpha_grid = [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14]
```

但标签应明确这是涡格法预失速曲线，不代表真实失速后的气动。

所有角度拟合必须统一单位。建议内部拟合使用弧度 `rad`，保存元数据时同时记录原始角度单位。

### 6.3 参考量一致性

每个样本必须固定并保存:

- `S_ref`
- `b_ref`
- `c_ref`
- `x_cg`
- `moment_reference_point`
- `Mach`
- `Re`
- `rho`、`V` 或等效工况定义

俯仰力矩 `Cm` 对参考点极其敏感。若不同样本参考点不一致，代理模型会学习到混乱标签。

## 7. 寄生阻力与总阻力定义

### 7.1 CD0 来源

`CD0` 应来自 OpenVSP 寄生阻力工具或独立阻力 build-up 方法，而不是 VSPAERO 涡格结果。

`CD0` 至少应包含:

- 机翼、尾翼、鸭翼的摩擦阻力。
- 机身摩擦和形状阻力。
- 短舱或发动机外形阻力。
- 可选的干扰阻力修正。

### 7.2 总阻力组合

基础组合定义:

```text
CD_total(alpha) = CD0 + CDi_vspaero(alpha)
```

如果寄生阻力工具能给出随工况变化的 `CD0(Mach, Re)`，则:

```text
CD_total(alpha; Mach, Re) = CD0(Mach, Re) + CDi_vspaero(alpha)
```

如果后续引入剖面阻力随升力变化的修正，可以扩展:

```text
CD_total(alpha) = CD0 + CDi_vspaero(alpha) + CD_profile_lift_dependent(alpha)
```

但在第一版数据集中，建议先使用清晰的两项定义:

```text
CD_total = CD0 + CDi
```

并把缺失的升力相关剖面阻力作为模型边界记录，而不是隐式混入标签。

### 7.3 极曲线形式

推荐保存两类阻力标签:

1. 迎角参数形式:

```text
CD_total(alpha)
```

2. 极曲线形式:

```text
CD_total(CL)
```

常用二次极曲线:

```text
CD_total(CL) = CD0_fit + k1 * CL + k2 * CL^2
```

或以最小阻力升力系数为中心:

```text
CD_total(CL) = CDmin + k * (CL - CL_minD)^2
```

第一版建议保存:

- 原始离散点。
- 二次多项式系数。
- 拟合残差。

不要只保存多项式系数，否则后续无法检查 VSPAERO 或寄生阻力异常。

## 8. 曲线代理模型标签定义

代理模型可预测曲线系数，而不是每个 alpha 点的离散值。

### 8.1 升力曲线

在线性预失速范围内:

```text
CL(alpha) = CL0 + CL_alpha * alpha
```

如果需要覆盖较宽 alpha:

```text
CL(alpha) = a0 + a1 * alpha + a2 * alpha^2
```

建议第一版标签:

- `CL0`
- `CL_alpha`
- `CL_fit_rmse`

### 8.2 诱导阻力曲线

迎角形式:

```text
CDi(alpha) = d0 + d1 * alpha + d2 * alpha^2
```

极曲线形式:

```text
CDi(CL) = ki0 + ki1 * CL + ki2 * CL^2
```

如果 VSPAERO 输出中存在小的数值偏置，不应强制 `ki0 = 0`。第一版可以让二次项自由拟合，并记录残差。

### 8.3 总阻力极曲线

使用:

```text
CD_total(CL) = CD0 + CDi(CL)
```

若 `CDi(CL)` 使用二次拟合:

```text
CD_total(CL) = CD0 + ki0 + ki1 * CL + ki2 * CL^2
```

为了避免 `CD0` 与 `ki0` 混淆，建议标签中分开保存:

- `CD0_parasite`
- `CDi_poly_coeffs`
- `CD_total_poly_coeffs`

### 8.4 俯仰力矩曲线

基础线性形式:

```text
Cm(alpha) = Cm0 + Cm_alpha * alpha
```

较宽范围可用二次形式:

```text
Cm(alpha) = m0 + m1 * alpha + m2 * alpha^2
```

建议第一版标签:

- `Cm0`
- `Cm_alpha`
- `Cm_fit_rmse`

稳定性筛选可使用:

```text
Cm_alpha < 0
```

注意: 鸭翼和飞翼布局可能需要考虑控制面偏角或配平状态。若当前几何没有控制面建模，`Cm(alpha)` 应定义为未配平静态曲线。

## 9. 配平与可行性判据

### 9.1 第一版可行性判据

每个样本至少满足:

- OpenVSP 几何生成成功。
- VSPAERO 网格生成成功。
- VSPAERO alpha sweep 全部或大部分收敛。
- `CL(alpha)` 拟合残差低于阈值。
- `CDi(alpha)` 非负或无明显数值异常。
- `CD0 > 0`。
- `CD_total > CDi`。
- `Cm(alpha)` 有合理趋势。

如果训练目标包含稳定布局，则增加:

```text
Cm_alpha < 0
```

### 9.2 配平标签

如果后续加入控制面或尾翼安装角扫描，可以定义配平点:

```text
Cm(alpha_trim, delta_e) = 0
CL(alpha_trim, delta_e) = CL_target
```

配平标签可保存:

- `alpha_trim`
- `delta_e_trim`
- `CL_trim`
- `CD_trim`
- `L_over_D_trim`
- `trim_success`

第一版若尚未建控制面，建议不要伪造配平标签，只保存未配平曲线。

## 10. 数据表结构建议

### 10.1 样本主表

```text
sample_id
layout_family
status
failure_reason
vsp3_path
created_at
```

### 10.2 参数表

```text
sample_id
high_level_params_json
derived_params_json
reference_values_json
```

### 10.3 原始气动表

```text
sample_id
alpha_deg
alpha_rad
CL
CDi
Cm
run_status
```

### 10.4 寄生阻力表

```text
sample_id
Mach
Re
CD0_parasite
parasite_drag_breakdown_json
run_status
```

### 10.5 曲线系数表

```text
sample_id
CL_coeffs_json
CDi_alpha_coeffs_json
CDi_CL_coeffs_json
CD_total_CL_coeffs_json
Cm_coeffs_json
fit_metrics_json
validity_flags_json
```

## 11. 训练目标建议

第一版代理模型建议预测以下目标:

```text
CL0
CL_alpha
CD0_parasite
CDi_CL_poly_coeffs
CD_total_CL_poly_coeffs
Cm0
Cm_alpha
validity_score
```

如果模型直接预测曲线离散点，则也应同时保存多项式系数作为辅助监督或评估指标。

多任务损失建议分组:

- 升力曲线损失。
- 阻力极曲线损失。
- 俯仰力矩曲线损失。
- 有效性分类损失。

阻力相关量数值较小，训练时应做尺度归一化，避免被 `CL` 和 `Cm` 损失淹没。

## 12. 第一阶段实施顺序

推荐先实现最小闭环:

1. 常规布局参数化生成器。
2. OpenVSP 几何导出。
3. VSPAERO alpha sweep。
4. 寄生阻力 `CD0` 计算。
5. `CD_total = CD0 + CDi` 合成。
6. `CL(alpha)`、`CD(CL)`、`Cm(alpha)` 曲线拟合。
7. 样本过滤与数据落盘。

常规布局闭环稳定后，再加入:

- 鸭翼布局。
- 飞翼布局。
- 控制面/配平扫描。
- 主动采样。
- latent surrogate。

## 13. 关键原则

- 参数化生成器是独立系统，不复用 GRASS 树结构作为数据源。
- VSPAERO 主要提供升力、诱导阻力和力矩曲线。
- `CD0` 必须来自寄生阻力工具或独立 build-up。
- 总阻力标签使用显式公式组合，第一版定义为 `CD_total = CD0 + CDi`。
- 所有曲线必须保存原始离散点、拟合系数和拟合误差。
- 俯仰力矩曲线必须统一参考点，否则不同样本不可比较。
- 第一版不伪造失速和配平能力；涡格法标签只代表预失速、无强分离的气动趋势。
