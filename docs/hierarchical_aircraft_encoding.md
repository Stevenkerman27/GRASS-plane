# 分层飞行器序列编码定义

## 范围

当前 flying-wing schema 中，机身与右主翼都是单个变长序列叶节点。发动机不属于本次迁移，仍保留其旧 10D OBB 接口；飞翼数据集不生成发动机。

树的后序叶顺序固定为：

```text
fuselage, main_wing_right, SYM(main_wing_right), ADJ
```

`SYM` 只镜像右主翼，不复制机身。

## 序列 Payload

所有序列组件的有效截面数 `section_count` 在 `[2, 8]`，是 padding 和停止的唯一权威定义。`.pt` 中的截面张量补零到 `max_sections=8`；JSON 只保存有效截面，不能保存独立 mask 或 EOS。

```text
fuselage.sections[8, 5] = [x, y, z, width, height]
wing.sections[8, 29]    = [CST24, leading_edge_x, leading_edge_y,
                           leading_edge_z, chord, twist]
```

机身截面按机头到机尾排序，`x` 必须严格递增，`width` 和 `height` 必须为正。机翼截面按 `+Y` root-to-tip 排序，`chord` 必须为正。原始数据保存物理单位；训练归一化统计量若需要，只能作为训练工件单独保存。

## 网络

每类序列组件都使用一层经典 `torch.nn.RNN` encoder：

```text
valid_sections[0:N, D] -> RNN -> final_hidden[feature_size]
```

机身和机翼的 decoder 都是自回归 RNN：首步输入可学习 BOS token，后续步在训练中使用前一真实截面（teacher forcing）。decoder 从部件 feature 初始化 hidden state，最多产生 8 个截面，并额外预测 `section_count` logits。生成时由预测 count 选择有效前缀，不使用 EOS。

机身 decoder 将首站 `x` 固定为 0，并将后续 `x` 解码为正增量的累积值；`width`、`height` 用 `softplus` 保证为正。机翼 decoder 保持 CST `N1/N2` 与 chord 的正值约束。

## 损失与边界

序列字段损失只在由真实 `section_count` 导出的有效 mask 上计算，且 count 使用分类损失。机翼继续拆分 position、chord、twist、CST code 和 decoded-curve 损失；机身使用 position、size 和 count 损失。组件类别由独立 classifier 预测。

`util.py` 是截面数范围、padding mask、字段维度和损失权重的唯一配置来源。旧 13D box 数据和旧“三段 10D 机身 OBB”数据不得通过补零或截断混入该 schema。
