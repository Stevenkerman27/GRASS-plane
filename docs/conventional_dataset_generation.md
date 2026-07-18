# 固定常规飞机数据集生成

本文档定义固定构型常规飞机数据集的生成流程。第一版仅覆盖三段机身、左右各一台翼吊短舱、主翼、常规 T 尾；发动机数量和尾翼布局不在本 schema 的随机范围内，后续以新 topology/schema 扩展。

## 环境与入口

OpenVSP 模型必须在 `vsppytools` 环境生成，原因是共享 OpenVSP 基础设施和该环境中的 OpenVSP 3.50.5 绑定都在其中。翼型的 CST 拟合及 `.pt` 写入必须在含 PyTorch 的 `myml` 环境运行。

```powershell
C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe data\generate_conventional_dataset.py
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\precompute_airfoil_codes.py --workers 4
C:\Users\zyx20\anaconda3\envs\myml\python.exe data\json_to_typed_obb_dataset.py
```

默认输出目录为 `data/conventional_dataset`，其中每架飞机各有一组同名文件：

```text
sample_0000.vsp3
sample_0000.json
...
sample_0199.vsp3
sample_0199.json
conventional_dataset.pt
```

独立 JSON 使失败样本可以使用固定样本序号重跑，而不影响其他样本。生成器的样本随机种子为 `root_seed + sample_index`；默认 `root_seed=20260711`。例如重建 50 至 99 号样本：

```powershell
D:\Software\anaconda\envs\vsppytools\python.exe data\generate_conventional_dataset.py --start-index 50 --count 50 --overwrite
```

`--overwrite` 只应在确定重新生成对应索引样本时使用；默认遇到已有 `.vsp3` 或 JSON 会直接失败，避免意外破坏一一对应关系。

## 几何与拓扑约定

- 单侧主翼根点固定在 `(x, 0, 0)`，单侧半展固定 `0.5 m`；每个 `.vsp3` 保持半模，不对主翼、短舱或平尾设置 `SYM_XZ`。
- OBB 树仍使用三个镜像 `SYM` 节点表达完整飞机：主翼、发动机短舱和平尾各从右侧叶节点镜像一次。因此 OpenVSP 文件用于半模检查，OBB 用于完整布局表达。
- 机身和短舱均使用椭圆截面、非零首尾站位和 `ROUND_END_CAP`。全部机身、短舱和翼面均使用 `tess_int=0.025 m`。
- 每架飞机从本项目 `foildata/processed_foil` 独立抽取主翼、垂尾、平尾的根截面和尖截面翼型，共六个来源，允许偶然抽到同一文件。JSON 按 `wing_airfoil_sources.{component}.{root|tip}` 记录绝对路径。左右镜像翼沿用右侧根/尖翼型。
- 垂尾的 OpenVSP 根/尖截面均由其来源文件的上表面关于弦线镜像构成；OBB CST code 对同一上表面的形状系数反序取负构造下表面，并共享 `N1`、`N2`，因此上下表面严格对称。临时 `.dat` 文件只用于写入 `.vsp3`，不会成为数据集工件。
- 树只保存右侧部件：3 个机身段、右主翼、右短舱、垂尾、右平尾。主翼、短舱和平尾各由一个 `SYM` 节点镜像，形成完整飞机。

随机范围集中定义在 `data/aircraft_dataset_common.py`：机身长度 `0.75-1.35 m`，机身宽度 `0.09-0.16 m`，机身高度 `0.10-0.18 m`，翼弦与尾翼/短舱尺寸在文件中定义的受限范围内采样。该模块也是 JSON schema、部件顺序、树操作和 OpenVSP 创建逻辑的唯一权威来源。

## JSON 与 PT 契约

每个 JSON 使用 `conventional_twin_engine_dataset_v2` schema，并记录米制单位、样本索引、随机种子、翼展、网格尺度、六个翼型路径、半机结构化部件、后序 `ops` 和 `syms`。JSON 不保存翼型 CST code。

先运行 `data/precompute_airfoil_codes.py`。该脚本以 CPU `ProcessPoolExecutor` 并行拟合本项目 `foildata/processed_foil` 中全部 `.dat` 文件，默认使用 4 个 worker、每 worker 1 个 PyTorch CPU 线程，并写入被 Git 忽略的 `data/cst_airfoil_code_cache.pt`。缓存以翼型内容 SHA-256、`cst_airfoil_codec.py` 中的拟合配置和算法版本作为身份；任一输入变化都会形成新缓存条目。内容完全相同的不同文件会共用一个 cache entry，但每个文件路径仍可独立命中该 entry。

转换器读取 200 个 `sample_*.json` 后仅从该缓存取 code，再写入一个结构化 `.pt`；缓存缺少任一翼型会立即报错，不会静默回退到串行或 CUDA 拟合。后续生成数据集时，已缓存的翼型无需再次优化。其中：

- 机身和发动机：10D 几何。
- 翼面：8D 几何加根/尖各 24D、共 48D CST 翼型 code。
- `ops` 与 `syms` 保持 JSON 中的固定树拓扑，可由 `StructuredGRASSDataset` 直接读取。

`.pt` 只包含现有 `StructuredGRASSDataset` 需要的 `list[dict]`，不额外嵌入生成配置或归一化统计量。

从旧的 Rational-Bezier 数据集迁移时，必须先使用当前 `foildata/processed_foil` 与相同随机种子重新生成 JSON/VSP，再转换 PT；不得仅修改 JSON 内的绝对路径，因为已删除或替换的翼型将使几何与 code 不再一致。

`data/visualize_dataset.py` 直接读取该 `.pt`，随机选择一架飞机并从其 `SYM` 节点还原完整 OBB，再显示交互式 3D 窗口。它不重新拟合翼型。可选 `--index` 指定样本，或 `--seed` 复现随机选择：

```powershell
D:\Software\anaconda\envs\myml\python.exe data\visualize_dataset.py
D:\Software\anaconda\envs\myml\python.exe data\visualize_dataset.py --index 17
```

默认情况下，脚本会先使用 `D:\3D\Projects\OpenVSP-3.51.0-win64\vsp.exe` 打开同索引的 `sample_XXXX.vsp3`，再显示 OBB 图窗。可用 `--vsp-exe` 或 `--vsp-dir` 覆盖路径；无 GUI 校验或只需 OBB 时传入 `--no-open-vsp`。

## 归一化

第一版 `.pt` 保留原始米制几何值，不进行 OBB 特征归一化。虽然主翼展已经固定为 1 米，其余长度、翼弦和安装位置仍保留真实尺度变化。后续训练阶段应只用训练集计算逐特征归一化统计量，并将统计量作为独立训练工件保存；不得反写或替换原始几何数据集。

## 校验

预拟合与转换脚本均采用 fail-fast 校验：缺少翼型、缓存身份/编码长度错误、JSON schema/部件维度错误、样本索引重复或数量不是预期的 200 时都会停止。转换器保存后会以 `weights_only=True` 重载 `.pt` 并对每个样本执行 `grassdata.validate_structured_sample(...)`。
