# Member 4 — Training / Evaluation / Ablation Code

> 统一训练与评估脚本。成员4负责，三组实验（Original / +AdaIN / +Ours）共用同一套代码，
> 唯一差别是读入的 manifest 不同。

## 0. 环境

```
torch 2.11.0+cu128   (RTX 4070 Laptop 8GB)
torchvision 0.26.0
numpy / scipy / opencv-python / pillow / tqdm
pip install lpips     # 可选，3.6 的 LPIPS 指标需要；缺失时脚本会自动降级并提示
```

数据根目录由环境变量控制，所有成员一致：

```powershell
$env:NNDL_DATA_ROOT = '<path>\to\data'
```

不设置时默认用 `<repo>/data`。所有路径统一由仓库根目录的 `paths.py` 解析（`config.py` 只是转发），细节见 [`../ENVIRONMENT.md`](../ENVIRONMENT.md)。对应的 `config.py`：

```python
DATA_ROOT   = paths.DATA_ROOT        # <repo>/data 或 $NNDL_DATA_ROOT
IMAGE_ROOT  = DATA_ROOT / "PACS"     # 原始图片
AUG_ROOT    = DATA_ROOT / "augmented" # 风格化图片（成员2/3 产出）
OUTPUT_ROOT = paths.RUNS_ROOT        # <repo>/../runs 或 $NNDL_RUNS_ROOT
```

运行结果（checkpoint、summary）默认写到仓库根目录的 `runs/`，该目录已在 `.gitignore` 中排除，不会被提交。已产出的结果 JSON 汇总在仓库根目录的 `results/`。

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `config.py` | **唯一真源**：三组共用的超参、split 路径、transform。`config_hash()` 用于公平性校验 |
| `model.py` | ResNet-18 构造 + 随机种子设置 |
| `data.py` | Dataset：原始图片与风格化图片走同一套代码路径 |
| `train.py` | 统一训练脚本（step-budget 制） |
| `evaluate.py` | 在六个 split 上评估，输出 JSON |
| `run_all.py` | 一个配置跑多 seed + 汇总 |
| `summarize_results.py` | 生成统一结果表（plain / markdown / latex 三种格式） |
| `fairness_check.py` | **公平性机器校验**：各组 manifest 是否满足 3.7 的全部约束 |
| `stats_compare.py` | **配对统计检验**：同 seed 设计下的配对 t 检验、95% CI、Wilcoxon 交叉验证 |
| `final_analysis.py` | **最终汇总**：主表、配对检验、强度-准确率关系、按类别细分、机制分析 |
| `structure_mechanism.py` | **机制检验**：逐图 EdgeScore 配对比较 + 结构密度相关性 |
| `build_style_pool.py` | 冻结风格图池 + 冻结 原图→风格 配对（**含防泄露检查**） |
| `structure_metrics.py` | 3.6 的结构指标：LPIPS + EdgeScore |
| `verify_splits.py` | 数据校验：六个 split 对本地 PACS 的一致性检查 |
| `make_synthetic_aug.py` | 造 40 张假增强集，用于打通流程（**不可用于真实结果**） |
| `convert_aug_manifest.py` | 把成员2/3 的生成结果转成训练用的 manifest，并核对配对表 |
| `_sync_sections.py` | 由 `sections/member4_*.tex` 生成 `sections/member4_sections.tex` |
| `project_splits/` | 成员1 的六份清单（只读，不修改） |
| `style_pool/images/` | **10 张外部风格图**（不属于 PACS） |
| `style_pool/style_pool.json` | 冻结的风格图池 + 配对表 |
| `style_pool/SOURCES.txt` | 风格图来源与排除规则 |
| `GENERATOR_INTERFACE.md` | 给成员2、成员3 的一次性对接说明 |

## 2. 关键协议（已写入报告 3.3 / 3.7）

- **step-budget 制**：三组都是 660 个 optimizer step，与数据集大小无关。
  两个增强组图片数翻倍 → 对数据的遍历次数相应变少，保证优化预算一致。
- **不使用 checkpoint 选择**：报告固定预算下的**最后一个** checkpoint。
  原因见报告 3.3：Photo Validation 只有 150 张、60 步内就饱和到 97–98%，
  按 val 选点等于选噪声，把跨域准确率的 seed 间方差从 2.71 抬到 7.22。
- **不使用光度增强**：训练 transform 只用几何增强（RandomResizedCrop + flip）。
  color jitter / grayscale 会改变风格统计量，而那正是被研究的变量。
- **风格图池在 PACS 之外**：Art/Cartoon/Sketch 是**测试域**，风格图若取自其中
  就等于把测试外观送进训练。冻结的 10 张风格图全部是外部画作，
  `build_style_pool.py` 与 `fairness_check.py` 都会用内容哈希与本地 PACS 逐一比对，
  发现任一风格图就是 PACS 图就直接报错。
- **同 seed 集**：{0, 1, 2}。

## 2.5 轻量试跑（smoke test）：先证明管线通，再跑正式实验

正式实验（三组 × 3 seeds × 660 步）之前，先用一份**缩小的**协议把所有接缝跑通一遍。
这一节的命令不会产生任何可写进报告的数字，它只回答"管线能不能跑通"。

### 为什么必须先做

这个项目里有多条接缝**从未在真实数据上跑通过**：`fairness_check.py` 只在 40 张假图上跑过，
`data.py` 的 `group="adain"/"ours"` 分支从未被 `train.py` 调用过，
成员2 的生成器从未在 1,349 张规模上跑过。这些问题应该在 20 分钟内暴露，而不是在 3 小时后。

### 缩小的协议（`NNDL_SUBSET`）

```powershell
$env:NNDL_SUBSET = '150'    # 每个实验组用 150 张原图
$env:NNDL_RUNS_ROOT = '<path>\to\runs_smoke'
```

`NNDL_SUBSET` 做三件事，且三组完全一致：

- 训练集限制为**类别均衡**的前 N 张原图（`photo_train.txt` 按类别排序，
  直接取前 N 张会全是 dog，模型只学会一个类，验证准确率约 11% ≈ 随机，什么也说明不了）；
- 步数按 `N / 1349` 等比缩小，**每组跑同样的步数**，所以通过数据的次数相同；
- 选图逻辑集中在 `config.balanced_train_subset()`，成员2/3 用同一份定义生成对应的增强图。

子集是**缩小版**而不是**抽样版**：每个 epoch 看到的类别比例与正式实验一致，
步数按比例缩放，得到的仍然是"同优化量"的比较。

### 完整流程（约 25 分钟）

```powershell
# 0) 环境（nn.ps1 会自动挑一个能跑 CUDA 的解释器）
cd <repo>
$env:NNDL_DATA_ROOT = '<path>\to\data'

# 1) 生成 150 张 Ours + 匹配的 AdaIN 对照（--balanced 保证类别均衡）
python -m member3_structure_preserving.generate build --limit 150 --balanced `
    --alpha 0.5 --beta 0.7 --with-adain

# 2) 公平性校验（真实 manifest，不加 --subset）
python member4_training_eval/fairness_check.py `
    --adain  $env:NNDL_DATA_ROOT\aug_sp\previews\build_n150\a0p5_b0p7\manifest_adain.txt `
    --ours   $env:NNDL_DATA_ROOT\aug_sp\previews\build_n150\a0p5_b0p7\manifest_ours.txt

# 3) 三组各跑 1 个 seed
$env:NNDL_SUBSET = '150'
python member4_training_eval/run_all.py --group original --seeds 0
python member4_training_eval/run_all.py --group adain --seeds 0 --aug-manifest <上面的 manifest_adain.txt>
python member4_training_eval/run_all.py --group ours  --seeds 0 --aug-manifest <上面的 manifest_ours.txt>

# 4) 出表 + 结构指标
python member4_training_eval/summarize_results.py --format latex
python member4_training_eval/structure_metrics.py --manifest <上面的 manifest_ours.txt>
```

### 逐条对着看的检查项

| 检查项 | 期望 |
|---|---|
| `fairness_check.py` | `PASS`，0 problems；尤其是"content→style 配对一致"和"冻结配对一致" |
| manifest 路径 | 全部是**相对 `DATA_ROOT`** 的路径，`check_images_exist` 会逐张确认文件存在 |
| 三组训练图片数 | original = 150，adain = ours = 300；**步数三组相同** |
| `eval_last.json` | 三个组都产出，含 `cross_domain_avg` |
| `summarize_results.py --format latex` | 能直接贴进 `sections/member4_37.tex` 的 `tab:results` |
| `structure_metrics.py` | `lpips_available: true`，不崩 |

### 一个必须知道的坑

`run_all.py` 以前只看 `eval_last.json` 存不存在就跳过训练。
改了 `NNDL_SUBSET` 之后，它会**沿用上一次的结果**，三组数字就来自不同的协议，表格静默失真。
现在 `run_all.py` 会比对 `config_hash`，配置变了就重训；确认要重训用 `--force`。

### 另一个必须知道的坑：风格强度混杂

$\alpha(x,y) = \alpha_0(1-\beta M)$ 在 $M$ 大的地方必然小于 $\alpha_0$，所以**空间图的整体风格强度
必然低于同 $\alpha_0$ 的均匀 AdaIN**。实测：

| 设置 | 平均风格强度 |
|---|---|
| `+AdaIN` (uniform $\alpha_0 = 0.5$) | 0.5000 |
| `+Ours` ($\alpha_0=0.5, \beta=0.7$) | **0.3736** |

**如果只比 `+AdaIN` 和 `+Ours`，两组同时差了"空间调制"和"风格更弱"两件事**，
赢或输都解释不清。因此本项目引入**强度匹配对照**：

| 组 | 空间调制 | 平均强度 |
|---|---|---|
| `+AdaIN` | 无 | 0.5000 |
| `+AdaIN (strength-matched)` | 无 | **= `+Ours` 的逐图均值** |
| `+Ours` | 有 | 0.3736 |

生成方式（成员3 的 CLI，`--with-matched`）：
每张图用**自己的** $\alpha = \text{mean}(\alpha_{spatial})$ 做一次均匀 AdaIN，
逐图对齐强度，而不是用一个全局常数近似。

还有一个**完全等强度**的版本：把 $\alpha_0$ 调到使得空间图均值正好等于 0.5，
即 $\alpha_0 = 0.5 / (1 - \beta \overline{M}) = 0.669215$（$\beta=0.7$）。
此时 `+Ours` 与 `+AdaIN` 的**唯一差别就是空间调制**，是最干净的对照。
校准脚本：`member3_structure_preserving/calibrate_equal_strength.py`。

## 3. 常用命令

```bash
# 0) 校验数据（已通过：9990 张，数量/类别/sha256 全对）
python verify_splits.py

# 1) 冻结风格池（只需跑一次；换风格图要在生成前重跑并通知成员2/3）
python build_style_pool.py --n 10 --alpha 0.5

# 2) 跑一组实验（3 seeds）
python run_all.py --group original
python run_all.py --group adain --aug-manifest <manifest>
python run_all.py --group ours  --aug-manifest <manifest>

# 3) 公平性校验（各组 manifest 齐了之后）
#    --runs 可选，用来核对各组 run 的训练配置指纹是否一致
python fairness_check.py --adain <adain manifest> --ours <ours manifest> \
    --runs ../runs/original_s0 ../runs/original_s1 ../runs/original_s2

# 3b) 三路校验：加上强度匹配对照（排除"风格变弱"这个混杂）
python fairness_check.py --adain <adain manifest> \
    --adain-matched <strength-matched manifest> --ours <ours manifest>

# 4) 结构指标
python structure_metrics.py --manifest <manifest>

# 5) 出表
python summarize_results.py --format latex

# 5b) 统计检验与最终汇总
python stats_compare.py --per-class --out ../runs/stats_report.json
python final_analysis.py

# 5c) 机制检验（逐图 EdgeScore 配对 + 结构密度相关性）
python structure_mechanism.py

# 6) 重新合并 3.1/3.3/3.6/3.7/5.2（改的是 sections/member4_*.tex）
python _sync_sections.py

# 7) 把成员2/3 的生成结果转成训练用的 manifest
#    （格式不对时用这个脚本转，它会核对配对表和风格池）
python convert_aug_manifest.py --group adain \
    --pairs <生成结果目录>/pairs.txt --alpha 0.5
python convert_aug_manifest.py --group ours \
    --pairs <生成结果目录>/pairs.txt --alpha 0.5 --beta 0.7
```

## 4. manifest 格式（成员2/3 交付物）

```
<增强图片相对 DATA_ROOT 的路径> <标签1..7> <原图相对路径> <风格名> <alpha> [beta]
augmented/adain/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5
augmented/ours/056_0022__scene_de_rue.jpg 1 photo/dog/056_0022.jpg scene_de_rue 0.5 0.7
```

风格名必须是 `style_pool/images/` 里的 10 个外部画作之一（不带扩展名）。
原图相对路径必须是 `photo_train.txt` 里出现的路径（不带 `PACS/` 前缀）。

## 5. 当前结果（2026-11-06）

Original 组，3 seeds，660 steps，最终 checkpoint：

| | Photo | Art | Cartoon | Sketch | Avg |
|---|---|---|---|---|---|
| Original | 99.0±0.3 | 63.2±3.7 | 25.1±1.8 | 31.5±4.2 | **39.9±2.7** |

实验 1（domain gap）已完成。实验 2/3 等两组 manifest 到位后启动。

## 6. 目录说明

- `runs/<tag>/` — 每个 run 的 checkpoint、训练历史、评估 JSON
- `runs/summary_<group>.json` — 多 seed 汇总
- `runs/_archive_bestval/` — 旧的 best-val 选点结果（保留作对比，说明为什么弃用）
