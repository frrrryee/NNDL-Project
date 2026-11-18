# 项目任务分配与论文结构（第二阶段：实验 / 消融 / 分析 / 终稿）

> 适用版本：五组主实验已完成（Original / +AdaIN / +AdaIN 强度匹配 / +Ours / +Ours 等强度）
> 负责协调：成员 4
> 配套文档：`repo/ENVIRONMENT.md`（各成员在自己电脑上的环境与路径配置）、`repo/paths.py`（路径解析入口）

---

## 0. 当前成果摘要

所有数字均为 3 个种子（seed 0/1/2）、固定 660 步预算、报告最后一个 checkpoint 的结果。
交叉域准确率 = Art Painting / Cartoon / Sketch 三个域的不加权平均。

| 训练方法 | Photo | Art | Cartoon | Sketch | 交叉域均值 | 平均风格强度 |
|---|---|---|---|---|---|---|
| Original | 99.0±0.3 | 63.2±3.7 | 25.1±1.8 | 31.5±4.2 | 39.92±2.7 | 0 |
| +AdaIN (uniform 0.5) | 99.2±0.3 | 69.0±0.7 | 32.2±1.7 | 40.2±1.3 | **47.12±0.8** | 0.500 |
| +AdaIN (强度匹配 0.374) | 99.4±0.0 | 68.0±1.3 | 29.7±1.2 | 38.4±2.5 | 45.36±0.3 | 0.374 |
| +Ours (空间调制, mean 0.374) | 99.8±0.3 | 68.5±0.6 | 30.0±0.9 | 36.7±3.9 | 45.09±1.5 | 0.374 |
| +Ours (等强度, mean 0.500) | 99.6±0.3 | 68.1±0.8 | 31.1±0.3 | 41.8±3.3 | **47.00±0.9** | 0.500 |

结构指标（全部 1,349 张，配对逐图比较）：

| 数据集 | EdgeScore | LPIPS |
|---|---|---|
| +Ours | 0.3436 ± 0.0628 | 0.2258 |
| +AdaIN (uniform 0.5) | 0.3211 ± 0.0653 | 0.2787 |
| +AdaIN (强度匹配) | 0.3467 ± 0.0602 | 0.2191 |

- +Ours − +AdaIN(uniform) = **+0.0225**，p = 9.5e-294
- +Ours − +AdaIN(强度匹配) = **−0.0031**，p = 9.3e-23
- 增益与图像结构密度的相关性：Spearman ρ = +0.078（p = 0.004），密集半区与稀疏半区增益均为 −0.003

### 目前可以下的结论

1. **风格增强带来了 +7.20 点的提升**，且种子间标准差从 2.7 降到 0.8。但需注意：n=3 时该差异的配对 t 检验 p=0.070，尚未达显著；且训练集同时翻倍，"风格"与"数据量"两个因素尚未分离。本阶段将补足种子数与数据量对照，再给出确定结论。
2. **空间调制没有额外贡献**：控制住平均风格强度后，+Ours 与 +AdaIN 的交叉域准确率差异为 −0.12 点（p = 0.89），EdgeScore 差异为 −0.003。这一结论已有强度匹配对照支撑，不依赖上述补强实验。
3. **准确率几乎完全由"施加了多少风格"决定**，与"风格如何空间分布"无关。

### 现有结论尚存的四个缺口

1. **数据量未被单独控制**（最主要）。+AdaIN 组训练集是 Original 组的两倍，"风格"与"样本数量"两个因素混在一起。第 1.5 节给出补法。
2. **主结论的统计功效不足**。+7.20 点在 n=3 时 p=0.070，未达显著。任务 6.6 把主结果扩到 5 个种子。
3. **只测了 ResNet-18**，结论的普适性还可以加强（任务 6.7 之外的加分项）。
4. **没有和其他域泛化方法直接比过**（MixStyle 只被引用，未实现）。

### 本阶段要回答的问题

- 风格增强的 7.2 点，收益到底来自哪一部分？（数据量 / 通用外观多样性 / 风格 / 空间调制）
- 风格强度与跨域准确率之间是什么关系？这是论文的主图。
- 空间调制为什么没有额外贡献？机制上解释清楚。
- 风格增强是均匀地帮助所有类别，还是只帮助某几类？
- 风格池本身是不是一个敏感因素？

---

## 1. 终稿论文结构

终稿沿用 IEEE conference 模板，8 页以内（含图、表、参考文献）。
现进度报告共 7 页，其中第 I 节约 1.3 页、第 II 节约 0.7 页、第 III 节约 4 页、第 IV、V 节各约 1 页。
终稿页数预算：Abstract 0.25 / I 1.0 / II 0.9 / III 2.0 / IV 2.8 / V 0.5 / References 0.8。
当前第 III 节偏长，第 IV、V 节性质属于项目管理文档，这两处是本次改造的重点。

**III 节偏紧的处理预案**：III 有 7 个小节却只有 2.0 页，平均每节约 0.29 页，而 III.B（划分细节 + Table I）、III.F（四个指标定义）、III.G（公平性机制 + step-budget + 配置指纹）三块内容天然占空间。若写完溢出，按以下优先级压缩：

1. **III.B**——划分细节（逐类数量、无交集验证）可压缩成一段，完整表格保留
2. **III.F**——EdgeScore 的公式推导可精简，只留定义与取值范围
3. **III.A**——整体框架可与 Fig. 1 框图配合，用图承担大部分叙述

III.G 的公平性机制不建议压缩，那是本论文方法论上最有价值的部分之一。

### 1.1 章节骨架

```text
Abstract

I.   Introduction
     A. Problem Statement and Research Questions
     B. Motivation and Background
     C. PACS Dataset and Task Overview
     D. Contributions and Outline

II.  Related Work
     A. Cross-Domain Image Classification and PACS
     B. Style Transfer and AdaIN
     C. Style Augmentation for Image Classification
     D. Structure Preservation and Structure-Related Metrics

III. Methodology
     A. Overall Framework
     B. Dataset and Task Definition
     C. ResNet-18 Baseline
     D. AdaIN Style Augmentation
     E. Structure-Preserving Style Augmentation
     F. Evaluation Metrics
     G. Experimental Design and Fair Comparison

IV.  Experiments and Results
     A. Experimental Setup
     B. Main Results
     C. Style Strength versus Cross-Domain Accuracy
     D. Statistical Significance
     E. Structural Mechanism Analysis
     F. Failure Case Analysis
     G. Ablation Studies
     H. Discussion and Limitations

V.   Conclusion and Future Work

References
```

### 1.2 与进度报告的对应关系

| 进度报告章节 | 终稿章节 | 处理方式 |
|---|---|---|
| Abstract | Abstract | 重写。内容已就位，不再是占位符 |
| 1.1 Problem Statement | I.A | 改写：研究问题从"结构保护能否提升性能"改为"风格增强的收益来自什么" |
| 1.2 Motivation and Background | I.B | 基本照搬，补一句对贡献的引导 |
| 1.3 PACS Dataset and Task Overview | I.C | 照搬 |
| 无 | I.D Contributions | **新增**：四条贡献，每条对应一个对照实验 |
| 2.1–2.4 Related Works | II.A–II.D | 扩写，补 Related Work 对比表 |
| 3.1–3.6 Methodology | III.A–III.F | 照搬，按需要压缩 |
| 3.7 Experimental Design | **拆为 III.G + IV.A/IV.B** | 设计部分留在方法节，结果部分移入第 IV 节 |
| 4.1 Data Preparation and Baseline | IV.A | 改为实验设置描述 |
| 4.2 AdaIN Augmentation Progress | IV.B | 改为 AdaIN 组结果 |
| 4.3 Structure-Preserving Method Progress | IV.B / IV.E | 改为 Ours 组结果与机制分析 |
| 4.4 Training, Evaluation and Ablation Preparation | IV.A | 改为实验设置（step-budget 制、checkpoint 策略、配置指纹） |
| 4.5 Visualization, Demo and Report Integration | IV.F | 改为失败案例分析 |
| 4.6 Current Risks and Open Issues | IV.H | **性质转换**：待办风险改写为研究局限 |
| 5.1 Milestone and Weekly Plan | — | **删除** |
| 5.2 Experiment Completion Plan | IV.A | 改为实验设置的实现细节 |
| 5.3 Ablation and Analysis Plan | IV.G | 改为消融的实际结果 |
| 5.4 Demo, Report and Presentation Plan | V | 压缩为一段 |
| 5.5 Team Division and Coordination | — | **删除**（学术论文不写分工） |
| References | References | 补齐新增引用 |

### 1.3 图表编号（终稿统一编号）

| 编号 | 内容 | 负责人 |
|---|---|---|
| Fig. 1 | 整体框架图 | 成员 5（已有 `framework.png`，需重绘为矢量并对齐新流程） |
| Fig. 2 | **风格强度 → 交叉域准确率曲线（主图）** | 成员 4 |
| Fig. 3 | 配对差值森林图，带 95% CI | 成员 4 |
| Fig. 4 | EdgeScore 逐图配对累积分布图 | 成员 4 |
| Fig. 5 | 失败案例图集 | 成员 3 提供素材，成员 5 排版 |
| Fig. 6 | β 消融三联图（EdgeScore / 平均 α / 准确率） | 成员 3 |
| Fig. 7 | PACS 四域七类样例图 + 各类别数量统计图 | 成员 1 |
| Fig. 8 | 风格池敏感性对比图 | 成员 2 |
| Fig. 9 | **逐类准确率热力图（Original / AdaIN / Ours × 7 类）** | 成员 4（数据）/ 成员 5（排版） |

| 编号 | 内容 | 负责人 |
|---|---|---|
| Table I | 数据划分与实验角色 | 成员 1 |
| Table II | **主结果：全部实验组 × 四域准确率** | 成员 4 |
| Table III | 显著性总览表（差值 / CI / t / p / 校正 p / 效应量） | 成员 4 |
| Table IV | Related Work 对比表 | 成员 4 起草，成员 5 补全 |
| Table V | 结构指标汇总（EdgeScore / LPIPS / 平均 α） | 成员 3 |
| Table VI | **收益来源分解表**（数据量 / 通用多样性 / 风格 / 空间调制） | 成员 4 |

### 1.4 标题

现标题 `Structure-Preserving Style Augmentation for Single-Source Cross-Domain Image Classification on PACS` 承诺的是"结构保持方法有效"，与实际结论不符，需改写。
建议方向：以前半句提出问题、后半句点明方法，例如

> *What Does Style Augmentation Actually Buy? Style Strength versus Spatial Structure Preservation for Single-Source Cross-Domain Classification on PACS*

定稿由成员 4 汇总各成员意见后确定，成员 5 同步更新首页与摘要。

### 1.5 必须补上的对照组（当前结论的唯一软肋）

现有五组实验无法排除一个替代解释：**+AdaIN 相对 Original 的 7.2 点，究竟来自"风格"，还是来自"训练集里多了一倍不重复的图片"？**

Original 组训练集 1,349 行，+AdaIN 与 +Ours 组 2,698 行（1,349 原图 + 1,349 增强图），但步数预算相同（660 步 × 64 batch = 42,240 次采样）。数据集规模翻倍本身就会改变模型看到的信息量。

需要补三组对照，才能把收益来源锁死：

| 组 | 训练集行数 | 不重复图片数 | 外观多样性 | 作用 |
|---|---|---|---|---|
| Original | 1,349 | 1,349 | 无 | 已有 |
| **+Duplicate（新增）** | 2,698 | 1,349 | 无 | 数据量对照：把每张原图复制一份（不同文件名、字节相同）。预期 ≈ Original；若两者差很多，说明协议本身有问题，这是一个自检 |
| **+NonStyle（新增）** | 2,698 | 1,349 | 有（非风格） | 通用外观多样性对照：用色彩抖动 / 灰度 / 高斯模糊等非风格手段生成第二份，隔离"外观多样性"与"风格" |
| +AdaIN | 2,698 | 1,349 | 有（风格） | 已有 |

补上这三行之后，`+AdaIN − +NonStyle` 才是"风格本身的贡献"，`+Duplicate − Original` 是"纯数据量贡献"，论文的主结论才闭合。

成本：`+Duplicate` 只需复制字节，不生成新像素；`+NonStyle` 必须真正生成 1,349 张变换图（色彩抖动/灰度/模糊），约 120 MB。两组各训练 5 个种子，单次训练约 190 秒。

### 1.6 其他建议补强的实验

按性价比排序：

| 优先级 | 实验 | 成本 | 价值 |
|---|---|---|---|
| 高 | **数据量对照**（+Duplicate、+NonStyle） | 生成 + 训练 | 见第 1.5 节。这是主结论能否闭合的关键 |
| 高 | **种子 3 → 5** | 每次训练约 190 秒（第 2 周批次 E） | n=3 时主结论 p=0.070，扩到 5 个种子是让它站住的前提 |
| 高 | **MixStyle 基线**（特征统计混合） | 改 `model.py` + 训练 | 论文的 bib 里已有该文献引用，但从未实现。真实的域泛化方法，与我们的"空间风格调制"同类不同法，放在一起才构成"和别人比过" |
| 中 | **逐类准确率热力图** | 零训练成本 | `summary_*.json` 里已记录 `per_class`，可直接画。风格增强是**均匀**地帮助所有类别，还是只帮助某几类？这个差别对结论的解释力很强 |
| 中 | **跨架构验证**（ResNet-50） | 需让 `model.py` 支持架构参数 | 只测 ResNet-18，结论可能被质疑是架构特有。产出较新，只在时间充裕时做 |
| 低 | **收敛轨迹**（准确率 vs 步数） | 零额外成本 | `train.py` 已有 `--save-every` 与 `--eval-targets`。展示 660 步预算是否合理 |
| 低 | **风格池敏感性**（已列在任务 2.3） | 生成 + 训练 | 风格池规模是否影响主结论 |

注意：MixStyle 与 ResNet-50 会改变 `config_hash`，属于独立实验组，必须单独汇报，不得与 `ecc6090e5b29` 的结果混在同一张表里直接比较。

---

## 2. 共享纪律（所有成员必须遵守）

违反其中任何一条，对应产出不得进入报告。

1. **参数网格预先固定**。α ∈ {0.25, 0.5, 0.75, 1.0}，β ∈ {0, 0.3, 0.5, 0.7, 1.0}。不得增加网格外的值，也不得根据目标域测试结果调整或选优。
2. **目标域只用于最终评价**。Art Painting / Cartoon / Sketch 不参与参数选择、checkpoint 选择、增强集筛选。参数选择只能依据 Photo Validation 或预先声明的规则。
3. **不修改已有结果**。任何重跑必须换新的 `tag`，并在 `runs/` 下新建目录。禁止覆盖 `original_s*`、`adain_s*`、`adain_matched_s*`、`ours_s*`、`ours_equal_s*`。
4. **路径一律走 `repo/paths.py`**。禁止在任何脚本、默认参数、配置或 manifest 里写绝对路径。
   三个环境变量，全部可选，不设时用仓库内默认位置：

   | 变量 | 含义 | 默认值 |
   |---|---|---|
   | `NNDL_DATA_ROOT` | 包含 `PACS/` 的目录 | `<repo>/data` |
   | `NNDL_RUNS_ROOT` | 运行产物目录 | `<repo>/../runs` |
   | `NNDL_PYTHON` | Python 解释器路径 | 由 `nn.ps1` 自动探测 |

   组员在自己的电脑上只需设前两个（或完全不用设），具体步骤见 `repo/ENVIRONMENT.md`。
   写入 manifest 的路径必须**相对于 data root**，这样 manifest 才能跨机器复用。
5. **Python 环境**。用 `.\nn.ps1` 调用，该脚本会自动挑选能真正跑通 CUDA 卷积的解释器。要求 torch ≥ 2.0、torchvision ≥ 0.13、numpy、scipy、Pillow、matplotlib、opencv-python、lpips。
   已有结果产出于 torch 2.11.0+cu128 / torchvision 0.26.0。
6. **所有数字必须可追溯**。终稿里的每个数字都要能从 `runs/` 下的 JSON 重新算出来。禁止手输数字、禁止在图里写死数值。
7. **生成任务串行**。增强集生成统一由成员 2、成员 3 在指定时间窗口内执行，其他成员不要重复生成同一个配置。
8. **训练配置指纹**。所有 run 的 `config_hash` 必须是 `ecc6090e5b29`。新增或修改超参前，先和成员 4 确认，否则该 run 无法与已有结果比较。
   两类改动属于例外，会产生**新的 `config_hash`**，必须作为独立实验组单独汇报，不得与 `ecc6090e5b29` 的结果混在同一张表里直接比较：
   - **模型结构变更**：MixStyle（任务 6.7）、ResNet-50 等
   - **训练超参变更**：学习率、步数预算、增强方式等
   种子数量**不属于**例外（`config_hash` 不含 seeds），扩到 5 个种子后仍是 `ecc6090e5b29`。
9. **进度报告与终稿并行维护**。进度报告的第 4 节继续按现有风格更新；终稿的改动不直接写进 `progress_report.tex`，一律先落到各自 `sections/` 片段。
10. **磁盘管理**。当前 C 盘剩余约 26 GB。单次训练约 190 秒，一个 checkpoint 约 43 MB。
    旧的 `best.pt` 因 checkpoint 策略已改为 `last` 而不再使用，可在确认后删除以回收空间（约 0.67 GB）。
    训练完的 `last.pt` 若已产出 summary 且不再需要评估，可在整合阶段清理。

---

## 3. 成员 1：数据接口与实验配置

**代码位置**：`repo/member1_data/`
**负责章节**：III.B（配合）、IV.A（数据部分）、Table I、Fig. 7
### 任务 1.1 — 扩展数据接口，支持实验变体

给 `pacs_dataset.py` 增加 `variant` 参数，使同一张 Photo 原图的不同增强版本（不同 α、不同 β、强度匹配版本）能被同一个 Dataset 类读取。

- 新增 `variant` 参数字符串或元组，形式建议：`variant=("uniform", 0.5)`、`variant=("spatial", 0.5, 0.7)`、`variant=("matched",)`
- 根据 variant 解析出实际图片路径的规则要写在一个独立函数里，不散落在 `__getitem__` 内
- 保留现有 `PACSDataset(data_root, split_file, transform)` 的调用方式兼容（不传 variant 时行为不变）

**交付物**：`repo/member1_data/pacs_dataset.py` 修改版 + `repo/member1_data/example_load_data.py` 中增加一个 variant 切换示例。

**验收**：不修改任何其他代码，通过传参即可在 4 个 α 变体之间切换。

### 任务 1.2 — 增强集统计与质量检查脚本

新增 `repo/member1_data/augmentation_stats.py`，输入任意增强集 manifest，输出：

- 每类的图片数量，以及与 `photo_train.txt` 的覆盖对比
- 文件 SHA256 的唯一性统计（识别重复文件）
- 与对应 PACS 原图**字节完全相同**的图片数量（必须为 0，否则说明增强集里混入了原图副本）
- 每一行 manifest 指向的文件是否存在
- 增强图与原图的尺寸分布

**交付物**：`repo/member1_data/augmentation_stats.py` + 对现有五个增强集各跑一次的 JSON 输出，存入 `repo/member1_data/results/`。

**验收**：脚本对故意构造的脏数据（重复文件、缺失文件、原图副本）能报出问题。

### 任务 1.3 — 划分一致性的独立复核

新增 `repo/member1_data/verify_splits_independent.py`，**不 import 成员 4 的任何代码**，独立实现一遍校验：

- 六份 split 的图片数：photo_train 1,349 / photo_val 150 / photo_test 171 / art_painting 2,048 / cartoon 2,344 / sketch 3,928
- 每份 split 的图片路径都能在本地 PACS 中找到
- 每个 split 内无重复图片
- 六个 split 之间无交集
- 每行路径的父目录名与标签一致
- 与 `project_splits/protocol.json` 中记录的 sha256 比对

**交付物**：脚本 + 一次运行输出（保存为 JSON）。

**验收**：全部检查通过；任一项失败时以非零退出码结束并打印具体问题。

### 任务 1.4 — 终稿 III.B 数据小节 + Table I + Fig. 7

- 撰写终稿 **III.B Dataset and Task Definition** 的数据与划分部分。现有进度报告 3.2 节的文字可直接迁移，需补一句"六份列表覆盖全部 9,990 张可用图片，交叉无重复"
- 产出 **Table I**（数据划分与实验角色）的 LaTeX 片段
- 产出 **Fig. 7**（PACS 四域七类样例图 + 各类别数量统计图）的**可复现生成脚本**。图已存在，需要补脚本
- 统计图脚本从 `project_splits/` 读取真实数量，不得写死数值
- 另需产出一份进度报告 4.1 节（`member1_41.tex`），内容与终稿 III.B 一致但更简短

**交付物**：`repo/member1_data/sections/member1_iiib.tex`（终稿）、`repo/member1_data/sections/member1_41.tex`（进度报告）、Fig. 7 生成脚本。

### 任务 1.5 — 对照组数据集的构造器（终稿 IV.B 的关键输入）

配合第 1.5 节的两组新对照，成员 1 提供构造 manifest 的工具，因为这三组对照的本质都是"训练集行数的安排"。

- 新增 `repo/member1_data/build_control_manifests.py`
- **+Duplicate 组**：把 `photo_train.txt` 的每张原图复制一份到 `<DATA_ROOT>/augmented/duplicate/images/`，文件名加 `_dup` 后缀，**字节内容与原图完全相同**
- **+NonStyle 组**：对每张原图施加一种确定性的非风格变换，生成第二份图片。可选变换：色彩抖动（ColorJitter，亮度/对比度/饱和度）、灰度、高斯模糊。建议采用**随机分配**（每张图从变换集合中随机取一个，seed 固定），使各变换都被覆盖
- **manifest 必须是 6 列，一列都不能省**。`read_aug_manifest` 用 `line.split()` 按空白分割，列数错位会让 `data.py` 把 style 列当成原图列：
  `<图片路径> <标签1-7> <原图相对路径> <style名> <alpha> <beta>`
  - +Duplicate 与 +NonStyle 的 `style` 列填固定占位符 `none`（**不能留空**，留空会导致该行只有 5 列，列错位）
  - `alpha` 与 `beta` 填 `0.00`
- 两个 manifest 的路径都必须**相对于 data root**
- 脚本要能接受 `--data-root` 参数，默认取 `paths.DATA_ROOT`
- **必须验证**并打印结果：
  - +Duplicate 的图片与原图 SHA256 完全一致
  - +NonStyle 的图片与原图 SHA256 **不一致**（否则变换没生效）
  - 两个 manifest 的行数、列数、覆盖的原图集合与 `photo_train.txt` 完全一致

**交付物**：`repo/member1_data/build_control_manifests.py` + 两个 manifest + 一次验证输出。

**验收**：两个 manifest 各 1,349 行、**每行 6 列**，与 `photo_train.txt` 的图片集合一一对应；训练集总行数均为 2,698，与 +AdaIN 组一致。

---

## 4. 成员 2：AdaIN 生成器与风格强度主线

**代码位置**：`repo/member2_adain/`
**负责章节**：II.B、II.C、III.D、IV.C（配合）、IV.G（风格池部分）、Fig. 8

成员 2 负责终稿主图 Fig. 2 的 AdaIN 一侧数据。

> **注意**：进度报告第 2 节目前只有 2.1（Cross-Domain Image Classification and PACS，成员 4 写），2.2 / 2.3 / 2.4 仍是 TODO 标记。终稿的 II.B、II.C 需要成员 2 补写，II.D 需要成员 3 补写。终稿的 Related Work 还要补一张对比表（Table IV），成员 4 起草框架、成员 5 补全条目。

### 任务 2.1 — 统一 α 扫描的输出目录结构

`build_adain_aug.py` 目前在多 α 时输出到 `alpha_0.25/` 等子目录，而成员 3 的扫描输出到 `sweeps/sweep_full/a0p669215_b0p7/`，两者结构不一致，导致成员 4 无法用同一套脚本遍历。

成员 3 扫描档位的**实际结构**（已核对）：

```text
aug_sp/sweeps/sweep_full/a0p669215_b0p7/
├── adain/                 ← 同一档里的均匀 AdaIN 对照
├── images/                ← 空间调制结果
├── analysis.json
├── generation.json
├── manifest_adain.txt
├── manifest_ours.txt
├── pairs.txt
└── photo_train.txt
```

注意：sweep 档位里**没有** `manifest_adain_matched.txt`（该文件只存在于 `aug_sp/` 根目录），改造时以实际结构为准，不要凭空新增文件。

- 改造 `build_adain_aug.py`，让多 α 输出与上表同构（同样的 `images/`、`adain/`、`manifest_ours.txt`、`manifest_adain.txt`、`generation.json`、`pairs.txt`、`photo_train.txt`）
- 保持单 α 时的现有行为不变
- **必须保持的性质**：α = 0.5 那一档生成的 1,349 张图，与现在 `augmented/adain/` 中的文件**逐字节一致**（SHA256 完全相同）。改造后要用 SHA256 验证这一点

**交付物**：修改后的 `build_adain_aug.py` + 一份 SHA256 比对报告（改造前 vs 改造后）。

**验收**：α = 0.5 的输出与现有 1,349 张逐字节相同；其他 α 档目录结构与上表一致。

### 任务 2.2 — 强度匹配均匀 AdaIN 的生成器

把"均匀 AdaIN 按逐图给定强度生成"的能力从成员 3 的 `stylizer.py` 中独立出来，放到成员 2 的地盘。

- 新增 `repo/member2_adain/uniform_adain_batch.py`
- 输入：内容图列表、风格图、**每张图的目标 α**（从空间调制图算出均值）
- 输出：均匀 AdaIN 增强集 + 成员 4 格式的 6 列 manifest
- 与成员 4 的 `fairness_check.py` 兼容（provenance 字段格式完全一致）

**交付物**：`repo/member2_adain/uniform_adain_batch.py` + 对现有 1,349 张重生成一次并与 `aug_sp/adain_matched/` 做 SHA256 比对。

**验收**：输出与现有 `aug_sp/adain_matched/` 逐字节相同。

### 任务 2.3 — 风格池敏感性分析

当前只用了 10 张外部画作作为风格池，这个选择本身是一个未验证的自由参数。

- 把 10 张风格图拆成两个子池（例如 5 + 5、以及 10 张整体），各自生成增强集并训练
- 对比不同子池下的交叉域准确率
- 交付：增强集生成 + 3 seeds 训练 + **Fig. 8** 对比图
- 写作：终稿 **IV.G Ablation Studies** 中风格池敏感性一段；进度报告对应写进 3.2 节

**验收**：结论明确——风格池规模是否影响"+7.2 点"这一主结论。

### 任务 2.4 — 终稿 II.B、II.C 与 III.D 撰写

**II.B Style Transfer and AdaIN**（进度报告 2.2 目前是 TODO，需补写）：

- 风格迁移的发展脉络与基本思路：基于 CNN 特征的风格表示、逐特征统计量匹配
- AdaIN 的具体做法与 $\alpha$ 的含义，与本项目设定的关系
- 引用 AdaIN 原文（`huang2017adain`），说明它是本项目的生成基座

**II.C Style Augmentation for Image Classification**（进度报告 2.3 目前是 TODO，需补写）：

- 风格增强用于分类的既有做法：把风格迁移当作数据增强、风格随机化、域随机化等方向
- **明确写出 gap**：现有工作在引入风格增强时，**有没有单独控制"施加了多少风格"这个变量？** 本项目的答案是基本没有——这正是本项目的贡献落点
- 与 MixStyle 等特征空间混合方法的区别（它们在特征统计量上混合，本文在生成图像上做）

**III.D AdaIN Style Augmentation**（方法节）：

- AdaIN 的特征空间混合原理、$\alpha$ 的含义、与本项目设定的关系
- 风格池的来源、冻结方式、与 PACS 的字节级去重证明
- 引用已有的 `member2_essay.tex` / `member2_essay_en.tex` 内容，但**必须按本阶段的结论改写**——尤其是风格增强的收益来源
- 现有的进度报告 3.4 节可作为迁移起点

**交付物**：`repo/member2_adain/sections/member2_iib.tex`（II.B）、`member2_iic.tex`（II.C）、`member2_iiid.tex`（III.D）+ `repo/member2_adain/sections/member2_41.tex`（进度报告 4.2）。

---

## 5. 成员 3：机制分析与失败案例

**代码位置**：`repo/member3_structure_preserving/`
**负责章节**：II.D、III.E、IV.E、IV.F、IV.G（β 部分）、Fig. 5、Fig. 6、Table V

> **与成员 4 的接口**：任务 5.2 若新增结构度量（如 GMSD、SSIM 结构分量），成员 4 必须在终稿 **III.F Evaluation Metrics** 里补上对应定义。成员 3 交付 `mechanism_analysis.json` 时，请在 JSON 中为每个新增度量写明**完整定义、取值范围、计算代码位置**，方便成员 4 直接引用，不必二次沟通。

### 任务 5.1 — 终稿 II.D Related Work 小节

进度报告第 2.4 节目前仍是 TODO 标记，需要补写：

- **II.D Structure Preservation and Structure-Related Metrics**
- 内容：结构保持类图像编辑/合成工作的思路（结构引导的 style transfer、语义结构约束、edge-aware 生成）
- 内容：用于评估结构保持程度的常见度量（Canny 边缘一致性、SSIM、GMSD 等），以及它们的局限——这些度量衡量的是"边缘是否还在"，不衡量"边缘是否对分类有用"
- 这一节要为第 IV 节的负面结论做铺垫：**结构度量上的提升，与下游分类性能之间，并没有我们原本假设的那种对应关系**
- 引用 AdaIN、LPIPS、Canny 之外的相关文献

**交付物**：`repo/member3_structure_preserving/sections/member3_iid.tex`。

### 任务 5.2 — 机制检验脚本

`repo/member4_training_eval/structure_mechanism.py` 目前是一个原型，归入成员 3 维护并扩展。

- 逐图 EdgeScore 配对比较（+Ours vs +AdaIN uniform vs +AdaIN 强度匹配），n = 1,349
- 结构密度与增益的相关性分析（Pearson + Spearman）
- 按类别细分（7 类）
- **扩展结构度量**：在 EdgeScore 之外增加至少一个独立指标，例如 GMSD（梯度幅值相似度）或 SSIM 的结构分量，验证结论是否稳健
- 输出机器可读的 JSON

**交付物**：`repo/member3_structure_preserving/structure_mechanism.py` + `repo/member3_structure_preserving/results/mechanism_analysis.json`。

**验收**：结论在多个结构度量下一致；不一致时必须明确报告出来而不是只报有利的那一个。

### 任务 5.3 — 失败案例分析（终稿 IV.F）

这是终稿中最能体现理解深度的一节。

现有数据显示空间调制对密集结构和稀疏结构的效果相同（两半区增益均为 −0.003），说明它的"保护"缺乏选择性。需要把这一点讲清楚：

- 挑选 12–18 张失败案例，覆盖以下情形：
  - 纹理背景被误判为结构（假保护）
  - 细长结构（琴颈、屋脊、象鼻）未被有效保护
  - 风格强度过弱导致的解码损伤
  - 保护过度导致类别线索被风格污染
- 每张案例配齐：内容图 / 风格图 / M 图 / α 图 / uniform AdaIN 结果 / 空间调制结果
- 每张写 2–3 句失效分析，说明"空间图到底在保护什么"
- 定量支撑：统计失败案例的共性（如 M 图在背景纹理区域的高响应占比）
- 与成员 5 协作产出 **Fig. 5** 图集

**交付物**：`repo/member3_structure_preserving/figures/failure_cases/` 下的图集 + `repo/member3_structure_preserving/sections/member3_ivf.tex`（分析文字）。

### 任务 5.4 — β 全量消融（含强度匹配对照）

当前 β 消融只有 7 张 dog 图，且存在已知混杂：β 越大，平均风格强度越低，EdgeScore 单调上升可能只是"风格变弱"的平凡结果。

- 全量 1,349 张，β ∈ {0, 0.3, 0.5, 0.7, 1.0}
- **每一档 β 都要同时生成强度匹配对照**，否则曲线无法解释
- 统计每档的 (EdgeScore, 平均 α, LPIPS)
- 训练：每档 1 个种子即可（用于画趋势），关键档位补齐 3 seeds
- 产出 **Fig. 6** 三联图与 **Table V** 的 β 部分

**交付物**：5 个增强集 + 对应强度匹配集 + 训练结果 + Fig. 6。

### 任务 5.5 — 终稿 III.E 方法小节 + IV.E 机制分析

- **III.E Structure-Preserving Style Augmentation**：已有 `handoff_member5/sections/03_5_method.tex` 可作基础，但需补上**强度混杂的说明**（$\alpha(x,y)=\alpha_0(1-\beta M)$ 必然导致平均强度低于同 $\alpha_0$ 的均匀混合），并说明我们如何用两个对照处理它
- **IV.E Structural Mechanism Analysis**：把任务 5.2 的结果组织成正文，含 EdgeScore 配对统计与结构密度相关性
- 进度报告对应改写 4.3 节（`member3_43.tex`），内容改为已完成状态

**交付物**：`repo/member3_structure_preserving/sections/member3_iiie.tex`、`member3_ive.tex`、`member3_43.tex`。

---

## 6. 成员 4：实验编排、统计与整合

**代码位置**：`repo/member4_training_eval/`
**负责章节**：I.A–I.D、**II.A**、III.A、III.C、III.F、III.G、IV.A–IV.D、IV.H、V、Table II、Table III、Table IV、Table VI、Fig. 2、Fig. 3、Fig. 4、Fig. 9

> **II.A** 指 Cross-Domain Image Classification and PACS。进度报告里这一节是直接写在 `progress_report.tex` 正文中的（未拆成片段），终稿需要把它单独拆成 `member4_iia.tex`，并补上文献引用——现在它是全文唯一没有引用的 Related Work 小节。

> **成员 4 已有的章节片段**（`repo/report/sections/`）与终稿小节的对应关系，迁移时必须一并处理：
>
> | 现有片段 | 对应终稿小节 | 处理 |
> |---|---|---|
> | `member4_31.tex`（3.1 Overall Framework） | **III.A** | 迁移到 `member4_iiia.tex`，补一个描述完整实验流程的框图引用 |
> | `member4_33.tex`（3.3 ResNet-18 Baseline） | **III.C** | 迁移到 `member4_iiic.tex` |
> | `member4_36.tex`（3.6 Evaluation Metrics） | **III.F** | 迁移到 `member4_iiif.tex`，**并按任务 5.2 新增的结构度量扩写** |
> | `member4_37.tex`（3.7 Experimental Design） | **拆为 III.G + IV.A/IV.B** | 见任务 6.4 |
> | `member4_52.tex`（5.2 Experiment Completion Plan） | 无对应 | **仅保留在进度报告**，终稿不迁移 |
>
> `report/sections/member3_method.tex` 现有内容是 `\input{.../handoff_member5/sections/03_5_method.tex}`，指向旧的 3.5 节。终稿里 III.E 由成员 3 的 `member3_iiie.tex` 提供，**两个入口同时 `\input` 会造成 `\label` 重复定义、编译报错**。处理方式：把 `member3_method.tex` 的内容改为指向 `member3_iiie.tex`，或直接删除该兼容入口并在 `progress_report.tex` 中直接引用成员 3 的新文件。

### 任务 6.1 — 统一实验编排器

新增 `repo/member4_training_eval/run_sweep.py`，把分散的手工命令收敛成一条命令：

- 支持的扫描维度：α 网格、β 网格、配置类型（uniform AdaIN / spatial Ours / strength-matched AdaIN）、seed 列表
- 每组训练前自动执行 `fairness_check.py`
- 复用已有的 `config_hash` 机制自动跳过已完成的组（已完成则跳过，不重训）
- 输出统一的 `summary_<tag>.json`
- 支持 `--dry-run` 打印将要执行的全部命令而不实际运行

**交付物**：`repo/member4_training_eval/run_sweep.py` + 一份用它跑通全部五组的命令记录。

**验收**：一条命令能跑完 α 扫描并生成完整结果表；重复执行时正确跳过已完成的组。

### 任务 6.2 — 统计检验最终版

升级 `repo/member4_training_eval/stats_compare.py`：

- 在配对 t 检验之外增加 **bootstrap 置信区间**（n = 3 时 t 检验对单点异常敏感，bootstrap 更稳健）
- 增加 **多重比较校正**（Benjamini-Hochberg FDR），本阶段比较组数多，不校正会显著抬高假阳性率
- 增加 **效应量**（Cohen's d 与配对版本），不只报 p 值
- 输出 **Table III 显著性总览表**（比较组、差值、95% CI、t、p、校正后 p、效应量、结论）

**同时修复 `final_analysis.py` 的表头**：它现在把临界值写死成 `crit |t|=4.303`（对应 df=2）。种子扩到 5 之后 df=4、临界值变为 2.776，这个字符串会印错。必须改为由 `stats.t.ppf(0.975, n-1)` 动态计算。

（`stats_compare.py` 本身用的就是 `stats.t.ppf(0.975, n - 1)`，无需改动。）

**交付物**：升级后的 `stats_compare.py` + 修正后的 `final_analysis.py` + `runs/stats_report.json`。

### 任务 6.3 — 终稿主图生成

扩展 `repo/member4_training_eval/plot_results.py`：

| 图号 | 内容 | 要求 |
|---|---|---|
| Fig. 2（主图） | 风格强度 → 交叉域准确率曲线，三条线（Original / Uniform AdaIN / Spatial Ours），含误差棒 | 直接讲完整个故事 |
| Fig. 3 | 配对差值森林图（forest plot），带 95% CI | 展示效应量与不确定性 |
| Fig. 4 | EdgeScore 逐图配对累积分布图 | 机制证据 |

- 所有图的数值从 `runs/*.json` 读取，不写死
- 输出矢量 PDF 或 300 dpi PNG，可直接进 LaTeX
- 配色需色盲友好

**交付物**：`repo/member4_training_eval/plot_results.py` + `repo/report/figures/` 下的图文件。

### 任务 6.4 — 终稿 III.A、III.C、III.F、III.G 与 IV.A–IV.D 撰写

**方法类小节（III.A / III.C / III.F）**——从现有进度报告片段迁移，文字基本照搬：

- **III.A Overall Framework**：来自 `member4_31.tex`。补上对完整实验流程的描述（训练集构造 → 风格增强 → 训练 → 三域评测），并引用 Fig. 1 框图
- **III.C ResNet-18 Baseline**：来自 `member4_33.tex`，照搬
- **III.F Evaluation Metrics**：来自 `member4_36.tex`。**必须按任务 5.2 新增的结构度量扩写**——若成员 3 加了 GMSD 或 SSIM 结构分量，指标定义要同步加入，否则第 IV 节会出现正文里没定义过的指标
- **III.G Experimental Design and Fair Comparison**：从现有进度报告 3.7 节迁移，保留公平性检查机制、step-budget 制、配置指纹的描述，**删除其中所有结果表和结果讨论**（这些移到 IV 节）

**结果类小节（IV.A–IV.D）**：

- **IV.A Experimental Setup**：全部实验组的定义表、种子、超参、硬件、评测协议
- **IV.B Main Results**：Table II 主结果 + 文字解读，需包含第 1.5 节三组新对照的结果
- **IV.C Style Strength versus Cross-Domain Accuracy**：Fig. 2 + 讨论，这是全文核心段落
- **IV.D Statistical Significance**：Table III + Fig. 3

**交付物**：`repo/report/sections/member4_iia.tex`、`member4_iiia.tex`、`member4_iiic.tex`、`member4_iiif.tex`、`member4_iiig.tex`、`member4_iva.tex`、`member4_ivb.tex`、`member4_ivc.tex`、`member4_ivd.tex`。

### 任务 6.5 — 两组数据量对照的训练与汇报（终稿 IV.B 的核心）

配合第 1.5 节，这是成员 4 优先级最高的**新**任务，因为它补的是结论的软肋。

- 等待成员 1 交付 `build_control_manifests.py` 产出的两个 manifest（+Duplicate、+NonStyle）
- 产出 **Table VI 收益来源分解表**：

  | 收益来源 | 对照 | 计算方式 |
  |---|---|---|
  | 纯数据量 | +Duplicate − Original | 数据行数翻倍、不重复图片数不变、无外观变化 |
  | 通用外观多样性 | +NonStyle − Original | 行数翻倍、通用变换带来外观变化 |
  | 风格本身 | +AdaIN − +NonStyle | 风格相对通用变换的额外贡献（**上界估计**，见表注） |
  | 空间调制 | +Ours − +AdaIN(强度匹配) | 在同等风格强度下，空间分布的贡献 |

- 这一张表是全文最重要的表，它把"风格增强有效"拆成可归因的四个部分
- **表注必须写明**：第三行「风格本身」中同时包含"风格化 vs 通用变换"和"特定变换族 vs 随机变换族"两个因素，因此它是**上界估计**，不是严格的因果归因

#### 6.5.1 需要新增实验组

`config.GROUPS` 扩展为 `("original", "adain", "adain_matched", "ours", "ours_equal", "duplicate", "nonstyle")`

- `duplicate` 与 `nonstyle` 进入 `AUGMENTED_GROUPS`，训练集构成与 `adain` 完全一致（1,349 原图 + 1,349 manifest 图片）

#### 6.5.2 需要修改的四组硬编码

**这四处都维护着一份实验组列表，新增组后必须同步，否则新组不会出现在表里或图里。** 现有内容如下（截至本次核对）：

| 文件 | 常量 | 现有内容 | 缺什么 |
|---|---|---|---|
| `config.py` | `GROUPS` | 4 组 | 缺 `ours_equal`、`duplicate`、`nonstyle` |
| `config.py` | `AUGMENTED_GROUPS` | `adain`, `adain_matched`, `ours` | 缺 `ours_equal`、`duplicate`、`nonstyle` |
| `summarize_results.py` | `PRETTY` | 4 组 | 缺 `ours_equal`、`duplicate`、`nonstyle` |
| `summarize_results.py` | `GROUP_ORDER` | 4 组 | 缺 `ours_equal`、`duplicate`、`nonstyle` |
| `final_analysis.py` | `LABELS` | 5 组 | 缺 `duplicate`、`nonstyle` |
| `final_analysis.py` | `ORDER` | 5 组 | 缺 `duplicate`、`nonstyle` |
| `plot_results.py` | `LABELS` | **仅 3 组**（original/adain/ours） | 缺 `adain_matched`、`ours_equal`、`duplicate`、`nonstyle` |
| `plot_results.py` | `COLORS` | **仅 3 项** | 同上，新组要配新颜色 |

注意 `plot_results.py` 现在只画 3 组，而 `final_analysis.py` 已出 5 组——**现有的 `figures/results_baseline.png` 与结果表不一致**，本任务一并修正。

#### 6.5.3 公平性检查器的放宽规则

`fairness_check.py` 现有五项约束中，第 2 项（style 必须在冻结池内）、第 4 项（内容-风格配对一致）、第 5 项（风格图不得是 PACS 图）都不适用于这两组，因为它们的 style 列是占位符 `none`，不来自风格池。

放宽规则：

- `duplicate` 与 `nonstyle` 组**跳过**第 2、4、5 项检查
- 但**必须保留**第 1 项（覆盖同样的原图）与第 3 项（行数与其他 manifest 相同）——这两项正是本实验要控制的量，绝不能放宽
- 实现方式：给 `check()` 增加一个"无风格组"的参数或白名单，让这三项检查对该组返回空集合，而不是整体跳过 `check()`

**交付物**：扩展后的 `config.py`、`summarize_results.py`、`final_analysis.py`、`plot_results.py`、`fairness_check.py`，两组对照的训练结果，`repo/report/sections/member4_tablevi.tex`。

**验收**：Table VI 四行全部有数字，且每行都标注了差值的 95% CI 与 p 值；`figures/results_baseline.png` 的组数与 Table II 一致。

### 任务 6.6 — 种子扩展到 5 个

- 现有全部实验组是 3 个种子（0/1/2），统计功效偏弱。主结论 +7.20 点在 n=3 时 p=0.070，扩到 5 个种子是让它能站住的前提
- `config.SEEDS` 改为 `(0, 1, 2, 3, 4)`
- **必须显式跑全部五个种子**，命令形如：

  ```powershell
  .\nn.ps1 member4_training_eval\run_all.py --group adain --seeds 0 1 2 3 4 --aug-manifest <path>
  ```

  不能只跑 `--seeds 3 4`。`run_all.py` 写 summary 时用的是本次调用的 `args.seeds`：

  ```python
  summary = {"group": args.group, "seeds": args.seeds, "config_hash": config.config_hash()}
  ```

  只跑 3 4 会把 `summary_<group>.json` 重写成只含两个种子，**原来的 0/1/2 直接丢失**。跑全部五个时，前三个会因为 `eval_last.json` 已存在且 `config_hash` 匹配而自动跳过训练（跳过逻辑会先比对 `canonical_hash_of_run`），不会浪费时间也不会覆盖结果
- 注意：`config_hash` 不包含 seeds，所以 `ecc6090e5b29` 保持不变，已有结果仍可比
- 补跑后需要重新生成 `summary_*.json`（由 `run_all.py` 写出），不需要额外汇总步骤
- `stats_compare.py` 的配对检验用的是 `stats.t.ppf(0.975, n-1)`，n=5 时临界值会从 4.303 变为 2.776，这是预期行为，无需改代码

**交付物**：补跑结果 + 更新后的 `summary_*.json`。

**验收**：主结果表的每个组有 5 个种子的均值与标准差；`summary_*.json` 的 `seeds` 字段为 `[0, 1, 2, 3, 4]`。

### 任务 6.7 — MixStyle 基线

MixStyle 是真实的域泛化方法，在特征层做实例级统计量混合。论文的参考文献里已经收录了该文献（`zhou2021mixstyle`），但至今没有实现，只靠引用它而没有和它比较是不充分的。

**实现部分（第 1 周，不占 GPU）**：

- 在 `model.py` 中增加 MixStyle 层，插在 ResNet 的 stage 之间
- **不能改动 `build_model()` 的默认行为**，否则 `config_hash` 会变、已有结果失效。改为 `build_model(arch="resnet18", mixstyle=False)`，默认参数保持原样
- MixStyle 只作用于训练阶段，推理时关闭

**训练部分（第 2 周批次 F）**：

- 作为独立实验组（`mixstyle` 组，训练集 = 1,349 原图，不做增强），单独汇报
- 由于改了模型结构，MixStyle 组的 `config_hash` 会不同，必须单独成组，不与 `ecc6090e5b29` 的结果混在同一张表里直接比较
- 3 个种子

**交付物**：`model.py` 的 MixStyle 实现 + `mixstyle` 组的训练结果 + 在 Table IV Related Work 对比表中的对应行。

**验收**：MixStyle 组在 3 个种子上跑完，数字写入独立的小节，不能与其他组直接做显著性检验。

### 任务 6.8 — 逐类准确率分析（Fig. 9）

`summary_*.json` 里已经记录了 `per_class`（每个 split、每个类、每个种子的准确率），数据是现成的，**零训练成本**。

- 从所有实验组的 `summary_*.json` 读取 `per_class`，画热力图：行 = 实验组，列 = 7 个类别，值 = 交叉域准确率
- 回答一个对结论解释力很强的问题：**风格增强是均匀地帮助所有类别，还是只帮助某几类？**
  - 如果只有少数类别受益（比如都是轮廓清晰的物体），那"风格增强有效"的机制就清楚了：它在补缺失的轮廓信息
  - 如果均匀提升，说明作用更可能是通用的正则化效应
- 同步输出一张逐类增益表（+AdaIN − Original 每类的差值）
- 注意跨域每类样本量差异很大（Sketch 的 dog 只有几十张），热力图要标注每类的样本数，避免把噪声当成信号

**交付物**：`repo/member4_training_eval/plot_per_class.py` + Fig. 9 数据与初版图 + 逐类增益表。

> 图的**定稿排版**由成员 5 的任务 7.4 接手：本任务负责数据读取、统计计算与绘图脚本，成员 5 负责最终视觉规范（配色、单元格内数值标注、每类样本数标注）。

### 任务 6.9 — 终稿 I.D、IV.H、V 节与全局审查

- **I.D Contributions and Outline**：四条贡献，每条对应一个对照实验
- **IV.H Discussion and Limitations**：把进度报告 4.6 的"风险与未决问题"改写为研究局限，客观陈述负面结果
- **V Conclusion and Future Work**：结论 + 未来工作
- **全局一致性审查**：
  - Introduction 的研究问题与第 V 节结论一致
  - 摘要每个数字与 Table II 一致
  - 全篇数字可从 `runs/` 复算
  - 各成员片段合并后编译通过、交叉引用不报错
  - **没有重复 `\input`**（旧的 `member3_method.tex` 与新的 `member3_iiie.tex`）
  - 参考文献补齐（AdaIN、LPIPS、Canny 之外新增：MixStyle、域泛化综述等）
  - 页数控制在 8 页内
- **终稿骨架搭建**：新建 `repo/report/final_report.tex`，迁移 1–3 节并做好 3.7 的拆分，各成员片段按新编号接入。骨架要预留 III.A–III.G 与 IV.A–IV.H 全部小节的 `\input` 位置（内容可先空着），让各成员有明确的落点

**交付物**：`repo/report/final_report.tex` + `repo/report/sections/member4_ivh.tex` + 审查清单 + 编译通过的 `final_report.pdf`。

---

## 7. 成员 5：可视化、demo 与报告整合

**代码位置**：新建 `repo/member5_visualization/`
**负责章节**：Abstract、V、Fig. 1、Fig. 5 排版、Fig. 7 协助、Fig. 9 排版

### 任务 7.1 — 图表生成管线

- 新建 `repo/member5_visualization/make_all_figures.py`，一条命令生成终稿所需的全部图
- 图的数据来源统一为 `runs/*.json` 与增强集 manifest，不手输数值
- 制定并落实统一视觉规范：字号、配色、图例位置、色盲友好配色、导出格式
- 与成员 4 的任务 6.3 协调：Fig. 2–4 由成员 4 实现，本任务负责 Fig. 1、Fig. 5 排版、Fig. 7、Fig. 8 纳入同一条管线
- 重绘 **Fig. 1 整体框架图**：现有 `framework.png` 需更新为反映终稿流程（含两个对照实验的位置），并输出矢量版本

**交付物**：`repo/member5_visualization/make_all_figures.py` + `repo/report/figures/` 下的全部图。

### 任务 7.2 — 交互式 demo

- 上传内容图与风格图 → 依次显示内容图、风格图、M 图、α 图、uniform AdaIN 结果、空间调制结果
- 分别送入 baseline 分类器与增强分类器，显示预测标签与置信度对比
- **必须包含"强度匹配对照"开关**：同一张图可切换查看 uniform AdaIN（逐图匹配强度），让"空间调制到底改变了什么"在 demo 里可直接观察

**交付物**：`repo/member5_visualization/demo.py` + 一段 1 分钟的演示录屏或截图序列。

### 任务 7.3 — 失败案例图集排版

- 成员 3 提供失败案例的分析文字与候选图（任务 5.3）
- 本任务负责做成终稿可用的 **Fig. 5** 网格图集：统一标注、统一尺寸、每张图配一句话说明

**交付物**：`repo/report/figures/failure_cases.png` + 图注文字。

### 任务 7.4 — 逐类热力图排版与收益来源分解图（Fig. 9）

- 成员 4 负责数据与计算（任务 6.8），本任务负责把结果做成视觉上可直接读的图
- **Fig. 9 逐类准确率热力图**：行 = 实验组（含 +Duplicate、+NonStyle），列 = 7 个类别，色深 = 交叉域准确率
- 要求：
  - 单元格内直接标注数值，不只靠颜色传达信息
  - 标注每类在各域的样本数（跨域每类样本量差异很大，避免把噪声看成信号）
  - 旁边配一张"逐类增益"的差值图（+AdaIN − Original），突出哪些类别受益最多
- 另可做一张**收益来源分解的瀑布图**：从 Original 出发，依次叠加"纯数据量"、"通用外观多样性"、"风格本身"、"空间调制"，直观呈现每一层的贡献大小。这是 Table VI 的图形版，很适合做汇报。

**交付物**：`repo/report/figures/per_class_heatmap.png`、`repo/report/figures/gain_waterfall.png` + 图注。

### 任务 7.5 — 终稿摘要、V 节与 Related Work 对比表补全

- **Abstract**：全文最后写，**150–250 词、单段、无引用、无公式**（IEEE 惯例）。必须包含：问题设定、方法、核心实验设计（含强度匹配对照与数据量对照）、主要定量结果、以及那句负面结论。摘要里的每个数字都要与 Table II 一致
- **V Conclusion and Future Work**：结论是负面结果，需要客观陈述并说明局限与后续方向
- **Table IV Related Work 对比表**：成员 4 起草框架，本任务补全条目、核查每篇文献的报告数字与设定。MixStyle 一行必须填入我们实测的数字（任务 6.7），不能只引用原文报告值
- 进度报告对应：5.1 / 5.4 / 5.5 节与摘要。进度报告的 `member4_52.tex`（5.2 Experiment Completion Plan）**保留在进度报告里，不迁移到终稿**

**交付物**：`repo/member5_visualization/sections/member5_abstract.tex`、`member5_v.tex`、`member5_51.tex`、`member5_54.tex`、`member5_55.tex`。

### 任务 7.6 — 打包与交付检查脚本

- 新建 `repo/member5_visualization/check_package.py`
- 检查项：所有引用的图文件存在；所有 `\ref` / `\cite` 有对应定义；PDF 能编译；无遗留 TODO 标记；所有相对路径可解析；页数在限制内；仓库中无大文件误入（checkpoint、生成图不进仓库）
- **新增检查项**：全篇 LaTeX 与脚本中不得出现绝对路径（扫描 `C:\`、`D:\` 这类字面量）。组员各自在本地机器上工作，路径一旦写进终稿就会在别人机器上失效
- 输出逐项检查结果

**交付物**：检查脚本 + 一次全通过的检查报告。

---

## 8. 排期与 GPU 占用

### 第 1 周：并行写代码（不占用 GPU）

| 成员 | 任务 |
|---|---|
| 成员 1 | 1.1、1.2、**1.5**（对照组构造器） |
| 成员 2 | 2.1、2.2、**2.4 的 II.B/II.C 部分**（写作，不占 GPU） |
| 成员 3 | 5.1（II.D 写作）、5.2（用现有 1,349 张，无需生成） |
| 成员 4 | 6.1、6.2、**6.5.1–6.5.3**（新组与硬编码列表）、**6.7 实现部分**、终稿骨架搭建 |
| 成员 5 | 7.1、7.2 |

### 第 2 周：GPU 集中执行（成员 4 统一编排）

| 批次 | 内容 | 生成量 | 训练次数 | 预计耗时 |
|---|---|---|---|---|
| A | α 扫描：4 档 × {uniform, spatial, matched} | 9 套 ≈ 1.1 GB | 9 × 3 = 27 | ≈ 2.8 h |
| B | β 消融：5 档 + 5 套 matched | 10 套 ≈ 1.2 GB | 5 × 1（关键档补 3 seeds） | ≈ 1.2 h |
| C | 风格池敏感性：2 个子池 | 2 套 ≈ 0.25 GB | 2 × 3 = 6 | ≈ 0.6 h |
| **D** | **数据量对照：+Duplicate、+NonStyle** | **2 套 ≈ 0.24 GB** | **2 × 5 = 10** | **≈ 0.4 h** |
| **E** | **种子扩展：主结果各组补 seed 3/4** | 无 | **约 20** | **≈ 1.1 h** |
| **F** | **MixStyle 基线（3 seeds）** | 无 | **3** | **≈ 0.2 h** |

- 批次 A–F **按顺序串行**，成员 4 用任务 6.1 的编排器执行
- 批次 D 依赖成员 1 的任务 1.5 交付，需在第 1 周内完成
- 批次 E 建议排在 D 之后，让 Table VI 先有结果再补种子
- 批次 E、F 的命令必须显式列出全部种子（例如 `--seeds 0 1 2 3 4`），见第 10 节陷阱清单
- 各成员在等待训练时推进写作与图集工作
- 磁盘占用估算：增强集约 2.8 GB，checkpoint 约 2.5 GB。**训练完成后及时清理不再需要的 `last.pt`**（每个 43 MB）

### 第 3 周：写作、整合与审查

| 成员 | 任务 |
|---|---|
| 成员 1 | 1.3、1.4 |
| 成员 2 | 2.3（依赖批次 C）、2.4 |
| 成员 3 | 5.3、5.4 整合、5.5 |
| 成员 4 | 6.3、6.4、**6.5**（Table VI）、**6.6**、**6.8**、6.9 |
| 成员 5 | 7.3、**7.4**、7.5、7.6 |
---

## 9. 交付物清单

| 成员 | 代码 | 图表 | 报告片段 |
|---|---|---|---|
| 成员 1 | `pacs_dataset.py`(扩展)、`augmentation_stats.py`、`verify_splits_independent.py`、**`build_control_manifests.py`**、统计图脚本 | Fig. 7 | 终稿 `member1_iiib.tex`；进度报告 `member1_41.tex` |
| 成员 2 | `build_adain_aug.py`(改造)、`uniform_adain_batch.py` | Fig. 8 | 终稿 `member2_iib.tex`、`member2_iic.tex`、`member2_iiid.tex`；进度报告 `member2_41.tex` |
| 成员 3 | `structure_mechanism.py`、失败案例分析脚本 | Fig. 5、Fig. 6、Table V | 终稿 `member3_iid.tex`、`member3_iiie.tex`、`member3_ive.tex`、`member3_ivf.tex`；进度报告 `member3_43.tex` |
| 成员 4 | `run_sweep.py`、`stats_compare.py`(升级)、**`final_analysis.py`(临界值动态化)**、`plot_results.py`(扩展)、**`plot_per_class.py`**、`model.py`(MixStyle)、**`config.py`/`summarize_results.py`/`final_analysis.py`/`plot_results.py`/`fairness_check.py`(新组与硬编码列表)**、**`final_report.tex`** | Fig. 2、Fig. 3、Fig. 4、**Fig. 9**、Table II、Table III、Table IV、**Table VI** | 终稿 `member4_iia.tex`、`member4_iiia.tex`、`member4_iiic.tex`、`member4_iiif.tex`、`member4_iiig.tex`、`member4_iva.tex`–`member4_ivd.tex`、`member4_ivh.tex`、**`member4_tablevi.tex`** |
| 成员 5 | `make_all_figures.py`、`demo.py`、`check_package.py` | Fig. 1、Fig. 5 排版、**Fig. 9 排版**、**收益瀑布图**、全部图管线 | 终稿 `member5_abstract.tex`、`member5_v.tex`；进度报告 `member5_51.tex`、`member5_54.tex`、`member5_55.tex` |

### 片段接入方式

- **终稿**：各成员片段放到 `repo/<成员目录>/sections/`，由成员 4 合并进 `repo/report/final_report.tex`。命名规则 `member<N>_<罗马数字小节>.tex`，例如 `member3_ivf.tex` 表示终稿第 IV.F 节。
- **进度报告**：沿用现有 `member<N>_<数字>.tex` 命名（如 `member1_41.tex` 对应进度报告 4.1 节），由成员 4 用 `member4_training_eval/_sync_sections.py` 合并进 `repo/report/sections/member4_sections.tex`，最终并入 `repo/report/progress_report.tex`。
- 两套文件名不冲突，可并行维护。
- **`member4_52.tex` 只保留在进度报告**，终稿不迁移。
- **旧 `member3_method.tex` 的兼容入口必须处理**，否则终稿与进度报告同时 `\input` 同一份内容会造成 `\label` 重复定义、编译报错。

---

## 10. 已知陷阱清单（执行前必读）

以下每一条都是核对现有代码后确认的实际行为，不是假设。

| 陷阱 | 实际行为 | 应对 |
|---|---|---|
| 种子扩展会覆盖旧数据 | `run_all.py` 用本次调用的 `args.seeds` 写 summary。只跑 `--seeds 3 4` 会让 `summary_<group>.json` 只剩两个种子，原有 0/1/2 丢失 | 必须显式跑 `--seeds 0 1 2 3 4`（任务 6.6） |
| manifest 列数错位 | `read_aug_manifest` 用 `line.split()` 分割，style 列留空会导致整行只有 5 列，`data.py` 的 `meta[0]` 取到错误字段 | 对照组的 style 列填占位符 `none`，保持 6 列（任务 1.5） |
| 实验组列表散落五处 | `config.py`/`summarize_results.py`/`final_analysis.py`/`plot_results.py` 各有一份，且现在彼此不一致（`plot_results.py` 只画 3 组，`final_analysis.py` 已有 5 组） | 新增组时五处同步改（任务 6.5.2） |
| 临界值被写死 | `final_analysis.py` 表头印 `crit |t|=4.303`（df=2）。种子扩到 5 后 df=4、临界值 2.776 | 改为 `stats.t.ppf(0.975, n-1)` 动态计算（任务 6.2） |
| 公平性检查器误报 | `fairness_check.py` 的第 2/4/5 项检查 style 字段，对照组的占位符会被判为"不在冻结池内" | 这三项对无风格组返回空，第 1/3 项必须保留（任务 6.5.3） |
| `\input` 重复 | `report/sections/member3_method.tex` 与成员 3 的新 `member3_iiie.tex` 指向同一内容 | 改指向或删除兼容入口（成员 4 的章节映射表） |
| 风格池只出现在根目录 | `manifest_adain_matched.txt` 只在 `aug_sp/` 根目录，sweep 档位里没有 | 成员 2 改造时以实际结构为准，不凭空新增（任务 2.1） |
| `best.pt` 占用空间 | 16 个 `best.pt` 共 0.67 GB，因 checkpoint 策略已改为 `last` 而不再使用 | 确认后可删除回收空间（纪律 10） |

---

## 11. 路径与环境（各成员在自己电脑上执行）

代码中的所有路径都由 `repo/paths.py` 统一解析。**任何脚本、配置、manifest 里都不允许出现绝对路径。**

```text
<repo>/
├── paths.py                 ← 唯一的路径解析入口
├── ENVIRONMENT.md           ← 新机器的完整配置步骤
├── nn.ps1                   ← 自动挑选可用的 Python 解释器
├── member1_data/
├── member2_adain/
├── member3_structure_preserving/
├── member4_training_eval/
└── report/
```

三个环境变量，全部可选：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `NNDL_DATA_ROOT` | 包含 `PACS/` 的目录 | `<repo>/data` |
| `NNDL_RUNS_ROOT` | 运行产物目录 | `<repo>/../runs` |
| `NNDL_PYTHON` | Python 解释器 | 由 `nn.ps1` 自动探测 |

组员在自己电脑上只需做两件事：

1. 把 PACS 放到 `<repo>/data/PACS`，**或**设一次 `NNDL_DATA_ROOT` 指向自己的数据位置
2. 用 `.\nn.ps1` 而不是直接 `python` 调用脚本

验证环境是否就绪：

```powershell
.\nn.ps1 member4_training_eval\verify_splits.py
```

应输出 `PASS: all six splits are consistent with protocol.json and the local PACS copy.`

写新代码时的四条硬性要求：

1. 不要手工拼路径，一律从 `paths`（或 `config` 的转发）取
2. manifest 里写**相对于 data root** 的路径，这样 manifest 能跨机器复用
3. CLI 的默认值必须来自 `paths.py`，不能是字面量
4. 不要新增 splits 的副本。规范位置是 `member4_training_eval/project_splits/`，成员 2 那份只是回退副本

已经完成的路径改造（成员 4 负责，已验证）：

| 文件 | 改动 | 验证 |
|---|---|---|
| `repo/paths.py` | 新增，全仓库唯一的路径解析入口 | 有无环境变量均正确 |
| `member4_training_eval/config.py` | 改用 `paths.*` | `config_hash` 仍为 `ecc6090e5b29`，已有结果可比 |
| `member4_training_eval/verify_splits.py` | 改用 `paths.IMAGE_ROOT` | 六份 split 全部 PASS |
| `member2_adain/adain_paths.py` | `project_root()` 改为返回仓库根目录；split 定位改用 `paths.SPLIT_DIR` | 原来在无环境变量时会找错目录，已修复 |
| `member3_structure_preserving/protocol.py` | 已经通过 `config` 取路径，自动继承 | 无需改动 |
