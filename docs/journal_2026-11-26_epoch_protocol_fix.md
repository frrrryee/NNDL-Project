# 工作日志：Epoch 协议修复（固定步数 → 按数据量反推 epoch）

**文档 v1.0** · 2026-11-26 · 状态：**已实施 + 已验证**

---

## 0. 一页摘要

| 项 | 内容 |
|---|---|
| **问题** | `TOTAL_STEPS = 660` 是按 1,349 张原图算的固定步数，但被**无差别**发给所有组 |
| **后果** | 增强组（2,698 张）只看到 **15.7** epoch，基线组看到 **31.3** epoch → 增强方法被系统性低估 |
| **修复** | 步数改为按各组**自身数据量**反推：`steps = 30 × ⌈N / 64⌉` |
| **关键不变量** | `original` 组步数仍为 **660**、指纹仍为 `ecc6090e5b29` → **磁盘上已有结果全部保持有效** |
| **增强组新预算** | 2,698 张 → **1,290 步**（30.6 epoch），原为 660 步（15.7 epoch） || **验证结果** | `duplicate`（字节相同副本，唯一变量就是训练协议）从 **37.47% → 38.38%**（同种子），不再因协议而落后；`adain_a1p0_b0p7` 52.87%，单种子差异在种子噪声内，**需 5 种子重跑定论** |
| **附带修复** | 重建了丢失的 `aug_sp/manifest_adain.txt`（1,349 行，全部图片可解析，公平性检查通过） |
| **附带修复 2** | 修掉 `fairness_check` 的两个"记账层"误判：旧 hash 定义下的存档 run 曾被误报为协议违规（`original_s0/s1/s2` 的 `12fff826824b`）。详见 §3.5 |
| **影响范围** | 所有增强组的结果都需要重跑才能与新协议下的 `original` 对比；**这是必须的**，否则比较仍然带偏置 |
| **不影响** | `original`（40.83% ± 2.31，5 种子）与 `mixstyle` 的结果**完全有效**，是模块 A / B 的现成基准 |

---

## 1. 缺陷的精确描述

### 1.1 根因

`repo/member4_training_eval/config.py`：

```python
ORIGINAL_EPOCHS = 30
TOTAL_STEPS = 660      # = 30 * ceil(1349/64)，按 1349 张原图算
```

`train.py` 无条件把 `TOTAL_STEPS` 发给每一组：

```python
total_steps = config.TOTAL_STEPS
```

数据量由 `data.load_train_rows()` 决定：**Photo Training split（1,349 张）+ 该组的 manifest 行数（1,349 行）= 2,698 张**。

于是：

| 组 | 训练图数 | 修复前步数 | 修复前有效 epoch | 修复后步数 | 修复后有效 epoch |
|---|---|---|---|---|---|
| `original` | 1,349 | 660 | **31.3** | 660 | 31.3 |
| `mixstyle` | 1,349 | 660 | **31.3** | 660 | 31.3 |
| 所有增强组 | 2,698 | 660 | **15.7** | **1,290** | **30.6** |

### 1.2 为什么这不只是"理论上的不平等"

`duplicate` 组是**干净的判据**：它每张原图配一份**字节完全相同**的副本，所以里面**没有任何增强内容**，唯一的变量就是 epoch 数。

| 组 | 跨域 avgacc | 有效 epoch |
|---|---|---|
| `original` | **40.83%** | 31.3 |
| `duplicate` | **37.69%** | 15.7 |

**掉 3.14 点，且原因只可能是训练协议。** 这证明这个缺陷是**真实的、可测量的**，而不是"理论上可能有点影响"。

### 1.3 一句话推论

> **此前所有增强方法的结果，都被这个欠训练系统性压低了。**
> 它们之间的**相对**比较大体仍成立（都同样欠训练），但它们与 `original` 基线的**绝对**差距被夸大了。

---

## 2. 修复方案

### 2.1 核心规则

$$\text{total\_steps} = \text{ORIGINAL\_EPOCHS} \times \left\lceil \frac{N}{64} \right\rceil$$

**固定的是 epoch，不是步数。** 上取整到整个 batch，保证每个组的最后一个 epoch 是完整的。

### 2.2 代码改动

| 文件 | 改动 |
|---|---|
| `config.py` | 新增 `total_steps_for(n_images)`、`train_images_of_group(group, manifest)`、`total_steps_of_group(group, manifest)`；`TOTAL_STEPS` 保留但降级为**基线参考值**，不再是发给所有组的常量 |
| `config.py` | `training_config(group, manifest)`、`config_hash(group, manifest)` 及其派生指纹（mixstyle / foreground / line-art）全部改为 **group-aware** |
| `config.py` | 新增 `run_matches_protocol(run_dir, group, manifest)`：复用检查的**协议**部分 |
| `train.py` | `total_steps = config.total_steps_of_group(args.group, args.aug_manifest)`；记录的 training_config 带上本组预算 |
| `run_sweep.py` / `run_all.py` | 复用检查改为**两段式**：协议（步数+超参）+ 方法（地图/manifest 指纹） |
| `fairness_check.py` | 期望指纹按组计算；旧协议的 run 会被明确指出"步数 660 != 1290" |
| `check_fingerprint_isolation.py` | 同上 |
| `show_epoch_counts.py` | 改为按各 run **实际记录的步数**计算 epoch |

### 2.3 指纹为什么必须 group-aware

步数现在是**数据的函数**，所以两组即使超参完全相同，只要数据量不同，指纹就**必须**不同 —— 否则复用检查会把"训练预算不同的两个 run"当成可比较的。

派生指纹链：

```
config_hash(group, manifest)                    ← 基线 + 本组 total_steps
  ├─ mixstyle_config_hash()                     ← MixStyle（1349 张 → 660 步，指纹不变）
  ├─ foreground_config_hash(kind, group, m)     ← 前景修复（带地图）
  ├─ foreground_center_config_hash(group, m)
  ├─ foreground_center_matched_config_hash(group, m)
  └─ line_art_config_hash(unit, m, group)       ← 线稿族（带 manifest 内容哈希）
```

### 2.4 复用检查的两段式设计

这是本次修复里最容易出错的地方，所以单独说明：

| 检查段 | 比较对象 | 回答的问题 |
|---|---|---|
| **协议** | 记录的训练块 canonical hash **+ epoch 不变式** | "这个 run 是不是用当前协议训练的？" |
| **方法** | run 自己记录的 `config_hash` | "这个 run 是不是我要跑的那个方法？" |

**epoch 不变式**（`run_matches_protocol`）是一个**自足**的判据：

> run 记录的步数，必须等于它**自己记录的数据量**所蕴含的步数。

660 步 / 2,698 张 = 15.7 epoch ≠ 30 → **旧协议的 run 自动被拒**，即使 manifest 丢失也不会漏判。这条不变式是本次加的保险，因为 manifest 缺失会让"按组算预期指纹"这条路走不通。

---

## 3. 验证结果

### 3.1 协议层验证（全部通过）

```
[PASS] 1349 images -> 660 steps
[PASS] 2698 images -> 1290 steps
[PASS] the budget is not a constant
[PASS] original budget is still 660
[PASS] original fingerprint is still ecc6090e5b29
[PASS] mixstyle fingerprint is still e3bcbe6568bb
[PASS] 所有 15 个组都是 30~31.5 epoch
[PASS] 旧协议的 run 被判为 stale
[PASS] 新协议的 run 被接受
[PASS] 公平性检查拒绝旧协议、接受新协议
[PASS] original_s0..s4 仍然匹配当前协议（已有结果保持有效）
[PASS] 两种 hash 定义确实不同
[PASS] 旧定义下的快照被接受
[PASS] 当前定义下的快照被接受
[PASS] 两种定义都不匹配的快照被拒绝
[PASS] 无记录块时快照无法校验
[PASS] original_s0..s4 的存储快照与各自记录一致
```

**负向测试**（关键：修复不能把"真过期"也放过）：

```
$ fairness_check.py --runs .../ours_s0 .../duplicate_s0
FAIL (2 problems):
  - ours_s0: step budget 660 != 1290; the run was trained under the old
             fixed-budget protocol and has to be retrained
  - duplicate_s0: step budget 660 != 1290; ...
```

**验收工具本身也验证过**：

```
$ fairness_check.py --adain <adain manifest> --ours <ours manifest> --runs <original_s0..s4>
PASS: the augmented manifests satisfy all fairness constraints.
```

### 3.2 下游验证（各 1 种子，seed 0 对 seed 0）

| 组 | 修复前 CROSS-AVG | 修复后 CROSS-AVG | 差 |
|---|---|---|---|
| `duplicate` | 37.47% | **38.38%** | **+0.91** |
| `adain_a1p0_b0p7` | 55.26% | **52.87%** | **−2.39** |

**`duplicate` 涨了，符合预期** —— 它的唯一变量就是训练预算，多给一半 epoch 自然好一点。

**`adain_a1p0_b0p7` 掉了 2.39 点，但这不能直接解读为"更多训练更差"。** 原因：

1. **种子噪声就够解释**。该组旧协议的 5 种子是 `55.26 / 53.86 / 51.27 / 54.79 / 55.96`（均值 54.23，std **1.82**）。seed 0 的旧值 55.26 是**该组的最高点之一**，用它和新协议的 seed 0 比，是在拿"旧组最好的一个"对"新组的一个"。新协议 seed 0 的 52.87 落在旧组 5 种子分布的**正中间**。
2. **更多训练不一定更好**。这个组的 Photo Val 在第 780 步就到 100%，之后 510 步纯粹在**过拟合**。跨域泛化在源域已经饱和后继续训，经常是负收益 —— 这一点和之前 `diag_budget_s0`（1,320 步）观察到的"cross-domain 在窄带里波动、不随步数系统改善"是一致的。
3. **结论需要 5 种子才能下**。1 个种子的差值在 ±2.4 点以内时，**什么都证明不了**。

### 3.3 现有数据的整体图景（旧协议，5 种子，供重跑后对照）

| 组 | 均值 | std | 5 个种子 |
|---|---|---|---|
| `original` | 40.83% | 2.31 | 37.37 / 39.63 / 42.78 / 41.80 / 42.58 |
| `mixstyle` | 40.29% | — | — |
| `nonstyle` | 39.88% | 0.53 | 40.12 / 39.87 / 40.61 / 39.58 / 39.23 |
| `duplicate` | 37.69% | 1.86 | 37.47 / 35.06 / 39.79 / 39.12 / 37.03 |
| `adain_a0p25_b0p7` | 41.95% | 1.02 | 42.35 / 40.62 / 42.30 / 43.23 / 41.25 |
| `ours` | 45.22% | 1.22 | 45.89 / 46.07 / 43.32 / 46.15 / 44.68 |
| `adain_matched` | 45.39% | 0.29 | 45.04 / 45.53 / 45.52 / 45.72 / 45.15 |
| `ours_equal` | 47.21% | 1.12 | 46.35 / 48.04 / 46.61 / 48.76 / 46.28 |
| `adain_a1p0_b0p7` | **54.23%** | 1.82 | 55.26 / 53.86 / 51.27 / 54.79 / 55.96 |

**注意 `duplicate` (37.69) 比 `original` (40.83) 低 3.14 点，而 `nonstyle` (39.88) 几乎追平 `original`。**
这个组合本身就是证据：`nonstyle` 加的是**真实的外观变化**（模糊/灰度/抖动），`duplicate` 加的是**字节相同的副本**。
在旧协议下，`duplicate` 掉得比 `nonstyle` 多 —— 如果增强内容本身有用，`duplicate` 至少应该和 `nonstyle` 差不多。
新协议下如果 `duplicate` 追上 `nonstyle`，就说明**旧协议的 3.14 点缺口是预算造成的，不是数据造成的**。这是重跑时最值得盯的一个数。

### 3.4 指纹隔离检查（通过）

```
  12fff826824b    3 runs  groups=['original']
  e3bcbe6568bb    5 runs  groups=['mixstyle']
  ecc6090e5b29  125 runs  groups=['adain','adain_matched','duplicate','nonstyle','original','ours','ours_equal']
  ...
  PASS: every run is consistent with its own record
```

### 3.5 附带发现：两个"记账层"的 bug（与 epoch 修复无因果关系）

在把修复推上远端之前，对全部 173 个 run 做了一次指纹审计。审计暴露出两个**与 epoch 修复无关、但会让队友无法使用验收工具**的缺陷。

#### Bug A：`fairness_check` 把"指纹定义演进"误判成"协议违规"

**现象**：`fairness_check.py --runs results/runs/original_s0..s4` 报 FAIL：

```
- original_s0: config_hash 12fff826824b != current ecc6090e5b29
- original_s1: config_hash 12fff826824b != current ecc6090e5b29
- original_s2: config_hash 12fff826824b != current ecc6090e5b29
```

**根因**：`original_s0/s1/s2` 是在**更早的代码版本**下跑的，当时 `config_hash` 的定义是"把整个 `training_config` 块逐字哈希"。后来定义改成"只哈希显式的行为字段白名单"。于是：

| run | 存储的 hash | 定义 | 重算的 canonical hash |
|---|---|---|---|
| `original_s0/s1/s2` | `12fff826824b` | 旧（整块哈希） | `ecc6090e5b29` |
| `original_s3/s4` | `ecc6090e5b29` | 新（白名单哈希） | `ecc6090e5b29` |

**`12fff826824b` = 旧定义下整块 `training_config` 的哈希**（已用 `sha256(json.dumps(block, sort_keys=True))` 复算验证，完全吻合）。

**关键判断：这三个 run 的训练行为与当前协议完全一致。** 逐字段比对 10 个行为字段全部相同：

```
batch_size 64 | total_steps 660 | lr 0.01 | momentum 0.9 | weight_decay 5e-4
label_smoothing 0.1 | eval_every 60 | train_transform 同 | eval_transform 同
arch resnet18 | pretrained imagenet
```

`checkpoint_selection` 确实是旧值（`best photo_val accuracy`），但它**不在行为字段白名单里** —— 它是策略的文档说明，且 `config.py` 的注释明确写了"不计入指纹，因为改写措辞不应作废已有 run"。

**修法**：`config.py` 新增 `snapshot_is_consistent(stored, recorded_config)` —— 存储的快照在**两种定义下**分别从 run 自己的记录块重算，只要有一种吻合就接受。两种 hash 都是**从记录重算**，没有一处是信任存档字符串。

```python
def snapshot_is_consistent(stored, recorded_config) -> bool:
    if stored is None: return False
    if not isinstance(recorded_config, dict): return False
    if stored == config_hash_of(recorded_config): return True
    payload = json.dumps(recorded_config, sort_keys=True).encode()
    return stored == hashlib.sha256(payload).hexdigest()[:12]
```

#### Bug B：只有 hash 字符串的存档产物会被误判

`fairness_check` 的 fallback 分支（产物只有 `config_hash`、没有 `training_config` 块）同样把快照与当前值直接比较。没有记录块就无法重算，因此**无法区分"旧定义"和"不同方法"**。改为报 **warning** 而非 problem。

#### 顺带查清的一件事：`original` 的 5 个种子其实是同质的

审计过程中发现 `original_s0/s1/s2` 记录的是 `checkpoint_selection = "best photo_val accuracy"`，而 `original_s3/s4` 是 `"final checkpoint of the fixed step budget"` —— **两者的 checkpoint 选择策略不同**。这看起来像污染，需要核实：

| run | `eval_best.json` | `eval_last.json` |
|---|---|---|
| `original_s0` | 30.30 | **37.37** |
| `original_s1` | 44.74 | **39.63** |
| `original_s2` | 37.71 | **42.78** |
| `original_s3` | （无） | **41.80** |
| `original_s4` | （无） | **42.58** |

- `eval_best` 的 3 种子均值 = **37.58%**，std **7.22**
- `eval_last` 的 5 种子均值 = **40.83%**，std **2.31**

**结论：报告引用的是 `eval_last`，即 40.83% ± 2.31，与"最终 checkpoint、无选择"协议一致。** 报告里写的"选择 best 会把种子间差异从 2.71 扩大到 7.22 点"指的正是上面那个 37.58 / 7.22 的组合。`original` 的 5 个种子**全部是 final-checkpoint 协议**，没有污染。

（旧 run 目录里保留了 `eval_best.json` 是历史遗留——`SAVE_BEST_CHECKPOINT` 现在默认关闭，但旧 run 存过。**报告数字一律取 `eval_last`**。）

---

## 4. 附带修复：`aug_sp/manifest_adain.txt` 丢失

**发现方式**：新代码要求读取 manifest 行数来算训练集大小，`adain` 组的 manifest 读不到 → 直接报错。

**原因**：早前做 α/β sweep 时，生成器把 `aug_sp/manifest_adain.txt` 挪到了 `aug_sp/sweeps/sweep_full/<cell>/` 下，原路径没留副本。`adain_s0..s4` 这 5 个 run 的 summary 里记的还是旧路径。

**重建方式**（`manifest_ours.txt` 的配对 + `aug_sp/adain/photo/` 的实际文件）：
- 1,349 行，行数与 `manifest_ours.txt` 一致
- **0 行**的图片不存在
- **0 个**磁盘上有但未被引用的孤立文件
- **公平性检查通过**（同源图、同风格池、同配对、同计数、无测试泄漏）

（`aug_sp/manifest_adain_matched.txt` 一直在，未受影响。）

---

## 5. 重跑时最该盯的三个数

修复本身只是"把训练量拉平"，它**不会**让增强方法突然变好。真正的问题是：**在训练量拉平之后，哪些结论还站得住？** 重跑时按这个顺序看：

| # | 观察点 | 旧协议 | 怎么读 |
|---|---|---|---|
| **1** | `duplicate` vs `nonstyle` | 37.69% vs 39.88%（**差 2.19**） | 新协议下如果差距**缩小或消失**，说明旧的缺口是**预算**造成的，`duplicate` 作为对照组恢复了意义。**如果差距不变甚至变大**，说明 `nonstyle` 的优势来自数据本身（外观变化），那才是真正的增强效应 |
| **2** | `adain_a1p0_b0p7`（α=1.0 均匀） | 54.23%（最优） | 它是全项目最优组。如果新协议下它**掉出第一**，说明旧的领先部分来自欠训练（欠训练的组更容易在源域欠拟合、在某些域更"保守"） |
| **3** | `ours` / `ours_equal` vs `adain_matched` | 45.22 / 47.21 vs 45.39 | 这是**空间调制 vs 均匀 AdaIN** 的正面对决（控制掉总强度）。旧协议下三者几乎打平，说明空间调制净效应 ≈ 0。**新协议下如果空间调制开始领先，α 地图搜索才值得继续做；如果还是打平，就该彻底转向"更强样式"这条线** |

**这三条同时决定下一步方向**：
- 1 变好 → 增强内容本身有效，可以继续在增强方向上做
- 2 掉队 → 转向纯 AdaIN / 更强样式方向
- 3 仍打平 → **放弃空间调制这条线**，把精力投到 α=1.0 体制 + 别的正交维度（线稿、软投票）上

---

## 6. 接下来必须做的事

**这一步只修了协议，之前的增强结果都还是旧协议的，不能直接和新的 `original` 对比。**

### 5.1 当前哪些结果是有效的

`show_epoch_counts.py`（已更新）现在会直接标出 stale 状态：

```
  group                        images  steps  epochs        now  epochs
  adain                          2698    660    15.7  <- stale   1290    30.6
  adain_matched                  2698    660    15.7  <- stale   1290    30.6
  duplicate                      2698    660    15.7  <- stale   1290    30.6
  mixstyle                       1349    660    31.3        660    31.3
  nonstyle                       2698    660    15.7  <- stale   1290    30.6
  original                       1349    660    31.3        660    31.3
  ours                           2698    660    15.7  <- stale   1290    30.6
  ours_equal                     2698    660    15.7  <- stale   1290    30.6
  ours_fg_center                 2698    660    15.7  <- stale   1290    30.6
  ours_fg_center_matched         2698    660    15.7  <- stale   1290    30.6
  ours_line_attn                 2698    660    15.7  <- stale   1290    30.6
  ours_line_plain_fg             2698    660    15.7  <- stale   1290    30.6
  ours_line_plain_total          2698    660    15.7  <- stale   1290    30.6
  ours_line_vote                 2698    660    15.7  <- stale   1290    30.6
```

- **`original` / `mixstyle` 的 5 种子结果仍然有效**（预算未变，指纹未变）。
- **12 个增强组全部 stale**，必须重跑。

### 6.2 三个独立模块的基准各自是否受影响

这一节回答"重跑要挡谁的路"。判据是**该模块的对照基准落在哪个组**。

| 模块 | 改动对象 | 对照基准 | 基准是否受 epoch 修复影响 | 能否立刻开工 |
|---|---|---|---|---|
| **A 类别平衡与采样** | 采样权重（不生成新图） | `original`（A1 为 A2/A3 的基准） | **否** —— 1,349 张 → 660 步，预算未变 | **可以** |
| **B 决策边界平滑化** | 训练过程（不生成新图） | `original`（B4 为 B1/B2/B3 的基准） | **否** —— 同上 | **可以** |
| **C 剪影强化增强** | 特征层调制 + 生成新图 | `adain_a1p0_b0p7`（同为 $\alpha_0=1.0$） | **是** —— 2,698 张 → 1,290 步，旧值 54.23% 作废 | 代码与生成**可以**；最终对比需等基准重建 |

**可直接引用的基准数值**：

- 模块 A / B 的基准：`original` 5 种子，**40.83% ± 2.31**（final checkpoint，660 步）—— **现成有效**。
- 模块 C 的基准：`adain_a1p0_b0p7` 5 种子，**待重跑**。新协议 seed 0 已得 52.87%，需补齐其余 4 个种子。

> C1 还有一项额外约束：验收标准要求 C1 与 `adain_a1p0_b0p7` 的 **mean α 差异 < 0.05**。mean α 从 manifest / 图像算，但基准组必须用同一协议重新生成，否则配对检验不成立。

**结论**：重跑**只挡模块 C 的最终对比表**（约 35 分钟），模块 A 和 B 完全不受影响，三个模块的代码工作都可以立即并行开工。

### 6.3 重跑清单

| 优先级 | 动作 | 成本 |
|---|---|---|
| **P0** | 重跑主表所有增强组（`duplicate`/`nonstyle`/`adain`/`adain_matched`/`ours`/`ours_equal`）× 5 种子 | 6 组 × 5 种子 × ~8.5 min ≈ **4.3 小时** |
| **P0** | 重跑 `adain_a1p0_b0p7`（当前最优 54.23%）× 5 种子 | ~35 min |
| **P1** | 重跑前景组 / 线稿组（`ours_fg_center`、`ours_line_vote` 等） | ~2 小时 |
| **P1** | 在新协议下**重做 α 地图搜索**（这是修复的直接目的） | 视选型范围 |
| P2 | 更新报告中所有"固定 660 步"的表述 | 已改主要几处 |

**不要做的事**：不要在旧协议的结果和新协议的结果之间做比较。指纹机制会强制重训，但手工看数字时容易犯错。

---

## 7. 一句话总结

> **把"固定步数"改成"固定 epoch"之后，增强组第一次拿到和基线一样的训练量；此前所有"增强方法打不过基线"的结论都需要在新协议下重新测量 —— 包括 `duplicate` 那 3.14 点。**
