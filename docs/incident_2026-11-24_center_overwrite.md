# 事故记录：中心先验增强集覆盖了原 Sobel 数据集

**日期**：2026-11-24 · **状态**：已完全恢复 · **影响范围**：一个数据集，已逐字节重建

---

## 1. 发生了什么

启动"中心先验"（`center`）前景调制的数据集生成时，命令是：

```
generate build --modulation foreground --foreground-kind center --cell-tag fgcenter --overwrite
```

**`--cell-tag fgcenter` 是个错误**。生成器只把精确的 `foreground` 标签当作"修复变体"来处理；任何其他标签都会落到共享目录 `aug_sp/` —— 也就是**原方法（Sobel）数据集的家**。于是这一轮把 1349 张原 Sobel 增强图**全部替换**成了中心先验版本，并覆盖了 `aug_sp/manifest_ours.txt`。

两个叠加的因素让它没有被立刻发现：

1. `--overwrite` 关掉了"设置不一致"保护
2. 生成器的"未记录文件"保护只看目录里有没有非 `previews`/`sweeps` 的东西，`fgcenter` 目录当时还不存在，所以没触发

## 2. 影响

| 项 | 状态 |
|---|---|
| `aug_sp/images/`（原 Sobel，1349 张） | **被覆盖**，已重建 |
| `aug_sp/manifest_ours.txt` | **被覆盖**，已重建 |
| `augmented/manifest_ours.txt` | 被覆盖，已重建 |
| `aug_sp/foreground/`（spectral_center） | 未受影响 |
| `aug_sp/sweeps/`（β-sweep、α-sweep、子池） | 未受影响 |
| `aug_sp/adain_matched/` | 未受影响 |
| 正在运行的训练作业 | 未受影响（用的是 `sweeps/` 的 manifest） |
| **已发布的主表结果** | **未被污染**（见第 4 节） |

## 3. 恢复过程

### 3.1 先保住新数据

覆盖发生后第一件事不是重建，而是**把中心先验数据集移出共享目录**，避免二次覆盖：

```
aug_sp/images  ->  aug_sp/fgcenter/images
```

`aug_sp/fgcenter/manifest_ours.txt` 里的 1349 行路径随后被改写到新位置，并逐行确认文件存在。

### 3.2 确认生成器是确定性的

重建要成立，必须证明同样的输入会产出同样的字节。检查方法：把**同一条命令跑两次**，写进不同的 cell tag，然后逐字节比对：

| 模式 | 逐字节相同 | 不同 |
|---|---|---|
| `edge`（Sobel） | 14 | 0 |
| `foreground` + `center` | 14 | 0 |

生成器内无任何随机数使用，风格由 style-pool manifest 固定。**确定性成立。**

### 3.3 重建原数据集

用原参数（α=0.5, β=0.7, `--modulation edge`）重新生成 1349 张。

### 3.4 与**旧代码**的产物比对

这是关键一步。`previews/build_n5/a0p5_b0p7/` 里存着 11-18 由**生成主表的旧代码**（`stylizer.py=ce67229bcf56`）产出的 α=0.5/β=0.7 样本：

```
identical 5, different 0, missing 0
```

**重建结果与旧代码产物逐字节一致。** 这证明：

- `edge` 代码路径在前景修复提交中未被改动（diff 显示前景提交只新增了分支，`edge` 模式下调制图仍是 Sobel 结构图）
- **主表的已发布结果描述的正是现在磁盘上的数据集** —— 结论未受污染

### 3.5 完整性复查

| 数据集 | 行数 | 缺失文件 |
|---|---|---|
| `ours`（Sobel，重建） | 1349 | 0 |
| `ours_fg_center`（中心先验） | 1349 | 0 |
| `ours_fg`（spectral_center） | 1349 | 0 |
| `adain_matched`（对照） | 1349 | 0 |

**公平性检查 PASS**：新组满足与主组完全相同的约束——同样的行数、同样的内容-风格配对、同样的原图覆盖。

## 4. 为什么已发布的结果没有被污染

主表的 `ours` 组读的是 `aug_sp/manifest_ours.txt`，指向 `aug_sp/images/`。这正是被覆盖的部分。

但**训练早已完成**：主表的 5 个种子在 11-24 03:48 之前就已跑完并写入 `runs/`，训练产物（`eval_last.json` 等）**独立于数据集文件存在**。数据集被覆盖只影响**之后**用该数据集训练的新作业，不影响已经算出的数字。

而重建证明磁盘上的数据集与训练时用的那份**逐字节相同**，所以连"新作业与旧结果是否可比"这个问题也不存在。

## 5. 已修的根因

**生成器**（`member3_structure_preserving/generate.py`）：

- 任何带 cell tag 的 build 现在都写进 `aug_sp/<tag>/` 自己的目录
- 只有**不带** tag 的 build（即原方法）才写共享目录 `aug_sp/`，也才发布 `augmented/manifest_ours.txt`
- manifest 发布的判断条件从 `!= "foreground"` 改为 `is None`，语义更准确

**训练侧**（`member4_training_eval/`）：

- 新组 `ours_fg_center` 注册，**独立 config_hash**（与 `ours_fg` 也不同，因为地图不同 → 方法不同）
- `data.py` / `run_sweep.py` / `config.py` 改为按 `FOREGROUND_GROUPS` / `GROUP_MANIFESTS` 通用处理，不再硬编码单一组名

## 6. 留下的检查脚本

| 脚本 | 用途 |
|---|---|
| `check_determinism.py` | 同一命令跑两次并比对字节 —— 重建的前提 |
| `verify_rebuild_against_old_code.py` | 重建结果 vs 旧代码产物 |
| `verify_restored_datasets.py` | 各数据集行数、缺失文件、公平性 |
| `check_overwrite_damage.py` | 事后评估覆盖范围 |
| `check_center_run_targets.py` | 定位某次生成到底写了哪些目录 |

## 7. 教训

**生成器的"保护"只保护已知的状态，不保护共享目录的所有权。**

cell tag 原本只为子池实验设计（避免同 α/β 互相覆盖），语义是"给 sweep 目录加个前缀"。把它复用到 build 模式时，它落到了共享目录的**默认分支**上 —— 一个静默的失败。

教训落到三处：

1. **共享目录要有明确的所有者**，且只有所有者能写
2. **任何写入共享位置的路径都要显式声明**，不能靠默认值
3. **确定性检查要成为重建的前置步骤**，不能靠"生成器应该没问题"的假设

以及最重要的一条：**这次能恢复，是因为生成器恰好是确定性的，而且旧代码的样本还留在磁盘上。** 这两件事都是运气，不能当作理所当然 —— 所以它们现在是脚本（`check_determinism.py`），而不是记忆。
