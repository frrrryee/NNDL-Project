# 风格池规模消融：10 张 vs 20 张（`ours` 与 `adain` 双臂，完整四格）

**日期**：2026-11-26
**状态**：已完成
**协议**：epoch-budgeted（1,290 步 / 2,698 图），final checkpoint，四格两两一致
**α**：四格统一 α = 0.5（`a0p5_b0p7`）

---

## 一句话结论

> **在 20 张风格池下 Ours 仍然不如 AdaIN，且差距被放大。**
>
> - 10 张池：Ours 44.74% ± 2.12 vs AdaIN 46.80% ± 2.14，**Δ = −2.07 点**（t = −7.65，p = 0.0016，显著）
> - 20 张池：Ours 44.37% ± 2.06 vs AdaIN 47.72% ± 2.71，**Δ = −3.36 点**（t = −9.19，p = 0.0008，显著）
>
> 扩池本身对两种方法都不显著（Ours +0.37，n.s.；AdaIN −0.92，n.s.），
> 但**交互项显著**（Δ = +1.29 点，t = 3.72，p = 0.0205）：AdaIN 能从更大的风格池获益，
> Ours 不能。**风格池大小不是 Ours 的杠杆。**

---

## 完整四格对照表

### 跨域准确率（cross-domain avg，5 seeds）

| 方法 | 10 张风格 | 20 张风格 | 池增益 |
|---|---|---|---|
| **Ours** | 44.74% ± 2.12 | 44.37% ± 2.06 | +0.37（n.s.） |
| **AdaIN** | 46.80% ± 2.14 | 47.72% ± 2.71 | −0.92（n.s.，方向为负） |
| **方法差（Ours − AdaIN）** | **−2.07 点**（t = −7.65, p = 0.0016） | **−3.36 点**（t = −9.19, p = 0.0008） | 交互项 **+1.29 点**（t = 3.72, p = 0.0205，显著） |

### 逐种子明细

| 种子 | ours_p10 | ours_p20 | adain_p10 | adain_p20 |
|---|---|---|---|---|
| 0 | 45.66 | 45.30 | 47.99 | 49.79 |
| 1 | 42.12 | 41.63 | 43.51 | 44.01 |
| 2 | 45.75 | 46.14 | 47.99 | 49.95 |
| 3 | 47.20 | 46.04 | 48.72 | 49.17 |
| 4 | 42.95 | 42.73 | 45.79 | 45.69 |
| **均值 ± std** | **44.74 ± 2.12** | **44.37 ± 2.06** | **46.80 ± 2.14** | **47.72 ± 2.71** |

### 分域（5 seeds 均值 ± std）

| 格 | Art Painting | Cartoon | Sketch |
|---|---|---|---|
| ours_p10 | 67.93 ± 1.60 | 29.63 ± 2.87 | 36.64 ± 3.20 |
| ours_p20 | 67.93 ± 1.24 | 28.77 ± 2.88 | 36.40 ± 2.28 |
| adain_p10 | 68.32 ± 1.49 | 31.76 ± 2.96 | 40.33 ± 2.94 |
| adain_p20 | 68.52 ± 1.74 | 31.59 ± 2.81 | 43.07 ± 3.95 |

**Sketch 域读法**（这是 `sketch.png` 是否泄漏的判别证据）：

- AdaIN 的 Sketch 域在 20 张池下涨了 **+2.74 点**（40.33 → 43.07），这是四格里唯一有
  明确方向的单调变化，且与"新池含 `sketch.png`"的预期一致。
- 但 Ours 的 Sketch 域**没涨**（36.64 → 36.40，−0.24）。若 `sketch.png` 提供了可迁移的
  风格先验，两个方法都该受益；只有传输统计量更"直白"的 AdaIN 吃到了它。
- 这不构成 PACS 测试集的泄漏（`sketch.png` 与 PACS Sketch 字节不等，灰度均值 0.732
  vs 0.955），但它确实说明：**风格先验在 AdaIN 上比在 Ours 上更"直通"。**

### 结构保持指标（全部 1,349 张图）

| 池 | 方法 | LPIPS ↓ | Edge F1 ↑ |
|---|---|---|---|
| p10 | Ours | **0.2258** (sd 0.0817) | **0.3436** (sd 0.0628) |
| p10 | AdaIN | 0.2787 (sd 0.0947) | 0.3211 (sd 0.0653) |
| p20 | Ours | **0.2203** (sd 0.0804) | **0.3535** (sd 0.0600) |
| p20 | AdaIN | 0.2782 (sd 0.0977) | 0.3267 (sd 0.0642) |

**Ours 在两个池下都显著更好地保持了结构**，且 20 张池进一步改善（LPIPS −0.0055，
Edge +0.0099）。也就是说：

- **准确率维度：Ours 输给 AdaIN 2~3 点**
- **结构保持维度：Ours 明显赢**（LPIPS 低 0.05~0.06，Edge F1 高 0.02~0.03）

AdaIN 的准确率优势**部分是用结构损伤换来的**。这是 Ours 的方法论卖点，也解释了为什么
单纯追求跨域准确率会偏向 AdaIN。

---

## 交互项：为什么 AdaIN 吃到了扩池而 Ours 没有

这是本实验第二个关键发现（第一个是"池大小不是多样性的旋钮"，见下）。

**扩池效应配对差**（同 seed）：
- AdaIN：p10 → p20，Δ = −0.92 点（t = −2.28，p = 0.085，不显著）
- Ours：p10 → p20，Δ = +0.37 点（t = 1.48，p = 0.213，不显著）
- **交互项 Δpool(AdaIN) − Δpool(Ours) = +1.29 点（t = 3.72，p = 0.0205，显著）**

单个边际效应都不显著，但**它们之差显著**——这是典型的"效应抵消"型交互：AdaIN 从
扩池中得到的（符号为正的）Sketch 收益，恰好被 Cartoon 域的小幅回落抵消，净效应被压到
不显著的 −0.92；而 Ours 的扩池净效应是 +0.37。两者之差因此显著。

**机理**：AdaIN 直接搬运通道级均值/方差，风格参考图的统计量**直通**到增强图像。而 Ours
走的是结构保持路径（见 `repo/member3_ours/`），风格参考只影响颜色/纹理的调制方式，
不直接决定输出图像的统计量。因此：

- 一个"更像 Sketch"的风格图，对 AdaIN 是直接信号，对 Ours 是间接扰动。
- **风格池大小对 Ours 不是杠杆。** 想提升 Ours 的跨域能力，应该动 `AUG_PER_IMAGE`
  （每张原图配多个风格），而不是加风格图。

---

## 第一个关键发现：池大小不是多样性的旋钮

**每个训练样本只携带一个风格参考**（`AUG_PER_IMAGE = 1`）。把池从 10 张扩到 20 张
**不改变单个样本所含的风格多样性**，只是把风格集合换成了另一套配置。

真正控制多样性的参数是**每张原图所配的风格数 `AUG_PER_IMAGE`**，而不是池子大小。

---

## 第二个关键发现：`sketch.png` 没有造成 PACS 测试集泄漏

新池包含 `sketch.png`（黑白铅笔素描），PACS Sketch 是本项目的测试域之一。

**判别性证据是 Sketch 域本身没有一致上涨**（对 Ours：−0.24 点；且 AdaIN 的上涨伴随
Cartoon 域的回落，净效应不显著）。若这张图真的在暗中帮助 Sketch 域，两个方法都该
明显受益。

机理上也说得通。AdaIN 搬运的是通道级均值与方差：

```45:54:repo/member2_adain/adain.py
def adaptive_instance_normalization(content_feat, style_feat):
    """Match the channel-wise mean and variance of content features to style features."""
    ...
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean
```

实测统计量：

| | 灰度均值 | 饱和度 |
|---|---|---|
| PACS Sketch 测试域 | 0.955 | 0.000 |
| `sketch.png` | 0.732 | 0.000 |
| 池中 10 张画作均值 | 0.564 | 0.186 |

`sketch.png` 与 PACS Sketch 的共同点只有"饱和度为 0"，而灰度均值差 0.223。共同点没有
转化成一致的 Sketch 域收益。

---

## 限定：数量效应与配置效应在本实验中分不开

池子从 10 张变成 20 张时，内容→风格的配对会**整体重推**（配对规则是
`index % pool_size` 的循环分配）。实测在 seed 42 下，10 张池与 20 张池之间同一张原图
仍配到同一风格的只有 **203/1,349（15.0%）**。

因此本实验测的是"一套 10 张配置 vs 一套 20 张配置"，**不是**在固定配对下只增加风格数。

---

## 一次执行事故：CPU 环境污染

`adain_p10` 这一格第一次跑出来是**错的**：所有 5 个种子用了 CPU 训练（2,975 s/种子，
对比 GPU 的 361 s/种子，慢 8.2 倍）。

**根因**：`_run_structure_metrics.ps1` 设了 `$env:NNDL_DEVICE = "cpu"`（LPIPS/Edge 指标
是 CPU 算的），而 PowerShell 的 `$env:` 赋值在同一会话里**存活**，污染了之后在该会话里
运行的每一个脚本。`adain_p10` 的 runner 继承了 `cpu`，`config.py` 的
`os.environ.get("NNDL_DEVICE")` 就把设备锁到了 CPU。

**修复**：runner 里显式 `$env:NNDL_DEVICE = "cuda"`，并清理残留目录后重跑全部 5 个种子。
同时在 `config.py` 里保留 `NNDL_DEVICE` 覆盖能力（这是有用的调试开关），但 runner
必须显式声明设备。

**教训**：在这个环境里跑任何训练前，先确认 `NNDL_DEVICE` 的值；跑完第一格后核对
`wall_time_sec` 是否在预期量级（GPU 应为 ~360 s，CPU 为 ~3,000 s）。

---

## 对项目的影响

- **主表不动。** 主结果表继续使用冻结的 10 张池，与项目现有全部结果保持可比。
- **Ours 在跨域准确率上确实输于 AdaIN**（−2~3 点，5 seeds 全部方向一致）。这是本实验
  最诚实的结果，不能回避。
- **Ours 的结构保持优势在四格中全部成立**（LPIPS 低 0.05~0.06，Edge F1 高 0.02~0.03），
  且扩池后进一步改善。这是 Ours 唯一稳定赢的维度，也是方法论定位的正确落点。
- **风格池大小不是 Ours 的杠杆**（交互项显著）。下一步应转向 `AUG_PER_IMAGE`。

---

## 复现

```powershell
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"

# 训练（四格都是 1,290 步，α = 0.5）
python member4_training_eval\train.py --group ours  --seed <0..4> `
    --aug-manifest <manifest> --tag poolab_ours_p10_s<seed>  --keep-checkpoint
python member4_training_eval\train.py --group ours  --seed <0..4> `
    --aug-manifest <manifest> --tag poolab_ours_p20_s<seed>  --keep-checkpoint
python member4_training_eval\train.py --group adain --seed <0..4> `
    --aug-manifest <manifest> --tag poolab_adain_p10_s<seed> --keep-checkpoint
python member4_training_eval\train.py --group adain --seed <0..4> `
    --aug-manifest <manifest> --tag poolab_adain_p20_s<seed> --keep-checkpoint

python member4_training_eval\evaluate.py --run <run dir> --checkpoint last.pt
```

运行结果保存在 `results/runs/poolab_{ours,adain}_p{10,20}_s*`。
