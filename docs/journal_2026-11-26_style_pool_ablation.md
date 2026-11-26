# 风格池规模消融：10 张 vs 20 张（`ours` 组）

**日期**：2026-11-26
**状态**：已完成，结果为**阴性**
**协议**：epoch-budgeted（1,290 步 / 2,698 图），final checkpoint，两臂完全一致

---

## 一句话结论

> **扩大风格池不改善跨域准确率。** 10 张 → 20 张，配对差 **−0.37 点**（t = −1.48，p = 0.21，
> bootstrap 95% CI [−0.84, +0.08]），不显著。三个目标域均无改善，Sketch 域差异为
> **−0.24 点**。

---

## 为什么做这个实验

项目中 AdaIN 与结构保持变体使用一个**冻结的 10 张外部画作**作为风格参考池。
一个自然的问题是：**池子更大是否会带来更好的跨域泛化？**

在此之前，项目里已有的"子池敏感性实验"（10 张切成两个不重叠的 5 张子池）显示两个半池
都低于全池（45.2 / 43.2 / 43.7），当时被解读为"风格多样性有价值"。这个实验是那个解读的
直接检验：把池子**扩大**到 20 张，看是否延续这条趋势。

上游 AdaIN demo 集共 20 张，项目用了其中 10 张。剩余 10 张全部纳入：

| 新增 | 说明 |
|---|---|
| `antimonocromatismo` | 超现实几何 |
| `en_campo_gris` | 风景油画 |
| `goeritz` | 超现实拼贴 |
| `impronte_d_artista` | 人物肖像 |
| `mondrian_cropped` | `mondrian` 的裁剪版 |
| `picasso_seated_nude_hr` | 人物肖像 |
| `sketch` | 黑白铅笔素描 |
| `trial` | 抽象 |
| `woman_in_peasant_dress` | 人物肖像 |
| `woman_in_peasant_dress_cropped` | 同上图的裁剪版 |

**泄漏检查**：20 张全部与本地 PACS 副本（9,991 张）字节不相等，由
`build_style_pool.py` 的 PACS 哈希比对保证。配对 seed 42。

| 池 | 风格数 | 池哈希 | 配对 seed |
|---|---|---|---|
| 冻结基线（项目主表使用） | 10 | `72cdbadb8970` | 42 |
| 消融对照 | 20 | `b6af7ba8659f` | 42 |

两臂的**训练配置指纹**相同（`3399564971c9`，即 epoch-budgeted 协议下的 `ours` 组），
因为风格池不进入训练配置，只进入生成的数据。

---

## 结果

| 种子 | p10（10 张） | p20（20 张） | 配对差 |
|---|---|---|---|
| 0 | 45.66 | 45.30 | −0.36 |
| 1 | 42.12 | 41.63 | −0.49 |
| 2 | 45.75 | 46.14 | +0.39 |
| 3 | 47.20 | 46.04 | −1.16 |
| 4 | 42.95 | 42.73 | −0.22 |
| **均值 ± std** | **44.74 ± 2.12** | **44.37 ± 2.06** | **−0.37** |

配对统计（n = 5，同种子对齐）：

| 量 | 值 |
|---|---|
| 均值差 | −0.37 点 |
| 配对 t | −1.480 |
| p（双尾） | 0.213 |
| bootstrap 95% CI | [−0.84, +0.08] |
| 判定 | **不显著**（临界值 \|t\| > 2.776） |

分域：

| 域 | p10 | p20 | 差 | t | 判定 |
|---|---|---|---|---|---|
| Art Painting | 67.93 | 67.93 | **+0.00** | 0.00 | 无差异 |
| Cartoon | 29.63 | 28.77 | −0.86 | −2.14 | 不显著 |
| **Sketch** | 36.64 | 36.40 | **−0.24** | −0.26 | 无差异 |

---

## 两个关键判断

### 1. `sketch.png` 没有造成泄漏

新池包含 `sketch.png`（黑白铅笔素描），而 PACS Sketch 是本项目的测试域之一，因此
"目标域外观进入训练集"的顾虑是合理的。

**判别性证据是 Sketch 域本身没有涨**：−0.24 点，t = −0.26。如果这张图真的在暗中
帮助 Sketch 域，该域必然出现明显上升。它没有。

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

`sketch.png` 与 PACS Sketch 的共同点只有"饱和度为 0"，而灰度均值差 0.223（PACS Sketch
接近纯白，`sketch.png` 是中灰）。这个共同点没有转化成任何 Sketch 域的收益。

### 2. 池子大小不是多样性的旋钮

这是本实验最重要的发现。

**每个训练样本只携带一个风格参考**（`AUG_PER_IMAGE = 1`）。因此把池子从 10 张扩到 20 张，
**并不改变单个样本所含的风格多样性**，只是把风格集合换成了另一套配置。

这解释了此前子池实验的观察：把池子切成两个半池时，每张原图仍然只配一个风格，但风格
**覆盖的统计量范围**变窄了；而扩到 20 张时覆盖范围换成了另一组，样本内的多样性没有增加。

**真正控制多样性的参数是每张原图所配的风格数 `AUG_PER_IMAGE`，而不是池子大小。**

---

## 限定：数量效应与配置效应在本实验中分不开

池子从 10 张变成 20 张时，内容→风格的配对会**整体重推**（配对规则是
`index % pool_size` 的循环分配）。实测在 seed 42 下，10 张池与 20 张池之间同一张原图
仍配到同一风格的只有 **203/1,349（15.0%）**。

因此本实验测的是"一套 10 张配置 vs 一套 20 张配置"，**不是**在固定配对下只增加风格数。
若结果有变化，数量效应与配置效应都无法单独归因。本实验的结果是"整体持平略负"，
两种解释都指向同一个结论（扩池不划算），所以这个限定不影响结论方向。

---

## 对项目的影响

- **主表不动。** 主结果表继续使用冻结的 10 张池，与项目现有全部结果保持可比。
- **本实验作为消融表存在**，结论为阴性但有价值：它排除了一个方向，并给出了更有价值的
  替代方向（`AUG_PER_IMAGE`）。
- **原始 10 张池未被修改。** 20 张池是独立目录
  `member4_training_eval/style_pool_pool20/`，通过 `NNDL_STYLE_POOL_DIR` 环境变量选择，
  默认行为不变。

---

## 复现

```powershell
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"

# 生成 20 张池的 ours 图像与 manifest
$env:NNDL_STYLE_POOL_DIR = "<repo>\member4_training_eval\style_pool_pool20"
# ... 运行成员 3 的生成器 ...

# 训练（两个臂都是 1,290 步）
python member4_training_eval\train.py --group ours --seed <0..4> \
    --aug-manifest <manifest> --tag poolab_ours_p20_s<seed> --keep-checkpoint
python member4_training_eval\evaluate.py --run <run dir> --checkpoint last.pt
```

运行结果保存在 `results/runs/poolab_ours_p10_s*` 与 `results/runs/poolab_ours_p20_s*`。
