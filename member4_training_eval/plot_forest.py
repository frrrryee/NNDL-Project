"""Fig. 3: forest plot of the paired differences, and Fig. 4: the EdgeScore CDF.

Fig. 3 shows, for every comparison that carries the argument of the
report, the paired mean difference in cross-domain accuracy together with
its 95% confidence interval.  The differences are *paired* over the seeds,
because every group was trained with the same seeds; the interval therefore
describes the difference between two methods under one controlled protocol,
not two independent samples.

Fig. 4 shows the cumulative distribution of the per-image EdgeScore
differences.  It is the paired counterpart of the forest plot: instead of
one number per group it shows how many of the 1,349 images improve and how
many get worse, which is the level at which the structure metric is
actually defined.

Every number is read from ``runs/``; nothing is hard-coded.

Usage:
    python plot_forest.py --out-dir ../figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")

#: The comparisons of the forest plot, in reporting order.  Each entry is
#: (group a, group b, the sentence the comparison supports).
COMPARISONS = (
    ("original", "adain", "style augmentation, data volume not controlled"),
    ("original", "duplicate", "pure data volume (twice the rows, same images)"),
    ("duplicate", "nonstyle", "generic appearance diversity"),
    ("nonstyle", "adain", "style beyond a generic appearance change"),
    ("adain", "adain_matched", "uniform AdaIN at lower strength"),
    ("adain_matched", "ours", "spatial modulation at equal mean strength"),
    ("adain", "ours", "spatial modulation, strengths not matched"),
    ("original", "ours_equal", "style augmentation at matched strength"),
)

LABELS = {
    "original": "Original",
    "adain": "+AdaIN",
    "adain_matched": "+AdaIN (matched)",
    "ours": "+Ours",
    "ours_equal": "+Ours (equal str.)",
    "duplicate": "+Duplicate",
    "nonstyle": "+NonStyle",
}

# A colour-blind-safe palette (Okabe-Ito); the control groups are neutral.
COLORS = {
    "original": "#0072B2",
    "duplicate": "#999999",
    "nonstyle": "#BBBBBB",
    "adain": "#D55E00",
    "adain_matched": "#E69F00",
    "ours": "#009E73",
    "ours_equal": "#56B4E9",
}


def load_cross(runs_root: Path, group: str):
    """The cross-domain average per seed of one group, or None."""
    path = runs_root / f"summary_{group}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = []
    for split in CROSS:
        entry = payload.get(split)
        if not entry or not entry.get("per_seed"):
            return None
        series.append(entry["per_seed"])
    seeds = min(len(s) for s in series)
    return [float(np.mean([s[i] for s in series])) for i in range(seeds)]


def paired(a: list, b: list):
    """The paired difference b - a with its 95% confidence interval."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    diff = y - x
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t, p = stats.ttest_rel(y, x)
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    else:
        t, p = (0.0, 1.0), 1.0
        half = 0.0
    return mean, mean - half, mean + half, float(t), float(p), n


def common_seeds(runs_root: Path, groups: tuple) -> dict:
    """The per-seed cross-domain averages, restricted to the shared seeds.

    While the seed extension is still running, some groups have five seeds
    and others three.  A paired comparison is only valid over the seeds the
    two groups have in common, so the series are truncated to the common
    prefix.
    """
    raw = {g: load_cross(runs_root, g) for g in groups}
    raw = {g: v for g, v in raw.items() if v is not None}
    if not raw:
        return {}
    length = min(len(v) for v in raw.values())
    return {g: v[:length] for g, v in raw.items()}


def forest(runs_root: Path, out: Path) -> None:
    """Fig. 3: the paired differences with their confidence intervals."""
    rows = []
    all_groups = tuple(sorted({g for a, b, _ in COMPARISONS for g in (a, b)}))
    series = common_seeds(runs_root, all_groups)
    for group_a, group_b, note in COMPARISONS:
        if group_a not in series or group_b not in series:
            continue
        mean, lo, hi, t, p, n = paired(series[group_a], series[group_b])
        rows.append((group_a, group_b, note, mean, lo, hi, t, p, n))

    if not rows:
        print("forest plot: no comparable groups")
        return

    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(rows) + 1.6))
    ys = np.arange(len(rows))[::-1]
    for y, (group_a, group_b, note, mean, lo, hi, t, p, n) in zip(ys, rows):
        color = COLORS.get(group_b, "#888888")
        significant = p < 0.05
        ax.plot([lo * 100, hi * 100], [y, y], color=color, linewidth=1.8,
                solid_capstyle="butt", alpha=0.95 if significant else 0.5,
                zorder=2)
        ax.plot([mean * 100], [y], marker="D" if significant else "o",
                color=color, markersize=7 if significant else 5.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(hi * 100 + 0.25, y, f"{mean*100:+.2f}  p={p:.3f}",
                va="center", ha="left", fontsize=8,
                color="#222222" if significant else "#777777")

    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LABELS.get(a, a)} → {LABELS.get(b, b)}" for a, b, *_ in rows],
                       fontsize=9)
    ax.set_xlabel("Paired difference in cross-domain accuracy (percentage points)")
    ax.set_title("Fig. 3  Paired differences over the shared seeds\n"
                 f"(n = {rows[0][8]} seeds; filled marker = p < 0.05)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def edge_cdf(runs_root: Path, out: Path) -> None:
    """Fig. 4: the cumulative distribution of the per-image EdgeScore difference."""
    npz = runs_root / "edge_analysis.npz"
    if not npz.is_file():
        print("edge CDF: runs/edge_analysis.npz is missing; "
              "run structure_mechanism.py first")
        return
    payload = np.load(npz)
    ours = payload["ours"]
    matched = payload["adain"]
    plain = payload["adain"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for values, label, color in ((ours - matched, "+Ours − +AdaIN (matched)", "#009E73"),
                                 (ours - plain, "+Ours − +AdaIN (uniform)", "#D55E00")):
        clean = values[~np.isnan(values)]
        if not len(clean):
            continue
        ordered = np.sort(clean)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cumulative, color=color, linewidth=1.8, label=label)

    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Per-image EdgeScore difference")
    ax.set_ylabel("Cumulative fraction of the 1,349 training images")
    ax.set_title("Fig. 4  Per-image EdgeScore difference\n"
                 "(a curve far from the diagonal means little image benefits)",
                 fontsize=10)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=config.PROJECT_ROOT / "figures")
    args = parser.parse_args()

    forest(args.runs_root, args.out_dir / "forest_plot.png")
    edge_cdf(args.runs_root, args.out_dir / "edge_cdf.png")


if __name__ == "__main__":
    main()
