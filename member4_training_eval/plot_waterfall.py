"""The waterfall of the gain decomposition (the graphic version of Table VI).

The waterfall starts at the Original group's cross-domain accuracy and adds
the four contributions one after another, so that the reader sees the size
of each factor directly instead of having to add up the differences of a
table.

Every value is read from the recorded summaries.

Usage:
    python plot_waterfall.py --out-dir ../figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")

#: The steps of the waterfall: (label, group a, group b, colour).
STEPS = (
    ("Pure data\nvolume", "original", "duplicate", "#999999"),
    ("Generic appearance\ndiversity", "duplicate", "nonstyle", "#BBBBBB"),
    ("Style itself\n(upper bound)", "nonstyle", "adain", "#009E73"),
    ("Spatial\nmodulation", "adain_matched", "ours", "#D55E00"),
)


def load_cross(runs_root: Path, group: str):
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
    return float(np.mean([float(np.mean([s[i] for s in series]))
                           for i in range(seeds)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=config.PROJECT_ROOT / "figures")
    args = parser.parse_args()

    base = load_cross(args.runs_root, "original")
    if base is None:
        print("waterfall: no 'original' summary")
        return

    steps = []
    for label, a, b, color in STEPS:
        level_a = load_cross(args.runs_root, a)
        level_b = load_cross(args.runs_root, b)
        if level_a is None or level_b is None:
            continue
        steps.append((label, level_b - level_a, color, level_a, level_b))
    if not steps:
        print("waterfall: no comparable groups")
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    running = base
    xs = ["Original"]
    ys = [running * 100]
    bottoms = [0.0]
    heights = [running * 100]
    colors = ["#0072B2"]
    for label, delta, color, _a, _b in steps:
        xs.append(label)
        bottoms.append(running * 100)
        heights.append(delta * 100)
        colors.append(color)
        running += delta

    bars = ax.bar(range(len(xs)), heights, bottom=bottoms, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.62)
    for index, (bar, height, bottom) in enumerate(zip(bars, heights, bottoms)):
        top = bottom + height
        label = f"{height:+.2f}" if index else f"{height:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, top + (0.35 if height >= 0 else -0.35),
                label, ha="center", va="bottom" if height >= 0 else "top",
                fontsize=8.5)

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=8.5)
    ax.set_ylabel("Cross-domain accuracy (%)")
    ax.set_title("Decomposition of the gain\n"
                 "(each bar adds its factor to the level reached by the bars "
                 "to its left)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "gain_waterfall.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
