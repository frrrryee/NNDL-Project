"""Fig. 9: the per-class accuracy heat map and the per-class gain bar chart.

The question this figure answers is: does the style augmentation help every
class equally, or only the classes whose identity is carried by a thin
contour?  The answer changes the interpretation of the whole result -- a
uniform gain points to a regularisation effect, a gain that concentrates on
a few classes points to a specific missing cue.

The heat map has one row per experimental group and one column per class.
Every cell shows the cross-domain accuracy of that class, averaged over the
three target domains and over the seeds.  The number of test images per
class is printed under the column label, because the classes are very
unequal in size and a small class produces a noisy number.

Every value is read from the ``per_class`` block of the summaries.

Usage:
    python plot_per_class.py --out-dir ../figures
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

GROUP_ORDER = ("original", "duplicate", "nonstyle", "adain", "adain_matched",
               "ours", "ours_equal")

LABELS = {
    "original": "Original",
    "adain": "+AdaIN",
    "adain_matched": "+AdaIN (matched)",
    "ours": "+Ours",
    "ours_equal": "+Ours (equal str.)",
    "duplicate": "+Duplicate",
    "nonstyle": "+NonStyle",
}


def load(runs_root: Path):
    """{group: {class: cross-domain accuracy}} and {class: test sample count}."""
    table = {}
    counts = {}
    for group in GROUP_ORDER:
        path = runs_root / f"summary_{group}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = payload.get("per_class")
        if not block:
            continue
        row = {}
        for name in config.CLASSES:
            values = []
            for split in CROSS:
                entry = (block.get(split) or {}).get(name)
                if entry and entry.get("mean") is not None:
                    values.append(entry["mean"])
            if values:
                row[name] = float(np.mean(values))
        if row:
            table[group] = row
            for split in CROSS:
                entry = (block.get(split) or {}).get(name)
                del entry
            # The sample counts come from the recorded per-class blocks.
            for split in CROSS:
                sub = block.get(split) or {}
                for name, entry in sub.items():
                    if name in config.CLASSES and entry.get("count"):
                        counts.setdefault(name, 0)
                        counts[name] += int(entry["count"])
    return table, counts


def heatmap(table: dict, counts: dict, out: Path) -> None:
    """The heat map of the per-class cross-domain accuracy."""
    groups = [g for g in GROUP_ORDER if g in table]
    classes = [c for c in config.CLASSES if any(c in table[g] for g in groups)]
    if not groups or not classes:
        print("per-class heat map: no per-class data in the summaries")
        return

    values = np.array([[table[g].get(c, np.nan) * 100 for c in classes]
                       for g in groups], dtype=float)

    fig, ax = plt.subplots(figsize=(1.55 * len(classes) + 3.2,
                                    0.52 * len(groups) + 2.4))
    image = ax.imshow(values, cmap="viridis", aspect="auto",
                      vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(
        [f"{c}\n(n={counts.get(c, 0)})" if counts.get(c) else c for c in classes],
        fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([LABELS.get(g, g) for g in groups], fontsize=9)
    ax.set_title("Fig. 9  Per-class cross-domain accuracy (%)\n"
                 "(averaged over Art Painting, Cartoon and Sketch)",
                 fontsize=10)

    threshold = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    fig.colorbar(image, ax=ax, shrink=0.85, label="Top-1 accuracy (%)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def gains(table: dict, out: Path) -> None:
    """The per-class gain of the style augmentation over the Original."""
    if "original" not in table or "adain" not in table:
        print("per-class gains: need the 'original' and 'adain' groups")
        return
    classes = [c for c in config.CLASSES if c in table["original"]]
    deltas = {c: (table["adain"].get(c, np.nan) - table["original"].get(c, np.nan)) * 100
              for c in classes}

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = ["#009E73" if v >= 0 else "#D55E00" for v in deltas.values()]
    ax.bar(range(len(classes)), [deltas[c] for c in classes], color=colors,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("Accuracy gain (percentage points)")
    ax.set_title("Per-class gain of +AdaIN over Original\n"
                 "(cross-domain average)", fontsize=10)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for index, c in enumerate(classes):
        ax.text(index, deltas[c], f"{deltas[c]:+.1f}", ha="center",
                va="bottom" if deltas[c] >= 0 else "top", fontsize=8)
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

    table, counts = load(args.runs_root)
    heatmap(table, counts, args.out_dir / "per_class_heatmap.png")
    gains(table, args.out_dir / "per_class_gains.png")


if __name__ == "__main__":
    main()
