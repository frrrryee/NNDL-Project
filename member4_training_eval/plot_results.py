"""Plot the unified results bar chart for the report.

Usage:
    python plot_results.py --out ../figures/results_baseline.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")

#: The display order of the groups.  The two data-volume controls sit next to
#: the Original group, because that is the comparison they exist for.
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

# A colour-blind-safe palette (Okabe-Ito), extended with two neutral tones
# for the control groups so that they read as a family.
COLORS = {
    "original": "#0072B2",
    "duplicate": "#999999",
    "nonstyle": "#BBBBBB",
    "adain": "#D55E00",
    "adain_matched": "#E69F00",
    "ours": "#009E73",
    "ours_equal": "#56B4E9",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=config.PROJECT_ROOT / "figures" / "results_baseline.png")
    args = parser.parse_args()

    summaries = {}
    for path in sorted(args.runs_root.glob("summary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries[payload.get("group", path.stem)] = payload

    if not summaries:
        print("no summaries found")
        return

    # Keep the canonical order and drop the groups that have no summary yet.
    ordered = [g for g in GROUP_ORDER if g in summaries]
    ordered += [g for g in sorted(summaries) if g not in ordered]

    splits = ["photo_test"] + list(CROSS)
    names = [LABELS.get(g, g) for g in ordered]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    width = 0.82 / len(ordered)
    x = list(range(len(splits)))
    for i, group in enumerate(ordered):
        summary = summaries[group]
        offset = (i - (len(ordered) - 1) / 2) * width
        vals = [summary[split]["mean"] * 100 if split in summary else 0 for split in splits]
        errs = [summary[split]["std"] * 100 if split in summary else 0 for split in splits]
        ax.bar([xi + offset for xi in x], vals, width,
               yerr=errs, capsize=2.5, label=LABELS.get(group, group),
               color=COLORS.get(group, "#888888"), edgecolor="white", linewidth=0.5)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi + offset, v + e + 1.5, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=6.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Photo", "Art", "Cartoon", "Sketch"])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Evaluation domain")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=min(len(ordered), 4), loc="upper right",
              fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
