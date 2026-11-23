"""Fig. 2: the style-strength / cross-domain-accuracy curve.

This is the main figure of the report.  Its x axis is the average style
strength that a configuration applies, and its y axis is the cross-domain
accuracy it reaches.  Three series are drawn:

  * the spatial variant of this project (+Ours),
  * the uniform AdaIN variant (+AdaIN),
  * the strength-matched uniform AdaIN control, which receives exactly the
    same average strength as the spatial variant.

If the spatial *distribution* of the style mattered, the spatial series
would lie clearly above the two uniform series.  It does not, and the
figure makes that visible.

The curve is built from the runs that are actually on disk: every summary
file whose tag records an alpha is one point of one of the series.  The
sweep therefore needs no hard-coded table, and the figure follows the
experiments automatically.

The average strength of a spatial variant is not its $\\alpha_0$ but the mean
of its $\\alpha$ map, $\\alpha_0(1-\\beta\\overline{M})$; with the measured
$\\overline{M}$ of 0.361 and $\\beta = 0.7$ the mean of the $\\alpha_0 = 0.5$
map is 0.374.  The uniform series use their $\\alpha$ directly.

Usage:
    python plot_strength_curve.py --out-dir ../figures
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")

#: The three series of the figure: (tag prefix, display label, colour, marker).
SERIES = (
    ("ours", "+Ours (spatial)", "#009E73", "o"),
    ("adain", "+AdaIN (uniform)", "#D55E00", "s"),
    ("adain_matched", "+AdaIN (strength-matched)", "#E69F00", "^"),
)

#: The mean of the spatial alpha map, measured on the 1,349 training images.
#: It converts an alpha0 into the average strength that is actually applied.
SPATIAL_MEAN_FACTOR = 1.0 - 0.7 * 0.361

#: The main groups and the average strength they were trained at.  These are
#: the points of the curves that the sweeps do not cover.
MAIN_POINTS = {
    "ours": ("+Ours (spatial, main)", 0.5 * SPATIAL_MEAN_FACTOR),
    "adain": ("+AdaIN (uniform, main)", 0.5),
    "adain_matched": ("+AdaIN (strength-matched, main)", 0.5 * SPATIAL_MEAN_FACTOR),
}


def load_cross(tag: str, runs_root: Path):
    """The cross-domain average of one run tag, or None."""
    path = runs_root / f"summary_{tag}.json"
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
    per_seed = [float(np.mean([s[i] for s in series])) for i in range(seeds)]
    mean = float(np.mean(per_seed))
    std = float(np.std(per_seed, ddof=1)) if seeds > 1 else 0.0
    return mean, std


def sweep_alpha(tag: str):
    """The alpha0 of a sweep tag such as ``ours_a0p25_b0p7``, or None."""
    match = re.match(r"^(?:ours|adain|adain_matched)_a(\d+)p(\d+)_b(\d+)p(\d+)$", tag)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=config.PROJECT_ROOT / "figures")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    plotted_any = False
    for prefix, label, color, marker in SERIES:
        points = []
        # The sweep cells of this series.
        for path in sorted(args.runs_root.glob(f"summary_{prefix}_a*_b*.json")):
            tag = path.stem[len("summary_"):]
            alpha = sweep_alpha(tag)
            if alpha is None:
                continue
            result = load_cross(tag, args.runs_root)
            if result is None:
                continue
            mean, std = result
            # The spatial series applies the mean of its alpha map, the
            # uniform series apply their alpha directly.
            strength = (alpha * SPATIAL_MEAN_FACTOR if prefix == "ours" else alpha)
            points.append((strength, mean, std))
        # The main run of this series.
        if prefix in MAIN_POINTS:
            result = load_cross(prefix, args.runs_root)
            if result is not None:
                mean, std = result
                points.append((MAIN_POINTS[prefix][1], mean, std))
        if not points:
            continue
        plotted_any = True

        # Points that share an x value are averaged, so that a cell that was
        # trained twice does not distort the curve.
        grouped = {}
        for strength, mean, std in points:
            grouped.setdefault(round(strength, 4), []).append((mean, std))
        xs = sorted(grouped)
        ys = [float(np.mean([m for m, _ in grouped[x]])) for x in xs]
        es = [float(np.mean([s for _, s in grouped[x]])) for x in xs]

        ax.errorbar(xs, [y * 100 for y in ys], yerr=[e * 100 for e in es],
                    marker=marker, color=color, linewidth=1.7, markersize=6,
                    capsize=3, label=label, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3)

    if not plotted_any:
        print("no sweep results found; run the alpha sweep first")
        return

    ax.set_xlabel("Average style strength $\\alpha$")
    ax.set_ylabel("Cross-domain accuracy (%)")
    ax.set_title("Fig. 2  Cross-domain accuracy vs. style strength\n"
                 "(PACS Photo $\\rightarrow$ Art / Cartoon / Sketch)",
                 fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "strength_curve.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
