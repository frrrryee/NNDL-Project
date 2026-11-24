"""Fig. 6: the beta ablation, as three linked panels.

The modulation depth $\\beta$ is the second axis of the method, and it is
the axis on which the claim under test actually varies.  The sweep varies
$\\beta$ at a fixed nominal strength $\\alpha_0 = 0.5$ and trains, at every
level, both the spatial variant and the strength-matched uniform control.

Three quantities are shown, because a single one would be misleading:

  1. the effective strength $\\bar{\\alpha}$, which falls as $\\beta$ grows;
  2. the structural retention (EdgeScore) of the generated images;
  3. the cross-domain accuracy.

The third panel is the one that carries the conclusion: the spatial variant
and the matched control lie on top of each other at every level, so the
modulation contributes nothing beyond its effect on the total strength.

The strengths are the *measured* ones, read back from each cell's
``analysis.json``, not the nominal parameters.

Usage:
    python plot_beta_ablation.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")
CELL = re.compile(r"^summary_(?P<group>[a-z_]+?)_a(?P<alpha>[0-9p]+)_b(?P<beta>[0-9p]+)$")


def number(text: str) -> float:
    return float(text.replace("p", "."))


def load_cross(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = []
    for split in CROSS:
        entry = payload.get(split)
        if not entry or not entry.get("per_seed"):
            return None
        series.append(entry["per_seed"])
    length = min(len(s) for s in series)
    return [float(np.mean([s[i] for s in series])) for i in range(length)]


def collect(runs_root: Path) -> dict:
    """The beta cells on disk, by beta level."""
    cells: dict = {}
    for path in sorted(runs_root.glob("summary_*.json")):
        match = CELL.match(path.stem)
        if not match:
            continue
        beta = number(match.group("beta"))
        if abs(beta - 0.7) < 1e-9:
            # beta == 0.7 with a sweep tag belongs to the alpha sweep.
            continue
        group = match.group("group")
        values = load_cross(path)
        if values is None:
            continue
        cells.setdefault(beta, {})[group] = values
    return cells


def measured_strength(beta: float):
    """The measured mean strengths of one beta cell."""
    tag = f"a0p5_b{str(beta).replace('.', 'p')}"
    path = (config.DATA_ROOT / "aug_sp" / "sweeps" / "sweep_full" / tag /
            "analysis.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ours = ((payload.get("ours") or {}).get("mean_alpha") or {}).get("mean")
    matched = payload.get("matched_alpha_mean")
    if ours is None or matched is None:
        return None
    return float(ours), float(matched)


def edge_retention(beta: float, group: str = "ours"):
    """The mean EdgeScore of one group of one beta cell."""
    tag = f"a0p5_b{str(beta).replace('.', 'p')}"
    path = (config.DATA_ROOT / "aug_sp" / "sweeps" / "sweep_full" / tag /
            "analysis.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    block = payload.get(group) or {}
    value = block.get("edge_score")
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = collect(args.runs_root)
    if not cells:
        print("no beta cell has been trained yet")
        return
    levels = sorted(cells)
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))

    # Panel 1: the effective strength.
    axis = axes[0]
    for group, marker, colour in (("ours", "o", "#C44E52"),
                                  ("adain_matched", "s", "#4C72B0")):
        xs, ys = [], []
        for level in levels:
            record = cells[level].get(group)
            if record is None:
                continue
            measured = measured_strength(level)
            if measured is None:
                continue
            xs.append(level)
            ys.append(measured[0] if group == "ours" else measured[1])
        if xs:
            axis.plot(xs, ys, marker=marker, color=colour, label=group)
    axis.set_xlabel(r"$\beta$")
    axis.set_ylabel(r"measured $\bar{\alpha}$")
    axis.set_title("Effective strength", fontsize=9)
    axis.legend(fontsize=7, frameon=False)

    # Panel 2: the structural retention.
    axis = axes[1]
    for group, marker, colour, label in (
            ("ours", "o", "#C44E52", "+Ours (spatial)"),
            ("adain_matched", "s", "#4C72B0", "+AdaIN (matched)")):
        xs, ys = [], []
        for level in levels:
            value = edge_retention(level, group)
            if value is not None:
                xs.append(level)
                ys.append(value)
        if xs:
            axis.plot(xs, ys, marker=marker, color=colour, label=label)
    axis.set_xlabel(r"$\beta$")
    axis.set_ylabel("EdgeScore")
    axis.set_title("Structural retention", fontsize=9)
    axis.legend(fontsize=7, frameon=False)

    # Panel 3: the cross-domain accuracy.
    axis = axes[2]
    for group, marker, colour in (("ours", "o", "#C44E52"),
                                  ("adain_matched", "s", "#4C72B0")):
        xs, means, lows, highs = [], [], [], []
        for level in levels:
            values = cells[level].get(group)
            if values is None:
                continue
            array = np.asarray(values)
            xs.append(level)
            means.append(array.mean() * 100)
            if array.size > 1:
                generator = np.random.default_rng(0)
                boot = np.array([generator.choice(array, array.size, replace=True).mean()
                                 for _ in range(2000)])
                lows.append((array.mean() - np.percentile(boot, 2.5)) * 100)
                highs.append((np.percentile(boot, 97.5) - array.mean()) * 100)
            else:
                lows.append(0.0)
                highs.append(0.0)
        if xs:
            axis.errorbar(xs, means, yerr=[lows, highs], marker=marker,
                          color=colour, capsize=3, elinewidth=1,
                          label="+Ours (spatial)" if group == "ours"
                          else "+AdaIN (matched)")
    axis.set_xlabel(r"$\beta$")
    axis.set_ylabel("cross-domain accuracy (%)")
    axis.set_title("Classification", fontsize=9)
    axis.legend(fontsize=7, frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(r"The modulation depth $\beta$ changes the strength, "
                    r"not the outcome", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out_dir / "beta_ablation.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")
    for level in levels:
        measured = measured_strength(level)
        print(f"  beta={level:.2f}  strength="
              f"{measured if measured else 'n/a'}  groups="
              f"{sorted(cells[level])}")


if __name__ == "__main__":
    main()
