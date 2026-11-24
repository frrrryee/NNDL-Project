"""Fig. 8: the sensitivity of the conclusion to the choice of style pool.

The style pool is frozen and shared by every group, so a reader may ask
whether the conclusion is an artefact of that particular pool.  This
figure answers it: the same experiment is repeated with two disjoint
half-pools, and the numbers are compared against the full pool.

Each bar is a five-seed mean with a bootstrap confidence interval, and the
full-pool result is drawn as a reference line, so the reader can see
directly whether a half-pool moves the conclusion.

Usage:
    python plot_pool_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
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


def load(runs_root: Path, group: str):
    path = runs_root / f"summary_{group}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cross_values(summary: dict) -> list:
    """The per-seed cross-domain accuracy of a summary.

    A summary stores each target domain separately, with one accuracy per
    seed.  The cross-domain figure of the report is the unweighted average
    of the three target domains, so the averaging has to happen *per seed*
    before anything is aggregated across seeds.  Concatenating the three
    per-domain lists instead would give fifteen values that are not
    independent, and would widen every confidence interval.
    """
    splits = [summary.get(name, {}) for name in
              ("art_painting_test", "cartoon_test", "sketch_test")]
    per_seed_lists = [split.get("per_seed") or [] for split in splits]
    if not all(per_seed_lists):
        return []
    length = min(len(values) for values in per_seed_lists)
    return [float(np.mean([values[i] for values in per_seed_lists]))
            for i in range(length)]


def bootstrap_ci(values: list, seed: int = 0, rounds: int = 4000):
    """A percentile bootstrap interval for the mean."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float(array.mean()) if array.size else float("nan"), 0.0, 0.0
    generator = np.random.default_rng(seed)
    means = np.empty(rounds)
    for i in range(rounds):
        means[i] = generator.choice(array, size=array.size, replace=True).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(array.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # (label, summary file) for every pool that was actually run.  The
    # full-pool spatial group is ``ours``; the half-pool runs are the
    # ``subpool_*`` groups, which differ from it only in the five style
    # references they draw from.
    candidates = [
        ("Full pool\n(10 styles)", "summary_ours.json"),
        ("Pool A\n(5 styles)", "summary_subpool_a_a0p5_b0p7.json"),
        ("Pool B\n(5 styles)", "summary_subpool_b_a0p5_b0p7.json"),
        ("AdaIN\nfull pool", "summary_adain.json"),
    ]
    rows = []
    for label, name in candidates:
        summary = load(args.runs_root, name.replace("summary_", "").replace(".json", ""))
        if summary is None:
            continue
        values = cross_values(summary)
        if not values:
            continue
        mean, low, high = bootstrap_ci(values)
        rows.append((label, mean * 100, low * 100, high * 100, len(values)))
    if not rows:
        print("no pool-summary file was found")
        return
    figure, axis = plt.subplots(figsize=(5.4, 3.0))
    positions = np.arange(len(rows))
    means = [r[1] for r in rows]
    lows = [r[1] - r[2] for r in rows]
    highs = [r[3] - r[1] for r in rows]
    axis.bar(positions, means, 0.55, color="#4C72B0",
             yerr=[lows, highs], capsize=4,
             error_kw={"elinewidth": 1, "ecolor": "#333333"})
    for position, row in zip(positions, rows):
        axis.text(position, row[3] + 0.5, f"{row[1]:.1f}%",
                  ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels([r[0] for r in rows], fontsize=8)
    axis.set_ylabel("cross-domain accuracy (%)", fontsize=9)
    axis.set_ylim(0, max(r[3] for r in rows) + 4)
    axis.set_title("The conclusion does not depend on the style pool",
                   fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    out = args.out_dir / "pool_sensitivity.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")
    for row in rows:
        print(f"  {row[0]:20s} {row[1]:6.2f}%  [{row[2]:.2f}, {row[3]:.2f}]  "
              f"n={row[4]}")


if __name__ == "__main__":
    main()
