"""Statistical comparison of the experimental groups.

The three (now four) groups are trained with the *same* seeds, so the design
is paired: for every seed the same initialisation and the same data order
are used, and only the augmentation set changes.  A paired test is therefore
the correct one and is strictly more powerful than an independent-samples
test, which matters a lot at n = 3 where the degrees of freedom are 2.

Reported for every pair of groups and every split:

  * the paired mean difference and its 95% confidence interval,
  * a paired t statistic and its two-sided p value,
  * a Wilcoxon signed-rank p value as a distribution-free cross-check,
  * the raw per-seed values, so that a gain that depends on a single
    favourable seed is visible instead of being hidden in the average.

A difference that does not reach significance is reported as such.  The
script never rounds an inconclusive result into a positive claim.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")
SPLITS = ("photo_test",) + CROSS

LABELS = {
    "original": "Original",
    "adain": "+AdaIN",
    "adain_matched": "+AdaIN (strength-matched)",
    "ours": "+Ours",
}
# The comparisons that carry the scientific argument, in reporting order.
PRIMARY_PAIRS = (
    ("original", "adain"),
    ("original", "ours"),
    ("adain", "ours"),
    ("adain_matched", "ours"),
    ("adain", "adain_matched"),
)


def load_group(runs_root: Path, group: str):
    """Return {split: [accuracy per seed]} for one group, or None if absent."""
    path = runs_root / f"summary_{group}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for split in SPLITS:
        entry = payload.get(split)
        if entry and entry.get("per_seed"):
            result[split] = list(entry["per_seed"])
    return result


def paired_test(a: list, b: list):
    """Paired comparison of ``b`` minus ``a`` over shared seeds."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t_stat, p_value = stats.ttest_rel(b, a)
        t_stat, p_value = float(t_stat), float(p_value)
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        ci = (mean - half, mean + half)
    else:
        # With identical values the difference is exactly zero: report it as
        # a deterministic tie rather than as an undefined statistic.
        t_stat, p_value = (0.0, 1.0) if mean == 0 else (float("inf"), 0.0)
        ci = (mean, mean)
    try:
        _, wilcoxon_p = stats.wilcoxon(b, a) if n >= 6 else (None, None)
    except ValueError:
        wilcoxon_p = None
    return {
        "n": n,
        "mean_diff": mean,
        "std_diff": sd,
        "ci95": [float(ci[0]), float(ci[1])],
        "t": t_stat,
        "p_paired_t": p_value,
        "p_wilcoxon": (float(wilcoxon_p) if wilcoxon_p is not None else None),
        "per_seed_diff": [float(x) for x in diff],
        "significant": bool(p_value < 0.05) if math.isfinite(p_value) else False,
    }


def cross_domain_means(group_results):
    """Average the three target domains per seed, so that the comparison is
    over the same unit (a seed) as every other statistic."""
    seeds = group_results.get("art_painting_test")
    if seeds is None:
        return None
    per_seed = []
    for index in range(len(seeds)):
        values = []
        for split in CROSS:
            series = group_results.get(split)
            if series is None or index >= len(series):
                values = None
                break
            values.append(series[index])
        if values is None:
            return None
        per_seed.append(float(np.mean(values)))
    return per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the full result as JSON to this path")
    parser.add_argument("--per-class", action="store_true",
                        help="also compare per-class accuracies")
    args = parser.parse_args()

    groups = {}
    for group in LABELS:
        data = load_group(args.runs_root, group)
        if data:
            groups[group] = data
    if not groups:
        print("no group summaries found; run run_all.py first")
        return

    # Attach the cross-domain average as another comparable series.
    for group, data in groups.items():
        means = cross_domain_means(data)
        if means is not None:
            data["cross_domain_avg"] = means

    report = {"groups": {g: {s: list(v) for s, v in d.items()} for g, d in groups.items()},
              "comparisons": []}

    print("=" * 78)
    print("HEADLINE: cross-domain average (Art / Cartoon / Sketch), per seed")
    print("=" * 78)
    for group in LABELS:
        if group not in groups:
            continue
        series = groups[group].get("cross_domain_avg")
        if not series:
            continue
        mean = float(np.mean(series))
        sd = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
        print(f"  {LABELS[group]:28s} {mean*100:6.2f} +/- {sd*100:4.2f}   "
              f"per seed: {[f'{v*100:.2f}' for v in series]}")

    print()
    print("=" * 78)
    print("PAIRED COMPARISONS (same seeds, so the design is paired)")
    print("=" * 78)
    pairs = [p for p in PRIMARY_PAIRS if p[0] in groups and p[1] in groups]
    for split in ("cross_domain_avg",) + SPLITS:
        for group_a, group_b in pairs:
            if split not in groups[group_a] or split not in groups[group_b]:
                continue
            a, b = groups[group_a][split], groups[group_b][split]
            if len(a) != len(b):
                continue
            result = paired_test(a, b)
            result["split"] = split
            result["group_a"] = group_a
            result["group_b"] = group_b
            report["comparisons"].append(result)
            flag = "*" if result["significant"] else " "
            ci = result["ci95"]
            print(f"  {split:20s} {LABELS[group_a]} -> {LABELS[group_b]:26s} "
                  f"{result['mean_diff']*100:+6.2f} pt "
                  f"[{ci[0]*100:+6.2f}, {ci[1]*100:+6.2f}] "
                  f"t={result['t']:+6.2f} p={result['p_paired_t']:.4f}{flag}")
        print()

    if args.per_class:
        print("=" * 78)
        print("PER-CLASS BREAKDOWN (cross-domain, averaged over the three targets)")
        print("=" * 78)
        per_class = {}
        for group in LABELS:
            path = args.runs_root / f"summary_{group}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            block = payload.get("per_class")
            if not block:
                continue
            per_class[group] = block
        if not per_class:
            print("  (no per-class data in the summaries)")
        else:
            classes = list(config.CLASSES)
            header = f"  {'class':<11}" + "".join(f"{LABELS[g][:14]:>16}" for g in per_class)
            print(header)
            for name in classes:
                row = f"  {name:<11}"
                for group, block in per_class.items():
                    values = []
                    for split in CROSS:
                        entry = (block.get(split) or {}).get(name)
                        if entry:
                            values.append(entry["mean"])
                    row += f"{(np.mean(values)*100 if values else float('nan')):>16.1f}"
                print(row)
            report["per_class"] = per_class

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    print()
    print("Note: '*' marks p < 0.05 in the paired t test with n-1 degrees of")
    print("freedom.  At n = 3 the critical value is |t| > 4.303, so only a")
    print("difference larger than about two seed standard deviations can reach")
    print("significance.  Differences that do not reach it are reported as")
    print("inconclusive rather than as an improvement.")


if __name__ == "__main__":
    main()
