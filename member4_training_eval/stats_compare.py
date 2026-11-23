"""Statistical comparison of the experimental groups.

All groups are trained with the *same* seeds, so the design is paired: for
every seed the same initialisation and the same data order are used, and
only the augmentation set changes.  A paired test is therefore the correct
one and is strictly more powerful than an independent-samples test, which
matters a lot at small n where the degrees of freedom are few.

Reported for every pair of groups and every split:

  * the paired mean difference and its 95% confidence interval,
  * a bootstrap confidence interval, which does not assume normality and is
    therefore more trustworthy than the t interval at n = 3 or n = 5;
  * a paired t statistic and its two-sided p value,
  * a Benjamini-Hochberg FDR-corrected p value, because the report compares
    many groups at once and an uncorrected threshold inflates the false
    positive rate;
  * Cohen's d (the paired version) as an effect size, so that a difference
    can be judged by its magnitude and not only by its p value;
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
    "ours_equal": "+Ours (equal strength)",
    "duplicate": "+Duplicate (data-volume control)",
    "nonstyle": "+NonStyle (appearance control)",
}

# The comparisons that carry the scientific argument, in reporting order.
PRIMARY_PAIRS = (
    # the main claim
    ("original", "adain"),
    ("original", "ours"),
    # the strength-matched controls
    ("adain", "adain_matched"),
    ("adain_matched", "ours"),
    ("adain", "ours"),
    ("adain", "ours_equal"),
    ("ours_equal", "adain_matched"),
    # the decomposition of where the gain comes from
    ("original", "duplicate"),
    ("duplicate", "nonstyle"),
    ("nonstyle", "adain"),
    ("original", "nonstyle"),
    ("duplicate", "adain"),
)

#: How many bootstrap resamples to draw for the confidence interval.
BOOTSTRAP_RESAMPLES = 10000
#: The seed of the bootstrap generator, so that the interval is reproducible.
BOOTSTRAP_SEED = 20240601


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


def bootstrap_ci(diff: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES,
                 alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for the mean difference.

    The paired differences are resampled with replacement, which is the
    non-parametric counterpart of the paired t test and does not assume
    that the differences are normally distributed.
    """
    n = len(diff)
    if n < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(diff, size=(resamples, n), replace=True).mean(axis=1)
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def cohens_d_paired(diff: np.ndarray) -> float:
    """The paired version of Cohen's d: mean difference over its own spread.

    For a paired design the relevant standard deviation is that of the
    differences, not the pooled one of the two samples.
    """
    n = len(diff)
    if n < 2:
        return 0.0
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


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
    boot = bootstrap_ci(diff)
    return {
        "n": n,
        "mean_diff": mean,
        "std_diff": sd,
        "ci95_t": [float(ci[0]), float(ci[1])],
        "ci95_bootstrap": ([float(boot[0]), float(boot[1])] if boot else None),
        "t": t_stat,
        "p_paired_t": p_value,
        "p_wilcoxon": (float(wilcoxon_p) if wilcoxon_p is not None else None),
        "cohens_d_paired": cohens_d_paired(diff),
        "per_seed_diff": [float(x) for x in diff],
        "significant": bool(p_value < 0.05) if math.isfinite(p_value) else False,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Returns the adjusted p value of every entry, in the order the entries
    were given.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    previous = 1.0
    for rank, index in enumerate(reversed(order), start=1):
        position = n - rank + 1
        value = p_values[index] * n / position
        previous = min(previous, value)
        adjusted[index] = min(1.0, previous)
    return adjusted


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

    # The FDR correction is applied across all the comparisons that were
    # actually performed, which is the family of hypotheses the report makes.
    p_values = [c["p_paired_t"] for c in report["comparisons"]
                if math.isfinite(c["p_paired_t"])]
    adjusted = benjamini_hochberg(p_values)
    index = 0
    for comparison in report["comparisons"]:
        if math.isfinite(comparison["p_paired_t"]):
            comparison["p_fdr_bh"] = adjusted[index]
            comparison["significant_fdr"] = bool(adjusted[index] < 0.05)
            index += 1
        else:
            comparison["p_fdr_bh"] = None
            comparison["significant_fdr"] = False

    for split in ("cross_domain_avg",) + SPLITS:
        shown = False
        for comparison in report["comparisons"]:
            if comparison["split"] != split:
                continue
            if not shown:
                print(f"--- {split} ---")
                shown = True
            group_a = comparison["group_a"]
            group_b = comparison["group_b"]
            flag = "*" if comparison["significant"] else " "
            ci = comparison["ci95_t"]
            boot = comparison["ci95_bootstrap"]
            boot_text = f" boot[{boot[0]*100:+6.2f},{boot[1]*100:+6.2f}]" if boot else ""
            print(f"  {LABELS[group_a]} -> {LABELS[group_b]:26s} "
                  f"{comparison['mean_diff']*100:+6.2f} pt "
                  f"[{ci[0]*100:+6.2f}, {ci[1]*100:+6.2f}]{boot_text} "
                  f"t={comparison['t']:+6.2f} p={comparison['p_paired_t']:.4f}"
                  f"{flag} d={comparison['cohens_d_paired']:+.2f} "
                  f"p_fdr={comparison['p_fdr_bh']:.4f}")
        if shown:
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
    n_seeds = max((c["n"] for c in report["comparisons"]), default=0)
    critical = stats.t.ppf(0.975, max(n_seeds - 1, 1)) if n_seeds else float("nan")
    print(f"Note: '*' marks p < 0.05 in the paired t test with n-1 degrees of")
    print(f"freedom.  At n = {n_seeds} the critical value is |t| > {critical:.3f}, so "
          f"only a difference larger than about")
    print("two seed standard deviations can reach significance.  Differences that do")
    print("not reach it are reported as inconclusive rather than as an improvement.")
    print("'p_fdr' is the Benjamini-Hochberg adjusted p value over all the")
    print("comparisons that were performed.")


if __name__ == "__main__":
    main()
