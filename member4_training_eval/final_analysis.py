"""Consolidated final analysis for all experimental groups.

Produces the numbers that go into the report:

  1. the headline table (per-domain, per-seed, mean +/- std);
  2. the paired significance table with 95% confidence intervals;
  3. the style-strength / accuracy relationship, which is the key finding;
  4. the per-class breakdown;
  5. the structural-mechanism analysis (edge retention vs classification).

Every number is read from the recorded run artefacts; nothing is re-typed.
The number of seeds is read from the summaries themselves, so the critical
values and the degrees of freedom are always consistent with the data.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT
CROSS = ("art_painting_test", "cartoon_test", "sketch_test")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")

LABELS = {
    "original": "Original",
    "adain": "+AdaIN (uniform 0.5)",
    "adain_matched": "+AdaIN (strength-matched 0.374)",
    "ours": "+Ours (spatial, mean 0.374)",
    "ours_equal": "+Ours (spatial, mean 0.500)",
    "duplicate": "+Duplicate (data-volume control)",
    "nonstyle": "+NonStyle (appearance control)",
}
ORDER = ("original", "duplicate", "nonstyle", "adain", "adain_matched",
         "ours", "ours_equal")

#: The average style strength of every group, used for the strength/accuracy
#: relation.  The control groups have no style at all.
STRENGTH = {"original": 0.0, "adain": 0.5, "adain_matched": 0.3736,
            "ours": 0.3736, "ours_equal": 0.5, "duplicate": 0.0,
            "nonstyle": 0.0}


def load(group):
    path = RUNS / f"summary_{group}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not all(split in payload for split in CROSS):
        return None
    payload["_cross_per_seed"] = [
        (a + b + c) / 3 for a, b, c in zip(
            payload["art_painting_test"]["per_seed"],
            payload["cartoon_test"]["per_seed"],
            payload["sketch_test"]["per_seed"])]
    return payload


groups = {name: load(name) for name in ORDER}
groups = {k: v for k, v in groups.items() if v is not None}

# The seed count is taken from the data, so the critical value printed in the
# header always matches the number of paired observations.
n_seeds = max((len(v["_cross_per_seed"]) for v in groups.values()), default=0)
df = max(n_seeds - 1, 1)
crit = stats.t.ppf(0.975, df)

print("=" * 100)
print(f"TABLE 1  Headline results (mean +/- std over {n_seeds} seeds; "
      f"final checkpoint, 660 steps)")
print("=" * 100)
header = f"  {'group':<30}{'Photo':>9}{'Art':>9}{'Cartoon':>9}{'Sketch':>9}{'CrossAvg':>11}"
print(header)
print("  " + "-" * 96)
for name in ORDER:
    if name not in groups:
        continue
    s = groups[name]

    def cell(split):
        e = s.get(split)
        return f"{e['mean']*100:.1f}±{e['std']*100:.1f}" if e else "--"

    cross = np.mean(s["_cross_per_seed"])
    cross_std = np.std(s["_cross_per_seed"], ddof=1) if len(s["_cross_per_seed"]) > 1 else 0.0
    print(f"  {LABELS[name]:<30}{cell('photo_test'):>9}{cell('art_painting_test'):>9}"
          f"{cell('cartoon_test'):>9}{cell('sketch_test'):>9}"
          f"{cross*100:>6.2f}±{cross_std*100:.1f}")

print()
print("=" * 100)
print(f"TABLE 2  Paired comparisons (same seeds -> paired t test, "
      f"n={n_seeds}, df={df}, crit |t|={crit:.3f})")
print("=" * 100)
PAIRS = [("original", "adain"), ("original", "ours"), ("adain", "adain_matched"),
         ("adain", "ours"), ("adain_matched", "ours"),
         ("adain", "ours_equal"), ("ours_equal", "adain_matched"),
         # the data-volume decomposition (Table VI)
         ("original", "duplicate"), ("duplicate", "nonstyle"),
         ("nonstyle", "adain"), ("original", "nonstyle"),
         ("duplicate", "adain")]
for a, b in PAIRS:
    if a not in groups or b not in groups:
        continue
    x = np.asarray(groups[a]["_cross_per_seed"])
    y = np.asarray(groups[b]["_cross_per_seed"])
    if len(x) != len(y):
        continue
    diff = y - x
    t, p = stats.ttest_rel(y, x)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
    flag = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"  {LABELS[a]:<30} -> {LABELS[b]:<30} {diff.mean()*100:+6.2f} pt "
          f"[{(diff.mean()-half)*100:+6.2f},{(diff.mean()+half)*100:+6.2f}]  "
          f"t={t:+6.2f} p={p:.4f}  {flag}")

print()
print("=" * 100)
print("TABLE 3  Style strength vs cross-domain accuracy (the central finding)")
print("=" * 100)
rows = []
for name in ORDER:
    if name not in groups:
        continue
    s = groups[name]
    rows.append((STRENGTH[name], name, np.mean(s["_cross_per_seed"]),
                 np.std(s["_cross_per_seed"], ddof=1) if len(s["_cross_per_seed"]) > 1 else 0.0))
for strength, name, mean, std in sorted(rows):
    if strength <= 0:
        continue
    print(f"  strength {strength:.4f}  {LABELS[name]:<30} {mean*100:6.2f} "
          f"± {std*100:.2f}")

print()
print("=" * 100)
print("TABLE 4  Where the gain comes from (the decomposition of Table VI)")
print("=" * 100)


def cross_mean(name):
    return float(np.mean(groups[name]["_cross_per_seed"]))


def paired(a, b):
    x = np.asarray(groups[a]["_cross_per_seed"])
    y = np.asarray(groups[b]["_cross_per_seed"])
    diff = y - x
    t, p = stats.ttest_rel(y, x)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
    return diff.mean(), (diff.mean() - half), (diff.mean() + half), t, p


DECOMPOSITION = [
    ("Pure data volume", "original", "duplicate",
     "twice the rows, same distinct images, no appearance change"),
    ("Generic appearance diversity", "duplicate", "nonstyle",
     "a non-stylistic transform on top of the duplicate"),
    ("Style itself (upper bound)", "nonstyle", "adain",
     "style over a generic appearance change; an upper bound, not a strict "
     "attribution"),
]
for label, a, b, note in DECOMPOSITION:
    if a not in groups or b not in groups:
        continue
    mean, lo, hi, t, p = paired(a, b)
    flag = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"  {label:<32} {LABELS[a]} -> {LABELS[b]}")
    print(f"      {mean*100:+6.2f} pt  95% CI [{lo*100:+6.2f}, {hi*100:+6.2f}]  "
          f"t={t:+6.2f} p={p:.4f}  {flag}")
    print(f"      ({note})")

if "adain_matched" in groups and "ours" in groups:
    mean, lo, hi, t, p = paired("adain_matched", "ours")
    flag = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"  {'Spatial modulation':<32} {LABELS['adain_matched']} -> {LABELS['ours']}")
    print(f"      {mean*100:+6.2f} pt  95% CI [{lo*100:+6.2f}, {hi*100:+6.2f}]  "
          f"t={t:+6.2f} p={p:.4f}  {flag}")
    print("      (the effect of the spatial distribution at equal mean strength)")

print()
print("=" * 100)
print("TABLE 5  Per-class breakdown (cross-domain accuracy per class)")
print("=" * 100)


def per_class_cross(group):
    """Mean cross-domain accuracy per class, averaged over the seeds."""
    payload = groups[group]
    totals = {name: [] for name in CLASSES}
    for split in CROSS:
        block = (payload.get("per_class") or {}).get(split)
        if not block:
            return None
        for name in CLASSES:
            entry = block.get(name)
            if entry is None:
                return None
            totals[name].append(entry["mean"])
    return {name: sum(values) / len(values) for name, values in totals.items()}


baseline = per_class_cross("original") if "original" in groups else None
header = f"  {'group':<30}" + "".join(f"{c[:7]:>9}" for c in CLASSES)
print(header)
for name in ORDER:
    if name not in groups:
        continue
    values = per_class_cross(name)
    if values is None:
        continue
    print(f"  {LABELS[name]:<30}" + "".join(f"{values[c]*100:>9.1f}" for c in CLASSES))

if baseline:
    print()
    print("  gains over Original:")
    for name in ORDER:
        if name not in groups or name == "original":
            continue
        values = per_class_cross(name)
        if values is None:
            continue
        gains = {c: values[c] - baseline[c] for c in CLASSES}
        best = max(gains, key=gains.get)
        worst = min(gains, key=gains.get)
        print(f"  {LABELS[name]:<30}" + "".join(f"{gains[c]*100:>+9.1f}" for c in CLASSES)
              + f"   best={best} worst={worst}")

print()
print("=" * 100)
print("TABLE 7  Style strength vs. cross-domain accuracy (every trained variant)")
print("=" * 100)
# Every summary that was written by a sweep carries a tag of the form
# ``<kind>_a<alpha>_b<beta>``; the average style strength of a variant is
# its alpha, and the sweep groups them by that value.
variants = []
for path in sorted(RUNS.glob("summary_*.json")):
    tag = path.stem[len("summary_"):]
    if tag in groups:
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not all(split in payload for split in CROSS):
        continue
    per_seed = [(a + b + c) / 3 for a, b, c in zip(
        payload["art_painting_test"]["per_seed"],
        payload["cartoon_test"]["per_seed"],
        payload["sketch_test"]["per_seed"])]
    alpha = None
    match = re.search(r"_a(\d+)p(\d+)_b(\d+)p(\d+)$", tag)
    if match:
        alpha = float(f"{match.group(1)}.{match.group(2)}")
    variants.append((tag, alpha, float(np.mean(per_seed)),
                     float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0))

if variants:
    print(f"  {'variant':<32}{'alpha':>8}{'CrossAvg':>10}{'std':>8}")
    for tag, alpha, mean, std in sorted(variants, key=lambda v: (v[1] is None, v[1] or 0, v[0])):
        alpha_text = f"{alpha:.2f}" if alpha is not None else "--"
        print(f"  {tag:<32}{alpha_text:>8}{mean*100:>10.2f}{std*100:>8.2f}")

    # The mean cross-domain accuracy per alpha, over the variants that
    # share it.  This is the curve of the main figure.
    print()
    print("  mean over the variants of each alpha:")
    by_alpha = {}
    for tag, alpha, mean, std in variants:
        if alpha is None:
            continue
        by_alpha.setdefault(alpha, []).append(mean)
    for alpha in sorted(by_alpha):
        values = by_alpha[alpha]
        print(f"    alpha {alpha:.2f}  {np.mean(values)*100:6.2f}  "
              f"({len(values)} variants: "
              f"{', '.join(f'{v*100:.2f}' for v in sorted(values))})")
else:
    print("  (no sweep variants found)")

print()
print("=" * 100)
print("TABLE 6  Structural metrics (EdgeScore / LPIPS)")
print("=" * 100)
for name in ("adain", "adain_matched", "ours"):
    path = RUNS / f"structure_metrics_{name}.json"
    if not path.is_file():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    edge = payload.get("edge_consistency", {})
    lpips = payload.get("lpips", {})
    print(f"  {LABELS[name]:<30} EdgeScore {edge.get('mean', 0):.4f} "
          f"± {edge.get('std', 0):.4f}   LPIPS {lpips.get('mean', float('nan')):.4f} "
          f"(n={edge.get('n')})")
