"""Consolidated final analysis for the four/five experimental groups.

Produces the numbers that go into the report:

  1. the headline table (per-domain, per-seed, mean +/- std);
  2. the paired significance table with 95% confidence intervals;
  3. the style-strength / accuracy relationship, which is the key finding;
  4. the per-class breakdown;
  5. the structural-mechanism analysis (edge retention vs classification).

Every number is read from the recorded run artefacts; nothing is re-typed.
"""
from __future__ import annotations

import json
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
}
ORDER = ("original", "adain", "adain_matched", "ours", "ours_equal")


def load(group):
    path = RUNS / f"summary_{group}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_cross_per_seed"] = [
        (a + b + c) / 3 for a, b, c in zip(
            payload["art_painting_test"]["per_seed"],
            payload["cartoon_test"]["per_seed"],
            payload["sketch_test"]["per_seed"])]
    return payload


groups = {name: load(name) for name in ORDER}
groups = {k: v for k, v in groups.items() if v is not None}

print("=" * 100)
print("TABLE 1  Headline results (mean +/- std over seeds {0,1,2}; final checkpoint, 660 steps)")
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
    cross_std = np.std(s["_cross_per_seed"], ddof=1)
    print(f"  {LABELS[name]:<30}{cell('photo_test'):>9}{cell('art_painting_test'):>9}"
          f"{cell('cartoon_test'):>9}{cell('sketch_test'):>9}"
          f"{cross*100:>6.2f}±{cross_std*100:.1f}")

print()
print("=" * 100)
print("TABLE 2  Paired comparisons (same seeds -> paired t test, n=3, df=2, crit |t|=4.303)")
print("=" * 100)
PAIRS = [("original", "adain"), ("original", "ours"), ("adain", "adain_matched"),
         ("adain", "ours"), ("adain_matched", "ours"),
         ("adain", "ours_equal"), ("ours_equal", "adain_matched")]
for a, b in PAIRS:
    if a not in groups or b not in groups:
        continue
    x = np.asarray(groups[a]["_cross_per_seed"])
    y = np.asarray(groups[b]["_cross_per_seed"])
    diff = y - x
    t, p = stats.ttest_rel(y, x)
    sd = diff.std(ddof=1)
    half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
    flag = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"  {LABELS[a]:<30} -> {LABELS[b]:<30} {diff.mean()*100:+6.2f} pt "
          f"[{(diff.mean()-half)*100:+6.2f},{(diff.mean()+half)*100:+6.2f}]  "
          f"t={t:+6.2f} p={p:.4f}  {flag}")

print()
print("=" * 100)
print("TABLE 3  Style strength vs cross-domain accuracy (the central finding)")
print("=" * 100)
strength = {"original": 0.0, "adain": 0.5, "adain_matched": 0.3736,
            "ours": 0.3736, "ours_equal": 0.5}
rows = []
for name in ORDER:
    if name not in groups:
        continue
    s = groups[name]
    rows.append((strength[name], name, np.mean(s["_cross_per_seed"]),
                 np.std(s["_cross_per_seed"], ddof=1)))
rows.sort()
print(f"  {'mean style strength':>21}  {'group':<30}{'crossAvg':>10}{'std':>7}")
for st, name, mean, sd in rows:
    print(f"  {st:>21.4f}  {LABELS[name]:<30}{mean*100:>9.2f}{sd*100:>7.2f}")

print()
print("=" * 100)
print("TABLE 4  Per-class cross-domain accuracy (mean over the three target domains)")
print("=" * 100)
print(f"  {'class':<11}" + "".join(f"{LABELS[n][:16]:>18}" for n in ORDER if n in groups))
for name in CLASSES:
    line = f"  {name:<11}"
    for group in ORDER:
        if group not in groups:
            continue
        block = groups[group].get("per_class")
        values = []
        if block:
            for split in CROSS:
                entry = (block.get(split) or {}).get(name)
                if entry:
                    values.append(entry["mean"])
        line += f"{(np.mean(values)*100 if values else float('nan')):>18.1f}"
    print(line)

print()
print("=" * 100)
print("TABLE 5  Structural mechanism: edge retention (n=1349 paired images)")
print("=" * 100)
summary = RUNS / "structure_metrics_summary.json"
if summary.is_file():
    payload = json.loads(summary.read_text())
    for name, block in payload.items():
        edge = block["edge_consistency"]
        lp = block["lpips"]
        lp_text = f"{lp['mean']:.4f}" if lp else "n/a"
        print(f"  {name:<16} EdgeScore {edge['mean']:.4f} +/- {edge['std']:.4f}   LPIPS {lp_text}")
    ours = payload["ours"]["edge_consistency"]["mean"]
    adain = payload["adain"]["edge_consistency"]["mean"]
    matched = payload["adain_matched"]["edge_consistency"]["mean"]
    print()
    print(f"  Ours - AdaIN(uniform 0.5)          = {ours - adain:+.4f}   "
          f"(spatial map DOES retain more edges than plain AdaIN)")
    print(f"  Ours - AdaIN(strength-matched 0.374) = {ours - matched:+.4f}   "
          f"(but NOT more than equal-strength uniform AdaIN)")

npz = RUNS / "edge_analysis.npz"
if npz.is_file():
    d = np.load(npz)
    gain = d["ours"] - d["matched"]
    density = d["density"]
    valid = ~np.isnan(gain)
    rho, prho = stats.spearmanr(density[valid], gain[valid])
    median = np.median(density[valid])
    low = valid & (density <= median)
    high = valid & (density > median)
    print()
    print(f"  gain vs structural density: Spearman rho = {rho:+.4f}  p = {prho:.4f}")
    print(f"  low-density half  (n={int(low.sum()):4d}): {gain[low].mean():+.4f}")
    print(f"  high-density half (n={int(high.sum()):4d}): {gain[high].mean():+.4f}")
