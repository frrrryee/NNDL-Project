"""Per-class, per-image structure retention: does the spatial modulation help
specifically where the class is defined by a thin, high-frequency contour?

If structure preservation works through the mechanism it claims, then the
per-image EdgeScore gain over a *strength-matched* uniform AdaIN should be
concentrated on images whose content has many thin contours, and should be
near zero (or negative) on images dominated by coarse texture.

Two quantities are reported per class:

  * the mean EdgeScore of each set, and
  * the paired difference against the strength-matched control.

A correlation between the per-image structural density (edge count of the
content image) and the paired gain is also reported, which is the direct
test of the mechanism claim.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "member4_training_eval"))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from structure_metrics import edge_consistency, edge_map  # noqa: E402
from data import read_aug_manifest  # noqa: E402

DATA = config.DATA_ROOT
CACHE = config.OUTPUT_ROOT


def per_image(manifest_path):
    rows = read_aug_manifest(manifest_path)
    scores, labels, density = [], [], []
    for rel, label, meta in rows:
        content = config.IMAGE_ROOT / meta[0]
        styled = DATA / rel
        scores.append(edge_consistency(edge_map(content), edge_map(styled)))
        # read_aug_manifest already converts the manifest's 1-based label to
        # a 0-based index, so it must not be decremented again.
        labels.append(label)
        # Structural density: how many edges the *content* image has.
        density.append(float(edge_map(content).sum()))
    return (np.asarray(scores, dtype=float),
            np.asarray(labels, dtype=int),
            np.asarray(density, dtype=float))


print("computing per-image scores and structural density ...", flush=True)
ours, labels, density = per_image(DATA / "aug_sp" / "manifest_ours.txt")
adain, _, _ = per_image(DATA / "aug_sp" / "manifest_adain.txt")
matched, _, _ = per_image(DATA / "aug_sp" / "manifest_adain_matched.txt")
np.savez(CACHE / "edge_analysis.npz", ours=ours, adain=adain, matched=matched,
         labels=labels, density=density)

print()
print("=" * 86)
print("PER-CLASS EDGE RETENTION (mean over the class's images)")
print("=" * 86)
print(f"  {'class':<11}{'n':>5}{'density':>10}{'+AdaIN':>10}{'+Match':>10}"
      f"{'+Ours':>10}{'Ours-Match':>13}")
rows = []
for index, name in enumerate(config.CLASSES):
    mask = labels == index
    n = int(mask.sum())
    d = density[mask].mean()
    a = adain[mask].mean()
    m = matched[mask].mean()
    o = ours[mask].mean()
    diff = ours[mask] - matched[mask]
    t, p = stats.ttest_rel(ours[mask], matched[mask])
    rows.append((name, n, d, a, m, o, diff.mean(), p))
    print(f"  {name:<11}{n:>5}{d:>10.0f}{a:>10.4f}{m:>10.4f}{o:>10.4f}"
          f"{diff.mean():>+13.4f}")

print()
print("=" * 86)
print("MECHANISM TEST: does the gain grow with the structural density of the image?")
print("=" * 86)
gain = ours - matched
valid = ~np.isnan(gain)
r, p = stats.pearsonr(density[valid], gain[valid])
rho, prho = stats.spearmanr(density[valid], gain[valid])
print(f"  Pearson  r = {r:+.4f}  p = {p:.3e}   (n={int(valid.sum())})")
print(f"  Spearman rho = {rho:+.4f}  p = {prho:.3e}")
print()
# Split the images at the median structural density.
median = np.median(density[valid])
low = valid & (density <= median)
high = valid & (density > median)
print(f"  median density = {median:.0f} edges")
print(f"  low-density  half (n={int(low.sum()):4d}): gain {gain[low].mean():+.4f} "
      f"(+/- {gain[low].std(ddof=1):.4f})")
print(f"  high-density half (n={int(high.sum()):4d}): gain {gain[high].mean():+.4f} "
      f"(+/- {gain[high].std(ddof=1):.4f})")
t, p = stats.ttest_ind(gain[high], gain[low], equal_var=False)
print(f"  Welch t-test high vs low: t={t:+.2f} p={p:.3e}")
print()
print("  If the spatial modulation preserves structure for its stated reason,")
print("  the high-density half must gain more than the low-density half.")
