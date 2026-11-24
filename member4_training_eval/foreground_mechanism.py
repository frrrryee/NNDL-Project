"""The mechanism measurement of the repaired modulation.

The measurement is deliberately identical to ``structure_mechanism.py``:
the same EdgeScore, the same content images, the same strength-matched
control, the same structural density.  The only difference is which
modulation map produced the images.

That is what makes the comparison meaningful.  The repaired variant is
judged on exactly the criterion the original variant failed:

    does the spatial map retain more edge structure than a uniform
    control at the same effective strength?

If the foreground map fixes the mechanism, the repaired variant's paired
difference against its own matched control is positive.  If it does not,
the difference is still negative, and the repair has failed.

The results are written to ``runs/edge_analysis_fg.npz`` so that the
report generators can read them without recomputing anything.

Usage:
    python foreground_mechanism.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from structure_metrics import edge_consistency, edge_map  # noqa: E402
from data import read_aug_manifest  # noqa: E402


def per_image(manifest_path: Path):
    """The EdgeScore of every image of one manifest."""
    rows = read_aug_manifest(manifest_path)
    scores, labels, density = [], [], []
    for rel, label, meta in rows:
        content = config.IMAGE_ROOT / meta[0]
        styled = config.DATA_ROOT / rel
        scores.append(edge_consistency(edge_map(content), edge_map(styled)))
        labels.append(label)
        density.append(float(edge_map(content).sum()))
    return (np.asarray(scores, dtype=float),
            np.asarray(labels, dtype=int),
            np.asarray(density, dtype=float))


def report(name: str, ours: np.ndarray, matched: np.ndarray,
           labels: np.ndarray, density: np.ndarray) -> dict:
    """The per-class table and the mechanism test of one variant."""
    difference = ours - matched
    print()
    print("=" * 88)
    print(f"{name}: PER-CLASS EDGE RETENTION")
    print("=" * 88)
    print(f"  {'class':<11}{'n':>5}{'density':>10}{'+Match':>10}{'Variant':>10}"
          f"{'Variant-Match':>15}")
    for index, class_name in enumerate(config.CLASSES):
        mask = labels == index
        diff = ours[mask] - matched[mask]
        print(f"  {class_name:<11}{int(mask.sum()):>5}{density[mask].mean():>10.0f}"
              f"{matched[mask].mean():>10.4f}{ours[mask].mean():>10.4f}"
              f"{diff.mean():>+15.4f}")

    valid = ~np.isnan(difference)
    gain = difference[valid]
    r, p = stats.pearsonr(density[valid], gain)
    rho, prho = stats.spearmanr(density[valid], gain)
    median = np.median(density[valid])
    low = valid & (density <= median)
    high = valid & (density > median)
    t, p_welch = stats.ttest_ind(gain[high], gain[low], equal_var=False)
    worse = int((difference < 0).sum())
    print()
    print(f"  mean difference vs the matched control: {difference.mean():+.4f} "
          f"(std {difference.std(ddof=1):.4f})")
    print(f"  images where the variant retains LESS: {worse}/{len(difference)} "
          f"({worse/len(difference)*100:.1f}%)")
    print(f"  Pearson  r = {r:+.4f}  p = {p:.3e}")
    print(f"  Spearman rho = {rho:+.4f}  p = {prho:.3e}")
    print(f"  low-density  half (n={int(low.sum()):4d}): "
          f"{difference[low].mean():+.4f} "
          f"(+/- {difference[low].std(ddof=1):.4f})")
    print(f"  high-density half (n={int(high.sum()):4d}): "
          f"{difference[high].mean():+.4f} "
          f"(+/- {difference[high].std(ddof=1):.4f})")
    print(f"  Welch t-test high vs low: t={t:+.2f} p={p_welch:.3e}")
    return {"mean_difference": float(difference.mean()),
            "fraction_worse": worse / len(difference),
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(prho),
            "low_density_mean": float(difference[low].mean()),
            "high_density_mean": float(difference[high].mean()),
            "welch_t": float(t), "welch_p": float(p_welch)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--fg-manifest", type=Path,
                        default=config.DATA_ROOT / "aug_sp" / "foreground" /
                        "manifest_ours.txt")
    parser.add_argument("--fg-matched-manifest", type=Path,
                        default=config.DATA_ROOT / "aug_sp" / "foreground" /
                        "manifest_adain_matched.txt")
    args = parser.parse_args()

    if not args.fg_manifest.is_file():
        print(f"{args.fg_manifest} is missing; build the foreground set first")
        return

    print("computing the per-image scores of the repaired variant ...",
          flush=True)
    fg, labels, density = per_image(args.fg_manifest)
    fg_matched, _, _ = per_image(args.fg_matched_manifest)

    # The original method's numbers, for the side-by-side comparison.
    original = args.runs_root / "edge_analysis.npz"
    original_result = None
    if original.is_file():
        payload = np.load(original)
        original_result = report("ORIGINAL (edge map)",
                                 payload["ours"], payload["matched"],
                                 payload["labels"], payload["density"])

    repaired_result = report("REPAIRED (foreground map)", fg, fg_matched,
                             labels, density)

    np.savez(args.runs_root / "edge_analysis_fg.npz",
             ours=fg, matched=fg_matched, labels=labels, density=density)

    print()
    print("=" * 88)
    print("SIDE BY SIDE")
    print("=" * 88)
    print(f"  {'variant':<28}{'mean diff':>12}{'% worse':>10}"
          f"{'low dens':>11}{'high dens':>11}")
    if original_result:
        print(f"  {'original (edge map)':<28}"
              f"{original_result['mean_difference']:>+12.4f}"
              f"{original_result['fraction_worse']*100:>9.1f}%"
              f"{original_result['low_density_mean']:>+11.4f}"
              f"{original_result['high_density_mean']:>+11.4f}")
    print(f"  {'repaired (foreground map)':<28}"
          f"{repaired_result['mean_difference']:>+12.4f}"
          f"{repaired_result['fraction_worse']*100:>9.1f}%"
          f"{repaired_result['low_density_mean']:>+11.4f}"
          f"{repaired_result['high_density_mean']:>+11.4f}")
    print()
    if repaired_result["mean_difference"] > 0:
        print("  The repaired variant retains MORE structure than the matched")
        print("  control: the mechanism now works in the intended direction.")
    else:
        print("  The repaired variant still retains LESS structure than the")
        print("  matched control: the repair did not fix the mechanism.")


if __name__ == "__main__":
    main()
