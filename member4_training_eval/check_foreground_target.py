"""Where does a candidate's mechanism gain land: the object, or the background?

The mechanism rank compares every candidate against its *own*
strength-matched uniform control, so the comparison is fair in terms of
total style energy.  But it does not say *which* edges the spatial map
protects.  Two very different situations produce the same EdgeScore gain:

  * the map protects the object's contour, so the class cue survives;
  * the map protects whatever happens to be in the protected region,
    which for a centre prior is often background as well.

Only the first one can help domain generalisation, because only the class
cue is what the classifier reads.

The test uses the structural density of the *content* image.  An image
with many edges contains a detailed object -- its edges are concentrated
on the object -- while an image with few edges is dominated by a smooth
background.  The gain is therefore reported separately for the two
halves of the density distribution:

  * a gain that is *larger* on the high-density half means the map
    protects detailed objects, which is the mechanism the method claims;
  * a gain that is *larger* on the low-density half means the map is
    protecting background, which is not the claimed mechanism.

Usage:
    python check_foreground_target.py --subset 200
"""
from __future__ import annotations

import argparse
import json
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

CANDIDATES = ("spectral", "center", "spectral_center")


def subset_dir(kind: str, subset: int) -> Path:
    return (config.DATA_ROOT / "aug_sp" / "previews" / f"build_n{subset}" /
            f"subset{subset}_{kind}_a0p5_b0p7")


def per_image(manifest_path: Path):
    rows = read_aug_manifest(manifest_path)
    scores, density = [], []
    for rel, _label, meta in rows:
        content = config.IMAGE_ROOT / meta[0]
        styled = config.DATA_ROOT / rel
        content_map = edge_map(content)
        scores.append(edge_consistency(content_map, edge_map(styled)))
        density.append(float(content_map.sum()))
    return np.asarray(scores, dtype=float), np.asarray(density, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=config.OUTPUT_ROOT / "foreground_target_check.json")
    args = parser.parse_args()

    print()
    print("=" * 90)
    print("WHERE DOES THE MECHANISM GAIN LAND?")
    print("=" * 90)
    print("  High density = a detailed object, its edges are concentrated on it.")
    print("  Low density  = a smooth background dominates the image.")
    print()
    print(f"  {'candidate':<20}{'low dens':>12}{'high dens':>12}"
          f"{'difference':>13}{'t':>8}{'p':>11}")

    results = {}
    for kind in CANDIDATES:
        root = subset_dir(kind, args.subset)
        manifest = root / "manifest_ours.txt"
        matched = root / "manifest_adain_matched.txt"
        if not manifest.is_file() or not matched.is_file():
            print(f"  {kind:<20}{'(missing)':>12}")
            continue
        ours, density = per_image(manifest)
        control, _ = per_image(matched)
        gain = ours - control
        valid = ~np.isnan(gain)
        median = np.median(density[valid])
        low = valid & (density <= median)
        high = valid & (density > median)
        low_mean = float(gain[low].mean())
        high_mean = float(gain[high].mean())
        t, p = stats.ttest_ind(gain[high], gain[low], equal_var=False)
        results[kind] = {
            "low_density_gain": low_mean,
            "high_density_gain": high_mean,
            "difference": high_mean - low_mean,
            "welch_t": float(t), "welch_p": float(p),
            "n_low": int(low.sum()), "n_high": int(high.sum()),
            "median_density": float(median),
        }
        print(f"  {kind:<20}{low_mean:>+12.4f}{high_mean:>+12.4f}"
              f"{high_mean - low_mean:>+13.4f}{t:>+8.2f}{p:>11.3e}")

    print()
    for kind, block in results.items():
        verdict = ("protects the object" if block["difference"] > 0
                   else "protects the background")
        print(f"  {kind:<20} {verdict:<24} "
              f"(high-density minus low-density = {block['difference']:+.4f})")
    print()
    print("  A candidate whose gain is larger on the detailed half is doing")
    print("  what the method claims.  A candidate whose gain is larger on the")
    print("  smooth half is protecting background, which is a different")
    print("  mechanism from the one the method claims -- whatever it does to")
    print("  the EdgeScore.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  written: {args.out}")


if __name__ == "__main__":
    main()
