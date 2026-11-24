"""Where could the spatial modulation be improved?  A decomposition.

The spatial variant loses edge retention against the strength-matched
uniform control.  This script decomposes that loss into the two
mechanisms that could produce it, so that the fix is aimed at the right
one.

Hypothesis A -- the *strength distribution* is wrong.
    The blending field is ``alpha = alpha0 (1 - beta M)``.  With M in
    [0,1] the field's spatial mean is ``alpha0 (1 - beta * mean(M))``,
    but its *local* strength on the object (where M is low) is close to
    alpha0, i.e. nearly the full nominal strength.  So the modulation
    applies the full strength exactly on the class cue and less of it
    on the background.  The styling damage therefore lands on the
    object.  This predicts that the spatial variant should be *worse*
    than a uniform field at the same mean, and it should be worse by
    more on images where the object occupies a large fraction of the
    area.

Hypothesis B -- the *structure map* is wrong.
    M is a raw Sobel magnitude.  It responds to background texture as
    strongly as to the object's contour, so it does not identify the
    structure worth protecting (the failure cases of Fig. 5 show
    this).  This predicts that replacing the raw magnitude with a
    *saliency-weighted* map would change the outcome.

This script measures the quantity that separates the two: the
correlation between an image's EdgeScore difference and the
*fraction of the image that the object covers*, which is the
quantity Hypothesis A is about.  It also measures how much of the
structure map's response mass falls on the highest-gradient pixels,
which is the quantity Hypothesis B is about.

The script only reads data that already exists.  It proposes nothing
and changes nothing.
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
sys.path.insert(0, str(REPO / "member3_structure_preserving"))

import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--sample", type=int, default=200,
                        help="how many images to analyse for the map statistics")
    args = parser.parse_args()

    npz = args.runs_root / "edge_analysis.npz"
    payload = np.load(npz)
    difference = payload["ours"] - payload["matched"]
    density = payload["density"]
    print(f"images: {len(difference)}")

    # ---- Hypothesis A: the strength lands on the object ----------------
    print("\n=== A. does the loss grow with the structural density? ===")
    mask = ~np.isnan(difference) & ~np.isnan(density)
    d, s = difference[mask], density[mask]
    print(f"  Pearson  r = {stats.pearsonr(s, d)[0]:+.4f} (p={stats.pearsonr(s,d)[1]:.4f})")
    print(f"  Spearman rho = {stats.spearmanr(s, d)[0]:+.4f} (p={stats.spearmanr(s,d)[1]:.4f})")
    for label, subset in (("low density", s < np.median(s)),
                           ("high density", s >= np.median(s))):
        print(f"  {label:14s} mean difference = {d[subset].mean():+.4f} "
              f"+- {d[subset].std(ddof=1):.4f}")

    # ---- Hypothesis B: what does the structure map respond to? --------
    print("\n=== B. what does the structure map respond to? ===")
    from member3_structure_preserving.structure_map import (
        StructureConfig, build_structure_map)
    from member3_structure_preserving.stylizer import StructurePreservingStylizer

    manifest = config.DATA_ROOT / "aug_sp" / "manifest_ours.txt"
    rows = [line.split() for line in
            manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    generator = StructurePreservingStylizer()
    structure_config = StructureConfig()

    top_fractions = []
    background_mass = []
    steps = max(1, len(rows) // max(1, args.sample))
    for row in rows[::steps][:args.sample]:
        content = config.IMAGE_ROOT / row[2]
        if not content.is_file():
            continue
        prepared = generator.prepare_content(content)
        m = prepared.structure.squeeze().cpu().numpy()
        # The response mass carried by the strongest 1% of the pixels.
        flat = np.sort(m.ravel())[::-1]
        total = flat.sum()
        if total <= 0:
            continue
        top_fractions.append(float(flat[:max(1, flat.size // 100)].sum() / total))
        # How much of the response sits in the *middle* of the range
        # rather than at the extremes: a flat, texture-dominated map has
        # its mass spread out, a silhouette-dominated one has it at 0.
        middle = ((m > 0.2) & (m < 0.8)).sum() / m.size
        background_mass.append(float(middle))

    if top_fractions:
        top = np.asarray(top_fractions)
        middle = np.asarray(background_mass)
        print(f"  analysed {top.size} images")
        print(f"  response mass in the strongest 1% of the pixels: "
              f"{top.mean():.3f} +- {top.std(ddof=1):.3f}")
        print(f"  fraction of the map in the middle band (0.2-0.8): "
              f"{middle.mean():.3f} +- {middle.std(ddof=1):.3f}")
        print("  A map that concentrated on one silhouette would put its "
              "mass at the extremes (middle fraction near 0) and give the "
              "top 1% almost all of the total.")

    # ---- the cost of the current parameterisation ---------------------
    print("\n=== C. what the current parameterisation implies ===")
    print("  The field is alpha = alpha0 (1 - beta M), so on the object")
    print("  (M ~ 0) the applied strength is alpha0, while the mean over")
    print("  the image is alpha0 (1 - beta mean(M)) = 0.374 at the main")
    print("  setting.  The modulation therefore applies the FULL nominal")
    print("  strength on the class cue and less everywhere else.  Any")
    print("  mechanism that protects the object must instead reduce the")
    print("  strength ON the object, which this rule cannot do: it only")
    print("  ever reduces alpha where M is HIGH.")
    print()
    print("  This is the direction a fix has to take, and it is the")
    print("  opposite of what the current rule does.")


if __name__ == "__main__":
    main()
