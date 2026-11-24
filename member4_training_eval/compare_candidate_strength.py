"""Compare the effective style strength of the three foreground candidates.

The selection over the three candidates picked ``center`` with a
mechanism gain of +0.0336 against the +0.0010 of ``spectral``.  That
difference is far too large to be about the *meaning* of the map, and
this script checks the most likely mechanical explanation.

The EdgeScore gain of a spatial map over its uniform control is not
only about *where* the style was reduced but also about *how much* style
was applied.  A map that covers more of the image reduces the strength
over a larger area, which lowers the mean strength of the styled image,
which raises the EdgeScore on its own -- independently of any
structure-preserving effect.

The three candidates differ in coverage:

  * ``spectral`` marks 2-3% of the pixels,
  * ``center`` marks about 53%,
  * ``spectral_center`` marks 3%.

So a coverage-matched comparison is necessary before any of the three
mechanism numbers can be compared with each other.  This script prints,
for every candidate:

  * the mean of the alpha field (the effective style strength),
  * the mean of the matched control's strength,
  * the strength difference, and
  * the EdgeScore gain,

the last two side by side, so that the share of the gain that is
explained by the strength difference can be seen directly.

A regression of the EdgeScore gain on the strength difference across the
candidates gives the answer: if the gain is proportional to the
strength difference, the gain measures "less style", not "better
structure", and the candidate ranking it produces is an artefact.

Usage:
    python compare_candidate_strength.py --subset 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def scores(manifest: Path):
    values = []
    for rel, _label, meta in read_aug_manifest(manifest):
        content = config.IMAGE_ROOT / meta[0]
        styled = config.DATA_ROOT / rel
        values.append(edge_consistency(edge_map(content), edge_map(styled)))
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=200)
    args = parser.parse_args()

    print()
    print("=" * 88)
    print("IS THE MECHANISM GAIN ABOUT STRUCTURE, OR ABOUT LESS STYLE?")
    print("=" * 88)
    print(f"  {'candidate':<18}{'map alpha':>11}{'ctrl alpha':>12}"
          f"{'d strength':>13}{'edge gain':>12}")
    rows = []
    for kind in CANDIDATES:
        root = subset_dir(kind, args.subset)
        record = root / "generation.json"
        map_alpha = None
        matched_alpha = None
        if record.is_file():
            payload = json.loads(record.read_text(encoding="utf-8"))
            entries = list((payload.get("outputs") or {}).values())
            map_values = [e["metrics"]["mean_alpha"] for e in entries
                          if e.get("metrics") and "mean_alpha" in e["metrics"]]
            matched_values = [e.get("matched_alpha") for e in entries
                              if e.get("matched_alpha") is not None]
            if map_values:
                map_alpha = float(np.mean(map_values))
            if matched_values:
                matched_alpha = float(np.mean(matched_values))

        ours = scores(root / "manifest_ours.txt")
        control = scores(root / "manifest_adain_matched.txt")
        gain = float((ours - control).mean())
        strength = ((map_alpha - matched_alpha)
                    if (map_alpha is not None and matched_alpha is not None)
                    else None)
        rows.append((kind, map_alpha, matched_alpha, strength, gain))
        strength_text = (f"{strength:>+13.4f}" if strength is not None
                         else f"{'(n/a)':>13}")
        alpha_text = (f"{map_alpha:>11.4f}" if map_alpha is not None
                      else f"{'(n/a)':>11}")
        control_text = (f"{matched_alpha:>12.4f}" if matched_alpha is not None
                        else f"{'(n/a)':>12}")
        print(f"  {kind:<18}{alpha_text}{control_text}{strength_text}{gain:>+12.4f}")

    available = [r for r in rows if r[3] is not None]
    if len(available) >= 2:
        strength = np.asarray([r[3] for r in available])
        gain = np.asarray([r[4] for r in available])
        print()
        print("  The strength difference is the mean of the alpha field of the")
        print("  candidate minus the mean of its uniform control: it is how")
        print("  much *less* total style the candidate applied.")
        if np.std(strength) > 0:
            slope, intercept = np.polyfit(strength, gain, 1)
            correlation = np.corrcoef(strength, gain)[0, 1]
            print("  regression of the gain on the strength difference:")
            print(f"    gain = {slope:+.4f} * d_strength {intercept:+.4f}"
                  f"   (r = {correlation:+.4f})")
            print()
            if abs(correlation) > 0.9:
                print("  The gain is almost entirely explained by the amount of")
                print("  style that was applied.  The candidates are therefore")
                print("  NOT comparable on this criterion: a map that covers")
                print("  more of the image always wins, whether or not it finds")
                print("  the object.")
            else:
                print("  The gain is not explained by the strength alone, so the")
                print("  criterion does carry information about the map.")
    print()


if __name__ == "__main__":
    main()
