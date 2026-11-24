"""Is the centre prior's mechanism gain real, or just less stylisation?

The selection picked the ``center`` map because its EdgeScore gain over
the strength-matched control is twenty times larger than the other two
candidates.  That is suspicious, and the reason is specific: the centre
prior reduces the style strength over a broad central region, so the
styled image stays closer to the content image.  The EdgeScore measures
how well the styled image preserves the *content's* edges, so simply
stylising less raises it -- whether or not the modulation is doing
anything principled.

The test is the effective style strength.  For every candidate the
script reports:

  * the mean of the spatial alpha field (the effective strength);
  * the mean EdgeScore of the styled image;
  * the mean EdgeScore of the *content* image's own edges -- the upper
    bound any styled image can reach;
  * the retention ratio, the styled score over the content score.

If the ``center`` candidate wins the mechanism test while having the
*lowest* effective strength and a retention ratio close to that of the
content image, then it is winning by stylising less, and the selection
has to be redone on a criterion that is not confounded with strength.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    args = parser.parse_args()

    print()
    print("=" * 92)
    print("IS THE CENTRE PRIOR'S GAIN REAL, OR JUST LESS STYLISATION?")
    print("=" * 92)
    print(f"  {'candidate':<20}{'eff alpha':>11}{'EdgeScore':>11}"
          f"{'content':>9}{'retention':>11}{'gain':>10}")
    rows = {}
    for kind in CANDIDATES:
        root = subset_dir(kind, args.subset)
        manifest = root / "manifest_ours.txt"
        analysis = root / "analysis.json"
        if not manifest.is_file():
            print(f"  {kind:<20}{'(missing)':>11}")
            continue
        effective = None
        if analysis.is_file():
            payload = json.loads(analysis.read_text(encoding="utf-8"))
            block = (payload.get("ours") or {}).get("mean_alpha") or {}
            effective = block.get("mean")

        entries = read_aug_manifest(manifest)
        styled_scores, content_scores = [], []
        for rel, _label, meta in entries:
            content = config.IMAGE_ROOT / meta[0]
            styled = config.DATA_ROOT / rel
            content_map = edge_map(content)
            styled_scores.append(edge_consistency(content_map, edge_map(styled)))
            content_scores.append(float(content_map.sum()))
        styled_mean = float(np.mean(styled_scores))

        # The upper bound: the EdgeScore of the content image against
        # itself.  A styled image can never exceed it, because a perfect
        # copy has identical edges and the score is defined on the
        # overlap of the two edge sets.
        upper = 1.0
        retention = styled_mean / upper

        # The gain over the matched control, recomputed here so that all
        # three candidates are measured the same way.
        control_manifest = root / "manifest_adain_matched.txt"
        gain = float("nan")
        if control_manifest.is_file():
            control = read_aug_manifest(control_manifest)
            control_scores = []
            for rel, _label, meta in control:
                content = config.IMAGE_ROOT / meta[0]
                styled = config.DATA_ROOT / rel
                control_scores.append(
                    edge_consistency(edge_map(content), edge_map(styled)))
            gain = styled_mean - float(np.mean(control_scores))

        rows[kind] = {"effective_alpha": effective, "edge_score": styled_mean,
                      "gain": gain, "retention": retention,
                      "mean_content_edges": float(np.mean(content_scores))}
        print(f"  {kind:<20}{(effective if effective is not None else float('nan')):>11.4f}"
              f"{styled_mean:>11.4f}{upper:>9.3f}{retention:>11.3f}{gain:>+10.4f}")

    print()
    print("  A candidate that wins by *stylising less* has a lower effective")
    print("  alpha than its competitors.  The mechanism claim is only")
    print("  supported if the winner reaches a high retention at a strength")
    print("  that is comparable to the others.")

    out = config.OUTPUT_ROOT / "foreground_strength_check.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
