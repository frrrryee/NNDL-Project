"""Build the failure-case figure set of the report (Fig. 5).

The claim under test is that the spatial modulation preserves the content's
structure where it matters.  The mechanism analysis of Section III.E found
that it does not: the EdgeScore gain over the strength-matched uniform
control is the same on structurally dense images as on sparse ones, and it
is slightly *negative* everywhere.

This script turns that finding into a picture.  It selects the images on
which the modulation does the most damage -- the ones with the smallest
paired EdgeScore difference against the matched control -- and renders,
for each of them, the six panels that make the failure legible:

    content | style | structure map M | alpha map | uniform AdaIN | ours

The panels are produced by the generator itself, so what is shown is
exactly what the generator computed, not a re-derivation of it.

The selection is deterministic: the images are ranked by their paired
EdgeScore difference and the requested number is taken from the worst
end.  Nothing here depends on a target-domain accuracy, so no evaluation
image influences which images are shown.

Usage:
    python make_failure_cases.py --count 12
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
sys.path.insert(0, str(REPO / "member3_structure_preserving"))

import config  # noqa: E402


def load_scores(runs_root: Path):
    """The per-image EdgeScores and the manifest rows they belong to."""
    npz = runs_root / "edge_analysis.npz"
    if not npz.is_file():
        return None
    payload = np.load(npz)
    rows = []
    manifest = config.DATA_ROOT / "aug_sp" / "manifest_ours.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(line.split())
    return payload["ours"], payload["adain"], payload["matched"], rows


def select(ours, matched, rows, count: int, worst: bool = True):
    """The images with the smallest (or largest) paired EdgeScore difference."""
    difference = ours - matched
    order = np.argsort(difference if worst else -difference)
    picked = []
    for index in order:
        if index >= len(rows) or np.isnan(difference[index]):
            continue
        picked.append((int(index), rows[index], float(difference[index])))
        if len(picked) >= count:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=12,
                        help="how many cases to render")
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "figures" / "failure_cases")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--best", action="store_true",
                        help="select the images the modulation helps most "
                             "instead of the ones it hurts most")
    args = parser.parse_args()

    payload = load_scores(args.runs_root)
    if payload is None:
        print(f"{args.runs_root}/edge_analysis.npz is missing; "
              f"run structure_mechanism.py first")
        return 1
    ours, adain, matched, rows = payload

    picked = select(ours, matched, rows, args.count, worst=not args.best)
    if not picked:
        print("no image could be selected")
        return 1

    # The generator is imported lazily so that a missing model only fails
    # when the figure is actually requested.
    from member3_structure_preserving.stylizer import StructurePreservingStylizer
    from member3_structure_preserving.analysis import comparison_panel

    generator = StructurePreservingStylizer()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for position, (i, row, difference) in enumerate(picked, 1):
        rel, label, content_rel = row[0], row[1], row[2]
        style_name = row[3] if len(row) > 3 else ""
        content_path = config.IMAGE_ROOT / content_rel
        style_path = None
        pool = config.STYLE_POOL_DIR / "images"
        if pool.is_dir():
            for candidate in pool.rglob("*.jpg"):
                if candidate.stem == style_name:
                    style_path = candidate
                    break
        if style_path is None:
            print(f"  {position:2d}. {Path(rel).name:44s} "
                  f"style '{style_name}' not found, skipped")
            continue

        prepared = generator.prepare_content(content_path)
        features = generator.encode_style(style_path)
        result = generator.render(prepared, features, args.alpha, args.beta,
                                  with_baseline=True, with_matched=True)
        panel = comparison_panel(result, style_path, args.alpha, args.beta)
        name = f"case{position:02d}_{Path(rel).stem}"
        target = args.out_dir / f"{name}.png"
        panel.save(target)

        index.append({
            "position": position,
            "content": str(content_rel),
            "style": style_name,
            "label": label,
            "edge_ours": float(ours[i]),
            "edge_adain": float(adain[i]),
            "edge_matched": float(matched[i]),
            "edge_difference": difference,
            "mean_alpha": float(result.alpha_map.mean()),
            "figure": str(target),
        })
        print(f"  {position:2d}. {Path(rel).name:44s} "
              f"ours={ours[i]:.4f} matched={matched[i]:.4f} "
              f"diff={difference:+.4f}  saved")

    manifest = args.out_dir / "index.json"
    manifest.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
