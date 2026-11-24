"""Choose the foreground map: a Photo Validation gate and a mechanism rank.

The selection problem
---------------------
Three candidate maps are available: ``spectral``, ``center`` and
``spectral_center``.  Which one the repaired variant uses has to be
decided by data.

Photo Validation alone cannot decide it.  The measurement in
``check_val_discrimination.py`` shows that every group of the main table
lands between 98.13% and 98.93% on that split -- a total spread of 2.00
percentage points, while a single validation image is worth 0.67 points
and the seed-to-seed spread is 0.30-0.60%.  The split is saturated: it
can exclude a candidate that is clearly worse, but it cannot rank
candidates that are close.

The two criteria
----------------
The map is therefore chosen by two criteria with different jobs:

1. **The Photo Validation gate.**  A candidate whose validation accuracy
   falls clearly below the others is rejected, whatever its mechanism
   score.  This is the part of the decision that the user asked to be
   made on Photo Validation, and it is the only part that touches a
   held-out split of the *source* domain.

2. **The mechanism rank.**  Among the candidates that pass the gate, the
   one with the highest EdgeScore gain over its *own* strength-matched
   uniform control is chosen.  This is the quantity the method claims to
   optimise, it is measured on the images themselves and not on any
   label, and it involves no target-domain image at all.

Neither criterion uses a target-domain image, so the target domains stay
untouched until the final five-seed evaluation.

Usage:
    python select_foreground_final.py
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


def subset_manifest(kind: str, subset: int) -> Path:
    """The manifest of one candidate's reduced set."""
    return (config.DATA_ROOT / "aug_sp" / "previews" / f"build_n{subset}" /
            f"subset{subset}_{kind}_a0p5_b0p7" / "manifest_ours.txt")


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


def mechanism_score(kind: str, subset: int) -> dict | None:
    """The mechanism numbers of one candidate.

    The comparison is against the candidate's *own* strength-matched
    uniform control, which is what the generator writes next to it.  A
    positive difference means the spatial map retains more edge structure
    than a uniform control at the same effective strength -- that is, the
    mechanism works.
    """
    manifest = subset_manifest(kind, subset)
    matched = manifest.parent / "manifest_adain_matched.txt"
    if not manifest.is_file() or not matched.is_file():
        return None
    ours, labels, density = per_image(manifest)
    control, _, _ = per_image(matched)
    difference = ours - control
    valid = ~np.isnan(difference)
    t, p = stats.ttest_rel(ours[valid], control[valid])
    return {
        "mean_difference": float(difference.mean()),
        "std": float(difference.std(ddof=1)),
        "fraction_worse": float((difference < 0).sum() / len(difference)),
        "paired_t": float(t), "paired_p": float(p),
        "n": int(valid.sum()),
    }


def validation_scores() -> dict:
    """The Photo Validation accuracy of every candidate that was trained."""
    scores = {}
    for kind in CANDIDATES:
        values = []
        for run_dir in sorted(config.OUTPUT_ROOT.glob(f"fgsel_{kind}_s*")):
            summary = run_dir / "train_summary.json"
            if not summary.is_file():
                continue
            payload = json.loads(summary.read_text(encoding="utf-8"))
            accuracy = payload.get("reported_val_acc")
            if accuracy is None:
                accuracy = payload.get("best_val_acc")
            if accuracy is not None:
                values.append(float(accuracy))
        if values:
            array = np.asarray(values)
            scores[kind] = {
                "per_seed": values,
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "n": len(values),
            }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--gate-tolerance", type=float, default=0.01,
                        help="a candidate is rejected when its Photo Validation "
                             "mean falls more than this far below the best one; "
                             "the default is one percentage point, which is "
                             "about 1.5 validation images")
    parser.add_argument("--out", type=Path,
                        default=config.OUTPUT_ROOT / "foreground_selection.json")
    args = parser.parse_args()

    validation = validation_scores()
    mechanisms = {kind: mechanism_score(kind, args.subset) for kind in CANDIDATES}

    print()
    print("=" * 84)
    print("CRITERION 1 -- PHOTO VALIDATION GATE")
    print("=" * 84)
    print(f"  {'candidate':<20}{'mean':>10}{'std':>9}{'n':>4}   per seed")
    for kind in CANDIDATES:
        block = validation.get(kind)
        if not block:
            print(f"  {kind:<20}{'(not trained)':>23}")
            continue
        seeds = ", ".join(f"{v*100:.2f}" for v in block["per_seed"])
        print(f"  {kind:<20}{block['mean']*100:>9.2f}%{block['std']*100:>8.2f}%"
              f"{block['n']:>4}   {seeds}")

    trained = [k for k in CANDIDATES if k in validation]
    best_validation = max((validation[k]["mean"] for k in trained), default=0.0)
    gate = args.gate_tolerance
    passed = [k for k in trained
              if validation[k]["mean"] >= best_validation - gate]
    rejected = [k for k in trained if k not in passed]
    print()
    print(f"  best mean {best_validation*100:.2f}%, gate +/-{gate*100:.2f} points")
    print(f"  passed : {', '.join(passed) if passed else '(none)'}")
    print(f"  rejected: {', '.join(rejected) if rejected else '(none)'}")

    print()
    print("=" * 84)
    print("CRITERION 2 -- MECHANISM RANK (EdgeScore vs own matched control)")
    print("=" * 84)
    print(f"  {'candidate':<20}{'mean diff':>12}{'std':>9}{'% worse':>9}"
          f"{'paired p':>12}")
    for kind in CANDIDATES:
        block = mechanisms.get(kind)
        if not block:
            print(f"  {kind:<20}{'(no matched control)':>20}")
            continue
        print(f"  {kind:<20}{block['mean_difference']:>+12.4f}"
              f"{block['std']:>9.4f}{block['fraction_worse']*100:>8.1f}%"
              f"{block['paired_p']:>12.3e}")

    eligible = [k for k in passed if mechanisms.get(k)]
    if not eligible:
        print()
        print("  no candidate passed the gate with a measured mechanism; "
              "the selection cannot be made.")
        return
    winner = max(eligible,
                 key=lambda k: mechanisms[k]["mean_difference"])
    print()
    print(f"  eligible: {', '.join(eligible)}")
    print(f"  SELECTED: {winner}")

    result = {
        "selected": winner,
        "criterion_1_gate": {
            "description": "Photo Validation mean accuracy, source-domain "
                           "held-out split; a candidate is rejected when it "
                           "falls more than the tolerance below the best one",
            "tolerance": gate,
            "per_candidate": validation,
            "passed": passed,
            "rejected": rejected,
        },
        "criterion_2_rank": {
            "description": "EdgeScore gain over the candidate's own "
                           "strength-matched uniform control, measured on the "
                           "images; no label and no target-domain image is "
                           "involved",
            "per_candidate": {k: v for k, v in mechanisms.items() if v},
        },
        "selection_set": "photo_val (gate) + image-level mechanism (rank)",
        "target_domain_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  written: {args.out}")


if __name__ == "__main__":
    main()
