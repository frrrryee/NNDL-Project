"""Select the foreground map on Photo Validation.

The repaired modulation has three candidate maps.  Which one is used must
be decided by data, not by preference, and the decision must not touch a
target domain.  Photo Validation is a held-out part of the *source*
domain, so it is a legitimate selection set: choosing a hyper-parameter
there is exactly what a validation split is for.

The selection uses the reduced protocol (``NNDL_SUBSET``): every candidate
is generated for the same 200 content images and trained for the same
reduced step budget, so the comparison is balanced exactly as the full
protocol is.  The winner is then trained under the full protocol.

For each candidate the script reports the Photo Validation accuracy of
every seed, its mean and the standard error of the mean.  The candidate
with the highest mean wins; ties are broken by the seed-to-seed spread,
because a candidate that only wins by luck is not a better choice.

Usage:
    python select_foreground_map.py --seeds 0 1 2 --subset 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

#: The candidates, in the order they are tried.
CANDIDATES = ("spectral", "center", "spectral_center")


def subset_paths(kind: str, subset: int) -> tuple[Path, Path]:
    """The generation directory and the manifest of one candidate.

    The reduced sets are built by ``build_foreground_subsets.py``, which
    uses the generator's ``build`` mode so that the manifest lands in the
    same directory as the images.  The subset is taken with
    ``config.balanced_train_subset``, the same helper the trainer uses, so
    the generator and the trainer see exactly the same content images.
    """
    root = (config.DATA_ROOT / "aug_sp" / "previews" / f"build_n{subset}" /
            f"subset{subset}_{kind}_a0p5_b0p7")
    return root, root / "manifest_ours.txt"


def ensure_subset(kind: str, subset: int, alpha: float, beta: float) -> Path:
    """The manifest of one candidate, generating the set if it is missing."""
    root, manifest = subset_paths(kind, subset)
    if manifest.is_file():
        print(f"\n--- {kind}: reusing the existing set ---", flush=True)
        return manifest
    tag = f"subset{subset}_{kind}"
    command = [
        sys.executable, "-u", "-m", "member3_structure_preserving.generate",
        "build",
        "--limit", str(subset),
        "--balanced",
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--modulation", "foreground",
        "--foreground-kind", kind,
        "--cell-tag", tag,
        "--no-metrics",
        "--overwrite",
    ]
    print(f"\n--- generating {kind} ---", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(REPO))
    if result.returncode != 0:
        raise SystemExit(f"generation of '{kind}' failed")
    if not manifest.is_file():
        raise SystemExit(f"no manifest was produced for '{kind}' at {manifest}")
    return manifest


def train(kind: str, seed: int, subset: int, manifest: Path) -> dict:
    """Train one candidate on one seed and read its validation accuracy."""
    tag = f"fgsel_{kind}_s{seed}"
    environment = dict(os.environ)
    environment["NNDL_SUBSET"] = str(subset)
    environment["NNDL_NUM_WORKERS"] = "2"
    environment["NNDL_DATA_ROOT"] = str(config.DATA_ROOT)
    command = [
        sys.executable, "-u", "train.py",
        "--group", config.FOREGROUND_GROUP,
        "--seed", str(seed),
        "--tag", tag,
        "--aug-manifest", str(manifest),
    ]
    print(f"  training {tag} ...", flush=True)
    result = subprocess.run(command, cwd=str(HERE), env=environment)
    if result.returncode != 0:
        raise SystemExit(f"training of '{tag}' failed")
    summary_path = config.OUTPUT_ROOT / tag / "train_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--kinds", nargs="+", default=list(CANDIDATES),
                        choices=list(CANDIDATES),
                        help="which candidates to compare; the default is all three")
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--out", type=Path,
                        default=config.OUTPUT_ROOT / "foreground_selection.json")
    args = parser.parse_args()

    results = {}
    for kind in args.kinds:
        manifest = ensure_subset(kind, args.subset, args.alpha, args.beta)
        accuracies = []
        for seed in args.seeds:
            payload = train(kind, seed, args.subset, manifest)
            # ``reported_val_acc`` is the accuracy of the *final* checkpoint
            # of the fixed step budget.  ``best_val_acc`` is the best value
            # seen during training, which is a selection over the same
            # validation split that is being used to choose the map here.
            # Using it would make the selection itself a maximum over
            # several noisy measurements, so the final value is used.
            accuracy = payload.get("reported_val_acc")
            if accuracy is None:
                accuracy = payload.get("best_val_acc")
            accuracies.append(accuracy)
            print(f"    {kind} seed {seed}: photo-val {accuracy}", flush=True)
        values = np.asarray([a for a in accuracies if a is not None], dtype=float)
        results[kind] = {
            "photo_val_per_seed": accuracies,
            "photo_val_mean": float(values.mean()) if values.size else None,
            "photo_val_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "manifest": str(manifest),
        }

    print()
    print("=" * 72)
    print("PHOTO VALIDATION SELECTION (source-domain held-out split)")
    print("=" * 72)
    print(f"  {'candidate':<20}{'mean':>10}{'std':>10}   per seed")
    for kind in args.kinds:
        block = results[kind]
        mean = block["photo_val_mean"]
        std = block["photo_val_std"]
        seeds = ", ".join(f"{a*100:.2f}" for a in block["photo_val_per_seed"])
        print(f"  {kind:<20}{mean*100:>9.2f}%{std*100:>9.2f}%   {seeds}")

    # The winner is the highest mean; a tie is broken by the smaller
    # seed-to-seed spread, because a candidate that only wins by luck is
    # not a better choice.
    ranked = sorted(args.kinds,
                    key=lambda k: (-(results[k]["photo_val_mean"] or 0),
                                   results[k]["photo_val_std"]))
    winner = ranked[0]
    results["selected"] = winner
    results["selection_set"] = "photo_val"
    results["selection_protocol"] = {
        "subset": args.subset, "seeds": args.seeds,
        "alpha": args.alpha, "beta": args.beta,
        "candidates": list(args.kinds),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print(f"  selected: {winner}")
    print(f"  written: {args.out}")
    print()
    print("  The selection used Photo Validation, a held-out part of the")
    print("  source domain.  No target-domain image was used to choose the")
    print("  map.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
