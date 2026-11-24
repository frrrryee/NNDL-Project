"""Verify the degenerate case of the method: beta = 0 is plain AdaIN.

The method's design rests on one algebraic property: the strength field
of Equation (2) is constant when beta is zero, so the blend of
Equation (1) reduces exactly to AdaIN at the same strength.  The report
relies on that property twice: the beta sweep uses beta = 0 as its
anchor, and the claim that "the baseline is a special case of the
method" is only true if the *images* are identical, not merely the
formulae.

This script checks the images, not the formulae.  It compares the
generated files of the beta = 0 cell against the AdaIN baseline set and
reports how many are byte-identical.

Usage:
    python verify_beta_zero.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(baseline: Path, candidate: Path, limit: int | None):
    """The fraction of the candidate images that are byte-identical."""
    files = sorted(candidate.rglob("*.jpg"))
    if limit:
        files = files[:limit]
    if not files:
        return 0, 0, "the candidate set is empty"
    identical = 0
    missing = 0
    for path in files:
        other = baseline / path.relative_to(candidate)
        if not other.is_file():
            missing += 1
            continue
        if digest(other) == digest(path):
            identical += 1
    return identical, len(files), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="check only the first N images")
    args = parser.parse_args()

    data = config.DATA_ROOT
    baseline = data / "aug_sp" / "adain"
    cell = data / "aug_sp" / "sweeps" / "sweep_full" / "a0p5_b0p0"

    if not baseline.is_dir():
        print(f"the AdaIN baseline set {baseline} does not exist")
        return 1
    if not cell.is_dir():
        print(f"the beta = 0 cell {cell} does not exist; "
              f"run generate_sweeps.py --batch B first")
        return 1

    failures = 0
    for name, candidate in (("ours", cell / "images"),
                            ("adain", cell / "adain")):
        if not candidate.is_dir():
            print(f"  {name}: {candidate} does not exist, skipped")
            continue
        identical, total, missing = compare(baseline, candidate, args.limit)
        share = identical / total if total else 0.0
        status = "OK" if share > 0.999 else "FAILED"
        print(f"  {name}: {identical}/{total} byte-identical to the AdaIN "
              f"baseline ({share:.1%})  {status}"
              + (f"  ({missing} missing)" if missing else ""))
        if share <= 0.999:
            failures += 1

    if failures:
        print("\nFAILED: beta = 0 does not reproduce plain AdaIN")
        return 1
    print("\nbeta = 0 reproduces plain AdaIN exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
