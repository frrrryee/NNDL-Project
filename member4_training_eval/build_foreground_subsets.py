"""Build the reduced sets used to select the foreground map.

The three candidate maps -- ``spectral``, ``center`` and ``spectral_center``
-- are compared on Photo Validation only.  The comparison uses the
reduced protocol: 200 training images and a proportionally reduced step
budget.  The subset is taken with the *same* helper the trainer uses
(``config.balanced_train_subset``), so the generator and the trainer see
exactly the same 200 content images.

Nothing here touches a target domain: the selection criterion is
Photo Validation accuracy, and the target domains are only read by the
final evaluation of the five-seed run.

Usage:
    python build_foreground_subsets.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from member4_training_eval import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", nargs="+",
                        default=("spectral", "center", "spectral_center"))
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()

    for kind in args.kinds:
        command = [
            sys.executable, "-u", "-m", "member3_structure_preserving.generate",
            "build",
            "--limit", str(args.subset),
            "--balanced",
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--modulation", "foreground",
            "--foreground-kind", kind,
            "--threads", str(args.threads),
            "--cell-tag", f"subset{args.subset}_{kind}",
            "--overwrite",
        ]
        print("running:", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=str(REPO))
        if result.returncode != 0:
            print(f"FAILED for {kind}", flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
