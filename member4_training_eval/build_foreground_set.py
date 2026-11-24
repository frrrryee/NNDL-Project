"""Generate the repaired-modulation augmentation set.

This is the full-set build of the foreground variant described in
``member3_structure_preserving/foreground_map.py``.  It writes:

    <DATA_ROOT>/aug_sp/foreground/images/...      the styled images
    <DATA_ROOT>/aug_sp/foreground/manifest_ours.txt
    <DATA_ROOT>/aug_sp/foreground/adain_matched/...
    <DATA_ROOT>/aug_sp/foreground/manifest_adain_matched.txt
    <DATA_ROOT>/aug_sp/foreground/analysis.json

The manifest lives in its own directory so that it can never be confused
with the manifest of the original method, and the group carries its own
configuration fingerprint for the same reason.

Usage:
    python build_foreground_set.py --kind spectral_center
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
    parser.add_argument("--kind", default=config.FOREGROUND_KIND,
                         choices=("spectral", "center", "spectral_center"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    command = [
        sys.executable, "-u", "-m", "member3_structure_preserving.generate",
        "build",
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--modulation", "foreground",
        "--foreground-kind", args.kind,
        "--threads", str(args.threads),
        "--with-matched",
        "--metrics",
        "--overwrite",
        "--cell-tag", "foreground",
    ]
    # The build mode writes into ``aug_sp/`` directly; the repaired
    # variant needs its own directory, so the cell tag is used to place
    # it in ``aug_sp/foreground/``.
    print("running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(REPO))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
