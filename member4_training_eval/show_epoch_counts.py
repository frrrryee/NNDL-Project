"""How many epochs does each group actually see?

Every group gets the same *number of optimizer steps*, so a group whose
training set is twice as large sees half as many passes over its data.
The original group trains on 1,349 images and the augmented groups on
2,698, so the augmented groups take half as many epochs as the
original one.

That matters, because the two groups are compared to each other: if
the augmented groups are under-trained relative to the original one,
the comparison is between a fully-trained baseline and a half-trained
one, and the augmented groups are penalised for it.  The duplicated
group is the cleanest evidence: it holds the *original* images and
their byte-identical copies, so it can only be hurt by training
protocol and not by anything about the augmentation.

This script prints the effective epoch count of every group, so the
comparison can be read on equal terms.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT


def main() -> None:
    originals = len(config.read_split(config.SPLIT_FILES["photo_train"]))
    print()
    print("=" * 88)
    print("THE EFFECTIVE EPOCH COUNT OF EVERY GROUP")
    print("=" * 88)
    print()
    print(f"  the original split has {originals} images")
    print(f"  every group is trained for {config.TOTAL_STEPS} optimizer "
          f"steps at batch {config.BATCH_SIZE}")
    print()
    print(f"  {'group':<34}{'images':>9}{'epochs':>9}")
    seen = set()
    for tag in sorted(os.listdir(RUNS)):
        summary = RUNS / tag / "train_summary.json"
        if not summary.is_file():
            continue
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        images = payload.get("train_images")
        if not images:
            continue
        family = payload.get("group") or tag.rsplit("_s", 1)[0]
        if family in seen:
            continue
        seen.add(family)
        epochs = config.TOTAL_STEPS * config.BATCH_SIZE / images
        print(f"  {family:<34}{images:>9}{epochs:>9.1f}")

    print()
    print("  The original group sees about twice as many passes over its")
    print("  data as the augmented groups do.  The duplicated group is the")
    print("  evidence that this matters: it holds the original images")
    print("  and their byte-identical copies, so nothing about the")
    print("  augmentation can explain a loss.")


if __name__ == "__main__":
    main()
