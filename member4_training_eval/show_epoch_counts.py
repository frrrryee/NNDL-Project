"""How many epochs does each group actually see?

Under the epoch-based protocol every group is trained for the same number
of *passes over its own training set*: the step budget is derived from the
size of the data (``30 * ceil(N / 64)``) instead of being a constant that
is handed out to every group.

That distinction matters.  The Original group trains on 1,349 images and
the augmented groups on 2,698, so the old fixed 660-step budget gave the
Original group 31.3 epochs and every augmented group only 15.7 -- the
augmented groups were compared against a baseline that had trained twice
as long.  The duplicated group is the cleanest evidence: it holds the
*original* images and their byte-identical copies, so it can only be hurt
by training protocol and not by anything about the augmentation.  It lost
3.14 points to the Original group for exactly that reason.

This script prints, for every group, both the budget the runs on disk
were actually trained with and the budget the current protocol assigns to
it, so that the state of the results is visible at a glance: a group whose
recorded budget differs from its current one has to be retrained before it
can be compared with the others.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT

#: The run directories that are diagnostics rather than reported groups.
SKIP = ("diag_", "_archive", "fgsel_", "epfix_")


def main() -> None:
    originals = len(config.read_split(config.SPLIT_FILES["photo_train"]))
    print()
    print("=" * 96)
    print("THE EFFECTIVE EPOCH COUNT OF EVERY GROUP")
    print("=" * 96)
    print()
    print(f"  the original split has {originals} images")
    print(f"  the protocol gives every group {config.ORIGINAL_EPOCHS} epochs over "
          f"its own training set at batch {config.BATCH_SIZE}")
    print()
    print(f"  {'group':<28}{'images':>7}{'steps':>7}{'epochs':>8}"
          f"{'':>4}{'now':>7}{'epochs':>8}")
    print(f"  {'':<28}{'':>7}{'':>7}{'':>8}{'':>4}{'steps':>7}{'':>8}")
    stale = []
    seen = set()
    for tag in sorted(os.listdir(RUNS)):
        summary = RUNS / tag / "train_summary.json"
        if not summary.is_file():
            continue
        if any(part.startswith(SKIP) for part in (tag,)):
            continue
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        images = payload.get("train_images")
        steps = payload.get("total_steps")
        if not images or not steps:
            continue
        family = payload.get("group") or tag.rsplit("_s", 1)[0]
        if family in seen:
            continue
        seen.add(family)
        epochs = steps * config.BATCH_SIZE / images
        # The budget the current protocol assigns to this group.
        try:
            manifest = config.DATA_ROOT / config.GROUP_MANIFESTS[family] \
                if family in config.GROUP_MANIFESTS else None
            wanted = config.total_steps_of_group(family, manifest)
        except (OSError, ValueError):
            wanted = 0
        mark = ""
        if wanted and wanted != steps:
            mark = "  <- stale"
            stale.append(family)
        wanted_epochs = (wanted * config.BATCH_SIZE / images) if wanted else 0
        print(f"  {family:<28}{images:>7}{steps:>7}{epochs:>8.1f}"
              f"{mark:>4}{wanted:>7}{wanted_epochs:>8.1f}")

    print()
    print(f"  The runs on disk were trained with the budget in the third column;")
    print(f"  the protocol now assigns the budget in the last two columns.  A group")
    print(f"  marked 'stale' was trained under the old fixed-step protocol and has")
    print(f"  to be retrained before it can be compared with the others.")
    if stale:
        print()
        print(f"  {len(stale)} group(s) are stale: {', '.join(sorted(stale))}")


if __name__ == "__main__":
    main()
