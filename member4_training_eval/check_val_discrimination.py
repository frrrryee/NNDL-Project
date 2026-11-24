"""Compare the Photo Validation accuracy of the main groups.

The question this answers is whether Photo Validation can be used to
*select* anything.  If every group reaches the same accuracy on it, the
split has no discriminative power, and a selection made on it is a
selection on noise.

The script prints the per-seed Photo Validation accuracy of every group
that has been trained under the full protocol, together with the spread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import config  # noqa: E402

GROUPS = ("original", "adain", "adain_matched", "ours", "duplicate",
          "nonstyle", "mixstyle")


def main() -> None:
    print(f"{'group':<18}{'n':>3}{'mean val':>12}{'std':>9}   per seed")
    print("-" * 72)
    all_values = []
    for group in GROUPS:
        values = []
        for run_dir in sorted(config.OUTPUT_ROOT.glob(f"{group}_s*")):
            summary = run_dir / "train_summary.json"
            if not summary.is_file():
                continue
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if payload.get("subset_size"):
                # A reduced-protocol run is not comparable with a full one.
                continue
            accuracy = payload.get("reported_val_acc")
            if accuracy is None:
                accuracy = payload.get("best_val_acc")
            if accuracy is not None:
                values.append(float(accuracy))
        if not values:
            continue
        array = np.asarray(values)
        all_values.append(array)
        seeds = ", ".join(f"{v*100:.2f}" for v in array)
        print(f"{group:<18}{len(array):>3}{array.mean()*100:>11.2f}%"
              f"{array.std(ddof=1)*100 if len(array) > 1 else 0:>8.2f}%   {seeds}")
    if all_values:
        pooled = np.concatenate(all_values)
        print("-" * 72)
        print(f"{'pooled':<18}{len(pooled):>3}{pooled.mean()*100:>11.2f}%"
              f"{pooled.std(ddof=1)*100:>8.2f}%")
        print()
        print(f"  the whole spread across every group is "
              f"{pooled.max()*100 - pooled.min()*100:.2f} percentage points, "
              f"and one image on the validation split is worth "
              f"{100/len(config.read_split(config.SPLIT_FILES['photo_val'])):.2f} points.")


if __name__ == "__main__":
    main()
