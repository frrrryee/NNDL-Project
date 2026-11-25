"""Every completed group, in one table.

The point of this script is to make the *whole* search space visible at
once.  A new direction is worth trying only if it is not a variation of
something that has already been measured, and the fastest way to know
whether that is the case is to have every result in front of one.

The table is sorted by the cross-domain accuracy, so the best method
found so far is at the top and the distance from it is visible.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT
TARGETS = ("art_painting_test", "cartoon_test", "sketch_test")


def read(tag: str) -> dict | None:
    path = RUNS / tag / "eval_last.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect() -> dict[str, list[dict]]:
    """Every run, grouped by the family its tag name belongs to."""
    groups = defaultdict(list)
    for tag in sorted(os.listdir(RUNS)):
        record = read(tag)
        if not record:
            continue
        if not all(split in record for split in TARGETS):
            continue
        family = re.sub(r"_s\d+$", "", tag)
        groups[family].append(record)
    return groups


def main() -> int:
    groups = collect()

    rows = []
    for family, records in groups.items():
        cross = np.array([sum(r[s] for s in TARGETS) / 3
                          for r in records])
        art = np.mean([r["art_painting_test"] for r in records])
        cartoon = np.mean([r["cartoon_test"] for r in records])
        sketch = np.mean([r["sketch_test"] for r in records])
        photo = np.mean([r.get("photo_val", 0.0) for r in records])
        rows.append((float(cross.mean()), float(cross.std(ddof=1))
                     if len(cross) > 1 else 0.0, family, len(records),
                     float(art), float(cartoon), float(sketch), float(photo)))

    rows.sort(reverse=True)

    print()
    print("=" * 96)
    print("EVERY COMPLETED GROUP, BY CROSS-DOMAIN ACCURACY")
    print("=" * 96)
    print(f"  {'group':<32}{'cross':>8}{'std':>7}{'photo':>8}{'art':>8}"
          f"{'cartoon':>9}{'sketch':>8}{'n':>4}")
    for cross, std, family, n, art, cartoon, sketch, photo in rows:
        print(f"  {family:<32}{cross*100:>7.2f}%{std*100:>6.2f}%"
              f"{photo*100:>7.2f}%{art*100:>7.2f}%{cartoon*100:>8.2f}%"
              f"{sketch*100:>7.2f}%{n:>4}")

    # The best result, and what it costs to beat it.
    best = rows[0]
    print()
    print("=" * 96)
    print("THE BEST RESULT SO FAR")
    print("=" * 96)
    print()
    print(f"  {best[2]:<32}{best[0]*100:.2f}%  ({best[3]} seeds)")
    print()
    print("  Anything new has to beat this, and it has to beat it with a")
    print("  mechanism that is not 'apply less style' -- because that")
    print("  reading of the earlier results turned out to be wrong.")

    out = RUNS / "summary_all_groups.json"
    out.write_text(json.dumps(
        [{"group": r[2], "cross": r[0], "std": r[1], "n": r[3],
          "art": r[4], "cartoon": r[5], "sketch": r[6], "photo_val": r[7]}
         for r in rows], indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
