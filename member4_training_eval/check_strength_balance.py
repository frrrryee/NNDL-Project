"""Is the main comparison strength-matched in fact?

The main experiment compares ``ours`` (a spatially modulated AdaIN)
against ``adain_matched`` (a uniform AdaIN whose strength is the mean of
the spatial alpha field).  The comparison is only meaningful if the two
really do apply the same amount of style.

The selection over the foreground candidates showed why this has to be
checked rather than assumed: a candidate whose map covers a large area
applies less total style, and a less-styled image keeps more of its
edges for a reason that has nothing to do with structure preservation.
When the strength was matched properly, the apparent +0.0336 gain of
the centre prior collapsed to +0.0002 with p = 0.74.

This script reads the recorded generation of every group and prints the
mean alpha field of each one next to the mean alpha of its matched
control, so that the balance of the main comparison can be read off
directly.

Usage:
    python check_strength_balance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

#: The generation records of the project, with their human-readable name.
RECORDS = {
    "ours (main set)": config.DATA_ROOT / "aug_sp" / "generation.json",
    "ours_fg (repaired)": (config.DATA_ROOT / "aug_sp" / "foreground" /
                           "generation.json"),
}


def means(path: Path):
    """The mean alpha field and the mean matched strength of one record."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = list((payload.get("outputs") or {}).values())
    if not entries:
        return None
    map_values = [e["metrics"]["mean_alpha"] for e in entries
                  if e.get("metrics") and "mean_alpha" in e["metrics"]]
    matched_values = [e.get("matched_alpha") for e in entries
                      if e.get("matched_alpha") is not None]
    if not map_values:
        return None
    return (float(sum(map_values) / len(map_values)),
            float(sum(matched_values) / len(matched_values))
            if matched_values else None,
            len(map_values))


def main() -> None:
    print()
    print("=" * 84)
    print("IS THE MAIN COMPARISON STRENGTH-MATCHED?")
    print("=" * 84)
    print(f"  {'set':<22}{'map alpha':>12}{'ctrl alpha':>12}{'difference':>13}"
          f"{'n':>7}")
    for name, path in RECORDS.items():
        result = means(path)
        if result is None:
            print(f"  {name:<22}{'(no record)':>16}")
            continue
        map_alpha, matched_alpha, count = result
        if matched_alpha is None:
            print(f"  {name:<22}{map_alpha:>12.4f}{'(none)':>12}{'':>13}{count:>7}")
            continue
        difference = map_alpha - matched_alpha
        print(f"  {name:<22}{map_alpha:>12.4f}{matched_alpha:>12.4f}"
              f"{difference:>+13.4f}{count:>7}")
    print()
    print("  The control is *constructed* to have the mean of the spatial")
    print("  alpha field, so the difference should be at the level of the")
    print("  numerical precision of the recorded means.  A difference of")
    print("  more than about 0.01 would mean the comparison is confounded")
    print("  with the amount of style applied.")


if __name__ == "__main__":
    main()
