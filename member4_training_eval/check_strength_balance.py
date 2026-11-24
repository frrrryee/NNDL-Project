"""Is every group's comparison strength-matched?

The main experiment compares a spatially modulated AdaIN against a
uniform AdaIN whose strength is the mean of the spatial alpha field.
The comparison is only meaningful if the two really do apply the same
amount of style.

This has to be checked rather than assumed, because a map that covers a
large area applies less total style, and a less-styled image keeps more
of its edges for a reason that has nothing to do with structure
preservation.  The centre prior covers 34% of the pixels, so its alpha
field is markedly lower than the Sobel one -- and a group whose
comparison is not matched would lose accuracy for that reason alone.

The generation records of every set are read here, and each map's mean
alpha is printed next to the mean alpha of its own matched control.
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
    "ours (Sobel)": config.DATA_ROOT / "aug_sp" / "generation.json",
    "ours_fg (spectral_center)": (config.DATA_ROOT / "aug_sp" / "foreground" /
                                   "generation.json"),
    "ours_fg_center (centre)": (config.DATA_ROOT / "aug_sp" / "fgcenter" /
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
            (float(sum(matched_values) / len(matched_values))
             if matched_values else None),
            len(map_values))


def main() -> None:
    print()
    print("=" * 84)
    print("IS EVERY COMPARISON STRENGTH-MATCHED?")
    print("=" * 84)
    print(f"  {'set':<28}{'map alpha':>12}{'ctrl alpha':>12}{'difference':>13}"
          f"{'n':>7}")
    unmatched = []
    for name, path in RECORDS.items():
        result = means(path)
        if result is None:
            print(f"  {name:<28}{'(no record)':>16}")
            continue
        map_alpha, matched_alpha, count = result
        if matched_alpha is None:
            print(f"  {name:<28}{map_alpha:>12.4f}{'(none)':>12}{'':>13}"
                  f"{count:>7}")
            unmatched.append(name)
            continue
        difference = map_alpha - matched_alpha
        print(f"  {name:<28}{map_alpha:>12.4f}{matched_alpha:>12.4f}"
              f"{difference:>+13.4f}{count:>7}")
    print()
    print("  The control is *constructed* to have the mean of the spatial")
    print("  alpha field, so the difference should be at the level of the")
    print("  numerical precision of the recorded means.  A difference of")
    print("  more than about 0.01 would mean the comparison is confounded")
    print("  with the amount of style applied.")
    print()
    print("  Note that the alpha fields differ *between* the sets, which is")
    print("  expected and not a problem: each set is compared with its own")
    print("  control.  What matters is the difference within a row.")
    if unmatched:
        print()
        print(f"  WARNING: {len(unmatched)} set(s) have no matched control:")
        for name in unmatched:
            print(f"    - {name}")
        print("  A group without a matched control cannot be compared with")
        print("  the main groups: its result would be confounded with the")
        print("  amount of style it applied.")


if __name__ == "__main__":
    main()
