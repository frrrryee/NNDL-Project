"""The per-class accuracy table, for every domain.

The per-class numbers are the only place where the *kind* of image is
visible.  A mean accuracy says that a domain is hard; the per-class
table says which objects are hard in it, and a class that is easy in
one domain and impossible in another is telling us something specific
about the domain rather than about the object.

Two readings matter here:

* the class ordering -- which objects survive the domain shift and
  which do not;
* the *spread* -- whether a class is uniformly mediocre or bimodal.

The bimodality matters because the failure the project has been
chasing is a failure of the method, and the failure being described
here is a failure of the *image*: an image in which the object is not
recognisable to a human should not be recoverable by any method.
Those two look identical in a mean and are separable per class.

Usage:
    python show_per_class.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RESULTS = REPO / "results" / "runs"

TARGETS = ("art_painting_test", "cartoon_test", "sketch_test")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")


def load(group: str) -> dict:
    """The summary of one group."""
    path = RESULTS / f"summary_{group}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def per_class(summary: dict, split: str) -> dict:
    block = (summary.get("per_class") or {}).get(split) or {}
    return {name: block[name]["mean"] for name in CLASSES if name in block}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="adain_a1p0_b0p7")
    parser.add_argument("--compare", default="original",
                        help="a second group to put side by side")
    args = parser.parse_args()

    summary = load(args.group)
    if not summary:
        print(f"  no summary for {args.group}")
        return 1
    other = load(args.compare)

    print()
    print("=" * 92)
    print(f"PER-CLASS ACCURACY -- {args.group}")
    print("=" * 92)
    print()
    header = f"  {'class':<12}{'photo':>9}{'art':>9}{'cartoon':>10}{'sketch':>9}{'mean x-domain':>15}"
    print(header)
    print("  " + "-" * 70)

    rows = []
    for name in CLASSES:
        values = {}
        for split in ("photo_val",) + TARGETS:
            block = per_class(summary, split)
            values[split] = block.get(name)
        cross = [values[s] for s in TARGETS if values[s] is not None]
        if not cross:
            continue
        mean = sum(cross) / len(cross)
        rows.append((mean, name, values))

    for mean, name, values in rows:
        cells = "".join(
            f"{values[s]*100:>9.1f}%" if values[s] is not None else f"{'-':>10}"
            for s in ("photo_val",) + TARGETS)
        print(f"  {name:<12}{cells}{mean*100:>14.1f}%")

    print()
    print(f"  {'domain':<12}{'mean':>9}{'worst class':>20}{'best class':>20}{'spread':>10}")
    print("  " + "-" * 66)
    for split in ("photo_val",) + TARGETS:
        block = per_class(summary, split)
        if not block:
            continue
        ordered = sorted(block.items(), key=lambda kv: kv[1])
        worst, worst_value = ordered[0]
        best, best_value = ordered[-1]
        average = sum(block.values()) / len(block)
        print(f"  {split:<12}{average*100:>8.1f}%"
              f"{worst+f' {worst_value*100:.1f}%':>20}"
              f"{best+f' {best_value*100:.1f}%':>20}"
              f"{(best_value-worst_value)*100:>9.1f}")

    if other:
        print()
        print("=" * 92)
        print(f"SAME TABLE FOR {args.compare} -- WHAT THE AUGMENTATION CHANGED")
        print("=" * 92)
        print()
        print(f"  {'class':<12}" + "".join(
            f"{s.replace('_test','').replace('_val',''):>12}"
            for s in ("photo_val",) + TARGETS))
        print("  " + "-" * 60)
        for mean, name, _ in rows:
            cells = ""
            for split in ("photo_val",) + TARGETS:
                a = per_class(summary, split).get(name)
                b = per_class(other, split).get(name)
                if a is None or b is None:
                    cells += f"{'-':>12}"
                else:
                    cells += f"{(a-b)*100:>+11.1f}%"
            print(f"  {name:<12}{cells}")

    print()
    print("  A class that is near zero in one domain and high in another")
    print("  is a domain problem, not an object problem.  A class that")
    print("  is near zero everywhere is an object problem, and no")
    print("  augmentation is going to fix it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
