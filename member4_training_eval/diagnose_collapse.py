"""Is the collapse explained by the training set, or by the domain?

The confusion matrices show a collapse: on the sketch domain almost
every image is called a giraffe, and on the cartoon domain almost every
image is called a guitar.  A collapse to *one* class is not the
signature of an unrecognisable image -- an unrecognisable image
produces a spread of wrong answers.  A collapse is the signature of a
model that has one class it defaults to.

Two things could produce it:

1. The training set is imbalanced, and the two classes that dominate
   the collapse are the ones that are over-represented in it.
2. The collapse target is the class whose visual signature survives
   the domain shift best -- the class that is a shape before it is a
   texture.

The two are separable, because they make different predictions about
*which* class the model falls back on.  The imbalance account says the
fallback is the largest class in the training set.  The visual account
says it is the class with the most distinctive silhouette.

This script prints both rankings side by side with the observed
fallback, so that the account can be checked rather than argued.

Usage:
    python diagnose_collapse.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
import data  # noqa: E402

CLASSES = list(config.CLASSES)

#: The observed fallback class of each target domain, read off the
#: confusion matrices of the trained runs.
OBSERVED = {
    "art_painting_test": None,
    "cartoon_test": "guitar",
    "sketch_test": "giraffe",
}


def training_counts() -> Counter:
    """How many training images each class has."""
    counts = Counter()
    for relative, label in config.read_split(config.SPLIT_FILES["photo_train"]):
        counts[CLASSES[label - 1]] += 1
    return counts


def test_counts(split: str) -> Counter:
    """How many test images each class has."""
    counts = Counter()
    for relative, label in config.read_split(config.SPLIT_FILES[split]):
        counts[CLASSES[label - 1]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="sketch_test")
    args = parser.parse_args()

    train = training_counts()
    total = sum(train.values())

    print()
    print("=" * 88)
    print("THE TRAINING SET")
    print("=" * 88)
    print()
    print(f"  {'class':<12}{'images':>9}{'share':>9}")
    for name, count in train.most_common():
        print(f"  {name:<12}{count:>9}{count/total*100:>8.1f}%")

    print()
    print("=" * 88)
    print("THE FALLBACK TARGETS OBSERVED IN THE CONFUSION MATRICES")
    print("=" * 88)
    print()
    print(f"  {'domain':<22}{'falls back to':>16}{'train share':>14}"
          f"{'test share':>13}")
    print("  " + "-" * 63)
    for split, fallback in OBSERVED.items():
        if not fallback:
            continue
        tests = test_counts(split)
        test_total = sum(tests.values())
        print(f"  {split:<22}{fallback:>16}"
              f"{train[fallback]/total*100:>13.1f}%"
              f"{tests[fallback]/test_total*100:>12.1f}%")

    print()
    print("  For the imbalance account to hold, the fallback class must")
    print("  be the largest class in the training set.")
    print()
    largest = train.most_common(1)[0][0]
    print(f"  the largest training class is '{largest}' "
          f"({train[largest]/total*100:.1f}%)")
    for split, fallback in OBSERVED.items():
        if not fallback:
            continue
        verdict = "MATCHES" if fallback == largest else "does NOT match"
        print(f"    {split:<22} falls back to '{fallback}'  ->  {verdict}")

    print()
    print("=" * 88)
    print("AND THE TEST SETS THEMSELVES")
    print("=" * 88)
    print()
    for split in ("art_painting_test", "cartoon_test", "sketch_test"):
        tests = test_counts(split)
        test_total = sum(tests.values())
        print(f"  {split}")
        print("    " + "  ".join(
            f"{name} {count/test_total*100:4.1f}%"
            for name, count in tests.most_common()))
    print()
    print("  A test set whose classes are balanced does not collapse")
    print("  because of its own composition; the collapse is coming from")
    print("  the model.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
