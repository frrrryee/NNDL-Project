"""The line-art family's results, with every difference signed correctly.

``compare_line_art.py`` compares the *gate* against each control, so
the difference it stores is ``gate - control``.  The soft vote is one
of those controls, which makes it easy to read its number as "the
vote beats the gate" when it in fact says the opposite.

This script states every comparison in both directions and prints the
per-domain split for the gate against its control and for the vote
against the gate, so that no sign is left to the reader's memory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT
TARGETS = ("art_painting_test", "cartoon_test", "sketch_test")

FAMILY = {
    "ours_line_attn": r"ours_line_attn_s\d+",
    "ours_line_vote": r"ours_line_vote_s\d+",
    "ours_line_plain_total": r"ours_line_plain_total_s\d+",
    "ours_line_plain_fg": r"ours_line_plain_fg_s\d+",
    "original": r"original_s\d+",
}


def tags_for(pattern: str) -> list[str]:
    import re
    found = [t for t in os.listdir(RUNS)
             if re.fullmatch(pattern, t) and (RUNS / t / "eval_last.json").is_file()]
    return sorted(found)


def read(tag: str) -> dict:
    return json.loads((RUNS / tag / "eval_last.json").read_text(encoding="utf-8"))


def cross(tag: str) -> float:
    record = read(tag)
    return sum(record[s] for s in TARGETS) / 3


def compare(a_tags: list[str], b_tags: list[str], label: str) -> None:
    """The paired statistics of ``a`` against ``b``, both directions."""
    a = np.array([cross(t) for t in a_tags[:len(b_tags)]])
    b = np.array([cross(t) for t in b_tags[:len(a_tags)]])
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / np.sqrt(n)) if sd > 1e-12 else 0.0
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1)) if n > 1 else 1.0
    except ImportError:
        p = float("nan")
    print(f"  {label:<46}{mean*100:>+8.2f} points   p = {p:.4f}   "
          f"n = {n}   a worse on {int((diff < 0).sum())}/{n} seeds")

    print(f"      {a.mean()*100:.2f}%  vs  {b.mean()*100:.2f}%")


def per_domain(a_tags: list[str], b_tags: list[str], label: str) -> None:
    print(f"  {label}")
    for split in list(TARGETS) + ["photo_val"]:
        a = np.mean([read(t)[split] for t in a_tags])
        b = np.mean([read(t)[split] for t in b_tags])
        print(f"      {split:<22}{a*100:>8.2f}%{b*100:>8.2f}%{(a-b)*100:>+9.2f}")


def main() -> None:
    table = {name: tags_for(pattern) for name, pattern in FAMILY.items()}
    for name, tags in table.items():
        if not tags:
            print(f"  {name}: not trained")
            return

    print()
    print("=" * 88)
    print("THE LINE-ART FAMILY, EVERY COMPARISON IN BOTH DIRECTIONS")
    print("=" * 88)
    print()
    for name, tags in table.items():
        values = np.array([cross(t) for t in tags])
        print(f"  {name:<26}{values.mean()*100:>7.2f}%   "
              f"std {values.std(ddof=1)*100:>5.2f}%   n = {len(tags)}")
    print()
    print("  cross-domain accuracy, five seeds")
    print()

    gate = table["ours_line_attn"]
    vote = table["ours_line_vote"]
    total = table["ours_line_plain_total"]
    fg = table["ours_line_plain_fg"]
    original = table["original"]

    print("=" * 88)
    print("PAIRED COMPARISONS (cross-domain accuracy)")
    print("=" * 88)
    print()
    compare(gate, total, "gate - plain_total (total-ink matched)")
    print()
    compare(gate, fg, "gate - plain_foreground (fg-ink matched)")
    print()
    compare(vote, gate, "vote - gate")
    print()
    compare(vote, total, "vote - plain_total (total-ink matched)")
    print()
    compare(vote, fg, "vote - plain_foreground (fg-ink matched)")
    print()
    compare(gate, original, "gate - original")
    print()
    compare(vote, original, "vote - original")

    print()
    print("=" * 88)
    print("PER-DOMAIN SPLIT")
    print("=" * 88)
    print()
    per_domain(gate, total, "gate - plain_total (the matched control)")
    print()
    per_domain(vote, gate, "vote - gate")


if __name__ == "__main__":
    main()
