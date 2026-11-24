"""Run the line-art training sweeps one group after another.

The four line-art groups are independent, but they are *not* run in
parallel: each training job holds a model, a dataloader and its
workers in memory, and four concurrent jobs exhaust the page file on a
machine with sixteen gigabytes of RAM.  The groups are therefore run
serially, which costs wall-clock time but keeps the machine healthy.

Usage:
    python run_line_art_sweep.py
    python run_line_art_sweep.py --groups ours_line_attn ours_line_vote
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import config  # noqa: E402

#: The line-art groups, in the order they are run.
DEFAULT_ORDER = ("ours_line_attn", "ours_line_vote",
                 "ours_line_plain_total", "ours_line_plain_fg")


def run_group(group: str, seeds: list[int]) -> int:
    """Run one group's sweep; return the exit status of the child."""
    print()
    print("=" * 78)
    print(f"=== {group}   seeds={seeds}")
    print("=" * 78)
    sys.stdout.flush()
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-u", "run_sweep.py", "--group", group,
         "--seeds", *[str(s) for s in seeds]],
        cwd=str(HERE), env={**dict(__import__("os").environ),
                            "PYTHONIOENCODING": "utf-8"})
    print(f"=== {group} finished in {(time.time() - started)/60:.1f} min "
          f"(exit {result.returncode})")
    sys.stdout.flush()
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_ORDER),
                        help="the groups to run, in order")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(config.SEEDS))
    args = parser.parse_args()

    for group in args.groups:
        if group not in config.LINE_ART_GROUPS:
            parser.error(f"'{group}' is not a line-art group")
        status = run_group(group, args.seeds)
        if status != 0:
            print(f"\n!!! {group} failed with exit status {status}; "
                  f"the remaining groups are not run")
            return status
    print()
    print("all line-art groups are done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
