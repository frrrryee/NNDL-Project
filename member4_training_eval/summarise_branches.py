"""Summarise every run that is not part of the main table.

The main table is the set of groups with the frozen fingerprint
``ecc6090e5b29``.  Everything else is either an ablation cell or an
exploratory branch, and this script prints one line per group so that
the state of those branches can be read at a glance.

Usage:
    python summarise_branches.py
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

MAIN_HASH = "ecc6090e5b29"
CROSS = ("art_painting_test", "cartoon_test", "sketch_test")


def cross_of(payload: dict) -> dict | None:
    """The cross-domain accuracies of one run, if they were recorded."""
    block = payload.get("cross_domain") or payload.get("targets")
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if isinstance(v, (int, float))}
    return None


def main() -> None:
    print(f"{'group':<30}{'n':>3}{'hash':>14}{'photo val':>12}"
          f"{'cross mean':>13}{'cross sd':>10}")
    print("-" * 84)
    groups: dict[str, list] = {}
    for run_dir in sorted(config.OUTPUT_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = run_dir / "train_summary.json"
        if not summary.is_file():
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        group = payload.get("group") or run_dir.name
        groups.setdefault(group, []).append((run_dir.name, payload))

    for group in sorted(groups):
        entries = groups[group]
        hashes = {p.get("config_hash", "?") for _n, p in entries}
        values, cross = [], []
        for _name, payload in entries:
            accuracy = payload.get("reported_val_acc")
            if accuracy is None:
                accuracy = payload.get("best_val_acc")
            if accuracy is not None:
                values.append(float(accuracy))
            block = cross_of(payload)
            if block:
                numbers = [v for v in block.values()
                           if isinstance(v, (int, float))]
                if numbers:
                    cross.append(float(np.mean(numbers)))
        hash_text = (hashes.pop() if len(hashes) == 1 else "MIXED")
        val_text = (f"{np.mean(values)*100:.2f}%" if values else "-")
        cross_text = (f"{np.mean(cross)*100:.2f}%" if cross else "-")
        cross_sd = (f"{np.std(cross, ddof=1)*100:.2f}%" if len(cross) > 1 else "-")
        marker = " " if hash_text == MAIN_HASH else "*"
        print(f"{marker}{group:<29}{len(entries):>3}{hash_text:>14}"
              f"{val_text:>12}{cross_text:>13}{cross_sd:>10}")
    print()
    print("  * = outside the main table's fingerprint (an ablation or an")
    print("      exploratory branch)")


if __name__ == "__main__":
    main()
