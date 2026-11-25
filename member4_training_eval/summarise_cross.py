"""Summarise the cross-domain accuracy of every branch.

The cross-domain accuracy is the number the paper reports, and it is
stored per run in ``eval_last.json``.  This script aggregates it over the
seeds of each group and prints it next to the main table's groups, so
that the ablation and the exploratory branches can be compared with the
main groups under one view.

Usage:
    python summarise_cross.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

MAIN_HASH = "ecc6090e5b29"
#: The order the main table is reported in.
MAIN_ORDER = ("original", "duplicate", "nonstyle", "adain", "adain_matched",
              "ours", "ours_equal")


def read_eval(run_dir: Path) -> dict | None:
    path = run_dir / "eval_last.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    groups: dict[str, list[tuple[str, dict]]] = {}
    for run_dir in sorted(config.OUTPUT_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        payload = read_eval(run_dir)
        if payload is None:
            continue
        group = payload.get("group") or run_dir.name
        groups.setdefault(group, []).append((run_dir.name, payload))

    print(f"{'group':<30}{'n':>3}{'cross':>9}{'sd':>8}"
          f"{'art':>8}{'cartoon':>9}{'sketch':>8}{'photo val':>11}  hash")
    print("-" * 100)
    ordered = [g for g in MAIN_ORDER if g in groups]
    ordered += sorted(g for g in groups if g not in MAIN_ORDER)
    for group in ordered:
        entries = groups[group]
        cross, art, cartoon, sketch, val = [], [], [], [], []
        hashes = set()
        for _name, payload in entries:
            hashes.add(payload.get("config_hash", "?"))
            value = payload.get("cross_domain_avg")
            if value is not None:
                cross.append(float(value))
            for key, sink in (("art_painting_test", art),
                              ("cartoon_test", cartoon),
                              ("sketch_test", sketch),
                              ("photo_val", val)):
                number = payload.get(key)
                if number is not None:
                    sink.append(float(number))
        hash_text = (hashes.pop() if len(hashes) == 1 else "MIXED")
        marker = " " if hash_text == MAIN_HASH else "*"

        def fmt(values):
            if not values:
                return "-"
            mean = f"{np.mean(values)*100:.2f}%"
            if len(values) > 1:
                mean += f"({np.std(values, ddof=1)*100:.1f})"
            return mean

        print(f"{marker}{group:<29}{len(entries):>3}{fmt(cross):>9}"
              f"{'':>8}{fmt(art):>8}{fmt(cartoon):>9}{fmt(sketch):>8}"
              f"{fmt(val):>11}  {hash_text}")
    print()
    print("  * = outside the main table's fingerprint")
    print("  the cross column is the mean over art painting, cartoon and sketch")


if __name__ == "__main__":
    main()
