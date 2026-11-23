"""Produce every table and figure of the final report from the recorded runs.

One command regenerates the whole set of artefacts, so that the tables and
the figures in the paper can never disagree with the runs:

    python make_all_outputs.py

It writes

  report/sections/member4_tableii.tex    the main results table
  report/sections/member4_tableiii.tex   the significance overview
  report/sections/member4_tablevi.tex    the decomposition of the gain
  report/sections/member4_ivb.tex        the main-results prose
  report/sections/member4_ivc.tex        the strength-relation prose
  report/sections/member5_abstract.tex   the abstract
  figures/results_baseline.png           the grouped bar chart
  figures/strength_curve.png             Fig. 2, the main figure
  figures/forest_plot.png                Fig. 3, the forest plot
  figures/edge_cdf.png                   Fig. 4, the EdgeScore CDF
  figures/per_class_heatmap.png          Fig. 9, the per-class heat map
  figures/per_class_gains.png            the per-class gain chart
  figures/gain_waterfall.png             the waterfall of the decomposition

A generator that cannot produce its output (because the experiment it needs
has not been run) reports that and is skipped; it never writes a
half-finished file.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

#: The generators, in the order their outputs are needed.
STEPS = (
    ("Table II  (main results)", "make_table_ii.py", []),
    ("Table III (significance)", "make_table_iii.py", []),
    ("Table VI  (decomposition)", "make_table_vi.py", []),
    ("Edge table (mechanism)", "make_edge_table.py", []),
    ("IV.B / IV.C (results prose)", "make_results_text.py", []),
    ("IV.D (ablation)", "make_ablation_text.py", []),
    ("Abstract", "make_abstract.py", []),
    ("Fig. 1  (grouped bars)", "plot_results.py", []),
    ("Fig. 2  (strength curve)", "plot_strength_curve.py", []),
    ("Fig. 3  (forest plot)", "plot_forest.py", []),
    ("Fig. 4  (EdgeScore CDF)", "plot_forest.py", []),
    ("Fig. 9  (per-class heat map)", "plot_per_class.py", []),
    ("Waterfall (decomposition)", "plot_waterfall.py", []),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only the steps whose name contains one of these")
    args = parser.parse_args()

    failures = []
    for label, script, extra in STEPS:
        if args.only and not any(token.lower() in label.lower() for token in args.only):
            continue
        command = [sys.executable, script] + extra
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run(command, cwd=HERE)
        if result.returncode != 0:
            failures.append(label)
            print(f"  FAILED ({label})")

    print("\n" + "=" * 60)
    if failures:
        print("the following steps failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("all outputs written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
