"""Assemble the failure cases into one figure for the report (Fig. 5).

The individual panels are written by ``make_failure_cases.py``.  This
script stacks them into a single multi-row figure, so that the report has
one picture to point at rather than twelve.

The layout is a grid: the cases are ordered from the worst to the least
bad, and the first ``--rows`` groups of ``--columns`` cases are laid out
row by row.

Usage:
    python compose_failure_cases.py --rows 3 --columns 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DIR = REPO / "figures" / "failure_cases"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=REPO / "figures" / "failure_cases.pdf")
    parser.add_argument("--png", type=Path,
                        default=REPO / "figures" / "failure_cases.png")
    args = parser.parse_args()

    index_path = args.dir / "index.json"
    if not index_path.is_file():
        print(f"{index_path} is missing; run make_failure_cases.py first")
        return 1
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [e for e in entries if Path(e["figure"]).is_file()]
    if not entries:
        print("no failure-case panel is on disk")
        return 1

    wanted = args.rows * args.columns
    if len(entries) < wanted:
        print(f"only {len(entries)} panels are available, "
              f"{wanted} were requested; the figure will be smaller")
        rows = max(1, len(entries) // args.columns)
        if rows * args.columns > len(entries):
            rows = len(entries) // args.columns
    else:
        rows = args.rows
    entries = entries[:rows * args.columns]

    panels = [Image.open(entry["figure"]).convert("RGB") for entry in entries]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * args.columns, height * rows), "white")
    for position, panel in enumerate(panels):
        x = (position % args.columns) * width
        y = (position // args.columns) * height
        canvas.paste(panel, (x, y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, "PDF", resolution=200)
    canvas.save(args.png, "PNG")
    print(f"wrote {args.out}")
    print(f"wrote {args.png}")
    print(f"{len(panels)} panels in {rows} x {args.columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
