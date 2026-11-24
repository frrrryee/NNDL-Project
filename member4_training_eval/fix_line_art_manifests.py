"""Rewrite the style column of the line-art manifests.

The line-art builder originally wrote the class name into the style
column of its manifest.  The manifest layout is ``path label original
style [alpha]``, and the style column names an image from the frozen
style pool -- so a class name there was never a style, and the fairness
checker rightly rejected the manifest for pairing classes against
styles.

A line drawing carries no style image, so the column takes the
placeholder ``none``, exactly as the data-volume controls do.  This
script rewrites the manifests in place; the images themselves are
untouched, so it is much cheaper than rebuilding the whole set.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import config  # noqa: E402

PLACEHOLDER = "none"


def rewrite(path: Path) -> tuple[int, int]:
    """Rewrite one manifest; return the row count and the rows changed."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    out = []
    changed = 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            out.append(line.rstrip())
            continue
        if parts[3] != PLACEHOLDER:
            changed += 1
        out.append(" ".join(parts[:3] + [PLACEHOLDER] + parts[4:]))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(out), changed


def main() -> int:
    print()
    print("=" * 78)
    print("REWRITING THE STYLE COLUMN OF THE LINE-ART MANIFESTS")
    print("=" * 78)
    for group, relative in config.LINE_ART_GROUPS.items():
        path = config.DATA_ROOT / relative
        if not path.is_file():
            print(f"  {group:<26} MISSING: {path}")
            continue
        rows, changed = rewrite(path)
        print(f"  {group:<26} {rows} rows, {changed} rewritten")
    print()
    print("  the placeholder 'none' marks a manifest that carries no style")
    print("  image, so the style-pool checks do not apply to it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
