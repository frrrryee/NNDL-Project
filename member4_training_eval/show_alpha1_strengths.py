"""Which dataset each alpha=1.0 run actually trained on.

The sweep cells differ in two ways at once: the *alpha* the generator
was asked for, and the strength of the uniform control that was
matched to the spatial map's mean.  A run's manifest says which
dataset it saw, and the generation record of that dataset says what
strength the control has.

This matters because the numbers are easy to misread: ``adain`` and
``adain_matched`` are not two strengths of the same thing, they are
the same strength reached two different ways.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT


def control_strength(manifest: Path) -> float | None:
    """The uniform strength of a manifest's dataset."""
    directory = manifest.parent
    record = directory / "generation.json"
    if not record.is_file():
        return None
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    values = [e.get("matched_alpha") for e in
              (payload.get("outputs") or {}).values()
              if e.get("matched_alpha") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    print()
    print("=" * 88)
    print("WHAT EACH alpha=1.0 RUN ACTUALLY TRAINED ON")
    print("=" * 88)
    print()
    print(f"  {'run':<30}{'uniform strength':>19}{'kind':>12}")
    for tag in sorted(os.listdir(RUNS)):
        if not tag.startswith(("adain_a1", "adain_matched_a1", "ours_a1")):
            continue
        summary = RUNS / tag / "train_summary.json"
        if not summary.is_file():
            continue
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        manifest = payload.get("aug_manifest")
        strength = (control_strength(Path(manifest)) if manifest else None)
        kind = ("spatial" if tag.startswith("ours") else "uniform")
        print(f"  {tag:<30}{(f'{strength:.4f}' if strength else '-'):>19}"
              f"{kind:>12}")

    print()
    print("  'uniform strength' is the strength of the uniform control")
    print("  that was matched to the spatial map's mean.  For the uniform")
    print("  runs it is the alpha they were generated at.")


if __name__ == "__main__":
    main()
