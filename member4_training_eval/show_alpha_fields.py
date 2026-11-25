"""The alpha field of every sweep cell, next to its uniform control.

The blend is ``alpha_map = alpha * (1 - beta * M)``, so the spatial
variant applies less style than a uniform mix of the same alpha.  The
mean of the spatial field is therefore the strength its uniform
control has to match, and the two numbers together say how much of
the frame the map actually modulates.

The table is what makes the alpha question answerable: if the map
becomes harmless -- or helpful -- as alpha grows, the mechanism that
is being tested changes with alpha, and a sweep run at one alpha says
nothing about the other.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402


def cells():
    """Every generation record under the sweep directory."""
    root = config.DATA_ROOT / "aug_sp" / "sweeps" / "sweep_full"
    if not root.is_dir():
        return
    for cell in sorted(p for p in root.iterdir() if p.is_dir()):
        record = cell / "generation.json"
        if record.is_file():
            yield cell, record


def main() -> None:
    print()
    print("=" * 88)
    print("THE ALPHA FIELD OF EVERY CELL, NEXT TO ITS UNIFORM CONTROL")
    print("=" * 88)
    print()
    print(f"  {'cell':<22}{'alpha':>7}{'beta':>7}{'map mean':>11}"
          f"{'ctrl':>9}{'drop':>9}")
    for cell, record_path in cells():
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        settings = payload.get("settings", {})
        outputs = list((payload.get("outputs") or {}).values())
        if not outputs:
            continue
        map_values = [e["metrics"]["mean_alpha"] for e in outputs
                      if e.get("metrics") and "mean_alpha" in e["metrics"]]
        control_values = [e.get("matched_alpha") for e in outputs
                          if e.get("matched_alpha") is not None]
        if not map_values:
            continue
        mean = sum(map_values) / len(map_values)
        control = (sum(control_values) / len(control_values)
                   if control_values else None)
        drop = (mean - control) if control is not None else None
        print(f"  {cell.name:<22}{settings.get('alpha', '?'):>7}"
              f"{settings.get('beta', '?'):>7}{mean:>11.4f}"
              f"{(f'{control:.4f}' if control is not None else '-'):>9}"
              f"{(f'{drop:+.4f}' if drop is not None else '-'):>9}")

    print()
    print("  The 'drop' is how much style the spatial map gives up against")
    print("  a uniform mix of the same alpha.  The map's own alpha and the")
    print("  control's alpha are the two things that are compared, so a")
    print("  fair comparison holds the map's mean fixed and varies only")
    print("  the map.")


if __name__ == "__main__":
    main()
