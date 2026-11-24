"""Measure the line-art mechanism, before any training result is read.

The line-art proposal makes two claims that can be checked on the
images alone, without training anything:

1. The attention gate puts the ink on the object.  The gated drawing
   should carry more ink inside the object outline than outside it, and
   the uniform controls should not.

2. The two controls hold different quantities fixed, so they are two
   controls and not one.  The total-matched control and the
   foreground-matched control must be different drawings.

The ink matching itself is checked here too, because the controls are
only controls if they hold the agreed quantity fixed.  The builder
reports the ink masses, but a builder that computes a quantity and
then draws something else would look fine to itself; this script
re-derives the masses from the *saved* images, so such a mistake shows
up.

A note on what "ink" means here.  The strength map of a drawing is
its darkness before it is turned into a picture, and the picture is
the source colour darkened by that strength.  The ink mass is
therefore proportional to the strength only when the source colour is
the same, which it is within one image -- so the mass of a saved
drawing is recovered as the amount by which it is darker than the white
paper, scaled by the colour it was drawn in.  This script uses the
mean channel, which is the quantity the builder matched on.

Usage:
    python measure_line_art.py --limit 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "member3_structure_preserving"))

import config  # noqa: E402
from member3_structure_preserving.attention_map import (  # noqa: E402
    AttentionConfig, ObjectnessAttention)
from member3_structure_preserving.line_art import (  # noqa: E402
    LineArtConfig, attention_from_map, foreground_mask, ink_masses)

UNITS = ("line_attn", "line_attn_vote",
         "line_plain_total", "line_plain_foreground")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=60,
                        help="how many images to measure")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    attention_model = ObjectnessAttention(AttentionConfig(), device=device)
    line_config = LineArtConfig()

    rows = config.read_split(config.SPLIT_FILES["photo_train"])[:args.limit]
    root = config.DATA_ROOT / "aug_sp" / "lineart"

    totals = {unit: [] for unit in UNITS}
    insides = {unit: [] for unit in UNITS}
    ratios = {unit: [] for unit in UNITS}
    identical = 0

    print()
    print("=" * 84)
    print("THE LINE-ART MECHANISM, MEASURED ON THE SAVED IMAGES")
    print("=" * 84)
    print(f"  images: {len(rows)}   device: {device}")
    print()

    for relative, label in rows:
        source = config.IMAGE_ROOT / relative
        image = Image.open(source).convert("RGB")
        pixels = np.asarray(image)
        tensor = torch.from_numpy(
            np.ascontiguousarray(pixels.astype(np.float32) / 255.0))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        attention = attention_model.objectness(tensor)
        attention_np = attention_from_map(attention, pixels.shape[0],
                                          pixels.shape[1],
                                          line_config.smooth_sigma)
        object_mask = foreground_mask(attention_np, line_config)

        stem = Path(relative).stem
        class_name = config.CLASSES[label]
        pictures = {}
        for unit in UNITS:
            path = (root / unit / "images" / "photo" / class_name /
                    f"{stem}.png")
            picture = np.asarray(Image.open(path).convert("RGB"))
            pictures[unit] = picture
            grey = picture.astype(np.float32).mean(axis=2) / 255.0
            # The strength map: 0 on the white paper, the ink's darkness
            # where a line was drawn.  ``ink_masses`` sums this.
            strength = 1.0 - grey
            mask = strength > 0.01
            total, inside = ink_masses(strength, mask, object_mask)
            totals[unit].append(total)
            insides[unit].append(inside)
            if total > 1e-6:
                ratios[unit].append(inside / total)
        if np.array_equal(pictures["line_plain_total"],
                          pictures["line_plain_foreground"]):
            identical += 1

    print(f"  {'unit':<26}{'total ink':>13}{'fg ink':>12}{'fg/total':>11}")
    for unit in UNITS:
        print(f"  {unit:<26}{np.mean(totals[unit]):>13.1f}"
              f"{np.mean(insides[unit]):>12.1f}"
              f"{np.mean(ratios[unit]):>11.3f}")

    print()
    print("=" * 84)
    print("CLAIM 1: THE GATE PUTS THE INK ON THE OBJECT")
    print("=" * 84)
    print()
    gated = float(np.mean(ratios["line_attn"]))
    plain_total = float(np.mean(ratios["line_plain_total"]))
    plain_fg = float(np.mean(ratios["line_plain_foreground"]))
    for unit in UNITS:
        share = float(np.mean(ratios[unit]))
        tag = "  (control)" if unit.startswith("line_plain") else ""
        print(f"  {unit:<26} {share*100:5.1f}% of the ink is inside the "
              f"object{tag}")
    print()
    print(f"  the gated drawing puts {gated*100:.1f}% of its ink on the "
          f"object;")
    print(f"  the total-matched control puts {plain_total*100:.1f}%, and the "
          f"foreground-matched")
    print(f"  control puts {plain_fg*100:.1f}%.  The gate moves "
          f"{(gated - plain_total)*100:+.1f} points of ink onto")
    print("  the object.")

    print()
    print("=" * 84)
    print("CLAIM 2: THE TWO CONTROLS HOLD DIFFERENT QUANTITIES FIXED")
    print("=" * 84)
    print()
    base_total = np.mean(totals["line_attn"])
    base_inside = np.mean(insides["line_attn"])
    total_err = abs(np.mean(totals["line_plain_total"]) - base_total) / base_total
    inside_err = (abs(np.mean(insides["line_plain_foreground"]) - base_inside)
                  / base_inside)
    print(f"  line_plain_total   vs line_attn, total ink: "
          f"{np.mean(totals['line_plain_total']):.1f} against "
          f"{base_total:.1f}   error {total_err*100:.4f}%")
    print(f"  line_plain_fg      vs line_attn, fg ink:    "
          f"{np.mean(insides['line_plain_foreground']):.1f} against "
          f"{base_inside:.1f}   error {inside_err*100:.4f}%")
    print(f"  the two controls are the same image on {identical}/{len(rows)}"
          f" images")
    print()

    problems = []
    if total_err > 0.02:
        problems.append(f"the total-matched control is off by "
                        f"{total_err*100:.4f}%")
    if inside_err > 0.02:
        problems.append(f"the foreground-matched control is off by "
                        f"{inside_err*100:.4f}%")
    if identical:
        problems.append(f"the two controls are the same image on "
                        f"{identical}/{len(rows)} images")

    if problems:
        print("  FAIL:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  PASS: each control holds its own quantity fixed, and the two")
    print("  controls are different drawings.")

    payload = {
        "images": len(rows),
        "units": {unit: {"total_ink": float(np.mean(totals[unit])),
                         "foreground_ink": float(np.mean(insides[unit])),
                         "foreground_share": float(np.mean(ratios[unit]))}
                  for unit in UNITS},
        "total_match_error": float(total_err),
        "foreground_match_error": float(inside_err),
        "identical_controls": identical,
    }
    out = config.OUTPUT_ROOT / "summary_line_art_mechanism.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
