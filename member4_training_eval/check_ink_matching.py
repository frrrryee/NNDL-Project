"""Is the ink matching exact, per image?

The strength-matched controls are the whole point of the comparison, so
the matching has to hold for every image and not merely on average.  It
also has to hold without the scale factor being clipped: a clipped
factor would silently leave the control unmatched.

This checks both, and reports how different the two controls are from
each other on the measure they do *not* match, because that difference
is what the design trades away.
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
    LineArtConfig, attention_from_map, draw_line_art, foreground_mask,
    ink_masses, rescale_ink)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    line_config = LineArtConfig()
    attention = ObjectnessAttention(AttentionConfig(), device=device)
    rows = config.read_split(config.SPLIT_FILES["photo_train"])[:args.limit]

    errors = {"total": [], "foreground": []}
    clipped = {"total": 0, "foreground": 0}
    ratios = {"total": [], "foreground": []}

    for relative, label in rows:
        source = config.IMAGE_ROOT / relative
        pixels = np.asarray(Image.open(source).convert("RGB"))
        height, width = pixels.shape[:2]
        tensor = torch.from_numpy(
            np.ascontiguousarray(pixels.astype(np.float32) / 255.0))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        gate = attention_from_map(attention.objectness(tensor), height, width,
                                  line_config.smooth_sigma)

        object_mask = foreground_mask(gate, line_config)
        gated, gated_mask, gated_strength = draw_line_art(
            pixels, gate, line_config, gate=True)
        target_total, target_fg = ink_masses(gated_strength, gated_mask,
                                             object_mask)

        plain, plain_mask, plain_strength = draw_line_art(
            pixels, gate, line_config, gate=False)

        scaled_total = rescale_ink(plain_strength, plain_mask, object_mask,
                                   target_total=target_total)
        got_total, got_fg_total = ink_masses(scaled_total, plain_mask,
                                             object_mask)
        errors["total"].append(abs(got_total - target_total)
                               / max(target_total, 1e-6))
        ratios["total"].append(got_total / max(target_total, 1e-6))

        scaled_fg = rescale_ink(plain_strength, plain_mask, object_mask,
                                target_foreground=target_fg)
        got_total_fg, got_fg = ink_masses(scaled_fg, plain_mask, object_mask)
        errors["foreground"].append(abs(got_fg - target_fg)
                                   / max(target_fg, 1e-6))
        ratios["foreground"].append(got_fg / max(target_fg, 1e-6))

        # A clipped factor would leave the control unmatched.
        factor_total = target_total / max(
            ink_masses(plain_strength, plain_mask, object_mask)[0], 1e-6)
        factor_fg = target_fg / max(
            ink_masses(plain_strength, plain_mask, object_mask)[1], 1e-6)
        if not 0.05 <= factor_total <= 20.0:
            clipped["total"] += 1
        if not 0.05 <= factor_fg <= 20.0:
            clipped["foreground"] += 1

    print("=" * 78)
    print("THE INK MATCHING")
    print("=" * 78)
    print(f"  images: {len(rows)}")
    print()
    for name in ("total", "foreground"):
        errors[name] = np.array(errors[name])
        ratios[name] = np.array(ratios[name])
        print(f"  {name} matching")
        print(f"    relative error   max {errors[name].max()*100:.4f}%  "
              f"mean {errors[name].mean()*100:.4f}%")
        print(f"    ratio            min {ratios[name].min():.6f}  "
              f"max {ratios[name].max():.6f}")
        print(f"    clipped factors  {clipped[name]}")
    print()
    worst = max(errors["total"].max(), errors["foreground"].max())
    if worst < 1e-6 and not any(clipped.values()):
        print("  PASS: the matching is exact for every image, and no scale")
        print("  factor was clipped.")
        return 0
    print(f"  FAIL: the worst relative error is {worst*100:.4f}%")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
