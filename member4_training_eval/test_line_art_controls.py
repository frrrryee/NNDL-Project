"""Check the matched controls on a handful of images.

This is the quick version of ``verify_line_art_controls.py``: it
renders the four units in memory and checks the two things that the
controls have to satisfy -- each control holds its own quantity fixed,
and the two controls are not the same drawing.

The full verifier reads the *saved* images, which is what the training
actually sees.  This one is for iterating on the rendering itself.

Usage:
    python test_line_art_controls.py [--limit 12]
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    import torch
    attention_model = ObjectnessAttention(AttentionConfig(), device="cuda")
    line_config = LineArtConfig()
    rows = config.read_split(config.SPLIT_FILES["photo_train"])[:args.limit]

    total_errors = []
    inside_errors = []
    identical = 0

    print()
    print("=" * 88)
    print(f"{'image':<14}{'gated tot':>11}{'plain tot':>11}{'err':>9}"
          f"{'gated fg':>11}{'plain fg':>10}{'err':>9}{'same?':>8}")
    print("=" * 88)

    for relative, _label in rows:
        image = np.asarray(
            Image.open(config.IMAGE_ROOT / relative).convert("RGB"))
        tensor = torch.from_numpy(
            np.ascontiguousarray(image.astype(np.float32) / 255.0))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        attention = attention_from_map(
            attention_model.objectness(tensor), image.shape[0],
            image.shape[1], line_config.smooth_sigma)
        object_mask = foreground_mask(attention, line_config)

        from build_line_art_sets import render
        gated_picture, gated_values = render(image, attention, line_config,
                                             "line_attn")
        total_picture, total_values = render(image, attention, line_config,
                                             "line_plain_total")
        inside_picture, inside_values = render(image, attention, line_config,
                                               "line_plain_foreground")

        gated_total = gated_values["total_ink"]
        gated_inside = gated_values["foreground_ink"]
        total_error = abs(total_values["total_ink"] - gated_total) / gated_total
        inside_error = (abs(inside_values["foreground_ink"] - gated_inside)
                        / gated_inside)
        total_errors.append(total_error)
        inside_errors.append(inside_error)
        same = np.array_equal(total_picture, inside_picture)
        identical += int(same)
        if not same:
            difference = np.abs(total_picture.astype(int)
                                - inside_picture.astype(int))
            changed = int((difference > 0).any(axis=2).sum())
            worst = int(difference.max())
            print(f"    {Path(relative).stem}: differ on {changed} pixels, "
                  f"worst {worst}")

        print(f"  {Path(relative).stem:<12}{gated_total:>11.1f}"
              f"{total_values['total_ink']:>11.1f}{total_error*100:>8.3f}%"
              f"{gated_inside:>11.1f}{inside_values['foreground_ink']:>10.1f}"
              f"{inside_error*100:>8.3f}%{'yes' if same else 'no':>8}")

    print()
    print(f"  mean total-match error:    {np.mean(total_errors)*100:.4f}%")
    print(f"  mean foreground-match error:{np.mean(inside_errors)*100:.4f}%")
    print(f"  the two controls identical on {identical}/{len(rows)} images")
    print()

    problems = []
    if np.mean(total_errors) > 0.01:
        problems.append(f"the total-matched control is off by "
                        f"{np.mean(total_errors)*100:.4f}%")
    if np.mean(inside_errors) > 0.01:
        problems.append(f"the foreground-matched control is off by "
                        f"{np.mean(inside_errors)*100:.4f}%")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
