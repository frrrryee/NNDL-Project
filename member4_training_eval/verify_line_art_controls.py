"""Verify the two ink-matched controls against the gated drawing.

The line-art family has one method and two controls, and the controls
are only controls if they hold the agreed quantity fixed:

``line_plain_total``
    the total ink mass is equal to the gated drawing's;
``line_plain_foreground``
    the ink mass inside the object is equal to the gated drawing's,
    and the background lines are left at their own weight.

The second one is the one that catches mistakes: a rescale that scales
the whole drawing by one factor satisfies the foreground matching
*and* the total matching at the same time, and the two controls then
turn into the same image.  The check here therefore verifies the
foreground matching *and* requires the two controls to differ.

Usage:
    python verify_line_art_controls.py [--limit 40]
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


def measure(path: Path, object_mask: np.ndarray):
    """The total and the foreground ink of one saved drawing."""
    picture = np.asarray(Image.open(path).convert("RGB"))
    grey = picture.astype(np.float32).mean(axis=2) / 255.0
    strength = 1.0 - grey
    mask = strength > 0.01
    return ink_masses(strength, mask, object_mask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    attention_model = ObjectnessAttention(AttentionConfig(), device=device)
    line_config = LineArtConfig()
    root = config.DATA_ROOT / "aug_sp" / "lineart"

    rows = config.read_split(config.SPLIT_FILES["photo_train"])[:args.limit]
    records = []
    identical = 0
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
        values = {}
        for unit in ("line_attn", "line_plain_total", "line_plain_foreground"):
            path = (root / unit / "images" / "photo" / class_name /
                    f"{stem}.jpg")
            values[unit] = measure(path, object_mask)
        values["paths"] = {
            unit: (root / unit / "images" / "photo" / class_name /
                   f"{stem}.jpg")
            for unit in ("line_plain_total", "line_plain_foreground")}
        values["paths"]["line_attn"] = (root / "line_attn" / "images" /
                                         "photo" / class_name / f"{stem}.jpg")
        records.append(values)
        if (values["paths"]["line_plain_total"].read_bytes() ==
                values["paths"]["line_plain_foreground"].read_bytes()):
            identical += 1

    print()
    print("=" * 84)
    print("THE TWO INK-MATCHED CONTROLS, AGAINST THE GATED DRAWING")
    print("=" * 84)
    print(f"  images: {len(records)}")
    print()

    total_err = []
    inside_err = []
    for values in records:
        gated_total, gated_inside = values["line_attn"]
        total, inside = values["line_plain_total"]
        if gated_total > 1e-6:
            total_err.append(abs(total - gated_total) / gated_total)
        if gated_inside > 1e-6:
            inside_err.append(abs(inside - gated_inside) / gated_inside)

    print("  the total-matched control:")
    print(f"    mean relative error of the total ink: "
          f"{np.mean(total_err)*100:.4f}%")
    print("  the foreground-matched control:")
    print(f"    mean relative error of the foreground ink: "
          f"{np.mean(inside_err)*100:.4f}%")
    print()
    print(f"  the two controls are byte-identical on {identical}/{len(records)}"
          f" images")
    print()

    problems = []
    if np.mean(total_err) > 0.01:
        problems.append(f"the total-matched control is off by "
                        f"{np.mean(total_err)*100:.4f}%")
    if np.mean(inside_err) > 0.01:
        problems.append(f"the foreground-matched control is off by "
                        f"{np.mean(inside_err)*100:.4f}%")
    if identical:
        problems.append(f"the two controls are the same image on "
                        f"{identical}/{len(records)} images, so they are "
                        f"one control and not two")

    if problems:
        print("  FAIL:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  PASS: each control holds its own quantity fixed, and the two")
    print("  controls are different images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
