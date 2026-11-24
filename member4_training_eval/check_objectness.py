"""Check the objectness map on the training images.

Before the objectness map is used for anything, two questions have to be
answered:

1. **Does it find the object?**  A useful object map covers a minority
   of the image, but not a negligible part of it, and it must not cover
   almost everything or almost nothing.

2. **Does it work for every class?**  This is the real risk.  ``house``
   and ``person`` are the classes where "the object" is least like a
   centred blob: a house scene's objectness may sit on the trees or the
   sky, and a person's may sit on the clothes rather than the body.
   A class whose coverage is an outlier needs to be handled, not
   averaged away.

For every class the script prints the fraction of pixels the map marks
as foreground, and writes a montage of the content image next to the map
so that the localisation can be inspected by eye.

Usage:
    python check_objectness.py --count 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "member3_structure_preserving"))

import config  # noqa: E402
from member3_structure_preserving.analysis import map_image_2d  # noqa: E402
from member3_structure_preserving.attention_map import (  # noqa: E402
    AttentionConfig, ObjectnessAttention)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=3,
                        help="how many images per class to inspect")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "figures" / "objectness_check")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = config.read_split(config.SPLIT_FILES["photo_train"])
    attention = ObjectnessAttention(AttentionConfig(), device=args.device)
    print(f"weights fingerprint: {attention.fingerprint()}")
    print(f"device: {args.device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_class = {name: [] for name in config.CLASSES}
    montage = Image.new("RGB", (256 * 2, 256 * len(config.CLASSES)), "white")
    draw = ImageDraw.Draw(montage)

    for index, class_name in enumerate(config.CLASSES):
        bucket = [(rel, label) for rel, label in rows if label == index]
        # A deterministic spread over the class, so the inspection is not
        # biased towards its first images.
        step = max(1, len(bucket) // max(1, args.count))
        chosen = bucket[::step][:args.count]
        fractions = []
        for position, (rel, _label) in enumerate(chosen):
            path = config.IMAGE_ROOT / rel
            image = Image.open(path).convert("RGB")
            tensor = torch_from_pil(image).to(args.device)
            objectness = attention.objectness(tensor)
            fraction = float((objectness > 0.5).float().mean())
            fractions.append(fraction)
            per_class[class_name].append(fraction)

            content = image.resize((256, 256))
            montage.paste(content, (0, index * 256))
            picture = map_image_2d(objectness).convert("RGB").resize((256, 256))
            montage.paste(picture, (256, index * 256))
            draw.text((6, index * 256 + 6),
                      f"{class_name}  fg={fraction*100:.1f}%", fill="yellow")
        print(f"  {class_name:<10} coverage over {args.count} images: "
              f"{np.mean(fractions)*100:5.1f}%  "
              f"(min {np.min(fractions)*100:.1f}%, max {np.max(fractions)*100:.1f}%)")

    montage.save(args.out_dir / "objectness_montage.png")
    print(f"\nwrote {args.out_dir / 'objectness_montage.png'}")

    print()
    print("PER-CLASS SUMMARY")
    summary = {}
    for name, values in per_class.items():
        if values:
            summary[name] = {"mean": float(np.mean(values)),
                             "min": float(np.min(values)),
                             "max": float(np.max(values))}
    print(json.dumps(summary, indent=2))
    out = config.OUTPUT_ROOT / "objectness_coverage.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written: {out}")


def torch_from_pil(image: Image.Image):
    import torch
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


if __name__ == "__main__":
    main()
