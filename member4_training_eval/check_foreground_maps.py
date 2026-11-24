"""Check the foreground maps on the failure cases.

Before generating 1,349 images, the foreground maps have to be inspected
on the images where the original method failed.  If the new map is any
good, it should be high on the object and low on the background -- the
exact opposite of the Sobel map, which was high on the background.

For each of the failure cases this script prints, side by side:

  * the Sobel structure map M (high = texture/edge);
  * each candidate foreground map (high = object);
  * the resulting strength field alpha = alpha0 (1 - beta M).

The alpha field is the quantity that decides the outcome: with the
Sobel map the object gets the full strength, and with a foreground map
the object gets the reduced strength.

Usage:
    python check_foreground_maps.py --count 6
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
sys.path.insert(0, str(REPO / "member3_structure_preserving"))

import config  # noqa: E402
from member3_structure_preserving.analysis import map_image  # noqa: E402
from member3_structure_preserving.foreground_map import (  # noqa: E402
    ForegroundConfig, build_foreground_map)
from member3_structure_preserving.structure_map import (  # noqa: E402
    StructureConfig, build_structure_map, spatial_alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "figures" / "foreground_check")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    args = parser.parse_args()

    index_path = REPO / "figures" / "failure_cases" / "index.json"
    if not index_path.is_file():
        print(f"{index_path} is missing; run make_failure_cases.py first")
        return
    cases = json.loads(index_path.read_text(encoding="utf-8"))[:args.count]

    from member3_structure_preserving.stylizer import StructurePreservingStylizer

    generator = StructurePreservingStylizer()
    structure_config = StructureConfig()
    foreground_config = ForegroundConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        content = config.IMAGE_ROOT / case["content"]
        prepared = generator.prepare_content(content)
        m = prepared.structure

        panels = [("Content", prepared.image[0].cpu()),
                  ("Sobel M (edge)", m)]
        # The alpha field of the original method.
        panels.append(("alpha: sobel", spatial_alpha(m, args.alpha, args.beta)))

        for kind in ("spectral", "center", "spectral_center"):
            foreground = build_foreground_map(prepared.image, kind, foreground_config)
            panels.append((f"FG {kind}", foreground))
            panels.append((f"alpha: {kind}",
                           spatial_alpha(foreground, args.alpha, args.beta)))

        # A quantitative check: what fraction of the image does the map
        # call foreground?  A useful object map covers a minority of the
        # pixels; the Sobel map covers most of them.
        width, height = 224, 254
        canvas = Image.new("RGB", (width * len(panels), height), "white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for position, (label, tensor) in enumerate(panels):
            draw.text((position * width + 6, 6), label, fill="black")
            values = tensor.detach().cpu().reshape(-1, *tensor.shape[-2:])[0]
            picture = map_image(values).convert("RGB")
            picture = picture.resize((width - 8, height - 30))
            canvas.paste(picture, (position * width + 4, 28))
        target = args.out_dir / f"{Path(case['content']).stem}.png"
        canvas.save(target)

        sobel_fraction = float((m > 0.5).float().mean())
        print(f"\n{Path(case['content']).name}")
        print(f"  sobel map above 0.5 on {sobel_fraction*100:5.1f}% of the pixels")
        for kind in ("spectral", "center", "spectral_center"):
            foreground = build_foreground_map(prepared.image, kind, foreground_config)
            fraction = float((foreground > 0.5).float().mean())
            # The strength the object would receive under the new map.
            alpha_field = spatial_alpha(foreground, args.alpha, args.beta)
            print(f"  {kind:16s} above 0.5 on {fraction*100:5.1f}% of the pixels, "
                  f"mean alpha {float(alpha_field.mean()):.3f}")
        print(f"  wrote {target}")


if __name__ == "__main__":
    main()
