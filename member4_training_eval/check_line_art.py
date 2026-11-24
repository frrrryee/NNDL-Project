"""The pre-flight check of the line-art idea.

The line-art idea can be rejected in fifteen minutes, before any training
is done.  Three numbers decide it:

1. **Coverage.**  How much of the image does the gate mark as
   foreground?  The spectral map that the repaired modulation uses marks
   2-3% of the pixels, and that is why its mechanism effect stayed tiny.
   A line-art map that does not cover substantially more of the object
   will not fix anything.

2. **Correlation with the Sobel map.**  The original method failed
   because its map was a *texture* response.  If the new map is highly
   correlated with the Sobel map, it is the same thing with new clothes
   and the idea must be dropped.  A clearly lower correlation means the
   semantics really drive the map.

3. **Per-class outliers.**  ``house`` and ``person`` are the classes
   where "the object" is least like a centred blob.  A class whose
   coverage or whose correlation is an outlier needs to be handled, not
   averaged away.

The script also writes a montage so that the drawing can be inspected by
eye, and it compares the gated drawing against the uniform one.

Usage:
    python check_line_art.py --count 4
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "member3_structure_preserving"))

import config  # noqa: E402
from member3_structure_preserving.attention_map import (  # noqa: E402
    AttentionConfig, ObjectnessAttention)
from member3_structure_preserving.foreground_map import (  # noqa: E402
    ForegroundConfig, build_foreground_map)
from member3_structure_preserving.line_art import (  # noqa: E402
    LineArtConfig, attention_from_map, draw_line_art)
from member3_structure_preserving.structure_map import (  # noqa: E402
    StructureConfig, build_structure_map)


def tensor_from(array):
    """A float tensor from a numpy array."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(array))


def sobel_map(image):
    """The Sobel edge magnitude of an image, as the original map.

    The structure map is built from a grey image, but the builder wants
    three channels, so the grey image is replicated.
    """
    import torch
    grey = image.astype(np.float32).mean(axis=2) / 255.0
    stacked = np.repeat(grey[:, :, None], 3, axis=2)
    tensor = tensor_from(stacked).permute(2, 0, 1).unsqueeze(0)
    structure = build_structure_map(tensor, StructureConfig())
    return structure.detach().cpu().numpy().reshape(grey.shape)


def correlation(a, b):
    """The Pearson correlation of two maps, flattened."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=4,
                        help="how many images per class to measure")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "figures" / "line_art_check")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = config.read_split(config.SPLIT_FILES["photo_train"])
    attention = ObjectnessAttention(AttentionConfig(), device=device)
    line_config = LineArtConfig()
    foreground_config = ForegroundConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}   weights: {attention.fingerprint()}")
    print()
    print("=" * 96)
    print("PRE-FLIGHT CHECK OF THE LINE-ART IDEA")
    print("=" * 96)

    per_class = {name: {"coverage": [], "corr": [], "sobel_coverage": [],
                        "spectral_coverage": []}
                 for name in config.CLASSES}
    montage = Image.new("RGB", (224 * 4, 224 * len(config.CLASSES)), "white")
    draw = ImageDraw.Draw(montage)

    for index, class_name in enumerate(config.CLASSES):
        bucket = [(rel, label) for rel, label in rows if label == index]
        step = max(1, len(bucket) // max(1, args.count))
        chosen = bucket[::step][:args.count]

        for position, (rel, _label) in enumerate(chosen):
            path = config.IMAGE_ROOT / rel
            image = Image.open(path).convert("RGB")
            pixels = np.asarray(image)
            height, width = pixels.shape[:2]

            rgb = tensor_from(pixels.astype(np.float32) / 255.0)
            rgb = rgb.permute(2, 0, 1).unsqueeze(0).to(device)

            objectness = attention.objectness(rgb)
            gate_map = attention_from_map(objectness, height, width,
                                          line_config.smooth_sigma)
            spectral = build_foreground_map(rgb, "spectral_center",
                                            foreground_config)
            spectral_map = attention_from_map(spectral, height, width,
                                              line_config.smooth_sigma)
            sobel = sobel_map(pixels)

            drawing, _mask = draw_line_art(pixels, gate_map, line_config,
                                           gate=True)
            plain, _plain_mask = draw_line_art(pixels, gate_map, line_config,
                                               gate=False)

            coverage = float((gate_map >= line_config.foreground_threshold).mean())
            sobel_coverage = float((sobel >= 0.5).mean())
            spectral_coverage = float((spectral_map >= 0.5).mean())
            corr = correlation(gate_map, sobel)

            per_class[class_name]["coverage"].append(coverage)
            per_class[class_name]["corr"].append(corr)
            per_class[class_name]["sobel_coverage"].append(sobel_coverage)
            per_class[class_name]["spectral_coverage"].append(spectral_coverage)

            if position == 0:
                montage.paste(image.resize((224, 224)), (0, index * 224))
                heat = Image.fromarray(
                    (gate_map * 255).astype(np.uint8)).convert("RGB")
                montage.paste(heat.resize((224, 224)), (224, index * 224))
                montage.paste(Image.fromarray(drawing).resize((224, 224)),
                              (448, index * 224))
                montage.paste(Image.fromarray(plain).resize((224, 224)),
                              (672, index * 224))
                draw.text((4, index * 224 + 4), class_name, fill="yellow")

    print()
    print(f"  {'class':<11}{'att cov':>10}{'spec cov':>10}{'sobel cov':>11}"
          f"{'corr(att,sobel)':>18}")
    print("  " + "-" * 60)
    summary = {}
    for name in config.CLASSES:
        block = per_class[name]
        if not block["coverage"]:
            continue
        coverage = float(np.mean(block["coverage"]))
        spec = float(np.mean(block["spectral_coverage"]))
        sob = float(np.mean(block["sobel_coverage"]))
        corr = float(np.nanmean(block["corr"]))
        summary[name] = {"attention_coverage": coverage,
                         "spectral_coverage": spec,
                         "sobel_coverage": sob,
                         "correlation_with_sobel": corr}
        print(f"  {name:<11}{coverage*100:>9.1f}%{spec*100:>9.1f}%"
              f"{sob*100:>10.1f}%{corr:>18.3f}")

    all_coverage = [v for b in per_class.values() for v in b["coverage"]]
    all_corr = [v for b in per_class.values() for v in b["corr"]]
    all_spec = [v for b in per_class.values() for v in b["spectral_coverage"]]
    print()
    print(f"  pooled attention coverage : {np.mean(all_coverage)*100:.1f}%")
    print(f"  pooled spectral coverage  : {np.mean(all_spec)*100:.1f}%")
    print(f"  pooled corr(att, sobel)    : {np.nanmean(all_corr):.3f}")

    montage.save(args.out_dir / "line_art_montage.png")
    out = config.OUTPUT_ROOT / "line_art_preflight.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out_dir / 'line_art_montage.png'}")
    print(f"  wrote {out}")

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    coverage = float(np.mean(all_coverage))
    corr = float(np.nanmean(all_corr))
    if coverage < 0.05:
        print("  REJECT: the gate covers too little of the image.  The object is")
        print("  still not protected, so the line art would not fix the")
        print("  mechanism that failed in the original method.")
    elif corr > 0.9:
        print("  REJECT: the gate is almost the Sobel map.  It would reproduce")
        print("  the texture response that the original method failed on.")
    else:
        print("  PASS: the gate covers a useful fraction of the image and is")
        print("  not a relabelled Sobel map.  The idea is worth training.")
        print()
        print(f"  coverage {coverage*100:.1f}% (target 20-40%), "
              f"correlation {corr:.3f} (must be clearly below 1.0)")


if __name__ == "__main__":
    main()
