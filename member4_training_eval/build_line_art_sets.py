"""Build the line-art datasets.

Three units are produced, and two of them are controls:

``line_attn``
    The attention-gated line drawing: the colour blocks decide where the
    boundaries are, the attention decides how dark they are.

``line_plain_total``
    The same boundaries, drawn uniformly, with the *total* ink mass
    matched to the gated drawing.  This is the control that excludes
    "less ink is easier to learn".

``line_plain_foreground``
    The same boundaries, drawn uniformly, with the ink mass *inside the
    object* matched to the gated drawing.  This is the control that
    excludes the narrower confound "the gate helped by drawing the
    object's outline more heavily".

The fourth unit is ``line_attn_vote``, the region-level soft vote: the
boundary darkness is decided by the attention of the two regions that
meet there, rather than by a binary foreground threshold.  It is the
variant the proposal actually asked for, and it is built alongside the
others so that the threshold-based version can be compared with it.

Nothing here uses a label, and the attention comes from an
ImageNet-pretrained network, so no PACS image other than the Photo
source is ever seen.
"""
from __future__ import annotations

import argparse
import json
import os
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
    ink_masses, solve_ink_weights)

#: The units that are produced, and the directory each one goes into.
UNITS = ("line_attn", "line_attn_vote", "line_plain_total",
         "line_plain_foreground")


def render(image: np.ndarray, attention: np.ndarray,
           config_line: LineArtConfig, unit: str
           ) -> tuple[np.ndarray, dict]:
    """Render one image for one unit, with its ink statistics."""
    object_mask = foreground_mask(attention, config_line)

    if unit == "line_attn":
        picture, mask, strength = draw_line_art(image, attention,
                                                 config_line, gate=True)
        total, inside = ink_masses(strength, mask, object_mask)
        return picture, {"total_ink": total, "foreground_ink": inside}

    if unit == "line_attn_vote":
        # The soft vote: every boundary is drawn, and its darkness is
        # decided by the attention of the regions on either side of it.
        # The strong/weak split is replaced by a continuous weight.
        picture, mask, strength = _soft_vote(image, attention, config_line)
        total, inside = ink_masses(strength, mask, object_mask)
        return picture, {"total_ink": total, "foreground_ink": inside}
    # The two matched controls.  The gated drawing is the reference, and
    # the control is built to match it on the agreed measure.
    #
    # The control must hold the ink mass fixed *without* touching the
    # relative weight of the object's outline against the background
    # lines, because that relative weight is the thing the gate
    # changes.  A uniform rescale would move the foreground ink as a
    # side effect, and the two controls would then be the same image.
    # So the control solves for the two weights instead.
    gated, gated_mask, gated_strength = draw_line_art(
        image, attention, config_line, gate=True)
    total, inside = ink_masses(gated_strength, gated_mask, object_mask)

    # The ungated drawing, at unit weights: the object's outline at
    # full strength and the background lines at the background strength.
    _, plain_mask, plain_strength = draw_line_art(
        image, attention, config_line, gate=False)

    if unit == "line_plain_total":
        foreground_weight, background_weight = solve_ink_weights(
            plain_strength, plain_mask, object_mask, target_total=total)
    else:
        foreground_weight, background_weight = solve_ink_weights(
            plain_strength, plain_mask, object_mask,
            target_foreground=inside)

    _, scaled_mask, scaled = draw_line_art(
        image, attention, config_line, gate=False,
        foreground_weight=foreground_weight,
        background_weight=background_weight)

    picture = render_from_strength(image, scaled)
    drawn = scaled > 0.05
    new_total, new_inside = ink_masses(scaled, drawn, object_mask)
    return picture, {"total_ink": new_total, "foreground_ink": new_inside,
                     "gated_total_ink": total, "gated_foreground_ink": inside,
                     "foreground_weight": foreground_weight,
                     "background_weight": background_weight}


def render_from_strength(image: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """The drawing that a strength map stands for.

    The ink's darkness *is* the strength: the picture is white paper
    with the line drawn on it, and the line is as dark as its strength
    says.  This is the same rule that ``draw_line_art`` applies, put in
    one place so that a drawing built from a *rescaled* strength map
    looks the way its map says it should.
    """
    picture = np.full_like(image, 255)
    drawn = strength > 0.05
    ink = 255.0 * (1.0 - strength)
    picture[drawn] = np.clip(ink[drawn][:, None].repeat(3, axis=1),
                             0, 255).astype(np.uint8)
    return picture


def _soft_vote(image: np.ndarray, attention: np.ndarray,
               config_line: LineArtConfig
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The region-level soft vote.

    Every boundary segment is drawn, and its darkness is the attention of
    the regions that meet there, combined as a weighted mean of the two
    sides.  A boundary between two foreground regions is dark, one
    between two background regions is faint, and the transition between
    them is continuous rather than a hard cut.
    """
    from member3_structure_preserving.line_art import (
        boundary_map, region_attention, superpixel_labels)

    labels = superpixel_labels(image, config_line)
    region = region_attention(labels, attention)
    height, width = labels.shape

    boundary = boundary_map(labels)
    ys, xs = np.nonzero(boundary)
    strength = np.zeros((height, width), dtype=np.float32)
    if len(ys):
        left = labels[ys, np.maximum(xs - 1, 0)]
        right = labels[ys, np.minimum(xs + 1, width - 1)]
        up = labels[np.maximum(ys - 1, 0), xs]
        down = labels[np.minimum(ys + 1, height - 1), xs]
        # The two regions that meet at this boundary are the ones the
        # pixel belongs to and its neighbour across the edge.
        here = labels[ys, xs]
        across = np.where(left != here, left, right)
        other = np.where(up != here, up, down)
        # A soft vote between the two sides: the mean of their attention,
        # with the stronger side counting slightly more.
        first = region[here]
        second = region[np.maximum(across, 0)]
        vote = 0.5 * (first + second) + 0.25 * np.maximum(first, second)
        strength[ys, xs] = np.clip(vote, 0.0, 1.0)

    if config_line.line_width > 1:
        dilated = strength.copy()
        for shift in range(1, config_line.line_width):
            dilated[:, shift:] = np.maximum(dilated[:, shift:],
                                            strength[:, :-shift])
            dilated[:, :-shift] = np.maximum(dilated[:, :-shift],
                                             strength[:, shift:])
            dilated[shift:, :] = np.maximum(dilated[shift:, :],
                                            strength[:-shift, :])
            dilated[:-shift, :] = np.maximum(dilated[:-shift, :],
                                             strength[shift:, :])
        strength = dilated

    mask = strength > 0.05
    picture = render_from_strength(image, strength)
    return picture, mask, strength


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = the whole Photo training split")
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--background-strength", type=float, default=0.35)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    line_config = LineArtConfig(n_segments=args.segments,
                                compactness=args.compactness,
                                background_strength=args.background_strength)
    attention = ObjectnessAttention(AttentionConfig(), device=device)

    rows = config.read_split(config.SPLIT_FILES["photo_train"])
    if args.limit:
        rows = rows[:args.limit]

    root = args.out_dir or (config.DATA_ROOT / "aug_sp" / "lineart")
    for unit in UNITS:
        (root / unit / "images" / "photo").mkdir(parents=True, exist_ok=True)

    print(f"device: {device}   weights: {attention.fingerprint()}")
    print(f"images:  {len(rows)}")
    print(f"units:   {', '.join(UNITS)}")
    print(f"output:  {root}")
    print()

    manifests = {unit: [] for unit in UNITS}
    stats = {unit: {"total_ink": [], "foreground_ink": []} for unit in UNITS}

    for index, (relative, label) in enumerate(rows, 1):
        source = config.IMAGE_ROOT / relative
        image = Image.open(source).convert("RGB")
        pixels = np.asarray(image)
        height, width = pixels.shape[:2]

        tensor = torch.from_numpy(
            np.ascontiguousarray(pixels.astype(np.float32) / 255.0))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        objectness = attention.objectness(tensor)
        gate = attention_from_map(objectness, height, width,
                                  line_config.smooth_sigma)

        class_name = config.CLASSES[label]
        for unit in UNITS:
            picture, values = render(pixels, gate, line_config, unit)
            stem = Path(relative).stem
            # The drawings are saved as PNG, not JPEG.  A line drawing is
            # mostly white paper with a few dark strokes, and JPEG's
            # chroma subsampling and quantisation alter the darkness of
            # those strokes -- which is exactly the quantity the two
            # controls are matched on.  A lossy round trip would leave
            # the matched controls off by tens of percent.
            out = root / unit / "images" / "photo" / class_name / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(picture).save(out, optimize=True)
            key = f"aug_sp/lineart/{unit}/images/photo/{class_name}/{stem}.png"
            # The manifest layout is ``path label original style [alpha]``.
            # A line drawing carries no style image, so the style column
            # holds the placeholder ``none`` -- exactly as the
            # data-volume controls do.  The class name is *not* a style:
            # writing it there would make the fairness checker compare
            # the class names against the style pool and reject the
            # manifest.
            manifests[unit].append(f"{key} 1 {relative} none")
            for name in ("total_ink", "foreground_ink"):
                if name in values:
                    stats[unit][name].append(values[name])

        if index % 100 == 0:
            print(f"  [{index}/{len(rows)}]")

    print()
    for unit in UNITS:
        total = stats[unit]["total_ink"]
        inside = stats[unit]["foreground_ink"]
        manifest = root / unit / "manifest_ours.txt"
        manifest.write_text("\n".join(manifests[unit]) + "\n",
                            encoding="utf-8")
        print(f"  {unit}")
        print(f"    rows {len(manifests[unit])}, "
              f"total ink {np.mean(total):.1f}, "
              f"foreground ink {np.mean(inside):.1f}")

    record = {"schema": 1, "method": "line-art-attention-gate-v1",
              "units": list(UNITS), "count": len(rows),
              "line_config": line_config.to_dict(),
              "attention": {"weights": attention.fingerprint()},
              "stats": {unit: {name: float(np.mean(values))
                               for name, values in block.items() if values}
                        for unit, block in stats.items()}}
    (root / "generation.json").write_text(json.dumps(record, indent=2),
                                          encoding="utf-8")
    print()
    print(f"  wrote {root / 'generation.json'}")


if __name__ == "__main__":
    main()
