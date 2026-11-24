"""The strength-matched comparison of the three foreground candidates.

The problem
-----------
The selection over the three candidates ranked ``center`` first with a
mechanism gain of +0.0336 against the +0.0010 of ``spectral``.  The
mean alpha field tells the story: ``center`` applies a mean strength of
0.3068 where ``spectral`` applies 0.4622.  The centre prior covers
about 53% of the image, so it reduces the style over a large area, and a
styled image with less style keeps more of its edges for a reason that
has nothing to do with the map finding the object.

The measurement is therefore confounded: it measures "how much style",
not "how well the structure was preserved".

The fix
-------
Every candidate has to be compared at the *same* effective strength.
This script re-renders each candidate with the alpha field rescaled so
that its mean equals a common target, and compares the EdgeScores of the
rescaled images against a uniform control of exactly that same strength.

Concretely, for a target strength ``a``:

  * the candidate's alpha field is rescaled to ``a * M / mean(M)`` and
    clamped to [0, 1];
  * the uniform control is a plain AdaIN at strength ``a``;
  * the EdgeScore of both is measured against the same content image.

Now every candidate applies the same amount of style, and the only thing
that differs is *where* it is applied.  A candidate that finds the
object reduces the style where the content's edges are, and scores
higher; a candidate that does not find the object scores no better than
the uniform control.

This is the same construction as the strength-matched control of the
main experiment, applied to the selection itself.

Usage:
    python compare_candidates_matched.py --subset 200 --target 0.46
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from structure_metrics import edge_consistency, edge_map  # noqa: E402
from data import read_aug_manifest  # noqa: E402
from member3_structure_preserving.foreground_map import (  # noqa: E402
    ForegroundConfig, build_foreground_map)
from member3_structure_preserving.structure_map import spatial_alpha  # noqa: E402
from member3_structure_preserving.stylizer import (  # noqa: E402
    StructurePreservingStylizer)
from member3_structure_preserving.protocol import FrozenProtocol  # noqa: E402
from member2_adain.adain import tensor_to_pil  # noqa: E402

CANDIDATES = ("spectral", "center", "spectral_center")


def subset_manifest(kind: str, subset: int) -> Path:
    return (config.DATA_ROOT / "aug_sp" / "previews" / f"build_n{subset}" /
            f"subset{subset}_{kind}_a0p5_b0p7" / "manifest_ours.txt")


def edge_of(image):
    """The Canny edge map of an in-memory image.

    ``structure_metrics.edge_map`` takes a path, because every image of
    the project is measured after it has been written to disk and encoded
    as JPEG.  The images of this script are measured in memory, so they
    go through the same Canny parameters directly; the encoder is the
    only difference from the rest of the project, and it is applied
    identically to the candidate and to the control.
    """
    import cv2
    grey = np.asarray(image.convert("L").resize((224, 224), Image.BILINEAR))
    edges = cv2.Canny(grey, 100, 200)
    return edges > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--target", type=float, default=0.46,
                        help="the common effective strength to compare at")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--out", type=Path,
                        default=config.OUTPUT_ROOT / "candidate_matched.json")
    args = parser.parse_args()

    protocol = FrozenProtocol()
    generator = StructurePreservingStylizer(
        foreground_config=ForegroundConfig(), modulation="foreground",
        content_size=512, style_size=512)
    job_by_original = {job.original: job for job in protocol.jobs()}
    style_cache = {}

    results = {kind: {"candidate": [], "control": [], "strength": []}
               for kind in CANDIDATES}
    for kind in CANDIDATES:
        manifest = subset_manifest(kind, args.subset)
        if not manifest.is_file():
            print(f"{manifest} is missing; skipping '{kind}'")
            continue
        # The generator must use *this* candidate's map.  The map is what
        # the comparison is about, so it has to be set per candidate.
        generator.foreground_kind = kind
        rows = read_aug_manifest(manifest)
        for index, (rel, _label, meta) in enumerate(rows, 1):
            job = job_by_original[meta[0]]
            key = job.style_path
            if key not in style_cache:
                style_cache[key] = generator.encode_style(job.style_path)
            style_features = style_cache[key]
            prepared = generator.prepare_content(job.content_path)

            # The candidate's own map, with the strength adjusted so that
            # the *mean* of the alpha field equals the common target.
            #
            # The strength is adjusted through alpha0 rather than by
            # scaling the map: scaling the map and clamping it would
            # saturate the large values and destroy the map's spatial
            # structure, which is exactly the thing under study.  With
            # alpha0 the map is left untouched and only the level moves.
            #
            # alpha(x) = alpha0 * (1 - beta * M) has mean
            # alpha0 * (1 - beta * mean(M)), so the alpha0 that puts the
            # mean at the target follows in closed form.
            raw = prepared.foreground
            map_mean = float(raw.mean())
            if map_mean <= 0:
                continue
            beta = args.beta
            denominator = 1.0 - beta * map_mean
            if denominator <= 1e-6:
                continue
            alpha0 = args.target / denominator
            alpha0 = min(1.0, alpha0)
            alpha_map = spatial_alpha(raw, alpha0, beta)
            feature_alpha = F.interpolate(alpha_map,
                                          size=prepared.features.shape[-2:],
                                          mode="area")
            effective = float(feature_alpha.mean())
            styled = generator.model.decode(
                generator.model.mix(prepared.features, style_features,
                                     feature_alpha))
            # The control is a uniform AdaIN at the candidate's own
            # effective strength, so the comparison is exactly the one
            # the main experiment makes.
            control = generator.model.decode(
                generator.model.mix(prepared.features, style_features,
                                     effective))

            content_edges = edge_map(job.content_path)
            results[kind]["candidate"].append(
                edge_consistency(content_edges, edge_of(tensor_to_pil(styled))))
            results[kind]["control"].append(
                edge_consistency(content_edges, edge_of(tensor_to_pil(control))))
            results[kind]["strength"].append(effective)
            if index % 50 == 0:
                print(f"  {kind}: {index}/{len(rows)}", flush=True)

    print()
    print("=" * 92)
    print(f"STRENGTH-MATCHED COMPARISON (every candidate at mean alpha = "
          f"{args.target})")
    print("=" * 92)
    print(f"  {'candidate':<18}{'candidate':>12}{'control':>12}{'gain':>12}"
          f"{'% worse':>10}{'paired p':>13}{'strength':>11}")
    summary = {}
    for kind in CANDIDATES:
        candidate = np.asarray(results[kind]["candidate"], dtype=float)
        control = np.asarray(results[kind]["control"], dtype=float)
        if candidate.size == 0:
            continue
        difference = candidate - control
        from scipy import stats
        valid = ~np.isnan(difference)
        t, p = stats.ttest_rel(candidate[valid], control[valid])
        worse = int((difference < 0).sum())
        strength_mean = float(np.mean(results[kind]["strength"]))
        print(f"  {kind:<18}{candidate.mean():>12.4f}{control.mean():>12.4f}"
              f"{difference.mean():>+12.4f}{worse/len(difference)*100:>9.1f}%"
              f"{p:>13.3e}{strength_mean:>11.4f}")
        summary[kind] = {
            "candidate_mean": float(candidate.mean()),
            "control_mean": float(control.mean()),
            "gain": float(difference.mean()),
            "fraction_worse": worse / len(difference),
            "paired_p": float(p),
            "effective_strength": strength_mean,
            "n": int(valid.sum()),
        }

    ranked = sorted(summary, key=lambda k: -summary[k]["gain"])
    print()
    print(f"  ranking at matched strength: {' > '.join(ranked)}")
    winner = ranked[0]
    print(f"  SELECTED: {winner}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"target_strength": args.target,
                                    "results": summary, "selected": winner},
                                   indent=2), encoding="utf-8")
    print(f"  written: {args.out}")


if __name__ == "__main__":
    import json
    main()
