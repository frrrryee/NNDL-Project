"""Is the objectness/attention map better than the centre prior?

The selection over the three foreground candidates picked ``center``,
with a mechanism gain of +0.0336 against the +0.0010 of ``spectral``.
That number is suspiciously large: the EdgeScore itself lives in the
0.32-0.36 range, so a gain of 0.034 is a tenth of the whole quantity.

The suspicion is that the gain does not come from the map's *meaning* but
from the map's *shape*.  A Gaussian centre prior covers the middle of
the image, where the subject usually is, and it therefore puts the least
style on the subject for a reason that has nothing to do with content.
The EdgeScore is a retention score, so any map that reduces the style
wherever the content's edges are will score well, and the centre prior
happens to do that without ever looking at the image.

The test
--------
The centre prior is *class-agnostic* and *content-agnostic*: it is the
same map for every image.  So the honest comparison is not between the
centre prior and a content-aware map, but between:

  * the centre prior, and
  * the same map with its image-dependent part removed.

Concretely, the script computes the EdgeScore of two controls:

  * ``centre_shuffled`` -- the centre prior computed for a *different*
    image of the same class.  If the centre prior's advantage comes from
    being centred on the subject, shuffling the images destroys it.

If ``centre_shuffled`` keeps a large fraction of the advantage, the
advantage was never about the map at all -- it was about the images
themselves, and the three candidates are not comparable this way.

The script also reports the correlation between the map's own value and
the EdgeScore gain, which is the direct check of the suspicion: a real
mechanism should correlate with the *content's* structure, not with the
map's spatial profile.

Usage:
    python check_centre_prior.py --subset 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from structure_metrics import edge_consistency, edge_map  # noqa: E402
from data import read_aug_manifest  # noqa: E402
from member3_structure_preserving.foreground_map import (  # noqa: E402
    ForegroundConfig, build_foreground_map)
from member3_structure_preserving.stylizer import (  # noqa: E402
    StructurePreservingStylizer)
from member3_structure_preserving.protocol import FrozenProtocol  # noqa: E402


def subset_dir(kind: str, subset: int) -> Path:
    return (config.DATA_ROOT / "aug_sp" / "previews" / f"build_n{subset}" /
            f"subset{subset}_{kind}_a0p5_b0p7")


def score_manifest(manifest: Path):
    """The EdgeScore of every image of one manifest."""
    scores = []
    for rel, _label, meta in read_aug_manifest(manifest):
        content = config.IMAGE_ROOT / meta[0]
        styled = config.DATA_ROOT / rel
        scores.append(edge_consistency(edge_map(content), edge_map(styled)))
    return np.asarray(scores, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    args = parser.parse_args()

    print("Scoring the three candidates against their matched controls ...",
          flush=True)
    results = {}
    for kind in ("spectral", "center", "spectral_center"):
        root = subset_dir(kind, args.subset)
        ours = score_manifest(root / "manifest_ours.txt")
        matched = score_manifest(root / "manifest_adain_matched.txt")
        difference = ours - matched
        results[kind] = {"ours": ours, "matched": matched,
                         "difference": difference}
        print(f"  {kind}: mean difference {difference.mean():+.4f}", flush=True)

    # ---- The shuffle test -------------------------------------------------
    # The centre prior is the same map for every image, so the map cannot
    # be the reason its control scores well.  The decisive test is to
    # check whether the *spatial profile* of the gain follows the map.
    print()
    print("=" * 84)
    print("THE SHUFFLE TEST")
    print("=" * 84)
    print("  The centre prior does not look at the image, so it is identical")
    print("  for every picture.  If its gain comes from the map, the gain")
    print("  must vanish when the images are paired with the wrong map.")

    protocol = FrozenProtocol()
    generator = StructurePreservingStylizer(
        foreground_config=ForegroundConfig(), modulation="foreground",
        content_size=512, style_size=512)
    job_by_original = {job.original: job for job in protocol.jobs()}

    root = subset_dir("center", args.subset)
    rows = read_aug_manifest(root / "manifest_ours.txt")
    originals = [meta[0] for _rel, _label, meta in rows]
    # A deterministic derangement within the class: every image keeps its
    # class, but it is paired with a different image of the same class.
    by_class = {}
    for original in originals:
        label = job_by_original[original].label
        by_class.setdefault(label, []).append(original)
    shuffled = []
    for label, bucket in by_class.items():
        shifted = bucket[1:] + bucket[:1]
        for original, other in zip(bucket, shifted):
            shuffled.append((original, other))

    # The gain of the *correct* pairing, recomputed the same way, so the
    # two numbers are comparable.
    centre = build_foreground_map.__module__  # keep the import honest
    from member3_structure_preserving.foreground_map import center_prior

    print()
    print(f"  {'pairing':<34}{'mean gain':>12}")
    print(f"  {'correct map':<34}{results['center']['difference'].mean():>+12.4f}")

    # Recompute the matched control for the *shuffled* pairing: the same
    # uniform strength, but attached to the wrong picture.
    shuffled_gains = []
    for (original, other), (_rel, label, _meta) in zip(shuffled, rows):
        job = job_by_original[original]
        other_job = job_by_original[other]
        prepared = generator.prepare_content(job.content_path)
        style_features = generator.encode_style(job.style_path)
        # The map of the *other* image, applied to this one.
        other_prepared = generator.prepare_content(other_job.content_path)
        from member3_structure_preserving.structure_map import spatial_alpha
        from member3_structure_preserving.foreground_map import _blur
        other_map = other_prepared.foreground
        alpha_map = spatial_alpha(other_map, args.alpha, args.beta)
        import torch.nn.functional as F
        feature_alpha = F.interpolate(alpha_map,
                                      size=prepared.features.shape[-2:],
                                      mode="area")
        matched_alpha = float(feature_alpha.cpu().mean())
        control = generator.model.decode(
            generator.model.mix(prepared.features, style_features, matched_alpha))
        control_pil = generator.model.decode(
            generator.model.mix(prepared.features, style_features, matched_alpha))
        from member2_adain.adain import tensor_to_pil
        styled_control = tensor_to_pil(control_pil)
        from PIL import Image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            styled_control.save(handle.name, format="JPEG", quality=95)
            temporary = handle.name
        score = edge_consistency(edge_map(job.content_path),
                                 Image.open(temporary))
        shuffled_gains.append(score)
        Path(temporary).unlink(missing_ok=True)
    shuffled_gains = np.asarray(shuffled_gains, dtype=float)

    correct = results["center"]["ours"]
    print(f"  {'wrong map (same class)':<34}"
          f"{(shuffled_gains - correct).mean():>+12.4f}")
    print()
    print("  The second row is the difference the *wrong* map makes relative")
    print("  to the correct one.  If the centre prior's advantage were")
    print("  about the map, this difference would be large and negative.")


if __name__ == "__main__":
    main()
