"""Generate the strength-matched uniform control of a reduced set.

The mechanism comparison needs, for every image, a uniform AdaIN whose
strength equals the mean of that image's spatial alpha field.  The reduced
sets were built without that control, and regenerating them with
``--with-matched`` would recompute all the styled images as well.

The control is cheap to produce on its own: the modulation map is a
deterministic function of the content image, so the matched strength can
be recomputed exactly, and only the uniform AdaIN has to be rendered.

Usage:
    python build_matched_controls.py --subset 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from member3_structure_preserving.foreground_map import ForegroundConfig  # noqa: E402
from member3_structure_preserving.stylizer import (  # noqa: E402
    StructurePreservingStylizer)
from member3_structure_preserving.protocol import FrozenProtocol  # noqa: E402
from data import read_aug_manifest  # noqa: E402


def subset_dir(kind: str, subset: int) -> Path:
    return (config.DATA_ROOT / "aug_sp" / "previews" / f"build_n{subset}" /
            f"subset{subset}_{kind}_a0p5_b0p7")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", nargs="+", default=("spectral", "center", "spectral_center"))
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    protocol = FrozenProtocol()
    generator = StructurePreservingStylizer(
        foreground_config=ForegroundConfig(),
        modulation="foreground",
        content_size=512, style_size=512)
    job_by_original = {job.original: job for job in protocol.jobs()}

    for kind in args.kinds:
        root = subset_dir(kind, args.subset)
        manifest = root / "manifest_ours.txt"
        if not manifest.is_file():
            print(f"{manifest} is missing; skipping '{kind}'")
            continue
        rows = read_aug_manifest(manifest)
        control_lines = []
        for index, (rel, label, meta) in enumerate(rows, 1):
            job = job_by_original.get(meta[0])
            if job is None:
                raise ValueError(f"unknown content image: {meta[0]}")
            prepared = generator.prepare_content(job.content_path)
            style_features = generator.encode_style(job.style_path)
            result = generator.render(prepared, style_features,
                                      args.alpha, args.beta,
                                      with_matched=True)
            target = root / "adain_matched" / Path(rel).parent / \
                (Path(rel).stem + ".jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            result.matched_baseline.save(target, format="JPEG", quality=95)
            control_lines.append(
                f"{target.resolve().relative_to(config.DATA_ROOT.resolve()).as_posix()} "
                f"{label + 1} {meta[0]} {meta[1] if len(meta) > 1 else 'none'} {args.alpha}")
            if index % 50 == 0:
                print(f"  {kind}: {index}/{len(rows)}", flush=True)
        (root / "manifest_adain_matched.txt").write_text(
            "\n".join(control_lines) + "\n", encoding="utf-8")
        print(f"  {kind}: wrote {len(control_lines)} control images", flush=True)


if __name__ == "__main__":
    main()
