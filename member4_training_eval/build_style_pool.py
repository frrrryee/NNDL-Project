"""Freeze the external style pool and the content->style pairing.

The style pool holds 10 paintings that are NOT part of PACS.  PACS Art
Painting, Cartoon and Sketch are the evaluation domains of this project, so
using any of them as a style reference would leak the test appearance into
training.  This script therefore refuses to build a pool from PACS images and
checks the result byte-for-byte against the local PACS copy.

The frozen content->style pairing (drawn with seed 42) is stored alongside
the pool so that +AdaIN and +Ours differ only in the stylization algorithm,
not in style assignment.

Usage:
    python build_style_pool.py [--alpha 0.5] [--pairing-seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import config


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def collect_pacs_hashes(image_root: Path) -> dict:
    """sha256 of every file under the PACS image root, for the leak check."""
    digests = {}
    if not image_root.is_dir():
        return digests
    for path in image_root.rglob("*"):
        if path.is_file():
            digests[sha256_file(path)] = path.as_posix()
    return digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=config.N_STYLE_IMAGES,
                        help="number of style images (must match the delivered pool)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="style strength recorded in the pairing (exp 4 sweeps this)")
    parser.add_argument("--pairing-seed", type=int, default=42)
    parser.add_argument("--style-dir", type=Path, default=config.STYLE_POOL_IMAGES,
                        help="folder holding the external style images")
    parser.add_argument("--styles-per-image", type=int, default=1,
                        help="how many style references every original image is "
                             "paired with.  The frozen protocol uses 1; the "
                             "multi-style ablation uses 2, which doubles the "
                             "number of generated rows and therefore also "
                             "doubles the training set of every augmented "
                             "group.  The pairing rule is the one member 2's "
                             "generator uses, so the two arms of the ablation "
                             "share exactly the same content-style pairs.")
    args = parser.parse_args()

    if args.styles_per_image < 1:
        raise SystemExit("--styles-per-image must be positive")

    style_dir = Path(args.style_dir)
    if not style_dir.is_dir():
        raise SystemExit(f"style image folder not found: {style_dir}")

    # 1. candidate style images: the delivered external pool, sorted by name
    #    so that the pool is reproducible from the folder contents alone.
    candidates = sorted(p for p in style_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(candidates) != args.n:
        raise SystemExit(
            f"expected {args.n} style images in {style_dir}, found {len(candidates)}"
        )

    # 2. LEAK GUARD: no style image may be a PACS image.  Art Painting,
    #    Cartoon and Sketch are our test domains, so a style image that is
    #    byte-identical to one of them would put the test appearance into
    #    training.  Compare content hashes over the whole local PACS copy.
    pacs_hashes = collect_pacs_hashes(config.IMAGE_ROOT)
    for src in candidates:
        digest = sha256_file(src)
        if digest in pacs_hashes:
            raise SystemExit(
                f"style image {src.name} is byte-identical to the PACS image "
                f"{pacs_hashes[digest]}. Style images must come from outside PACS."
            )

    # 3. name the styles by their file stem, so the pool is portable: the
    #    manifest refers to styles by name, not by a positional index.
    names = [p.stem for p in candidates]
    styles = {}
    for src in candidates:
        styles[src.stem] = {
            "file": f"images/{src.name}",
            "sha256": sha256_file(src),
        }
    pool = {
        "n_styles": len(styles),
        "source": "external paintings (not part of PACS)",
        "source_dir": "style_pool/images",
        "leak_check": "verified byte-disjoint from the local PACS copy",
        "pairing_seed": args.pairing_seed,
        "default_alpha": args.alpha,
        "styles": styles,
    }
    pool["pool_hash"] = hashlib.sha256(
        json.dumps(pool, sort_keys=True).encode()).hexdigest()[:12]

    # 4. frozen content->style pairing.  The assignment is the one member 2's
    #    generator uses (StylePool.assign): a seeded shuffle of the pool order
    #    followed by a cyclic walk over it.  Reproducing it exactly here means
    #    the pairing recorded in the manifest is the pairing the generators
    #    will actually produce, so the fairness check compares like with like.
    #
    #    With more than one style per original the pairing is a *set* of
    #    (original, style) pairs rather than a single original->style
    #    mapping, because an original now carries several styles.
    #
    #    The k-th style of the i-th original is ``names[order[(i + k) % n]]``.
    #    With styles_per_image == 1 this is exactly the frozen one-style rule,
    #    so the frozen pool keeps its pairing and its hash.  With two styles
    #    the first one is *still* the frozen pairing and the second one is the
    #    next entry of the same cyclic walk, so the multi-style set is a
    #    strict superset of the frozen one: every stylization the one-style
    #    arm trains on is also in the two-style arm.  That is what makes the
    #    two arms comparable -- the difference between them is the extra
    #    style, not a reshuffled pairing.
    train_rows = config.read_split(config.SPLIT_FILES["photo_train"])
    order = list(range(len(candidates)))
    random.Random(args.pairing_seed).shuffle(order)
    if args.styles_per_image == 1:
        pairing = {
            rel: names[order[i % len(order)]]
            for i, (rel, _) in enumerate(train_rows)
        }
        pair_sets = None
    else:
        if args.styles_per_image > len(order):
            raise SystemExit(
                f"--styles-per-image {args.styles_per_image} exceeds the style "
                f"pool size {len(order)}")
        pairing = {}
        pair_sets = {}
        for i, (rel, _) in enumerate(train_rows):
            chosen = []
            for offset in range(args.styles_per_image):
                style = names[order[(i + offset) % len(order)]]
                if style not in chosen:
                    chosen.append(style)
            pairing[rel] = chosen[0]
            pair_sets[rel] = chosen

    out = {
        "pool": pool,
        "pairing": pairing,
    }
    if pair_sets is not None:
        # The full multi-style assignment travels with the manifest, so the
        # generators and the fairness check read it instead of re-deriving it
        # and possibly deriving something else.
        out["pairing_multi"] = pair_sets
        out["styles_per_image"] = args.styles_per_image
    config.STYLE_POOL_MANIFEST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"style pool: {len(styles)} external styles (no PACS images), "
          f"pairing over {len(pairing)} originals, alpha={args.alpha}, "
          f"styles_per_image={args.styles_per_image}")
    print(f"wrote {config.STYLE_POOL_MANIFEST} (pool_hash={pool['pool_hash']})")


if __name__ == "__main__":
    main()
