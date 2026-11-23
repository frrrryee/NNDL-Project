"""Build the two half-pools used by the style-pool sensitivity experiment.

The frozen style pool holds ten external paintings.  The sensitivity
experiment asks whether the result depends on *how many* styles are used or
on *which* ones: the ten paintings are split into two disjoint sets of five,
and the whole pipeline is run again with each half as the only style pool.

The split is deterministic and is derived from the frozen pool, so both
halves are reproducible and neither one can be cherry-picked after the fact.

For each half a new pool manifest is written, with:

  * the five styles of that half,
  * a pairing that assigns every training image one of the five styles,
    using the *same* cyclic order as the original pool, so that every style
    is used by the same number of images and the assignment is balanced
    over the classes in the same way;
  * a new ``pool_hash``, computed exactly like the frozen one.

Usage::

    python member1_data/build_subpools.py
    python member1_data/build_subpools.py --data-root <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

#: The two half-pool names, as they appear in the run tags.
SUBPOOLS = ("subpool_a", "subpool_b")


def pool_hash_of(pool: dict) -> str:
    """The hash of a pool, computed the way the frozen one is."""
    payload = {k: v for k, v in pool.items() if k != "pool_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def split_styles(styles: dict) -> tuple[list[str], list[str]]:
    """Split the style names into two halves, deterministically.

    The styles are sorted alphabetically and then dealt round-robin into
    the two halves, so that the halves are balanced in the number of
    styles and each half contains a mix of the different painting styles.
    """
    names = sorted(styles)
    first, second = [], []
    for index, name in enumerate(names):
        (first if index % 2 == 0 else second).append(name)
    return first, second


def repair_pairing(original_pairing: dict, allowed: list[str],
                   rows: list[str]) -> dict:
    """Assign every training image one of the allowed styles.

    The original assignment is preserved wherever the style survived the
    split, so that the two halves differ as little as possible from the
    frozen pool and the comparison isolates the pool size.  The images
    whose style did not survive are dealt out over the allowed styles in
    order, which keeps every style used by a comparable number of images.
    """
    allowed_set = set(allowed)
    pairing = {}
    fallback = 0
    for row in rows:
        style = original_pairing.get(row)
        if style in allowed_set:
            pairing[row] = style
            continue
        pairing[row] = allowed[fallback % len(allowed)]
        fallback += 1
    return pairing


def build_half(source: dict, styles: dict, allowed: list[str],
               rows: list[str], name: str, out_dir: Path) -> dict:
    """Write one half-pool manifest and return its description."""
    pool = {
        "n_styles": len(allowed),
        "source": source["pool"].get("source", "external paintings"),
        "source_dir": source["pool"].get("source_dir", "style_pool/images"),
        "leak_check": source["pool"].get("leak_check", ""),
        "pairing_seed": source["pool"].get("pairing_seed", 42),
        "default_alpha": source["pool"].get("default_alpha", 0.5),
        "styles": {style: styles[style] for style in allowed},
    }
    pool["pool_hash"] = pool_hash_of(pool)
    payload = {"pool": pool, "pairing": repair_pairing(
        source["pairing"], allowed, rows)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "style_pool.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"name": name, "styles": allowed, "manifest": path,
            "pool_hash": pool["pool_hash"], "n_styles": len(allowed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="not used for the output; kept for symmetry")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="where the half-pool manifests are written; defaults "
                             "to <style pool dir>/subpools")
    args = parser.parse_args()

    source_path = paths.STYLE_POOL_DIR / "style_pool.json"
    if not source_path.is_file():
        print(f"ERROR: the frozen style pool is missing: {source_path}")
        return 1
    source = json.loads(source_path.read_text(encoding="utf-8"))
    styles = source["pool"]["styles"]
    rows = list(source["pairing"].keys())

    first, second = split_styles(styles)
    out_root = args.out_root or (paths.STYLE_POOL_DIR / "subpools")

    print(f"frozen pool: {len(styles)} styles, hash {source['pool']['pool_hash']}")
    print(f"  half A: {first}")
    print(f"  half B: {second}")

    built = [
        build_half(source, styles, first, rows, SUBPOOLS[0], out_root / SUBPOOLS[0]),
        build_half(source, styles, second, rows, SUBPOOLS[1], out_root / SUBPOOLS[1]),
    ]

    for entry in built:
        payload = json.loads(entry["manifest"].read_text(encoding="utf-8"))
        pool = payload["pool"]
        pairing = payload["pairing"]
        # every training image must be assigned, and only allowed styles
        problems = []
        if set(pairing) != set(rows):
            problems.append("the pairing does not cover every training image")
        if set(pairing.values()) - set(entry["styles"]):
            problems.append("the pairing uses a style outside the half-pool")
        counts = {style: 0 for style in entry["styles"]}
        for style in pairing.values():
            counts[style] += 1
        if min(counts.values()) == 0:
            problems.append(f"some style is never used: {counts}")
        recomputed = pool_hash_of(pool)
        if recomputed != pool["pool_hash"]:
            problems.append("the pool hash does not match its content")

        print(f"\n=== {entry['name']} ===")
        print(f"  manifest: {entry['manifest']}")
        print(f"  pool hash: {entry['pool_hash']}")
        print(f"  usage per style: {counts}")
        if problems:
            for problem in problems:
                print(f"  PROBLEM: {problem}")
        else:
            print("  verification: PASS")

    ok = True
    for entry in built:
        payload = json.loads(entry["manifest"].read_text(encoding="utf-8"))
        if set(payload["pairing"].values()) - set(entry["styles"]):
            ok = False
    print(f"\n{'PASS' if ok else 'FAIL'}: both half-pools are consistent")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
