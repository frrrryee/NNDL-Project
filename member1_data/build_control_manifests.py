"""Build the manifests of the two data-volume control groups.

The five reported groups cannot answer *where* the +7.2 points of style
augmentation come from, because every augmented group has twice as many
training rows as the Original group.  Two controls close that gap:

``duplicate``
    Every original image is copied a second time, byte for byte.  The
    training set has twice the rows but the same number of distinct images
    and no appearance change at all.  Any difference against Original is
    therefore attributable purely to the *amount of data*.

``nonstyle``
    A second copy of every original image is produced by a deterministic
    non-stylistic transform (colour jitter, greyscale or Gaussian blur).
    The training set has twice the rows, the same number of distinct
    images, and a generic appearance change.  The difference against
    ``duplicate`` isolates the contribution of *appearance diversity* in
    general, and the difference against ``adain`` isolates the contribution
    of *style* over a generic appearance change.

Both manifests use the six-column format that ``read_aug_manifest`` parses::

    <image path> <label 1..7> <original path> <style> <alpha> <beta>

The style column holds the placeholder ``none`` for both groups: they have
no style image, and the column must still be present, because the parser
splits on whitespace and a missing column would shift every field after it.

Usage::

    python member1_data/build_control_manifests.py
    python member1_data/build_control_manifests.py --data-root <path> --overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402  (the single entry point for path resolution)

#: The style column of a manifest that has no style image.
NO_STYLE = "none"

#: The transforms used for the generic appearance control.  Each original
#: image receives exactly one of them, chosen at random but reproducibly.
NONSTYLE_TRANSFORMS = ("jitter", "greyscale", "blur")


def resolve_data_root(explicit: Path | None) -> Path:
    """The folder that contains ``PACS/``."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return paths.DATA_ROOT


def read_split(path: Path) -> list[tuple[str, int]]:
    """Parse a split file: ``relative/path label_1_to_7``."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        relative, raw_label = line.rsplit(maxsplit=1)
        rows.append((relative.replace("\\", "/").removeprefix("./"),
                     int(raw_label)))
    return rows


def read_manifest(path: Path) -> list[tuple[str, int, str]]:
    """Parse a six-column manifest into (image, label, original).

    The manifest format is ``<image> <label> <original> <style> <alpha> <beta>``;
    only the first three columns are needed for the verification.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append((parts[0], int(parts[1]), parts[2]))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_nonstyle(source: Path, target: Path, kind: str, seed: int) -> None:
    """Write one deterministic non-stylistic variant of ``source``.

    The transform is a generic appearance change: it alters colour or
    sharpness but not the semantic content, and it never synthesises
    texture or brush strokes the way a style transfer does.
    """
    from PIL import Image, ImageFilter, ImageEnhance

    with Image.open(source) as handle:
        image = handle.convert("RGB")

    if kind == "jitter":
        # A per-image random state keeps the transform deterministic while
        # still varying its parameters across the dataset.
        rng = random.Random(seed)
        brightness = 1.0 + rng.uniform(-0.35, 0.35)
        contrast = 1.0 + rng.uniform(-0.35, 0.35)
        saturation = 1.0 + rng.uniform(-0.40, 0.40)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
    elif kind == "greyscale":
        image = image.convert("L").convert("RGB")
    elif kind == "blur":
        radius = 1.0 + (seed % 7) * 0.35
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        raise ValueError(f"unknown transform: {kind}")

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=95)


def write_manifest(path: Path, rows: list[tuple[str, int, str]]) -> None:
    """Write the six-column manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for image_rel, label, original_rel in rows:
            handle.write(f"{image_rel} {label} {original_rel} "
                         f"{NO_STYLE} 0.00 0.00\n")


def build_duplicate(data_root: Path, split_file: Path, out_dir: Path,
                    manifest_path: Path, overwrite: bool) -> dict:
    """Copy every training image a second time, byte for byte."""
    from PIL import Image

    image_root = data_root / "PACS"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    identical = 0
    for relative, label in read_split(split_file):
        source = image_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing original: {source}")
        target_rel = f"augmented/duplicate/images/{relative}"
        target = data_root / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or overwrite:
            shutil.copyfile(source, target)
        if sha256(source) == sha256(target):
            identical += 1
        rows.append((target_rel, label, relative))

    write_manifest(manifest_path, rows)
    return {
        "group": "duplicate",
        "rows": len(rows),
        "images": out_dir,
        "byte_identical_to_original": identical,
        "manifest": manifest_path,
    }


def build_nonstyle(data_root: Path, split_file: Path, out_dir: Path,
                   manifest_path: Path, overwrite: bool, seed: int) -> dict:
    """Produce one deterministic non-stylistic variant per training image."""
    image_root = data_root / "PACS"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    changed = 0
    counts = {name: 0 for name in NONSTYLE_TRANSFORMS}
    for relative, label in read_split(split_file):
        source = image_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing original: {source}")
        kind = rng.choice(NONSTYLE_TRANSFORMS)
        counts[kind] += 1
        target_rel = f"augmented/nonstyle/images/{relative}"
        target = data_root / target_rel
        if not target.is_file() or overwrite:
            apply_nonstyle(source, target, kind, seed + len(rows))
        if sha256(source) != sha256(target):
            changed += 1
        rows.append((target_rel, label, relative))

    write_manifest(manifest_path, rows)
    return {
        "group": "nonstyle",
        "rows": len(rows),
        "images": out_dir,
        "different_from_original": changed,
        "transform_counts": counts,
        "manifest": manifest_path,
    }


def verify(manifest_path: Path, split_file: Path, data_root: Path,
           expect_identical: bool) -> list[str]:
    """Check the manifest against the split and the images on disk."""
    problems = []
    rows = read_manifest(manifest_path)
    expected = read_split(split_file)

    if len(rows) != len(expected):
        problems.append(f"row count {len(rows)} != expected {len(expected)}")
    if len(rows) != len(expected):
        return problems

    for (image_rel, label, original_rel), (exp_rel, exp_label) in zip(rows, expected):
        if label != exp_label:
            problems.append(f"label {label} != {exp_label} for {image_rel}")
            break
        if original_rel != exp_rel:
            problems.append(f"original {original_rel} != {exp_rel}")
            break
        if len(image_rel.split()) != 1 or not (data_root / image_rel).is_file():
            problems.append(f"image missing: {image_rel}")
            break

    # every line must have exactly six whitespace-separated columns
    for number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            problems.append(f"line {number}: {len(parts)} columns, expected 6")
            break
        if parts[3] != NO_STYLE:
            problems.append(f"line {number}: style column is {parts[3]!r}, "
                            f"expected {NO_STYLE!r}")
            break

    # the content check: identical bytes for duplicate, different for nonstyle
    image_root = data_root / "PACS"
    same = 0
    for image_rel, _label, original_rel in rows:
        if sha256(image_root / original_rel) == sha256(data_root / image_rel):
            same += 1
    if expect_identical and same != len(rows):
        problems.append(f"only {same}/{len(rows)} are byte-identical to the "
                        f"original, expected all of them")
    if not expect_identical and same != 0:
        problems.append(f"{same} images are byte-identical to the original, "
                        f"the transform did not take effect")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="folder that contains PACS/; defaults to $NNDL_DATA_ROOT")
    parser.add_argument("--split-file", type=Path, default=None,
                        help="defaults to the frozen photo_train.txt")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the generated images go; defaults to "
                             "<data root>/augmented/<group>")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the transform assignment")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate images that already exist")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    if not (data_root / "PACS").is_dir():
        print(f"ERROR: no PACS images at {data_root / 'PACS'}")
        print("Set NNDL_DATA_ROOT or pass --data-root.")
        return 1

    split_file = args.split_file or (paths.SPLIT_DIR / "photo_train.txt")
    if not Path(split_file).is_file():
        print(f"ERROR: no split file at {split_file}")
        return 1

    report = {"data_root": str(data_root), "split": str(split_file)}
    for group, builder, expect_identical in (
            ("duplicate", build_duplicate, True),
            ("nonstyle", build_nonstyle, False)):
        out_dir = args.out_dir or (data_root / "augmented" / group)
        manifest_path = data_root / "augmented" / f"manifest_{group}.txt"
        if group == "nonstyle":
            info = builder(data_root, Path(split_file), out_dir, manifest_path,
                           args.overwrite, args.seed)
        else:
            info = builder(data_root, Path(split_file), out_dir, manifest_path,
                           args.overwrite)
        problems = verify(manifest_path, Path(split_file), data_root,
                          expect_identical)
        info["problems"] = problems
        info["verified"] = not problems
        report[group] = info
        print(f"\n=== {group} ===")
        for key, value in info.items():
            if key == "problems":
                continue
            print(f"  {key}: {value}")
        if problems:
            for problem in problems:
                print(f"  PROBLEM: {problem}")
        else:
            print("  verification: PASS")

    ok = all(report[group]["verified"] for group in ("duplicate", "nonstyle"))
    print(f"\n{'PASS' if ok else 'FAIL'}: both control groups are consistent")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
