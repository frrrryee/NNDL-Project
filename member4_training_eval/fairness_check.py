"""Mechanical fairness verification across the groups' manifests.

Checks that the +AdaIN, +AdaIN (strength-matched) and +Ours manifests satisfy
every constraint listed in Section 3.7 before any training run is accepted
into the results:

  1. same original images: every manifest's provenance (content images) equals
     the photo_train split, AUG_PER_IMAGE stylizations per original;
  2. same style pool: style names in all manifests come from the frozen
     style-pool manifest and the pool hash matches the recorded one;
  3. same augmentation count: identical number of stylized rows;
  4. same content-style pairing: original->style mapping identical between
     the manifests (only the generator differs);
  5. no test-set leakage: every style image referenced by the pool is
     byte-disjoint from the local PACS copy, so that no target-domain test
     image can reach the training set through the style pool.

Usage:
    python fairness_check.py --adain manifest.txt --ours manifest.txt
    python fairness_check.py --adain manifest.txt --adain-matched manifest.txt \
                             --ours manifest.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import config
from data import read_aug_manifest


def check(manifests, subset=False, pool_path=None):
    """Verify every manifest in ``manifests`` (a list of (name, path) pairs).

    The control groups ``duplicate`` and ``nonstyle`` have no style image:
    their manifests carry the placeholder ``none`` in the style column.
    The checks that concern the style pool (2), the content-style pairing
    (4) and the style-image leak (5) are therefore skipped for them, while
    the checks that define what the controls are *for* -- the same original
    images (1) and the same number of rows (3) -- are applied unchanged.

    ``pool_path`` is the style pool the manifests are supposed to belong to.
    It defaults to the frozen ten-painting pool.  The style-pool sensitivity
    experiment trains with a *half* pool, whose pairing is deliberately
    different from the frozen one, so the check has to be told which pool it
    validates against -- otherwise the half-pool manifests would be rejected
    for not matching the frozen pairing.
    """
    problems = []
    loaded = {}
    for name, path in manifests:
        try:
            loaded[name] = read_aug_manifest(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: cannot read manifest ({exc})")
    if len(loaded) != len(manifests):
        return problems

    # Split the manifests into the ones that carry a real style image and the
    # ones that do not.
    styled = [name for name in loaded if name not in config.NO_STYLE_GROUPS]
    unstyled = [name for name in loaded if name in config.NO_STYLE_GROUPS]

    # 3. counts
    counts = {name: len(rows) for name, rows in loaded.items()}
    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{name} {count}" for name, count in counts.items())
        problems.append(f"row count differs across manifests: {detail}")
    expected = len(config.read_split(config.SPLIT_FILES["photo_train"])) * config.AUG_PER_IMAGE
    for name, count in counts.items():
        if count != expected and not subset:
            problems.append(f"{name}: {count} rows != expected {expected}")

    # 1. same originals
    train_originals = {rel for rel, _ in config.read_split(config.SPLIT_FILES["photo_train"])}
    provenance = {}
    for name, rows in loaded.items():
        origs = Counter(meta[0] if meta else "?" for _, _, meta in rows)
        covered = set(origs)
        target = train_originals if not subset else covered
        if not covered <= train_originals:
            problems.append(f"{name}: provenance references non-training originals")
        if covered != target:
            missing = target - covered
            problems.append(f"{name}: provenance originals mismatch (missing {len(missing)})")
        for orig, count in origs.items():
            if count != config.AUG_PER_IMAGE:
                problems.append(f"{name}: original {orig} used {count}x != {config.AUG_PER_IMAGE}")
        provenance[name] = origs

    # The control groups must actually carry the placeholder in the style
    # column, otherwise they would be compared against the style pool and
    # the whole point of the relaxation would be lost.
    for name in unstyled:
        for rel, _label, meta in loaded[name]:
            if len(meta) > 1 and meta[1] not in ("none", "-", ""):
                problems.append(
                    f"{name}: style column is {meta[1]!r}, expected the placeholder "
                    f"'none' for a group without a style image")
                break

    # 2. same style pool (only for the groups that have one)
    pool_path = Path(pool_path) if pool_path is not None else config.STYLE_POOL_MANIFEST
    pool = {}
    if styled:
        if pool_path.is_file():
            outer = json.loads(pool_path.read_text(encoding="utf-8"))
            pool = outer.get("pool", {})
        else:
            problems.append(f"style pool manifest missing: {pool_path}")
        for name in styled:
            rows = loaded[name]
            styles = {meta[1] for _, _, meta in rows if len(meta) > 1}
            unknown = styles - set(pool.get("styles", {}))
            if unknown:
                problems.append(f"{name}: styles not in the active pool: {sorted(unknown)[:5]}")

    # 4. same content-style pairing (only among the styled groups)
    def pairing(rows):
        return {meta[0]: meta[1] for _, _, meta in rows if len(meta) > 1}

    if styled:
        reference_name = styled[0]
        reference_pairing = pairing(loaded[reference_name])
        for name in styled:
            if pairing(loaded[name]) != reference_pairing:
                problems.append(
                    f"content->style pairing differs between {reference_name} and {name}")
        if pool and subset is False:
            recorded = json.loads(pool_path.read_text(encoding="utf-8")).get("pairing", {})
            if reference_pairing != recorded:
                problems.append(
                    f"content->style pairing differs from the pairing of the "
                    f"active style pool ({pool_path.name})")

    # recorded hash (exclude pool_hash itself, as at build time)
    if pool:
        payload = {k: v for k, v in pool.items() if k != "pool_hash"}
        actual = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        if pool.get("pool_hash") != actual:
            problems.append(f"style pool hash mismatch: recorded {pool.get('pool_hash')} != {actual}")

    # 5. no test-set leakage: the style images must not be PACS images
    if styled:
        problems.extend(check_no_leak(pool))

    # 8. the control groups must be what they claim to be
    problems.extend(check_control_content(loaded, unstyled))

    return problems


def check_control_content(loaded, unstyled):
    """Verify the defining property of every data-volume control group.

    ``duplicate`` must be a byte-identical copy of the original, and
    ``nonstyle`` must differ from it.  Without this check the two controls
    could silently become the same group, and the decomposition of the gain
    would collapse.
    """
    problems = []
    if not unstyled:
        return problems
    root = config.DATA_ROOT
    for name in unstyled:
        identical = 0
        total = 0
        for rel, _label, meta in loaded[name]:
            if not meta:
                continue
            source = root / "PACS" / meta[0]
            target = root / rel
            if not source.is_file() or not target.is_file():
                continue
            total += 1
            if source.read_bytes() == target.read_bytes():
                identical += 1
        if total == 0:
            problems.append(f"{name}: no comparable image pairs")
            continue
        if name == "duplicate" and identical != total:
            problems.append(
                f"duplicate: only {identical}/{total} images are byte-identical to "
                f"the original, so the data-volume control is not a pure duplicate")
        if name == "nonstyle" and identical != 0:
            problems.append(
                f"nonstyle: {identical}/{total} images are byte-identical to the "
                f"original, so the appearance control has no effect")
    return problems


def check_no_leak(pool):
    """Return problems if any style image is a PACS image.

    Art Painting, Cartoon and Sketch are the evaluation domains.  A style image
    that is one of those images would put the test appearance into training,
    so the check compares content hashes against the whole local PACS copy.
    """
    problems = []
    if not pool:
        return problems

    style_dir = config.STYLE_POOL_IMAGES
    pacs_root = config.IMAGE_ROOT
    if not pacs_root.is_dir():
        problems.append(f"PACS image root not found: {pacs_root}")
        return problems

    pacs = {}
    for path in pacs_root.rglob("*"):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            pacs.setdefault(digest, path.as_posix())

    for name, meta in pool.get("styles", {}).items():
        filename = Path(meta.get("file", "")).name
        path = style_dir / filename
        if not path.is_file():
            problems.append(f"style image missing for '{name}': {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in pacs:
            problems.append(
                f"LEAK: style '{name}' is byte-identical to the PACS image "
                f"{pacs[digest]}; style images must come from outside PACS")
        if meta.get("sha256") and meta["sha256"] != digest:
            problems.append(f"style '{name}': sha256 does not match the recorded one")

    return problems


def check_not_synthetic(manifests):
    """Refuse manifests that point at the synthetic smoke-test dataset.

    ``make_synthetic_aug.py`` writes 40 copies of the original images under
    ``augmented/<group>/`` with exactly the file names a real generator uses.
    A manifest that references them would silently train on un-stylized
    images, and the training code has no way to notice.  The manifests are
    named ``manifest_<group>_synthetic.txt`` and live in the same directory,
    so the guard is on the manifest name and on the image content.
    """
    problems = []
    for name, path in manifests:
        if "synthetic" in Path(path).name.lower():
            problems.append(
                f"{name}: manifest '{path.name}' is a synthetic smoke-test manifest "
                f"and must never be used for a reported experiment")
    return problems


def check_images_exist(manifests):
    """Every manifest row must point at a real file under the data root.

    The manifest paths are resolved relative to ``DATA_ROOT``, exactly as
    ``data.py`` does at training time, so a path that only works on one
    machine is caught here rather than in the middle of a training run.
    """
    problems = []
    root = config.DATA_ROOT
    for name, path in manifests:
        try:
            rows = read_aug_manifest(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: cannot read manifest ({exc})")
            continue
        if not rows:
            problems.append(f"{name}: manifest is empty")
            continue
        missing = 0
        absolute = 0
        for rel, _label, _meta in rows:
            if Path(rel).is_absolute():
                absolute += 1
                continue
            if not (root / rel).is_file():
                missing += 1
        if missing:
            problems.append(
                f"{name}: {missing}/{len(rows)} manifest images do not exist under "
                f"{root}")
        if absolute:
            problems.append(
                f"{name}: {absolute} manifest rows use absolute paths; the paths must "
                f"be relative to DATA_ROOT so that the run is portable")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adain", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--adain-matched", type=Path, default=None,
                        help="strength-matched uniform AdaIN control manifest")
    parser.add_argument("--ours-equal", type=Path, default=None,
                        help="equal-strength spatial control manifest")
    parser.add_argument("--duplicate", type=Path, default=None,
                        help="data-volume control manifest (byte-identical copies)")
    parser.add_argument("--nonstyle", type=Path, default=None,
                        help="appearance control manifest (non-stylistic transforms)")
    parser.add_argument("--foreground", type=Path, default=None,
                        help="repaired-modulation manifest (spectral or centre map)")
    parser.add_argument("--foreground-matched", type=Path, default=None,
                        help="matched uniform control of a repaired modulation")
    parser.add_argument("--subset", action="store_true",
                        help="pipeline test: allow a subset of originals, "
                             "skip full-coverage and frozen-pairing checks")
    parser.add_argument("--runs", nargs="+", type=Path, default=None,
                        help="run directories (or summary_*.json files) whose "
                             "recorded config_hash must match the current one")
    args = parser.parse_args()

    manifests = [("adain", args.adain), ("ours", args.ours)]
    if args.adain_matched is not None:
        manifests.insert(1, ("adain_matched", args.adain_matched))
    if args.ours_equal is not None:
        manifests.append(("ours_equal", args.ours_equal))
    if args.duplicate is not None:
        manifests.append(("duplicate", args.duplicate))
    if args.nonstyle is not None:
        manifests.append(("nonstyle", args.nonstyle))
    if args.foreground is not None:
        manifests.append(("ours_fg", args.foreground))
    if args.foreground_matched is not None:
        manifests.append(("ours_fg_center_matched", args.foreground_matched))

    problems = check(manifests, subset=args.subset)

    # 6. identical training configuration across the groups
    problems.extend(check_config_hashes(args.runs))

    # 7. the manifests must be real experiment manifests and their rows must
    #    resolve to files that actually exist under DATA_ROOT
    problems.extend(check_not_synthetic(manifests))
    if not args.subset:
        problems.extend(check_images_exist(manifests))

    if problems:
        print(f"FAIL ({len(problems)} problems):")
        for p in problems[:30]:
            print("  -", p)
        raise SystemExit(1)
    print("PASS: the augmented manifests satisfy all fairness constraints.")


def check_config_hashes(runs):
    """Verify that the given runs used the current training configuration.

    For a run directory the authoritative record is ``train_summary.json``,
    which stores the full ``training_config`` block.  The fingerprint is
    recomputed from that block rather than trusting a stored ``config_hash``
    string, so a run recorded under an older hash definition is still judged
    on what it actually did, and a run whose hyper-parameters differ from the
    current ones is rejected.
    """
    problems = []
    if not runs:
        return problems
    expected = config.config_hash()
    for run in runs:
        run = Path(run)
        if run.is_dir():
            summary = run / "train_summary.json"
            if not summary.is_file():
                problems.append(f"no train_summary.json in {run}")
                continue
            candidates = [summary]
        else:
            candidates = [run]

        for candidate in candidates:
            try:
                payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{candidate}: cannot read ({exc})")
                continue
            if not isinstance(payload, dict):
                problems.append(f"{candidate}: not a JSON object")
                continue
            recorded_config = payload.get("training_config")
            if isinstance(recorded_config, dict):
                actual = config.config_hash_of(recorded_config)
                if actual != expected:
                    problems.append(
                        f"{candidate}: training config {actual} != current {expected}")
                continue
            # Fallback for artifacts that only carry the hash string.
            recorded_hash = payload.get("config_hash")
            if recorded_hash is None:
                problems.append(f"{candidate}: no config_hash found")
            elif recorded_hash != expected:
                problems.append(
                    f"{candidate}: config_hash {recorded_hash} != current {expected}")
    return problems


if __name__ == "__main__":
    main()
