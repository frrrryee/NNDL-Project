"""Verify that the training protocol is the epoch-based one, everywhere.

The protocol is "30 passes over each group's own training set", so the
step budget of a group is derived from the size of its data rather than
being a constant that is handed out to every group.  This script checks
the four things that could go wrong with that rule:

    1. the budget is a function of the data, not a constant;
    2. the Original group is untouched -- same budget, same fingerprint --
       so every result already on disk stays valid;
    3. every group gets 30 passes over its own data;
    4. the reuse check sees a run of the old fixed-budget protocol as
       stale, and a run of the current protocol as acceptable;
    5. a stored fingerprint snapshot that was written under an earlier
       hash definition is not mistaken for a different protocol.

Usage:
    python verify_epoch_protocol.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import config  # noqa: E402
from fairness_check import check_config_hashes  # noqa: E402

FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def groups_and_manifests():
    """Every group with the manifest it trains on."""
    table = {}
    for group in ("original", "mixstyle"):
        table[group] = None
    for group in ("adain", "adain_matched", "ours", "ours_equal",
                  "duplicate", "nonstyle"):
        table[group] = config.DATA_ROOT / config.GROUP_MANIFESTS[group]
    table[config.FOREGROUND_CENTER_GROUP] = \
        config.DATA_ROOT / config.FOREGROUND_CENTER_MANIFEST
    table[config.FOREGROUND_CENTER_MATCHED_GROUP] = \
        config.DATA_ROOT / config.FOREGROUND_CENTER_MATCHED_MANIFEST
    for group, relative in config.LINE_ART_GROUPS.items():
        table[group] = config.DATA_ROOT / relative
    return table


def main() -> int:
    print()
    print("=" * 78)
    print("1. THE BUDGET IS DERIVED FROM THE DATA")
    print("=" * 78)
    check("1349 images -> 660 steps",
          config.total_steps_for(1349) == 660,
          f"got {config.total_steps_for(1349)}")
    check("2698 images -> 1290 steps",
          config.total_steps_for(2698) == 1290,
          f"got {config.total_steps_for(2698)}")
    check("the budget is not a constant",
          config.total_steps_for(1349) != config.total_steps_for(2698))

    print()
    print("=" * 78)
    print("2. THE ORIGINAL GROUP IS UNTOUCHED")
    print("=" * 78)
    check("original budget is still 660",
          config.total_steps_of_group("original") == 660)
    check("original fingerprint is still ecc6090e5b29",
          config.config_hash_for_group("original") == "ecc6090e5b29",
          config.config_hash_for_group("original"))
    check("mixstyle fingerprint is still e3bcbe6568bb",
          config.config_hash_for_group("mixstyle") == "e3bcbe6568bb",
          config.config_hash_for_group("mixstyle"))
    check("the recorded training block of the Original group is reproduced",
          config.training_config("original")["total_steps"] == 660)

    print()
    print("=" * 78)
    print("3. EVERY GROUP GETS 30 EPOCHS OVER ITS OWN DATA")
    print("=" * 78)
    for group, manifest in groups_and_manifests().items():
        try:
            images = config.train_images_of_group(group, manifest)
            steps = config.total_steps_of_group(group, manifest)
        except (OSError, ValueError) as exc:
            check(f"{group}: the budget can be established", False, str(exc))
            continue
        epochs = steps * config.BATCH_SIZE / images
        check(f"{group:<26} {steps:>5} steps over {images:>5} images "
              f"= {epochs:5.1f} epochs",
              30.0 <= epochs <= 31.5, f"({epochs:.1f})")

    print()
    print("=" * 78)
    print("4. THE REUSE CHECK SEES THE OLD PROTOCOL AS STALE")
    print("=" * 78)
    manifest = config.DATA_ROOT / config.GROUP_MANIFESTS["duplicate"]
    old_run = config.OUTPUT_ROOT / "duplicate_s0"
    if (old_run / "train_summary.json").is_file():
        check("a run of the old fixed budget is NOT accepted",
              not config.run_matches_protocol(old_run, "duplicate", manifest))
    else:
        print("  (no run of the old protocol on disk to test against)")

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "duplicate_s1"
        fake.mkdir()
        payload = {
            "group": "duplicate",
            "seed": 1,
            "aug_manifest": str(manifest),
            "config_hash": config.config_hash_for_group("duplicate", manifest),
            "training_config": config.training_config("duplicate", manifest),
            "total_steps": config.total_steps_of_group("duplicate", manifest),
            "train_images": config.train_images_of_group("duplicate", manifest),
        }
        (fake / "train_summary.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
        check("a run of the current protocol IS accepted",
              config.run_matches_protocol(fake, "duplicate", manifest))
        check("the fairness checker accepts the current protocol",
              not check_config_hashes([fake])[0])
        if (old_run / "train_summary.json").is_file():
            check("the fairness checker rejects the old protocol",
                  bool(check_config_hashes([old_run])[0]))

    print()
    print("=" * 78)
    print("5. THE ORIGINAL RUNS ON DISK ARE STILL VALID")
    print("=" * 78)
    for seed in range(5):
        run = config.OUTPUT_ROOT / f"original_s{seed}"
        if not (run / "train_summary.json").is_file():
            continue
        check(f"original_s{seed} still matches the current protocol",
              config.run_matches_protocol(run, "original"))

    print()
    print("=" * 78)
    print("6. A STALE FINGERPRINT SNAPSHOT IS NOT A PROTOCOL VIOLATION")
    print("=" * 78)
    # A run recorded under the earlier hash definition -- the one that
    # hashed the whole training block -- carries a different string even
    # though its training behaviour is unchanged.  It must not be refused
    # for that reason, and a snapshot that agrees with neither definition
    # must still be refused.
    current = config.training_config("original")
    legacy_payload = json.dumps(current, sort_keys=True).encode()
    legacy = hashlib.sha256(legacy_payload).hexdigest()[:12]
    check("the two hash definitions really do differ",
          legacy != config.config_hash_of(current),
          f"{legacy} vs {config.config_hash_of(current)}")
    check("a snapshot under the earlier definition is accepted",
          config.snapshot_is_consistent(legacy, current))
    check("a snapshot under the current definition is accepted",
          config.snapshot_is_consistent(config.config_hash_of(current), current))
    check("a snapshot that matches neither definition is refused",
          not config.snapshot_is_consistent("deadbeef0000", current))
    check("a snapshot with no recorded block cannot be checked",
          not config.snapshot_is_consistent(config.config_hash_of(current), None))
    for seed in range(5):
        run = config.OUTPUT_ROOT / f"original_s{seed}"
        summary = run / "train_summary.json"
        if not summary.is_file():
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        stored = payload.get("config_hash")
        recorded = payload.get("training_config")
        check(f"original_s{seed}: its stored snapshot agrees with its record",
              config.snapshot_is_consistent(stored, recorded),
              f"stored {stored}")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} checks failed:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("PASS: every check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
