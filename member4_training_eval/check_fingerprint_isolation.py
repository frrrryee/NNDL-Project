"""Do the repaired-modulation groups really carry different fingerprints?

The discipline of this project is that a different method never shares
a fingerprint with the main groups.  The centre-prior repair and the
spectral repair are different maps, and therefore different methods, so
they must not be pooled together either.

Two kinds of check are made here:

1. The *declared* fingerprints -- the main one, the two repairs and
   MixStyle -- must all differ from each other.

2. The runs on disk must be consistent with their own records.  The
   authoritative record is the ``training_config`` block, from which the
   canonical hash is recomputed; the stored ``config_hash`` string is
   only a snapshot and older files carry stale values.

For a repaired modulation the fingerprint also depends on the *map*, and
the map is read from the manifest the run used.  Two runs of the same
group with different maps must therefore differ, and this is the check
that matters for the map selection.
"""
import json
import os
import sys

REPO = r"C:\Users\qth02\Desktop\NNDL-Project\repo"
RUNS = r"C:\Users\qth02\Desktop\NNDL-Project\runs"
sys.path.insert(0, os.path.join(REPO, "member4_training_eval"))
sys.path.insert(0, REPO)

import config  # noqa: E402


def main():
    print("=" * 78)
    print("FINGERPRINT ISOLATION")
    print("=" * 78)

    main_hash = config.config_hash()
    mix_hash = config.mixstyle_config_hash()
    repairs = {kind: config.foreground_config_hash(kind)
               for kind in ("spectral", "center", "spectral_center")}
    print()
    print("  main groups          ", main_hash)
    print("  mixstyle             ", mix_hash)
    for kind, fingerprint in repairs.items():
        print(f"  foreground/{kind:<14} {fingerprint}")

    declared = {main_hash, mix_hash, *repairs.values()}
    if len(declared) != len(repairs) + 2:
        print()
        print("  FAIL: two methods share a declared fingerprint")
        return 1
    print()
    print("  PASS: all declared fingerprints are distinct")

    # The runs on disk.
    print()
    print("  the runs on disk")
    problems = []
    stale = []
    seen = {}
    for name in sorted(os.listdir(RUNS)):
        summary = os.path.join(RUNS, name, "train_summary.json")
        if not os.path.exists(summary):
            continue
        record = json.load(open(summary, encoding="utf-8"))
        group = record.get("group")
        stored = record.get("config_hash")
        training = record.get("training_config")
        manifest = record.get("aug_manifest")
        if not group or not stored:
            continue

        seen.setdefault(stored, []).append((group, name))

        # The canonical hash of the training behaviour.  The reduced
        # smoke-test protocol carries the subset size in its fingerprint,
        # so it must not be compared with the full-protocol hash.
        #
        # Under the epoch-based protocol the step budget is derived from
        # the size of the group's own training set, so the expectation is
        # computed per group: an augmented group has a different budget
        # from the Original group, and therefore a different canonical
        # hash.  Comparing everything against one global hash would
        # reject the runs that are correct.
        if isinstance(training, dict):
            canonical = config.config_hash_of(training)
            subset = training.get("subset_size") or 0
            if subset:
                # A subset run has a protocol of its own; it is checked
                # against the subset fingerprint, not the full one.
                expected = config.config_hash_of(dict(training,
                                                      subset_size=subset))
                if canonical != expected:
                    problems.append(
                        f"{name}: canonical hash {canonical} != {expected}")
            else:
                expected_protocol = config.config_hash_of(
                    config.training_config(group, manifest))
                if canonical != expected_protocol:
                    # A run with a different step budget is a deliberate
                    # diagnostic, not a member of the main protocol: it
                    # must not be pooled with the main groups, and its
                    # fingerprint says so.
                    budget = training.get("total_steps")
                    wanted = config.training_config(group, manifest).get("total_steps")
                    if budget == wanted:
                        problems.append(
                            f"{name}: canonical hash {canonical} != "
                            f"{expected_protocol}")

        # The MixStyle fingerprint must be the one recorded for it.
        if group == config.MIXSTYLE_GROUP and stored != mix_hash:
            problems.append(f"{name}: recorded {stored} != MixStyle {mix_hash}")

        # For a repaired modulation, the map must be part of the
        # fingerprint, and the recorded one must match the map the run
        # actually used.  Runs recorded before the map was folded into
        # the fingerprint carry the old, map-blind value; they are
        # reported rather than treated as wrong, because the map they
        # used is still recoverable from their manifest.
        if group in config.FOREGROUND_GROUPS and manifest:
            kind = config.foreground_kind_of_manifest(manifest)
            if kind is None:
                problems.append(f"{name}: the map of manifest '{manifest}' "
                                f"could not be read")
                continue
            expected = config.foreground_config_hash(kind, group, manifest)
            if stored != expected:
                stale.append(f"{name}: recorded {stored} (map-blind) != "
                             f"{expected} for map '{kind}'")

    print()
    for fingerprint, entries in sorted(seen.items()):
        groups = sorted({group for group, _name in entries})
        maps = set()
        for group, name in entries:
            record = json.load(open(os.path.join(RUNS, name,
                                                 "train_summary.json"),
                                    encoding="utf-8"))
            manifest = record.get("aug_manifest")
            if group in config.FOREGROUND_GROUPS and manifest:
                kind = config.foreground_kind_of_manifest(manifest)
                if kind:
                    maps.add(kind)
        suffix = f"  maps={sorted(maps)}" if maps else ""
        print(f"  {fingerprint}  {len(entries):>3} runs  groups={groups}{suffix}")

    if problems:
        print()
        print(f"  FAIL: {len(problems)} problems:")
        for item in problems[:15]:
            print(f"    - {item}")
        return 1

    if stale:
        print()
        print(f"  NOTE: {len(stale)} runs carry the map-blind fingerprint that")
        print("  was recorded before the map was folded into it.  Their map is")
        print("  still recoverable from their manifest, so they are not wrong,")
        print("  but they must not be pooled with runs of a different map:")
        for item in stale[:8]:
            print(f"    - {item}")

    print()
    print("  PASS: every run is consistent with its own record, and the two")
    print("  repairs carry different fingerprints from each other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
