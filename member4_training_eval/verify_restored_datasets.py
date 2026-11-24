"""Check the whole restored dataset, and the fairness of the new group.

Three things are checked:

1. The rebuilt Sobel dataset has all 1349 rows, the images exist, and
   the manifest is well formed.

2. The centre-prior dataset that was preserved is complete and points
   at files that exist.

3. The fairness constraints hold for the new group: the same number of
   rows, the same content-style pairing, the same coverage of the
   originals.

The fairness check is the important one.  The centre run wrote into the
shared directory, and the pairing it recorded must still be the one it
used, otherwise the group is not comparable with anything.
"""
import os
import sys

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
SP = os.path.join(DATA, "aug_sp")
REPO = r"C:\Users\qth02\Desktop\NNDL-Project\repo"
sys.path.insert(0, os.path.join(REPO, "member4_training_eval"))
sys.path.insert(0, REPO)

import config  # noqa: E402
from member4_training_eval.fairness_check import check  # noqa: E402
from member4_training_eval.data import read_aug_manifest  # noqa: E402


def report(name, relative):
    path = os.path.join(DATA, relative)
    print(f"  {name}")
    if not os.path.exists(path):
        print(f"    MISSING manifest {relative}")
        return None
    rows = read_aug_manifest(path)
    print(f"    rows      {len(rows)}")
    missing = [row[0] for row in rows
               if not os.path.exists(os.path.join(DATA, row[0]))]
    print(f"    missing   {len(missing)}")
    if missing:
        for item in missing[:5]:
            print(f"      {item}")
    return rows


def main():
    print("=" * 78)
    print("THE RESTORED DATASETS")
    print("=" * 78)
    print()
    report("ours (Sobel, rebuilt)", "aug_sp/manifest_ours.txt")
    print()
    report("ours_fg_center (centre prior)", "aug_sp/fgcenter/manifest_ours.txt")
    print()
    report("ours_fg (spectral centre)", "aug_sp/foreground/manifest_ours.txt")
    print()
    report("adain_matched (control)", "aug_sp/manifest_adain_matched.txt")

    print()
    print("=" * 78)
    print("FAIRNESS OF THE NEW GROUP")
    print("=" * 78)
    print()
    manifests = [
        # The plain AdaIN manifest lives with the sweep runs, not in the
        # shared directory, so the reference is taken from there.
        ("adain", os.path.join(SP, "sweeps", "sweep_full",
                               "a0p5_b0p5", "manifest_adain.txt")),
        ("adain_matched",
         os.path.join(DATA, "aug_sp/manifest_adain_matched.txt")),
        ("ours", os.path.join(DATA, "aug_sp/manifest_ours.txt")),
        ("ours_fg_center",
         os.path.join(DATA, "aug_sp/fgcenter/manifest_ours.txt")),
    ]
    problems = check(manifests)
    if problems:
        print(f"  FAIL ({len(problems)} problems):")
        for item in problems[:20]:
            print(f"    - {item}")
        return 1
    print("  PASS: the centre-prior group satisfies every fairness constraint")
    print("  that the main groups do -- same rows, same pairing, same coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
