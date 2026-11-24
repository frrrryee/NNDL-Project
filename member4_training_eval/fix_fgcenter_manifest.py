"""Repair the manifest of the preserved centre-prior dataset.

The ``center`` run was moved out of the shared ``aug_sp`` directory into
``aug_sp/fgcenter`` after it had replaced the original dataset there.
The manifest it published still names the shared paths, so its rows no
longer resolve.  This script rewrites the image column to the new
location and checks that every row points at a file that exists.

The content-style pairing, the alpha and the beta are left untouched:
they are what the experiment is about, and they must not change.
"""
import os
import sys

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
SP = os.path.join(DATA, "aug_sp")
OLD_PREFIX = "aug_sp/images/"
NEW_PREFIX = "aug_sp/fgcenter/images/"


def main():
    manifest = os.path.join(SP, "fgcenter", "manifest_ours.txt")
    if not os.path.exists(manifest):
        print(f"  no manifest at {manifest}")
        return 1

    with open(manifest, encoding="utf-8") as handle:
        rows = handle.read().splitlines()

    print(f"  rows: {len(rows)}")
    changed = 0
    fixed = []
    for row in rows:
        parts = row.split(" ")
        if not parts:
            continue
        if parts[0].startswith(OLD_PREFIX):
            parts[0] = NEW_PREFIX + parts[0][len(OLD_PREFIX):]
            changed += 1
        fixed.append(" ".join(parts))

    print(f"  rewritten image paths: {changed}")

    missing = 0
    for row in fixed:
        path = row.split(" ")[0]
        if not os.path.exists(os.path.join(DATA, path)):
            missing += 1
            if missing <= 5:
                print(f"  MISSING {path}")
    print(f"  rows whose image does not exist: {missing}")

    if missing:
        print()
        print("  the manifest is not repaired -- some images are absent")
        return 1

    with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(fixed) + "\n")
    print(f"  wrote {manifest}")
    print(f"  first row: {fixed[0][:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
