"""Rebuild the original (Sobel) dataset that the center run replaced.

Background
----------
The ``center`` foreground run was started with the cell tag
``fgcenter``.  The generator only recognises the exact tag ``foreground``
in its build mode; every other tag falls through to the shared
``aug_sp`` directory -- the home of the *original* Sobel dataset.  The
run therefore replaced that dataset with the center-modulated one.

Recovery
--------
The generator is deterministic: it contains no random number use, and
the style of each image is fixed by the style-pool manifest.  The
``edge`` code path is unchanged since the sweeps were made (the
foreground commit only added a second branch; for ``edge`` the
modulation map is still the Sobel structure map).  So repeating the
original invocation reproduces the dataset exactly.

This script regenerates the dataset into ``aug_sp/images`` and checks
the result against the untouched ``aug_sp/sweeps`` copy of the same
alpha/beta cell.
"""
import hashlib
import os
import subprocess
import sys

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
REPO = r"C:\Users\qth02\Desktop\NNDL-Project\repo"
PYTHON = sys.executable
SP = os.path.join(DATA, "aug_sp")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root):
    """Map the relative path of every jpg under root to the file."""
    found = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jpg"):
                path = os.path.join(base, name)
                found[os.path.relpath(path, root)] = path
    return found


def main():
    print("=" * 78)
    print("REBUILDING THE ORIGINAL SOBEL DATASET")
    print("=" * 78)

    images = os.path.join(SP, "images")
    if not os.path.exists(images):
        print("  images/ is missing -- the rebuild has not run yet")
        return 1

    command = [PYTHON, "-u", "-m", "member3_structure_preserving.generate",
               "build", "--modulation", "edge", "--alpha", "0.5",
               "--beta", "0.7", "--data-root", DATA, "--with-matched"]
    print("  running:", " ".join(command[1:]))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(command, cwd=REPO, env=env)
    if result.returncode != 0:
        print(f"  the generator failed with code {result.returncode}")
        return 1

    print()
    print("  checking the rebuild against the untouched sweep copy")
    reference = collect(os.path.join(SP, "sweeps", "sweep_full",
                                     "a0p5_b0p7", "images"))
    rebuilt = collect(images)
    matched = mismatched = 0
    for rel, path in sorted(reference.items()):
        if rel not in rebuilt:
            print(f"  missing: {rel}")
            mismatched += 1
            continue
        if sha256(path) == sha256(rebuilt[rel]):
            matched += 1
        else:
            mismatched += 1
            if mismatched <= 5:
                print(f"  MISMATCH {rel}")
                print(f"    sweep   {sha256(path)[:16]}")
                print(f"    rebuild {sha256(rebuilt[rel])[:16]}")
    print(f"  identical {matched}, different {mismatched}")
    if mismatched:
        print()
        print("  VERDICT: the rebuild does not match the sweep copy -- the")
        print("  earlier results cannot be reproduced from it.")
        return 1
    print()
    print("  VERDICT: the rebuild is byte-identical to the sweep copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
