"""Is the generator deterministic?

The original Sobel dataset has to be rebuilt exactly, because the
already-computed results -- the main table, the beta sweep, the
mechanism measurements -- were all obtained from it.  Rebuilding is only
safe if the generator produces byte-identical output for the same input.

The check runs the *same* invocation twice, into two different cell
tags, and compares the two results byte for byte.  Comparing against
pre-existing files would not test anything, because the generator skips
work it has already done.
"""
import hashlib
import os
import subprocess
import sys

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
REPO = r"C:\Users\qth02\Desktop\NNDL-Project\repo"
PYTHON = sys.executable
SP = os.path.join(DATA, "aug_sp")
PREVIEWS = os.path.join(SP, "previews")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root):
    """Map a relative path under root to the file, ignoring png maps."""
    found = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jpg"):
                path = os.path.join(base, name)
                found[os.path.relpath(path, root)] = path
    return found


def generate(tag, modulation, kind=None):
    command = [PYTHON, "-u", "-m", "member3_structure_preserving.generate",
               "samples", "--limit", "7", "--modulation", modulation,
               "--data-root", DATA, "--cell-tag", tag]
    if kind:
        command += ["--foreground-kind", kind]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(command, cwd=REPO, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-1500:])
        print(result.stderr[-2500:])
        raise SystemExit(1)
    root = os.path.join(PREVIEWS, "samples_n7", f"{tag}_a0p5_b0p7")
    return collect(root)


def compare(left, right, label):
    matched = mismatched = 0
    for rel, path in sorted(left.items()):
        if rel not in right:
            print(f"  missing in the second run: {rel}")
            mismatched += 1
            continue
        if sha256(path) == sha256(right[rel]):
            matched += 1
        else:
            mismatched += 1
            print(f"  MISMATCH {rel}")
            print(f"    first  {sha256(path)[:16]}")
            print(f"    second {sha256(right[rel])[:16]}")
    print(f"  {label}: identical {matched}, different {mismatched}")
    return mismatched == 0 and matched > 0


def main():
    print("=" * 78)
    print("DETERMINISM CHECK")
    print("=" * 78)

    print()
    print("A. the edge (Sobel) settings, run twice")
    first = generate("dettest_a", "edge")
    second = generate("dettest_b", "edge")
    edge_ok = compare(first, second, "edge")

    print()
    print("B. the foreground settings, run twice")
    first = generate("dettest_c", "foreground", "center")
    second = generate("dettest_d", "foreground", "center")
    foreground_ok = compare(first, second, "foreground")

    print()
    print("=" * 78)
    if edge_ok and foreground_ok:
        print("VERDICT: the generator is deterministic in both modes -- the")
        print("original Sobel dataset can be rebuilt exactly.")
        return 0
    print("VERDICT: NOT byte-identical -- rebuilding would not reproduce")
    print("the dataset the earlier results were computed on.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
