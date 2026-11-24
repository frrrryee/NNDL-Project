"""Verify the rebuilt Sobel dataset against copies made by the old code.

The original dataset was rebuilt with the current generator code.  The
copies in ``previews/build_n5/a0p5_b0p7`` were made on 11-18 with the
code that produced the main table, at the same alpha and beta.  If the
rebuild is byte-identical to them, the earlier results still describe
the dataset that is now on disk.

This is the check that matters: it compares the rebuilt images against
images produced by *different* code, at the parameters of the main
table.
"""
import hashlib
import json
import os

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
SP = os.path.join(DATA, "aug_sp")
OLD = os.path.join(SP, "previews", "build_n5", "a0p5_b0p7")
NEW = os.path.join(SP, "images")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root):
    found = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jpg"):
                path = os.path.join(base, name)
                found[os.path.relpath(path, root).replace("\\", "/")] = path
    return found


def main():
    print("=" * 78)
    print("VERIFYING THE REBUILT DATASET AGAINST THE OLD CODE")
    print("=" * 78)

    record = json.load(open(os.path.join(OLD, "generation.json"),
                            encoding="utf-8"))
    settings = record.get("settings", {})
    print(f"  reference: alpha={settings.get('alpha')} beta={settings.get('beta')} "
          f"method={settings.get('method')}")
    print(f"             generate.py={settings.get('code', {}).get('generate.py', '')[:12]} "
          f"stylizer.py={settings.get('code', {}).get('stylizer.py', '')[:12]}")

    old = collect(os.path.join(OLD, "images"))
    new = collect(NEW)
    print(f"  reference images {len(old)}, rebuilt images {len(new)}")
    print()

    matched = mismatched = missing = 0
    for rel, path in sorted(old.items()):
        if rel not in new:
            print(f"  MISSING {rel}")
            missing += 1
            continue
        if sha256(path) == sha256(new[rel]):
            matched += 1
        else:
            mismatched += 1
            print(f"  MISMATCH {rel}")
            print(f"    old {sha256(path)[:20]}")
            print(f"    new {sha256(new[rel])[:20]}")

    print()
    print(f"  identical {matched}, different {mismatched}, missing {missing}")
    if mismatched == 0 and missing == 0 and matched > 0:
        print()
        print("  VERDICT: the rebuild is byte-identical to images produced by")
        print("  the code that generated the main table.  The earlier results")
        print("  describe the dataset that is now on disk.")
        return 0
    print()
    print("  VERDICT: the rebuild does not match -- the earlier results")
    print("  cannot be attributed to the dataset on disk.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
