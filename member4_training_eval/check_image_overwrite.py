"""Was the original (Sobel) dataset overwritten in place?

The ``center`` run wrote into the shared ``aug_sp`` directory.  This
checks the modification times of the individual image files, which say
whether the existing files were replaced or whether the generator
merely added new ones.
"""
import os
import time

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
RUN_START = time.time() - 45 * 60  # the center run began roughly here


def stamp(path):
    return time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def scan(root):
    """Return the newest and oldest file stamps under a directory."""
    stamps = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            try:
                stamps.append(os.path.getmtime(path))
            except OSError:
                pass
    if not stamps:
        return None, None, 0
    return max(stamps), min(stamps), len(stamps)


for rel in ["aug_sp/images", "aug_sp/adain_matched",
            "aug_sp/foreground/images", "aug_sp/foreground/adain_matched",
            "aug_sp/sweeps", "augmented"]:
    root = os.path.join(DATA, rel)
    newest, oldest, count = scan(root)
    if newest is None:
        print(f"  {rel}: empty")
        continue
    fresh = sum(1 for _ in ())
    print(f"  {rel}")
    print(f"      files   = {count}")
    print(f"      newest  = {stamp(root)}  ({time.strftime('%m-%d %H:%M', time.localtime(newest))})")
    print(f"      oldest  = {time.strftime('%m-%d %H:%M', time.localtime(oldest))}")

print()
print("=== files under aug_sp/images modified in the last 45 minutes ===")
recent = []
for base, _dirs, files in os.walk(os.path.join(DATA, "aug_sp", "images")):
    for name in files:
        path = os.path.join(base, name)
        try:
            if os.path.getmtime(path) > RUN_START:
                recent.append(path)
        except OSError:
            pass
print(f"  count = {len(recent)}")
for path in recent[:5]:
    print("   ", stamp(path), path.replace(DATA, ""))

print()
print("=== what the running jobs point at ===")
manifest = os.path.join(DATA, "aug_sp", "manifest_ours.txt")
if os.path.exists(manifest):
    with open(manifest, encoding="utf-8") as handle:
        print("   aug_sp/manifest_ours.txt ->", handle.readline().strip()[:110])
manifest = os.path.join(DATA, "augmented", "manifest_ours.txt")
if os.path.exists(manifest):
    with open(manifest, encoding="utf-8") as handle:
        print("   augmented/manifest_ours.txt ->", handle.readline().strip()[:110])
