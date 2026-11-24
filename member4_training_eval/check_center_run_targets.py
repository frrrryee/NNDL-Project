"""Which directories did the center run write into?

The ``center`` run was given the cell tag ``fgcenter``.  The generator
only recognises the exact tag ``foreground``, so it fell through to the
shared ``aug_sp`` directory -- the home of the *original* Sobel dataset
-- and replaced it.  This script finds every file the run touched, so
that the center dataset can be preserved before the original is rebuilt.
"""
import os
import time

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
# The center run started at about 19:09 and finished at 19:19.
START = time.mktime(time.strptime("2026-11-24 19:05", "%Y-%m-%d %H:%M"))


def count(root):
    if not os.path.exists(root):
        return 0
    return sum(len(files) for _, _, files in os.walk(root))


print("=== directories under the data root, with fresh-file counts ===")
for base, dirs, files in os.walk(DATA):
    dirs[:] = [d for d in dirs if d != "_synthetic_smoke"]
    if not files:
        continue
    fresh = 0
    for name in files:
        path = os.path.join(base, name)
        try:
            if os.path.getmtime(path) > START:
                fresh += 1
        except OSError:
            pass
    if fresh:
        print(f"  {fresh:>6} fresh / {len(files):>6} total   "
              f"{base.replace(DATA, '')}")

print()
print("=== the whole aug_sp tree, one level deep ===")
sp = os.path.join(DATA, "aug_sp")
for name in sorted(os.listdir(sp)):
    path = os.path.join(sp, name)
    if os.path.isdir(path):
        print(f"  DIR  {name:22} files={count(path)}")
    else:
        print(f"  FILE {name}")

print()
print("=== is the center matched-control set anywhere? ===")
found = False
for base, _dirs, files in os.walk(DATA):
    for name in files:
        if "adain_matched" in base and name.endswith(".jpg"):
            path = os.path.join(base, name)
            if os.path.getmtime(path) > START:
                print("   fresh:", path.replace(DATA, ""))
                found = True
if not found:
    print("   none: the center run did not write matched controls")
