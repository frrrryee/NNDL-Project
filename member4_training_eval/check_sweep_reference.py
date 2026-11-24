"""What survives in the sweep directory, and can it be used as a reference?

The sweep directory was not touched by the center run, so it holds
copies of the original Sobel dataset for several alpha/beta cells.  If
any cell matches the main table's settings (alpha 0.5, beta 0.7), the
rebuild can be verified against it byte for byte.
"""
import json
import os

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
SP = os.path.join(DATA, "aug_sp")
SWEEPS = os.path.join(SP, "sweeps", "sweep_full")


def count_images(root):
    total = 0
    for base, _dirs, files in os.walk(root):
        total += len([f for f in files if f.endswith(".jpg")])
    return total


print("=== sweep cells ===")
for name in sorted(os.listdir(SWEEPS)):
    path = os.path.join(SWEEPS, name)
    if not os.path.isdir(path):
        continue
    images = os.path.join(path, "images")
    count = count_images(images) if os.path.isdir(images) else 0
    settings = {}
    record = os.path.join(path, "generation.json")
    if os.path.exists(record):
        data = json.load(open(record, encoding="utf-8"))
        settings = data.get("settings", {})
    alpha = settings.get("alpha", "?")
    beta = settings.get("beta", "?")
    print(f"  {name:28} alpha={alpha} beta={beta} images={count}")

print()
print("=== the main table's cell (alpha 0.5, beta 0.7) ===")
for name in sorted(os.listdir(SWEEPS)):
    record = os.path.join(SWEEPS, name, "generation.json")
    if not os.path.exists(record):
        continue
    data = json.load(open(record, encoding="utf-8"))
    settings = data.get("settings", {})
    if settings.get("alpha") == 0.5 and settings.get("beta") == 0.7:
        images = os.path.join(SWEEPS, name, "images")
        print(f"  {name}  images={count_images(images)}")
        print(f"     method={settings.get('method')}  "
              f"modulation={settings.get('modulation')}")
