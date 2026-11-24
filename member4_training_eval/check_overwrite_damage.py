"""Where the original Sobel dataset went, and where the center one landed.

The ``center`` run was given the cell tag ``fgcenter``, which the
generator did not recognise.  It therefore wrote into the shared
``aug_sp`` directory -- the home of the *original* (Sobel) dataset --
instead of a directory of its own.  This script establishes exactly what
was overwritten and what is still intact, so that the repair can be
planned before any GPU time is spent.
"""
import json
import os
import time

DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
REPO = r"C:\Users\qth02\Desktop\NNDL-Project\repo"


def stamp(path):
    if not os.path.exists(path):
        return "MISSING"
    return time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def count(root):
    if not os.path.exists(root):
        return 0
    return sum(len(files) for _, _, files in os.walk(root))


print("=== the two generation records ===")
for rel in ["aug_sp/generation.json", "aug_sp/foreground/generation.json"]:
    path = os.path.join(DATA, rel)
    if not os.path.exists(path):
        print(f"  {rel}: MISSING")
        continue
    record = json.load(open(path, encoding="utf-8"))
    settings = record.get("settings", {})
    print(f"  {stamp(path)}  {rel}")
    print(f"      method={settings.get('method')}  fk={settings.get('foreground_kind')}")
    print(f"      alpha={settings.get('alpha')} beta={settings.get('beta')} "
          f"complete={record.get('complete')}")

print()
print("=== what the center run wrote (sample of its outputs) ===")
record = json.load(open(os.path.join(DATA, "aug_sp", "generation.json"),
                        encoding="utf-8"))
outputs = record.get("outputs", {})
for key in list(outputs)[:2]:
    print(f"  {key}")
    for name, value in outputs[key].get("files", {}).items():
        print(f"      {name:16} {value}")

print()
print("=== augmented/ layout ===")
aug = os.path.join(DATA, "augmented")
for name in sorted(os.listdir(aug)):
    path = os.path.join(aug, name)
    if os.path.isdir(path):
        print(f"  DIR  {stamp(path)}  {name:22} files={count(path)}")
    else:
        print(f"  FILE {stamp(path)}  {name}")

print()
print("=== manifests referenced by the training config ===")
import sys
sys.path.insert(0, os.path.join(REPO, "member4_training_eval"))
sys.path.insert(0, REPO)
import config  # noqa: E402

for group in ["original", "ours", "adain_matched", "ours_fg"]:
    manifest = config.group_manifest(group) if hasattr(config, "group_manifest") else None
    print(f"  {group:16} -> {manifest}")

print()
print("=== raw config constants ===")
for name in dir(config):
    if "MANIFEST" in name or "GROUP" in name:
        print(f"  {name} = {getattr(config, name)}")
