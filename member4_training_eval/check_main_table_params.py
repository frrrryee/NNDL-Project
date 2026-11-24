"""Which manifest and parameters did the main table's ``ours`` group use?

The rebuild has to reproduce the dataset that the already-reported
numbers were computed on, so the parameters come from the run records
rather than from an assumption.
"""
import json
import os

RUNS = r"C:\Users\qth02\Desktop\NNDL-Project\runs"
REPO = r"C:\Users\qth02\Desktop\NNDL-Project\repo"


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


print("=== run directories for the ours group ===")
for name in sorted(os.listdir(RUNS)):
    if "ours" not in name or "fg" in name or "a0p" in name:
        continue
    record = load(os.path.join(RUNS, name, "eval_last.json"))
    if not record:
        continue
    config = record.get("config", record)
    print(f"  {name}")
    for key in ("group", "aug_manifest", "alpha", "beta", "seed",
                "config_hash"):
        if key in config:
            print(f"      {key} = {config[key]}")

print()
print("=== any run that records an aug_manifest ===")
seen = {}
for name in sorted(os.listdir(RUNS)):
    record = load(os.path.join(RUNS, name, "eval_last.json"))
    if not record:
        continue
    config = record.get("config", record)
    manifest = config.get("aug_manifest") or config.get("manifest")
    if manifest:
        seen.setdefault(str(manifest), []).append(name)
for manifest, names in sorted(seen.items()):
    print(f"  {manifest}")
    print(f"      used by {len(names)} runs, e.g. {names[:3]}")
