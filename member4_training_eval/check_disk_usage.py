"""How much disk does each thing take, and what can be released?

The disk is close to full while several training jobs are running.  The
checkpoints of finished runs are the largest thing under our control:
they are kept only so that an evaluation can be repeated, and a run
that has already been evaluated does not need its weights any more.

This reports the sizes and names the candidates for release.  It never
deletes anything by itself.
"""
import os
import shutil

RUNS = r"C:\Users\qth02\Desktop\NNDL-Project\runs"
DATA = r"C:\Users\qth02\Desktop\nndl_pacs\data"
SP = os.path.join(DATA, "aug_sp")


def dir_size(root):
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def gb(value):
    return value / (1024 ** 3)


print("=== free space ===")
usage = shutil.disk_usage("C:\\")
print(f"  free {gb(usage.free):.1f} GB of {gb(usage.total):.1f} GB")
print()

print("=== checkpoints under runs/ ===")
total = 0
keepable = []
for name in sorted(os.listdir(RUNS)):
    path = os.path.join(RUNS, name)
    if not os.path.isdir(path):
        continue
    checkpoints = []
    for base, _dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pt"):
                checkpoints.append(os.path.join(base, file))
    if not checkpoints:
        continue
    size = sum(os.path.getsize(p) for p in checkpoints)
    total += size
    evaluated = os.path.exists(os.path.join(path, "eval_last.json"))
    keepable.append((name, size, evaluated, len(checkpoints)))

keepable.sort(key=lambda item: -item[1])
for name, size, evaluated, count in keepable[:15]:
    state = "evaluated" if evaluated else "NOT evaluated"
    print(f"  {gb(size):6.3f} GB  {count:>2} ckpt  {state:15} {name}")
print(f"  ... total over all runs: {gb(total):.2f} GB")

print()
print("=== the data directories ===")
for name in sorted(os.listdir(SP)):
    path = os.path.join(SP, name)
    if os.path.isdir(path):
        print(f"  {gb(dir_size(path)):6.2f} GB  aug_sp/{name}")

print()
print("=== what can be released ===")
released = sum(size for _name, size, evaluated, _c in keepable if evaluated)
print(f"  checkpoints of already-evaluated runs: {gb(released):.2f} GB")
unevaluated = sum(size for _name, size, evaluated, _c in keepable
                   if not evaluated)
print(f"  checkpoints of runs not yet evaluated: {gb(unevaluated):.2f} GB "
      f"(must be kept)")
print()
print("  The previews directory holds small samples that were kept for the")
print("  determinism check; they are disposable.")
previews = os.path.join(SP, "previews")
print(f"  aug_sp/previews: {gb(dir_size(previews)):.2f} GB")
