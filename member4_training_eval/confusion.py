"""The confusion matrix of a trained run, per target domain.

The per-class accuracies say that a class fails.  They do not say what
it fails *into*, and the two are very different problems:

* a class that fails into one other class is a confusion between two
  things that share a shape;
* a class that fails uniformly into everything is a class the model
  has no representation of at all.

The second one is what the per-class table looks like when a class
collapses, and it is the one that points at the training data rather
than at the method.

This script runs the checkpoint over the target splits and prints the
confusion matrix, so that a collapsed class can be told apart from a
confused one.

Usage:
    python confusion.py --run runs/adain_a1p0_b0p7_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
import data  # noqa: E402
from model import build_model, set_seed  # noqa: E402

TARGETS = ("art_painting_test", "cartoon_test", "sketch_test")


@torch.no_grad()
def confusion(model, loader):
    """The confusion matrix, indexed [true][predicted]."""
    n = len(config.CLASSES)
    matrix = np.zeros((n, n), dtype=np.int64)
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        pred = model(images).argmax(dim=1)
        for true, guess in zip(labels.cpu().numpy(), pred.cpu().numpy()):
            matrix[true, guess] += 1
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--split", default="sketch_test")
    args = parser.parse_args()

    summary_file = args.run / "train_summary.json"
    group = "original"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        group = summary.get("group") or group

    set_seed(0)
    model = build_model(mixstyle=group == config.MIXSTYLE_GROUP,
                        mixstyle_p=config.MIXSTYLE_P,
                        mixstyle_alpha=config.MIXSTYLE_ALPHA)
    state = torch.load(args.run / args.checkpoint, map_location=config.DEVICE,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(data.load_eval_dataset(args.split), batch_size=128,
                        shuffle=False, num_workers=2, pin_memory=True)
    matrix = confusion(model, loader)

    names = list(config.CLASSES)
    print()
    print("=" * 92)
    print(f"CONFUSION MATRIX -- {args.split}   ({args.run.name})")
    print("=" * 92)
    print()
    print("  rows = true class, columns = what the model said")
    print()
    print("  " + " " * 12 + "".join(f"{n[:7]:>9}" for n in names))
    print("  " + "-" * (12 + 9 * len(names)))
    for index, name in enumerate(names):
        row = matrix[index]
        total = row.sum()
        cells = "".join(
            f"{row[j]:>9}" if row[j] else f"{'.':>9}"
            for j in range(len(names)))
        accuracy = row[index] / total if total else 0.0
        print(f"  {name:<12}{cells}   {accuracy*100:5.1f}%")
    print("  " + "-" * (12 + 9 * len(names)))
    print(f"  {'predicted':<12}" + "".join(
        f"{matrix[:, j].sum():>9}" for j in range(len(names))))
    print()

    print("  where each class went:")
    for index, name in enumerate(names):
        row = matrix[index].copy()
        total = row.sum()
        if not total:
            continue
        row[index] = 0
        order = np.argsort(row)[::-1][:2]
        shares = ", ".join(f"{names[j]} {row[j]/total*100:.0f}%"
                           for j in order if row[j])
        print(f"    {name:<12} -> {shares or '(always correct)'}")

    print()
    print("  A class that collapses into every other class is not being")
    print("  confused with anything in particular; it has no")
    print("  representation.  A class that collapses into one other class")
    print("  is being confused with it, which is a different problem.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
