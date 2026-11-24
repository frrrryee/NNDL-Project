"""Evaluate a trained checkpoint on all six splits and write results JSON.

Usage:
    python evaluate.py --run runs/original_s0 [--splits photo_val photo_test ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
import data
from model import build_model, set_seed

ALL_SPLITS = ("photo_val", "photo_test", "art_painting_test",
              "cartoon_test", "sketch_test")


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_per_class(model, loader):
    """Per-class accuracy, so that gains can be attributed to classes.

    Structure preservation is supposed to help where the class is defined by
    a thin, high-frequency cue (the neck of a guitar, the outline of a
    house) and to matter less where the class is defined by coarse texture.
    A per-class breakdown therefore tests the *mechanism* of the method and
    not only its average effect.
    """
    model.eval()
    correct = [0] * len(config.CLASSES)
    total = [0] * len(config.CLASSES)
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        pred = model(images).argmax(dim=1)
        for index in range(len(config.CLASSES)):
            mask = labels == index
            if not bool(mask.any()):
                continue
            total[index] += int(mask.sum())
            correct[index] += int((pred[mask] == labels[mask]).sum())
    return {config.CLASSES[i]: (correct[i] / total[i] if total[i] else None)
            for i in range(len(config.CLASSES))}, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory with best.pt")
    parser.add_argument("--splits", nargs="+", default=list(ALL_SPLITS))
    parser.add_argument("--checkpoint", default=f"{config.CHECKPOINT}.pt")
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--group", default=None,
                        help="the group this run belongs to; selects the right "
                             "architecture and fingerprint.  It is read from the "
                             "run's train_summary.json when not given.")
    args = parser.parse_args()

    group = args.group
    manifest = None
    # The manifest is read from the run's own summary whatever the group
    # is: a repaired modulation's fingerprint depends on the map, and the
    # map is recorded with the manifest the run actually used.  Taking the
    # group from the command line must not lose that.
    summary_file = args.run / "train_summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if group is None:
            group = summary.get("group")
        recorded_manifest = summary.get("aug_manifest")
        if recorded_manifest:
            manifest = Path(recorded_manifest)
    if group is None:
        group = "original"

    set_seed(0)
    # The MixStyle layer is an identity at evaluation time, so the
    # architecture has to match the one the run was trained with: the
    # state dict of a MixStyle run has the extra layer in it.
    use_mixstyle = group == config.MIXSTYLE_GROUP
    model = build_model(mixstyle=use_mixstyle,
                        mixstyle_p=config.MIXSTYLE_P,
                        mixstyle_alpha=config.MIXSTYLE_ALPHA)
    state = torch.load(args.run / args.checkpoint, map_location=config.DEVICE,
                       weights_only=True)
    model.load_state_dict(state)

    results = {"run": str(args.run), "checkpoint": args.checkpoint,
               "group": group,
               "config_hash": config.config_hash_for_group(group, manifest)}
    per_class = {}
    for split in args.splits:
        loader = DataLoader(data.load_eval_dataset(split), batch_size=128,
                            shuffle=False, num_workers=2, pin_memory=True)
        acc = evaluate_model(model, loader)
        results[split] = acc
        print(f"{split:20s} {acc*100:6.2f}%")
        if args.per_class:
            accs, counts = evaluate_per_class(model, loader)
            per_class[split] = {"accuracy": accs, "count": counts}
            detail = "  ".join(f"{name} {value*100:5.1f}%" for name, value in accs.items()
                               if value is not None)
            print(f"{'  per-class':20s} {detail}")

    cross = [results.get(s) for s in ("art_painting_test", "cartoon_test", "sketch_test")]
    if all(a is not None for a in cross):
        results["cross_domain_avg"] = sum(cross) / 3
        print(f"{'cross-domain avg':20s} {results['cross_domain_avg']*100:6.2f}%")

    if per_class:
        results["per_class"] = per_class

    out = Path(args.run) / f"eval_{Path(args.checkpoint).stem}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
