"""Is the collapse caused by images whose subject cannot be seen?

The claim to test: a good part of the cross-domain failures are images
in which the object is not really there -- it fills the frame, its
outline is unclear, or the background is busy -- so no method can
recognise them.

That claim is testable without any training.  If it is true, the images
that the model gets wrong should be the images whose objectness map
says there is no object, or says the object is a small part of the
frame.  If it is false, the wrongly classified images should be as
recognisable as the correctly classified ones.

The test uses the label-free objectness map of the ImageNet
pretrained network that the project already owns.  For every image of a
split it measures:

* the *coverage*: the fraction of the frame the object occupies;
* the *peak*: the highest objectness value, which says how strongly
  the network believes an object is present;
* the *contrast*: how much the object stands out from its
  surroundings, which says whether the background is busy.

The images are then split into correctly and incorrectly classified
ones, and the three measures are compared between the two groups.

The outcome matters for the direction of the work.  If the wrong images
are the unrecognisable ones, the ceiling of the method is set by the
data and the best thing to do is to stop.  If they are as
recognisable as the right ones, the failure is the method's, and
there is room to work.

Usage:
    python diagnose_unrecognisable.py --run runs/ours_fg_center_s0
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

from member3_structure_preserving.attention_map import (  # noqa: E402
    AttentionConfig, ObjectnessAttention)


@torch.no_grad()
def image_measures(model, loader, attention):
    """The objectness measures of every image, and whether it was right."""
    records = []
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct = (pred == labels).cpu().numpy()
        labels = labels.cpu().numpy()

        # The objectness map is computed on the *unnormalised* image.  The
        # evaluation transform divides by the ImageNet statistics, which
        # puts the values outside [0,1], and the map's own input check
        # refuses that.  Undoing the normalisation gives back the
        # picture the classifier actually sees.
        mean = torch.tensor(config.IMAGENET_MEAN, device=images.device
                            ).view(1, 3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD, device=images.device
                           ).view(1, 3, 1, 1)
        raw = (images * std + mean).clamp(0, 1)
        maps = attention.maps(raw)
        for index in range(images.shape[0]):
            m = maps[index, 0].cpu().numpy()
            # The raw activation map is not in [0,1]; it is normalised
            # per image so that "coverage" means a fraction of the
            # frame and not an artefact of the activation scale.
            m = (m - m.min()) / max(float(m.max() - m.min()), 1e-6)
            peak = float(m.max())
            coverage = float((m > 0.5).mean())
            # The contrast: how far the peak stands above the median,
            # which says whether the object stands out from the frame.
            contrast = float(peak - np.median(m))
            records.append({"correct": bool(correct[index]),
                            "label": int(labels[index]),
                            "peak": peak, "coverage": coverage,
                            "contrast": contrast})
    return records


def report(records, split):
    right = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]
    print()
    print("=" * 92)
    print(f"DOES THE MODEL FAIL ON THE IMAGES THAT HAVE NO CLEAR SUBJECT? -- {split}")
    print("=" * 92)
    print()
    print(f"  {'measure':<26}{'correct':>12}{'wrong':>12}{'difference':>14}")
    print("  " + "-" * 62)
    for key, label in (("peak", "objectness peak"),
                       ("coverage", "subject covers frame"),
                       ("contrast", "stands out from bg")):
        a = np.mean([r[key] for r in right]) if right else float("nan")
        b = np.mean([r[key] for r in wrong]) if wrong else float("nan")
        print(f"  {label:<26}{a:>12.3f}{b:>12.3f}{b-a:>+14.3f}")

    print()
    print(f"  {len(right)} correct, {len(wrong)} wrong")

    # The decisive comparison: the worst a fifth of the images, judged
    # by each measure, against the overall accuracy.
    print()
    print("  accuracy of the least recognisable fifth, by each measure:")
    print()
    n = len(records)
    for key, label in (("peak", "objectness peak"),
                       ("coverage", "subject covers frame"),
                       ("contrast", "stands out from bg")):
        ordered = sorted(records, key=lambda r: r[key])
        fifth = ordered[:n // 5]
        rest = ordered[n // 5:]
        acc_a = np.mean([r["correct"] for r in fifth])
        acc_b = np.mean([r["correct"] for r in rest])
        print(f"    {label:<26} worst fifth {acc_a*100:5.1f}%   "
              f"rest {acc_b*100:5.1f}%   ({acc_a-acc_b:+.1%})")

    print()
    print("  If the wrongly classified images were the unrecognisable")
    print("  ones, the least recognisable fifth would carry almost all")
    print("  of the errors and the difference would be large.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--splits", nargs="+",
                        default=["art_painting_test", "cartoon_test", "sketch_test"])
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

    attention = ObjectnessAttention(AttentionConfig(), device=config.DEVICE)

    for split in args.splits:
        loader = DataLoader(data.load_eval_dataset(split), batch_size=16,
                            shuffle=False, num_workers=2, pin_memory=True)
        records = image_measures(model, loader, attention)
        report(records, split)
        del loader
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
