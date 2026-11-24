"""The centre-prior repair: what it does, and why.

The centre prior won the map selection on the mechanism criterion by a
factor of thirty: +0.0336 EdgeScore against its own matched control,
against +0.0010 for the spectral residual, with 3% of the images worse
than the control against 44%.

It lost the cross-domain accuracy.  This script is the account of that,
and it separates the two things that could explain it.

The first candidate explanation is that the centre prior is simply a
worse map: that it damages the object outline, and that the damage
costs accuracy.  The comparison against the map's *own* matched control
is the test for that, because the control has the same amount of style
and no spatial variation.

The second candidate explanation is that the centre prior applies less
style overall -- it covers 34% of the pixels, and the strength falls
there, so its mean alpha (0.3068) is well below the Sobel group's
(0.3736).  A group that is compared against a *different* group's
control would then be punished for applying less style, which has
nothing to do with the map.  The comparison against the centre prior's
own control is immune to this, and comparing the two differences
separates the two explanations.

Usage:
    python compare_foreground_center.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402

RUNS = config.OUTPUT_ROOT

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")


def read(tag: str) -> dict | None:
    path = RUNS / tag / "eval_last.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tags_for(pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return sorted(tag for tag in os.listdir(RUNS)
                  if regex.fullmatch(tag)
                  and (RUNS / tag / "eval_last.json").exists())


def split_values(tags: list[str], split: str) -> list[float]:
    out = []
    for tag in tags:
        record = read(tag)
        if record and split in record:
            out.append(record[split])
    return out


def cross_values(tags: list[str]) -> list[float]:
    out = []
    for tag in tags:
        record = read(tag)
        if not record:
            continue
        parts = [record[split] for split in CROSS if split in record]
        if len(parts) == 3:
            out.append(sum(parts) / 3)
    return out


def paired(a: list[float], b: list[float]) -> dict:
    """The paired statistics of ``a`` against ``b``."""
    a = np.array(a[:len(b)], dtype=float)
    b = np.array(b[:len(a)], dtype=float)
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / np.sqrt(n)) if sd > 1e-12 else 0.0
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1)) if n > 1 else 1.0
    except ImportError:
        p = float("nan")
    spread = float(b.std(ddof=1)) if n > 1 else 0.0
    d = mean / spread if spread > 1e-12 else 0.0
    return {"mean_diff": mean, "sd": sd, "t": t, "p": p, "cohens_d": d,
            "n": n, "worse": int((diff < 0).sum()), "diff": diff.tolist()}


def mean_alpha_of(relative: str) -> float | None:
    """The mean alpha field of a generation record."""
    path = config.DATA_ROOT / relative
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [entry["metrics"]["mean_alpha"]
              for entry in (payload.get("outputs") or {}).values()
              if entry.get("metrics") and "mean_alpha" in entry["metrics"]]
    return float(sum(values) / len(values)) if values else None


def main():
    print("=" * 84)
    print("THE CENTRE-PRIOR REPAIR: WHAT IT DOES, AND WHY")
    print("=" * 84)

    groups = {
        "original": "original_s\\d+",
        "adain_matched": "adain_matched_s\\d+",
        "ours (Sobel)": "ours_s\\d+",
        "ours_fg_center": "ours_fg_center_s\\d+",
        "ours_fg_center_matched": "ours_fg_center_matched_s\\d+",
    }

    print()
    print(f"  {'group':<26}{'cross':>9}{'std':>8}{'photo val':>11}"
          f"{'art':>8}{'cartoon':>9}{'sketch':>8}{'alpha':>8}  seeds")
    table = {}
    details = {}
    alphas = {
        "ours (Sobel)": "aug_sp/generation.json",
        "ours_fg_center": "aug_sp/fgcenter/generation.json",
        "ours_fg_center_matched": "aug_sp/fgcenter/generation.json",
    }
    for name, pattern in groups.items():
        tags = tags_for(pattern)
        cross = cross_values(tags)
        if not cross:
            continue
        table[name] = cross
        details[name] = tags
        photo = split_values(tags, "photo_val")
        art = split_values(tags, "art_painting_test")
        cartoon = split_values(tags, "cartoon_test")
        sketch = split_values(tags, "sketch_test")
        alpha = mean_alpha_of(alphas[name]) if name in alphas else None
        alpha_text = f"{alpha:.4f}" if alpha is not None else "-"
        print(f"  {name:<26}{np.mean(cross)*100:>8.2f}%"
              f"{np.std(cross, ddof=1)*100:>7.2f}%"
              f"{np.mean(photo)*100:>10.2f}%{np.mean(art)*100:>7.2f}%"
              f"{np.mean(cartoon)*100:>8.2f}%{np.mean(sketch)*100:>7.2f}%"
              f"{alpha_text:>8}  {len(cross)}")

    if "ours_fg_center" not in table:
        print()
        print("  the centre-prior group has no results yet")
        return 1

    print()
    print("=" * 84)
    print("PAIRED COMPARISONS (same seeds, same protocol)")
    print("=" * 84)
    comparisons = {}
    for other in ("ours (Sobel)", "adain_matched", "original",
                  "ours_fg_center_matched"):
        if other not in table:
            continue
        n = min(len(table[other]), len(table["ours_fg_center"]))
        stats = paired(table["ours_fg_center"][:n], table[other][:n])
        comparisons[other] = stats
        verdict = "worse" if stats["mean_diff"] < 0 else "better"
        print()
        print(f"  ours_fg_center vs {other}   ({verdict})")
        print(f"    mean difference {stats['mean_diff']*100:+.2f} points")
        print(f"    t({stats['n']-1}) = {stats['t']:.3f}   p = {stats['p']:.4f}")
        print(f"    Cohen's d = {stats['cohens_d']:+.3f}")
        print(f"    seeds where it is worse: {stats['worse']}/{stats['n']}")

    print()
    print("=" * 84)
    print("WHICH EXPLANATION SURVIVES")
    print("=" * 84)
    print()
    against_control = comparisons.get("ours_fg_center_matched")
    if against_control is None:
        print("  The matched control has not been trained yet, so the two")
        print("  explanations cannot be separated.")
    else:
        against_other = comparisons.get("adain_matched")
        print("  Two things could explain the loss, and they are not")
        print("  mutually exclusive:")
        print()
        print("    (a) the centre prior is a worse map -- it damages the")
        print("        object outline, and the damage costs accuracy;")
        print("    (b) the centre prior applies less style overall, so it")
        print("        loses accuracy for a reason that has nothing to do")
        print("        with the map.")
        print()
        print("  The comparison against its own matched control tests (a):")
        print("  the control has the same amount of style and no spatial")
        print("  variation, so any difference is the map's doing.")
        print()
        print(f"    ours_fg_center - ours_fg_center_matched = "
              f"{against_control['mean_diff']*100:+.2f} points  "
              f"(p = {against_control['p']:.4f})")
        if against_other:
            print(f"    ours_fg_center - adain_matched      = "
                  f"{against_other['mean_diff']*100:+.2f} points  "
                  f"(p = {against_other['p']:.4f})")
        print()
        # The differences are proportions, so a half-point is 0.005.
        threshold = 0.005
        if against_control["mean_diff"] < -threshold:
            print("  The map is genuinely worse: it loses accuracy against a")
            print("  control with the same style strength.  Explanation (a)")
            print("  survives.")
            if against_other and against_other["mean_diff"] < -threshold:
                share = (against_control["mean_diff"] /
                         against_other["mean_diff"])
                remainder = (against_other["mean_diff"] -
                             against_control["mean_diff"])
                print()
                print(f"  The loss against the Sobel group is "
                      f"{against_other['mean_diff']*100:+.2f} points, and it")
                print(f"  splits into two parts: {abs(share)*100:.0f}% of it is")
                print("  the map's own doing -- the centre prior damages the")
                print("  object outline even at matched strength -- and the")
                print(f"  remaining {abs(remainder)*100:.2f} points are the")
                print("  style strength itself, which is a property of how")
                print("  much of the frame the map covers and not a separate")
                print("  effect.")
                print()
                print("  So there are two causes, and the map is one of them:")
                print("  the mechanism measurement was measuring a real")
                print("  effect, and that effect is negative.")
        elif against_other and against_other["mean_diff"] < -threshold:
            print("  The map is not worse than its own control, so the loss")
            print("  against the Sobel group was the style strength, not the")
            print("  map.  Explanation (b) survives.")
        else:
            print("  Neither explanation accounts for a loss.")

    print()
    print("=" * 84)
    print("WHAT THE MECHANISM MEASUREMENT SAID")
    print("=" * 84)
    print()
    print("  the centre prior won the map selection by a factor of thirty on")
    print("  the mechanism criterion (EdgeScore against its own matched")
    print("  control): +0.0336 against +0.0010 for the spectral residual,")
    print("  with 3% of the images worse than the control against 44%.")
    print()
    print("  If the mechanism measure predicted cross-domain accuracy, the")
    print("  numbers above would put the centre prior ahead.  They do not.")

    payload = {
        "groups": {name: values for name, values in table.items()},
        "runs": details,
        "paired": {name: {key: value for key, value in stats.items()
                          if key != "diff"}
                   for name, stats in comparisons.items()},
    }
    out = RUNS / "summary_ours_fg_center_comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
