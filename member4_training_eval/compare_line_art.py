"""The line-art family: the method, its two controls, and where it wins.

The line-art proposal is a different hypothesis from the AdaIN style
transfer: it says that a *shape* bias, rather than a texture one, is
what generalisation needs.  The family has one method and two controls,
and the controls are matched on ink:

``line_attn``           the attention gate
``line_attn_vote``      the attention gate plus the superpixel soft vote
``line_plain_total``    uniform strokes, total ink matched to the gate
``line_plain_foreground`` uniform strokes, foreground ink matched to the gate

Two questions are answered here.

The first is whether the gate does anything.  The controls are the
answer: a control that matches the ink but not the *distribution* of it
will land on the same accuracy as the method, and the gate is then
just a different way of drawing the same information.

The second is the falsifiable prediction.  If the line drawing really
does establish a shape bias, the gain has to show up in Sketch, because
Sketch is the domain that keeps shape and drops everything else.  A
gain spread evenly over the three target domains is not a shape bias.
The script therefore prints the per-domain decomposition, and it says
plainly whether the prediction held.

Usage:
    python compare_line_art.py
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

TARGETS = ("art_painting_test", "cartoon_test", "sketch_test")

#: The groups of the line-art family, with the pattern that finds them.
FAMILY = {
    "original": r"original_s\d+",
    "ours_line_attn": r"ours_line_attn_s\d+",
    "ours_line_vote": r"ours_line_vote_s\d+",
    "ours_line_plain_total": r"ours_line_plain_total_s\d+",
    "ours_line_plain_fg": r"ours_line_plain_fg_s\d+",
}

#: The name each group is reported under.
PRETTY = {
    "original": "original",
    "ours_line_attn": "line_attn (gate)",
    "ours_line_vote": "line_attn_vote (soft vote)",
    "ours_line_plain_total": "line_plain_total (control)",
    "ours_line_plain_fg": "line_plain_foreground (control)",
}


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


def values(tags: list[str], split: str) -> list[float]:
    out = []
    for tag in tags:
        record = read(tag)
        if record and split in record:
            out.append(record[split])
    return out


def cross(tags: list[str]) -> list[float]:
    out = []
    for tag in tags:
        record = read(tag)
        if not record:
            continue
        parts = [record[split] for split in TARGETS if split in record]
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
    return {"mean_diff": mean, "t": t, "p": p,
            "cohens_d": mean / spread if spread > 1e-12 else 0.0,
            "n": n, "worse": int((diff < 0).sum())}


def main() -> int:
    print("=" * 88)
    print("THE LINE-ART FAMILY")
    print("=" * 88)

    table = {}
    for name, pattern in FAMILY.items():
        tags = tags_for(pattern)
        if tags:
            table[name] = tags

    if "ours_line_attn" not in table:
        print()
        print("  the line-art family has no results yet")
        return 1

    print()
    print(f"  {'group':<30}{'cross':>9}{'std':>8}{'photo val':>11}"
          f"{'art':>8}{'cartoon':>9}{'sketch':>8}  seeds")
    for name in FAMILY:
        if name not in table:
            continue
        tags = table[name]
        c = cross(tags)
        photo = values(tags, "photo_val")
        art = values(tags, "art_painting_test")
        cartoon = values(tags, "cartoon_test")
        sketch = values(tags, "sketch_test")
        print(f"  {PRETTY[name]:<30}{np.mean(c)*100:>8.2f}%"
              f"{np.std(c, ddof=1)*100:>7.2f}%"
              f"{np.mean(photo)*100:>10.2f}%{np.mean(art)*100:>7.2f}%"
              f"{np.mean(cartoon)*100:>8.2f}%{np.mean(sketch)*100:>7.2f}%"
              f"  {len(c)}")

    print()
    print("=" * 88)
    print("DOES THE GATE DO ANYTHING?")
    print("=" * 88)
    print()
    comparisons = {}
    method = cross(table["ours_line_attn"])
    for control in ("ours_line_plain_total", "ours_line_plain_fg",
                    "ours_line_vote", "original"):
        if control not in table:
            continue
        other = cross(table[control])
        n = min(len(method), len(other))
        stats = paired(method[:n], other[:n])
        comparisons[control] = stats
        verdict = "better" if stats["mean_diff"] > 0 else "worse"
        print(f"  line_attn vs {PRETTY[control]:<30} ({verdict})")
        print(f"    mean difference {stats['mean_diff']*100:+.2f} points"
              f"   t({stats['n']-1}) = {stats['t']:.3f}"
              f"   p = {stats['p']:.4f}")
        print(f"    Cohen's d = {stats['cohens_d']:+.3f}"
              f"   worse on {stats['worse']}/{stats['n']} seeds")
        print()

    print("=" * 88)
    print("THE FALSIFIABLE PREDICTION: THE GAIN SHOULD BE IN SKETCH")
    print("=" * 88)
    print()
    if "ours_line_plain_total" not in table:
        print("  the matched control has not been trained yet")
        return 0
    control_tags = table["ours_line_plain_total"]
    method_tags = table["ours_line_attn"]
    print(f"  {'domain':<24}{'gate':>10}{'control':>10}{'difference':>13}")
    print()
    differences = {}
    for split in list(TARGETS) + ["photo_val"]:
        method_values = values(method_tags, split)
        control_values = values(control_tags, split)
        if not method_values or not control_values:
            continue
        n = min(len(method_values), len(control_values))
        difference = (np.mean(method_values[:n]) -
                      np.mean(control_values[:n])) * 100
        differences[split] = difference
        print(f"  {split:<24}{np.mean(method_values[:n])*100:>9.2f}%"
              f"{np.mean(control_values[:n])*100:>9.2f}%"
              f"{difference:>+12.2f}")
    print()

    sketch = differences.get("sketch_test")
    others = [differences.get(s) for s in ("art_painting_test",
                                           "cartoon_test")]
    others = [d for d in others if d is not None]
    if sketch is not None and others:
        if sketch > max(others) and sketch > 0:
            print("  The prediction holds: the largest gain is in Sketch,")
            print("  which is the domain that keeps the shape and drops the")
            print("  texture.  That is what a shape bias looks like.")
        elif sketch > 0:
            print("  The gain in Sketch is positive but not the largest of")
            print("  the three target domains, so the prediction only")
            print("  partly holds.")
        else:
            print("  The prediction does NOT hold: Sketch did not gain.")
            print("  A gain spread over the domains is not a shape bias,")
            print("  and the line-art hypothesis is not supported by this")
            print("  run.")

    payload = {
        "groups": {name: list(table[name]) for name in table},
        "paired": {name: stats for name, stats in comparisons.items()},
        "per_domain_difference": differences,
    }
    out = RUNS / "summary_line_art_comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
