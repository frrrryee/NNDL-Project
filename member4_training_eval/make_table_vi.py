"""Table VI: the decomposition of where the gain comes from.

The table is generated from the recorded summaries, so that every number in
the paper can be recomputed from ``runs/``.  It is the most important table
of the report: it splits the headline difference into the four factors that
could explain it.

The rows are:

  pure data volume        +Duplicate - Original
                          (twice the training rows, the same distinct images,
                           no appearance change at all)

  generic appearance      +NonStyle - +Duplicate
                          (a non-stylistic transform on top of the copy:
                           colour jitter, greyscale, blur)

  style itself            +AdaIN - +NonStyle
                          (style on top of a generic appearance change).
                          This row contains two factors at once -- "styling
                          vs. a generic transform" and "a particular family
                          of transforms vs. a random one" -- so it is an
                          *upper bound* on the causal attribution of style,
                          not a strict attribution.  The table note says so.

  spatial modulation      +Ours - +AdaIN (strength-matched)
                          (the effect of the spatial distribution of the
                           style at an equal mean strength)

Usage:
    python make_table_vi.py --out ../report/sections/member4_tablevi.tex
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")

LABELS = {
    "original": "Original",
    "duplicate": "+Duplicate",
    "nonstyle": "+NonStyle",
    "adain": "+AdaIN",
    "adain_matched": "+AdaIN (matched)",
    "ours": "+Ours",
    "ours_equal": "+Ours (equal str.)",
}

#: The four rows of the table.  Each entry is
#: (row label, group a, group b, the note of the row).
ROWS = (
    ("Pure data volume", "original", "duplicate",
     "twice the training rows, the same distinct images, no appearance change"),
    ("Generic appearance diversity", "duplicate", "nonstyle",
     "a non-stylistic transform (colour jitter / greyscale / blur) on top of "
     "the duplicate"),
    ("Style itself (upper bound)", "nonstyle", "adain",
     "style on top of a generic appearance change"),
    ("Spatial modulation", "adain_matched", "ours",
     "the effect of the spatial distribution at an equal mean style strength"),
)


def load_cross(runs_root: Path, group: str):
    path = runs_root / f"summary_{group}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = []
    for split in CROSS:
        entry = payload.get(split)
        if not entry or not entry.get("per_seed"):
            return None
        series.append(entry["per_seed"])
    seeds = min(len(s) for s in series)
    return [float(np.mean([s[i] for s in series])) for i in range(seeds)]


def paired(a: list, b: list):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    diff = y - x
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t, p = stats.ttest_rel(y, x)
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    else:
        t, p, half = (0.0, 1.0), 1.0, 0.0
    return mean, mean - half, mean + half, float(t), float(p), n


def common_seeds(runs_root: Path, groups: tuple) -> dict:
    """The per-seed cross-domain averages, restricted to the shared seeds.

    While the seed extension is still running, some groups have five seeds
    and others three.  A paired comparison is only valid over the seeds the
    two groups have in common, so the series are truncated to the common
    prefix.
    """
    raw = {g: load_cross(runs_root, g) for g in groups}
    raw = {g: v for g, v in raw.items() if v is not None}
    if not raw:
        return {}
    length = min(len(v) for v in raw.values())
    return {g: v[:length] for g, v in raw.items()}


def build(runs_root: Path) -> tuple[list, list, int]:
    """The rows of the table, the absolute levels, and the seed count."""
    rows = []
    all_groups = tuple(sorted({g for _, a, b, _ in ROWS for g in (a, b)}))
    series = common_seeds(runs_root, all_groups)
    for label, a, b, note in ROWS:
        if a not in series or b not in series:
            continue
        mean, lo, hi, t, p, n = paired(series[a], series[b])
        rows.append({"label": label, "a": a, "b": b, "note": note,
                     "diff": mean, "lo": lo, "hi": hi, "t": t, "p": p, "n": n})
    levels = {}
    for group in ("original", "duplicate", "nonstyle", "adain", "adain_matched", "ours"):
        if group in series:
            levels[group] = float(np.mean(series[group]))
    n_seeds = max((r["n"] for r in rows), default=0)
    return rows, levels, n_seeds


def latex(rows: list, levels: dict, n_seeds: int) -> str:
    lines = [
        "% Generated by member4_training_eval/make_table_vi.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Decomposition of the cross-domain gain.  The headline "
        "difference of the style augmentation is split into the four factors "
        "that could explain it.  All numbers are percentage points of the "
        "cross-domain average (Art Painting / Cartoon / Sketch); the "
        "comparison is paired over the shared seeds.}",
        "\\label{tab:decomposition}",
        "\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Source of the gain & Difference & 95\\% CI & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        flag = "$^{\\ast}$" if row["p"] < 0.05 else ""
        lines.append(
            f"{row['label']} & {row['diff']*100:+.2f}{flag} & "
            f"[{row['lo']*100:+.2f}, {row['hi']*100:+.2f}] & "
            f"{row['p']:.3f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def notes(rows: list, levels: dict, n_seeds: int) -> str:
    """The LaTeX table note, with the absolute levels."""
    absolute = ", ".join(
        f"{LABELS[g]} {levels[g]*100:.2f}" for g in
        ("original", "duplicate", "nonstyle", "adain", "adain_matched", "ours")
        if g in levels)
    style = next((r for r in rows if r["label"].startswith("Style itself")), None)
    text = [
        "% Generated by member4_training_eval/make_table_vi.py -- do not edit by hand.",
        f"% Absolute cross-domain averages over {n_seeds} seeds: {absolute}",
        "% Note on the third row: it contains two factors at once -- 'styling",
        "% versus a generic transform' and 'a particular family of transforms",
        "% versus a random one' -- so it is an upper bound on the causal",
        "% attribution of style, not a strict attribution.",
    ]
    if style is not None:
        text.append(f"% third row: {style['diff']*100:+.2f} pt, p = {style['p']:.4f}")
    return "\n".join(text) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=config.PROJECT_ROOT / "report" / "sections" / "member4_tablevi.tex")
    parser.add_argument("--notes-out", type=Path,
                        default=config.PROJECT_ROOT / "report" / "sections" / "member4_tablevi_notes.tex")
    args = parser.parse_args()

    rows, levels, n_seeds = build(args.runs_root)
    if not rows:
        print("no comparable groups found; run the experiments first")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex(rows, levels, n_seeds), encoding="utf-8")
    args.notes_out.write_text(notes(rows, levels, n_seeds), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.notes_out}")


if __name__ == "__main__":
    main()
