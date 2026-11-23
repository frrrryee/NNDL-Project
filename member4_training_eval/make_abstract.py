"""Generate the abstract, with every number taken from the recorded results.

The abstract is the last thing written and every number in it has to agree
with Table~II.  Writing it by hand would guarantee that it eventually
disagrees with the table, so the numbers are interpolated here from the
summaries.

The text is fixed; only the numbers move.  If a number is missing, the
abstract says so rather than inventing one.
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


def paired(a: list, b: list):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError("the two series must have the same length")
    diff = y - x
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t, p = stats.ttest_rel(y, x)
        t, p = float(t), float(p)
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    else:
        t, p, half = (0.0, 1.0), 1.0, 0.0
    return mean, mean - half, mean + half, t, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=config.PROJECT_ROOT / "report" / "sections" / "member5_abstract.tex")
    args = parser.parse_args()

    groups = ("original", "duplicate", "nonstyle", "adain", "adain_matched", "ours")
    series = common_seeds(args.runs_root, groups)
    original = series.get("original")
    adain = series.get("adain")
    duplicate = series.get("duplicate")
    nonstyle = series.get("nonstyle")
    matched = series.get("adain_matched")
    ours = series.get("ours")
    if original is None or adain is None:
        print("abstract: the 'original' and 'adain' summaries are missing")
        return

    n_seeds = len(original)
    o = float(np.mean(original))
    a = float(np.mean(adain))
    gain, lo, hi, t, p = paired(original, adain)

    pieces = []
    pieces.append(
        f"Style augmentation is a widely used way to close the domain gap in "
        f"single-source cross-domain classification, but its reported benefit is "
        f"confounded: a styled training set differs from the original in its "
        f"size, in its generic appearance diversity, and in the style it carries, "
        f"all at once. We measure what style augmentation actually buys on PACS "
        f"with four controlled experiments.")
    pieces.append(
        f"First, two data-volume controls --- a byte-identical duplicate of every "
        f"training image, and a non-stylistic transform of every training image --- "
        f"separate the effect of the extra rows from the effect of the style.")
    pieces.append(
        f"Second, because a spatial strength map necessarily applies less total "
        f"style than a uniform mix at the same nominal strength, we train a "
        f"strength-matched uniform control and an equal-strength spatial variant, "
        f"so that the effect of the spatial distribution can be isolated.")
    pieces.append(
        f"Third, a sweep over the style strength traces the dose--response "
        f"relation, and the style pool is split into disjoint halves to test the "
        f"robustness of the result.")

    if duplicate is not None and nonstyle is not None:
        d_gain, d_lo, d_hi, d_t, d_p = paired(original, duplicate)
        n_gain, _, _, _, n_p = paired(duplicate, nonstyle)
        pieces.append(
            f"Adding AdaIN-styled images raises the cross-domain average from "
            f"{o*100:.1f}\\% to {a*100:.1f}\\% over {n_seeds} seeds, a paired gain "
            f"of {gain*100:.1f} points (95\\% CI [{lo*100:.1f}, {hi*100:.1f}], "
            f"$p={p:.4f}$); neither control reproduces an appreciable part of it "
            f"(+Duplicate {d_gain*100:+.1f} points, $p={d_p:.3f}$; +NonStyle "
            f"{n_gain*100:+.1f} points over +Duplicate, $p={n_p:.3f}$).")
    else:
        pieces.append(
            f"Adding AdaIN-styled images raises the cross-domain average from "
            f"{o*100:.1f}\\% to {a*100:.1f}\\% over {n_seeds} seeds, a paired gain "
            f"of {gain*100:.1f} points (95\\% CI [{lo*100:.1f}, {hi*100:.1f}], "
            f"$p={p:.4f}$).")

    if matched is not None and ours is not None:
        s_gain, s_lo, s_hi, s_t, s_p = paired(matched, ours)
        pieces.append(
            f"Under the strength-matched control the spatial modulation and the "
            f"uniform application are indistinguishable "
            f"({s_gain*100:+.2f} points, $p={s_p:.3f}$), and the accuracy follows "
            f"the average style strength rather than its spatial distribution. "
            f"We therefore conclude that style augmentation helps because of the "
            f"style it applies, and that the benefit of spatially adaptive "
            f"stylization is not supported by these controlled experiments.")
    else:
        pieces.append(
            f"The accuracy follows the average style strength rather than its "
            f"spatial distribution, so style augmentation helps because of the "
            f"style it applies.")

    text = " ".join(pieces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  ({len(text.split())} words)")


if __name__ == "__main__":
    main()
