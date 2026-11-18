"""Calibrate alpha0 so that the spatial map has the SAME mean style strength
as plain AdaIN at alpha = 0.5.

The current main setting (alpha0 = 0.5, beta = 0.7) gives a spatial map whose
mean is 0.3736, i.e. 0.126 weaker than plain AdaIN at alpha = 0.5.  That is
why "+Ours" and "+AdaIN" differ in two ways at once, and why the
strength-matched uniform control was needed to separate them.

For a perfectly clean "spatial vs uniform at equal strength" comparison, we
need an alpha0 such that

    mean(alpha_map) = alpha0 * (1 - beta * mean(M)) = 0.5

so that the spatial map and uniform AdaIN at alpha = 0.5 carry the same total
style energy and differ only in whether it is modulated in space.

mean(M) is estimated from the recorded per-image mean_alpha of the main run.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402

GEN = paths.DATA_ROOT / "aug_sp" / "generation.json"
ALPHA0_MAIN = 0.5
BETA = 0.7
TARGET_MEAN = 0.5

payload = json.loads(GEN.read_text(encoding="utf-8"))
means = [entry["metrics"]["mean_alpha"]
         for entry in payload["outputs"].values()
         if entry.get("metrics", {}).get("mean_alpha") is not None]
mean_map = sum(means) / len(means)
mean_M = (1.0 - mean_map / ALPHA0_MAIN) / BETA
print(f"recorded mean(alpha_map) at alpha0={ALPHA0_MAIN}, beta={BETA}: {mean_map:.6f}")
print(f"=> implied mean(M) = {mean_M:.6f}")

factor = 1.0 - BETA * mean_M
alpha0 = TARGET_MEAN / factor
print(f"factor (1 - beta*mean(M)) = {factor:.6f}")
print(f"alpha0 for mean(alpha_map) = {TARGET_MEAN}: {alpha0:.6f}")
print()
print(f"  check: alpha0 * factor = {alpha0 * factor:.6f}")
print()
print(f"USE: --alpha {alpha0:.6f} --beta {BETA}")
