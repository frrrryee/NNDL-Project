"""Generate the augmentation sets of the remaining sweeps, in order.

The generation is the slow part of the plan -- roughly six minutes per
cell of 1,349 images -- and it has to happen *before* the training of the
cell.  This script runs the cells one after another so that the GPU is
never asked for two generations at once.

The cells are:

    A  the alpha sweep: alpha in {0.25, 0.75, 1.0} at beta = 0.7
    B  the beta ablation: beta in {0.0, 0.3, 0.5, 1.0} at alpha = 0.5
    C  the style-pool sensitivity: the two half-pools at alpha = 0.5,
       beta = 0.7

The (alpha = 0.5, beta = 0.7) cell is *not* generated: it is the main
augmentation set, which already exists.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import paths  # noqa: E402

GENERATE = [sys.executable, "-u", "-m", "member3_structure_preserving.generate",
            "sweep", "--with-matched"]


def run(command: list[str], env: dict | None = None) -> float:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    if env:
        environment.update(env)
    started = time.time()
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"generation failed with exit code {result.returncode}")
    return time.time() - started


def cell(alpha: float, beta: float, pool: Path | None = None) -> None:
    """Generate one cell of the sweep."""
    command = list(GENERATE)
    command += ["--alphas", str(alpha), "--betas", str(beta)]
    env = None
    if pool is not None:
        env = {"NNDL_STYLE_POOL": str(pool)}
        command += ["--cell-tag", pool.parent.name]
    elapsed = run(command, env)
    print(f"  cell a={alpha} b={beta} done in {elapsed/60:.1f} min")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=("A", "B", "C"), default=None,
                        help="generate only this batch")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.75, 1.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 1.0])
    args = parser.parse_args()

    batches = [args.batch] if args.batch else ["A", "B", "C"]
    started = time.time()

    if "A" in batches:
        print("\n" + "=" * 70)
        print("BATCH A  the alpha sweep (beta = 0.7)")
        print("=" * 70)
        for alpha in args.alphas:
            cell(alpha, 0.7)

    if "B" in batches:
        print("\n" + "=" * 70)
        print("BATCH B  the beta ablation (alpha = 0.5)")
        print("=" * 70)
        for beta in args.betas:
            cell(0.5, beta)

    if "C" in batches:
        print("\n" + "=" * 70)
        print("BATCH C  the style-pool sensitivity")
        print("=" * 70)
        for name in ("subpool_a", "subpool_b"):
            pool = paths.STYLE_POOL_DIR / "subpools" / name / "style_pool.json"
            if not pool.is_file():
                raise SystemExit(f"the half-pool {pool} does not exist; "
                                 f"run member1_data/build_subpools.py first")
            cell(0.5, 0.7, pool)

    print(f"\nall generation finished in {(time.time()-started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
