"""Run the whole experiment plan, in the order the report needs it.

The batches are ordered by what the report needs first:

  D  the data-volume controls, which close the main conclusion
  E  the seed growth to five seeds, which makes the headline significant
  A  the alpha sweep, which produces the main figure
  B  the beta ablation, which shows that the spatial map is a null result
  C  the style-pool sensitivity, which shows the result is not a fluke of
     one particular set of paintings
  F  the MixStyle baseline, which places the work against the literature

Every step is idempotent: a run whose result is already on disk and was
produced with the current configuration is reused, not repeated.

Usage::

    # the whole plan
    python run_all_batches.py

    # one batch
    python run_all_batches.py --batch D

    # see what would happen
    python run_all_batches.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
import paths  # noqa: E402

#: The number of seeds of the reported table.  Three was the original
#: protocol; five is the extension that makes the headline difference
#: significant.
SEEDS = (0, 1, 2, 3, 4)


def sh(command: list[str], env: dict | None = None, dry_run: bool = False) -> None:
    import os

    environment = dict(os.environ)
    if env:
        environment.update(env)
    if dry_run:
        # The dry-run has to stop here: the sub-command would otherwise run
        # the real experiment.
        printable = " ".join(command)
        for key, value in (env or {}).items():
            printable = f"{key}={value} {printable}"
        print(f"\n$ {printable}")
        return
    print(f"\n$ {' '.join(command)}")
    result = subprocess.run(command, cwd=HERE, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"step failed with exit code {result.returncode}")


def sweep(alpha: float, beta: float, dry_run: bool = False,
          pool: Path | None = None) -> None:
    """Generate one cell of the alpha/beta sweep.

    The generator writes the spatial variant, the uniform AdaIN variant and
    the strength-matched control in one pass, because they share the same
    content/style pairs and the same structure map.

    When a half-pool is used, the cell gets its own directory (``<name>_a.._b..``)
    so that it cannot overwrite the full-pool run of the same alpha/beta.
    """
    env = {"PYTHONPATH": str(REPO)}
    command = [sys.executable, "-m", "member3_structure_preserving.generate",
               "sweep", "--alphas", str(alpha), "--betas", str(beta),
               "--with-matched"]
    if pool is not None:
        env["NNDL_STYLE_POOL"] = str(pool)
        command += ["--cell-tag", pool.parent.name]
    sh(command, env=env, dry_run=dry_run)


def batch_d(seeds, dry_run=False) -> None:
    """The data-volume controls."""
    sh([sys.executable, "run_sweep.py", "--batch", "controls",
        "--seeds", *[str(s) for s in seeds]] + (["--dry-run"] if dry_run else []),
        dry_run=dry_run)


def batch_e(seeds, dry_run=False) -> None:
    """The seed growth of the main table."""
    sh([sys.executable, "run_sweep.py", "--batch", "main",
        "--seeds", *[str(s) for s in seeds]] + (["--dry-run"] if dry_run else []),
        dry_run=dry_run)


def batch_a(seeds, alphas=(0.25, 0.75, 1.0), beta=0.7, dry_run=False) -> None:
    """The alpha sweep: the data of the main figure."""
    for alpha in alphas:
        print(f"\n--- generating alpha={alpha} ---")
        sweep(alpha, beta, dry_run)
    sh([sys.executable, "run_sweep.py", "--sweep", "alpha",
        "--alphas", *[str(a) for a in alphas], "--seeds",
        *[str(s) for s in seeds]] + (["--dry-run"] if dry_run else []),
        dry_run=dry_run)


def batch_b(seeds, betas=(0.0, 0.3, 0.5, 1.0), alpha=0.5, dry_run=False) -> None:
    """The beta ablation, with the strength-matched control."""
    for beta in betas:
        print(f"\n--- generating beta={beta} ---")
        sweep(alpha, beta, dry_run)
    sh([sys.executable, "run_sweep.py", "--sweep", "beta",
        "--betas", *[str(b) for b in betas], "--seeds",
        *[str(s) for s in seeds]] + (["--dry-run"] if dry_run else []),
        dry_run=dry_run)


def batch_c(seeds, dry_run=False) -> None:
    """The style-pool sensitivity: the ten paintings split into two halves."""
    for name in ("subpool_a", "subpool_b"):
        pool = paths.STYLE_POOL_DIR / "subpools" / name / "style_pool.json"
        print(f"\n--- generating with the half-pool {name} ---")
        sweep(0.5, 0.7, dry_run, pool)
    sh([sys.executable, "run_sweep.py", "--sweep", "pool", "--seeds",
        *[str(s) for s in seeds]] + (["--dry-run"] if dry_run else []),
        dry_run=dry_run)


def batch_f(seeds, dry_run=False) -> None:
    """The MixStyle baseline."""
    sh([sys.executable, "run_sweep.py", "--batch", "mixstyle", "--seeds",
        *[str(s) for s in seeds]] + (["--dry-run"] if dry_run else []),
        dry_run=dry_run)


BATCH_ORDER = {"D": batch_d, "E": batch_e, "A": batch_a,
               "B": batch_b, "C": batch_c, "F": batch_f}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=sorted(BATCH_ORDER), nargs="+",
                        default=sorted(BATCH_ORDER),
                        help="the batches to run, in the given order")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be run and exit")
    args = parser.parse_args()

    print(f"config_hash: {config.config_hash()}")
    print(f"data root:   {config.DATA_ROOT}")
    print(f"runs root:   {config.OUTPUT_ROOT}")

    started = time.time()
    for code in args.batch:
        print(f"\n{'=' * 70}")
        print(f"BATCH {code}")
        print("=" * 70)
        BATCH_ORDER[code](args.seeds, dry_run=args.dry_run)
    elapsed = (time.time() - started) / 60
    print(f"\nall batches finished in {elapsed:.1f} min")

    if not args.dry_run:
        print("\n--- the results ---")
        sh([sys.executable, "final_analysis.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
