"""Unified experiment orchestrator.

One command runs a whole batch of experiments and produces the summary files
that the report is built from.  It exists so that the batches of the report
plan are reproducible and so that no experiment can silently enter the
results with a different configuration.

What it does for every group and seed:

  1. checks whether the result already exists and was produced with the
     current ``config_hash``; if so, the run is skipped;
  2. runs ``fairness_check.py`` on the group's manifest before training;
  3. trains with ``train.py``;
  4. evaluates with ``evaluate.py --per-class``;
  5. writes ``summary_<group>.json``.

The ``config_hash`` check is the important one.  A run whose recorded
configuration differs from the current one is *retrained* rather than
reused, so a table can never mix two protocols.

Usage::

    # the whole main-results table, five seeds per group
    python run_sweep.py --batch main --seeds 0 1 2 3 4

    # one group
    python run_sweep.py --group duplicate --seeds 0 1 2 3 4

    # see what would happen without running anything
    python run_sweep.py --batch main --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from fairness_check import check  # noqa: E402
from data import read_aug_manifest  # noqa: E402

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")


def manifest_of(group: str) -> Path | None:
    """The manifest of a group, relative to the data root."""
    relative = config.GROUP_MANIFESTS.get(group)
    if relative is None:
        return None
    return config.DATA_ROOT / relative


def verify_manifest(group: str) -> list[str]:
    """Run the fairness constraints on a single group's manifest."""
    path = manifest_of(group)
    if path is None:
        return [f"{group}: no manifest registered"]
    if not path.is_file():
        return [f"{group}: manifest missing: {path}"]
    return check([(group, path)])


def run_command(command: list[str], cwd: Path | None = None,
                env: dict | None = None) -> None:
    environment = dict(os.environ)
    if env:
        environment.update(env)
    # ``-u`` makes the child write its progress line by line instead of
    # letting the pipe buffer hold them until the process exits.  Without it
    # a sweep of several hours produces no visible progress at all, and a
    # stalled run cannot be told apart from a running one.
    if command[0] == sys.executable and "-u" not in command:
        command = [command[0], "-u", *command[1:]]
    result = subprocess.run(command, cwd=cwd or config.CODE_ROOT,
                            env=environment)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}: "
                         f"{' '.join(command)}")


def evaluate(run_dir: Path, group: str | None = None) -> dict:
    """Evaluate one run and return the parsed result.

    The evaluation needs the trained weights, so the checkpoint is kept for
    the duration of the call and removed afterwards: the reported numbers
    live in the evaluation JSON, and a full sweep would otherwise leave
    dozens of 43 MB checkpoints behind.
    """
    eval_file = run_dir / f"eval_{config.CHECKPOINT}.json"
    if not eval_file.is_file():
        command = [sys.executable, "evaluate.py", "--run", str(run_dir),
                   "--checkpoint", f"{config.CHECKPOINT}.pt", "--per-class"]
        if group is not None:
            command += ["--group", group]
        run_command(command)
    return json.loads(eval_file.read_text(encoding="utf-8"))


def cleanup_checkpoint(run_dir: Path) -> None:
    """Remove the trained weights of a finished run."""
    for name in ("last.pt", "best.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            candidate.unlink()


def train_and_evaluate(group: str, seed: int, tag_prefix: str | None = None,
                        force: bool = False, manifest: Path | None = None,
                        env: dict | None = None) -> dict:
    """Train one (group, seed) and return its evaluation record."""
    tag = f"{tag_prefix or group}_s{seed}"
    run_dir = config.OUTPUT_ROOT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_file = run_dir / f"eval_{config.CHECKPOINT}.json"

    manifest = manifest if manifest is not None else manifest_of(group)

    reusable = eval_file.is_file() and not force
    if reusable:
        recorded = config.canonical_hash_of_run(run_dir)
        current = config.config_hash_for_group(group, manifest)
        if recorded is not None and recorded != current:
            print(f"    {tag}: stored result has config {recorded} != current "
                  f"{current}; retraining")
            reusable = False
    if reusable:
        print(f"    {tag}: already present and up to date, skipping")
        return json.loads(eval_file.read_text(encoding="utf-8"))

    command = [sys.executable, "train.py", "--group", group,
               "--seed", str(seed), "--tag", tag, "--keep-checkpoint"]
    if manifest is not None:
        command += ["--aug-manifest", str(manifest)]
    run_command(command, env=env)

    # The evaluation needs the checkpoint.  If it is missing, the training
    # script did not keep it and the run cannot be evaluated: fail loudly
    # instead of writing a summary without the numbers.
    checkpoint = run_dir / f"{config.CHECKPOINT}.pt"
    if not checkpoint.is_file():
        raise SystemExit(
            f"{tag}: the checkpoint {checkpoint} is missing, so the run cannot "
            f"be evaluated.  Train it again with --keep-checkpoint.")

    result = evaluate(run_dir, group)
    cleanup_checkpoint(run_dir)
    return result


def write_summary(group: str, seeds: list[int], records: list[dict],
                  tag_prefix: str | None = None) -> Path:
    """Aggregate the per-seed records into ``summary_<group>.json``."""
    summary = {
        "group": group,
        "seeds": seeds,
        "config_hash": config.config_hash(),
    }
    for split in ("photo_test",) + CROSS:
        values = [record[split] for record in records if split in record]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        summary[split] = {"mean": mean, "std": variance ** 0.5, "per_seed": values}

    cross_means = [summary[split]["mean"] for split in CROSS if split in summary]
    if len(cross_means) == 3:
        summary["cross_domain_avg"] = {"mean": sum(cross_means) / 3}

    per_class = {}
    for record in records:
        for split, block in (record.get("per_class") or {}).items():
            for name, value in (block.get("accuracy") or {}).items():
                if value is None:
                    continue
                per_class.setdefault(split, {}).setdefault(name, []).append(value)
    if per_class:
        summary["per_class"] = {
            split: {name: {"mean": sum(values) / len(values), "per_seed": values}
                    for name, values in classes.items()}
            for split, classes in per_class.items()}

    out = config.OUTPUT_ROOT / f"summary_{tag_prefix or group}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def run_group(group: str, seeds: list[int], force: bool = False,
              dry_run: bool = False, tag_prefix: str | None = None,
              env: dict | None = None) -> None:
    """Run every seed of one group and write its summary."""
    print(f"\n=== group '{group}'  seeds={seeds} ===")
    manifest = manifest_of(group)
    if manifest is not None:
        problems = verify_manifest(group)
        if problems:
            print(f"  fairness check FAILED for '{group}':")
            for problem in problems[:10]:
                print(f"    - {problem}")
            raise SystemExit(f"refusing to train '{group}': the manifest does not "
                             f"satisfy the fairness constraints")
        print(f"  fairness check: PASS")
    elif group == config.MIXSTYLE_GROUP:
        # MixStyle trains on the *original* Photo images and mixes the feature
        # statistics in the network, so there is no stylisation manifest to
        # check.  What must be checked is that the architecture is the
        # MixStyle one, which the environment variable guarantees.
        print("  fairness check: n/a (MixStyle trains on the original images; "
              "no stylisation manifest)")

    if dry_run:
        for seed in seeds:
            tag = f"{tag_prefix or group}_s{seed}"
            print(f"  would train {tag}")
        return

    started = time.time()
    records = []
    for seed in seeds:
        records.append(train_and_evaluate(group, seed, tag_prefix, force,
                                          manifest, env))
    out = write_summary(group, seeds, records, tag_prefix)
    elapsed = time.time() - started
    print(f"  wrote {out}  ({elapsed/60:.1f} min)")


def mixstyle_env() -> dict:
    """The environment that turns the MixStyle layer on.

    The layer is a no-op at evaluation time, so the training *and* the
    evaluation of a MixStyle run have to agree on the architecture.  The
    evaluation reads the group from the run summary and therefore builds the
    layer on its own; the training script reads the environment variable.
    """
    return {"NNDL_MIXSTYLE": "1"}


# --------------------------------------------------------------- batches ----

#: The batches of the experiment plan.  Each entry maps a group to the seeds
#: it should be run with.
BATCHES = {
    # the two data-volume controls that close the main conclusion
    "controls": {"duplicate": None, "nonstyle": None},
    # the main results table, every group on the same seeds
    "main": {"original": None, "duplicate": None, "nonstyle": None,
             "adain": None, "adain_matched": None, "ours": None,
             "ours_equal": None},
    # the MixStyle baseline: a different architecture, its own fingerprint
    "mixstyle": {"mixstyle": None},
    # the repaired modulation: a different method, its own fingerprint.
    # It is reported in its own subsection and never pooled with the main
    # groups.  The centre prior is a separate batch: it is the map that
    # won the selection, and it is a different method again.
    "foreground": {config.FOREGROUND_GROUP: None},
    "foreground_center": {config.FOREGROUND_CENTER_GROUP: None},
}


def run_variant(group: str, seeds: list[int], tag_prefix: str,
                manifest: Path | None, force: bool = False,
                dry_run: bool = False, env: dict | None = None,
                label: str = "") -> None:
    """Run one (group, seeds) combination under a specific manifest.

    A variant is a group trained on a different augmentation set: a
    different alpha, a different beta, or a different style pool.  It gets
    its own run tags and its own summary file, so that it can be compared
    with the main groups without overwriting them.
    """
    print(f"\n=== {label or tag_prefix}  seeds={seeds} ===")
    if manifest is not None:
        # The style-pool sensitivity experiment runs against a half-pool, so
        # the check has to validate against *that* pool and not against the
        # frozen ten-painting one.
        pool_path = (env or {}).get("NNDL_STYLE_POOL")
        problems = check([(group, manifest)], pool_path=pool_path)
        if problems:
            print(f"  fairness check FAILED for '{label or tag_prefix}':")
            for problem in problems[:10]:
                print(f"    - {problem}")
            raise SystemExit("refusing to train: the manifest does not satisfy "
                             "the fairness constraints")
        print("  fairness check: PASS")

    if dry_run:
        for seed in seeds:
            print(f"  would train {tag_prefix}_s{seed}")
        return

    started = time.time()
    records = []
    for seed in seeds:
        records.append(train_and_evaluate(group, seed, tag_prefix, force,
                                          manifest))
    out = write_summary(group, seeds, records, tag_prefix)
    elapsed = time.time() - started
    print(f"  wrote {out}  ({elapsed/60:.1f} min)")


def sweep_manifest(alpha: float | None, beta: float, kind: str,
                   tag: str | None = None) -> Path | None:
    """The manifest of one cell of the alpha/beta sweep.

    The generator writes its output under
    ``<data root>/aug_sp/sweeps/sweep_full/a<alpha>_b<beta>/`` and names the
    three manifests after the kind of image they describe.

    ``alpha`` may be ``None`` when only beta varies: the beta sweep is run at
    a fixed alpha, and the *tag* of the cell (which is what the directory is
    named after) is then supplied explicitly.  Without that the beta sweep
    could not find its manifests at all.
    """
    names = {"ours": "manifest_ours.txt",
             "adain": "manifest_adain.txt",
             "adain_matched": "manifest_adain_matched.txt"}
    name = names.get(kind)
    if name is None:
        return None
    if tag is None:
        if alpha is None:
            return None
        tag = f"a{str(alpha).replace('.', 'p')}_b{str(beta).replace('.', 'p')}"
    root = config.DATA_ROOT / "aug_sp" / "sweeps" / "sweep_full" / tag
    return root / name


def reuse_summary(group: str, tag_prefix: str, seeds: list[int]) -> Path | None:
    """Copy the result of an already-trained group under a sweep tag.

    Some cells of the sweeps are *the same experiment* as one of the main
    groups: the alpha sweep runs at beta = 0.7, and beta = 0.7 is the main
    setting, so the cell (alpha = 0.5, beta = 0.7) is generated by exactly
    the same code with exactly the same parameters and produces exactly the
    same images.  Re-training it would only burn GPU time and would
    reproduce the same numbers.

    The copy keeps the per-seed values untouched and only changes the tag, so
    that the sweep figures can read the cell from its own summary file.
    """
    source = config.OUTPUT_ROOT / f"summary_{group}.json"
    if not source.is_file():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not all(split in payload for split in CROSS):
        return None
    payload["group"] = tag_prefix
    payload["seeds"] = seeds
    payload["reused_from"] = group
    out = config.OUTPUT_ROOT / f"summary_{tag_prefix}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def run_alpha_sweep(seeds: list[int], alphas: list[float], beta: float,
                    force: bool = False, dry_run: bool = False) -> None:
    """Train every cell of the alpha sweep.

    For each alpha the three configurations are trained: the spatial
    variant, the uniform AdaIN variant, and the strength-matched uniform
    AdaIN variant.  The last one is the control that makes the comparison
    fair: it receives the same average stylization strength as the spatial
    variant, so the only difference between the two is the *distribution*
    of the strength over the image.

    The cell (alpha = 0.5, beta = 0.7) of the *spatial* variant is the main
    experiment itself, so it is reused instead of re-trained.
    """
    for alpha in alphas:
        for kind, group in (("ours", "ours"), ("adain", "adain"),
                            ("adain_matched", "adain_matched")):
            tag_prefix = f"{kind}_a{str(alpha).replace('.', 'p')}_b{str(beta).replace('.', 'p')}"
            # The main runs *are* the (alpha = 0.5, beta = 0.7) cells: the
            # main augmentation was generated with exactly these parameters
            # by exactly the same code, so the images are identical and
            # re-training them would only burn GPU time.
            if (abs(alpha - 0.5) < 1e-9 and abs(beta - 0.7) < 1e-9
                    and not force):
                reused = reuse_summary(group, tag_prefix, seeds)
                if reused is not None:
                    print(f"\n=== alpha={alpha} beta={beta} {kind} ===")
                    print(f"  identical to the main run '{group}'; reusing "
                          f"{reused.name}")
                    continue
            manifest = sweep_manifest(alpha, beta, kind)
            if manifest is None or not manifest.is_file():
                print(f"\n=== alpha={alpha} {kind} ===")
                print(f"  manifest missing: {manifest}")
                print("  (run the generator first)")
                if not dry_run:
                    raise SystemExit(f"the sweep manifest {manifest} does not exist; "
                                     f"generate the images first")
                continue
            run_variant(group, seeds, tag_prefix, manifest, force, dry_run,
                        label=f"alpha={alpha} beta={beta} {kind}")


def run_beta_sweep(seeds: list[int], betas: list[float], alpha: float,
                   force: bool = False, dry_run: bool = False) -> None:
    """Train every cell of the beta ablation.

    The beta sweep runs at a fixed alpha and varies how strongly the style
    strength is modulated in space: beta = 0 is a constant map (i.e. plain
    uniform AdaIN), and beta = 1 is the strongest modulation.  The
    strength-matched uniform control is trained alongside, so that every cell
    can be compared at equal mean strength.
    """
    for beta in betas:
        for kind, group in (("ours", "ours"), ("adain_matched", "adain_matched")):
            cell = f"a{str(alpha).replace('.', 'p')}_b{str(beta).replace('.', 'p')}"
            manifest = sweep_manifest(None, beta, kind, tag=cell)
            if manifest is None or not manifest.is_file():
                print(f"\n=== beta={beta} {kind} ===")
                print(f"  manifest missing: {manifest}")
                if not dry_run:
                    raise SystemExit(f"the sweep manifest {manifest} does not "
                                     f"exist; generate the images first")
                continue
            tag_prefix = f"{kind}_a{str(alpha).replace('.', 'p')}_b{str(beta).replace('.', 'p')}"
            run_variant(group, seeds, tag_prefix, manifest, force, dry_run,
                        label=f"beta={beta} {kind}")


def run_pool_sensitivity(seeds: list[int], force: bool = False,
                          dry_run: bool = False) -> None:
    """Train the two half-pools.

    The environment variable ``NNDL_STYLE_POOL`` is what the generators and
    the fairness checker read, so the whole pipeline -- generation,
    checking and training -- runs against the half-pool.
    """
    import paths

    for name in ("subpool_a", "subpool_b"):
        pool = paths.STYLE_POOL_DIR / "subpools" / name / "style_pool.json"
        if not pool.is_file():
            print(f"\n=== {name} ===")
            print(f"  the half-pool manifest is missing: {pool}")
            print("  (run member1_data/build_subpools.py first)")
            if not dry_run:
                raise SystemExit(f"the half-pool {pool} does not exist; build it first")
            continue
        env = {"NNDL_STYLE_POOL": str(pool)}
        # The generator writes the half-pool run into a *separate* cell
        # directory (``subpool_a_a0p5_b0p7``), so that it cannot overwrite
        # the full-pool run of the same alpha/beta values.
        cell = f"{name}_a0p5_b0p7"
        manifest = config.DATA_ROOT / "aug_sp" / "sweeps" / "sweep_full" / \
            cell / "manifest_ours.txt"
        if not manifest.is_file():
            manifest = None
        run_variant("ours", seeds, cell, manifest, force,
                    dry_run, env=env, label=f"style pool {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=config.GROUPS, default=None,
                        help="run a single group")
    parser.add_argument("--batch", choices=sorted(BATCHES), default=None,
                        help="run a whole batch of groups")
    parser.add_argument("--sweep", choices=("alpha", "beta", "pool"), default=None,
                        help="run one of the parameter sweeps")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="the alpha values of the alpha sweep")
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7, 1.0],
                        help="the beta values of the beta sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS),
                        help="the seeds to run")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if a matching result already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be run and exit")
    args = parser.parse_args()

    if not args.group and not args.batch and not args.sweep:
        parser.error("one of --group, --batch or --sweep is required")

    print(f"config_hash: {config.config_hash()}")
    print(f"data root:   {config.DATA_ROOT}")
    print(f"runs root:   {config.OUTPUT_ROOT}")
    print(f"seeds:       {args.seeds}")

    if args.sweep == "alpha":
        run_alpha_sweep(args.seeds, args.alphas, 0.7, args.force, args.dry_run)
    elif args.sweep == "beta":
        run_beta_sweep(args.seeds, args.betas, 0.5, args.force, args.dry_run)
    elif args.sweep == "pool":
        run_pool_sensitivity(args.seeds, args.force, args.dry_run)
    else:
        groups = ([args.group] if args.group
                  else list(BATCHES[args.batch].keys()))
        for group in groups:
            env = mixstyle_env() if group == config.MIXSTYLE_GROUP else None
            run_group(group, args.seeds, force=args.force, dry_run=args.dry_run,
                      env=env)

    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
