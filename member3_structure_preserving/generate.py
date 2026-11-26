"""Generate member 3 previews, official augmentation, and alpha/beta sweeps.

Run from the repository root:
  python -m member3_structure_preserving.generate samples --limit 7
  python -m member3_structure_preserving.generate build --alpha 0.5 --beta 0.7
  python -m member3_structure_preserving.generate sweep --limit 7 --betas 0 0.3 0.7 1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from member4_training_eval import config
from member3_structure_preserving.analysis import (
    StructureMetrics, comparison_panel, map_image, map_image_2d, summarize,
)
from member3_structure_preserving.foreground_map import ForegroundConfig
from member3_structure_preserving.protocol import FrozenProtocol, sha256
from member3_structure_preserving.structure_map import StructureConfig
from member3_structure_preserving.stylizer import StructurePreservingStylizer


def _atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".jpg":
        image.save(temporary, format="JPEG", quality=95)
    else:
        image.save(temporary, format="PNG")
    temporary.replace(path)


def _number(value):
    # Round-trip representation prevents two nearby float settings colliding.
    return str(float(value)).replace(".", "p")


@dataclass
class Run:
    root: Path
    manifest: Path
    alpha: float
    beta: float
    settings: dict
    outputs: dict

    @property
    def record_path(self):
        return self.root / "generation.json"

    def save(self, complete=False):
        _json(self.record_path, {"settings": self.settings,
                                "outputs": self.outputs, "complete": complete})


def _locations(args, data_root, alpha, beta):
    tag = f"a{_number(alpha)}_b{_number(beta)}"
    # The style-pool sensitivity experiment runs the same alpha/beta cell
    # with a *different* style pool.  Without a distinct cell tag the
    # half-pool run would write into the directory of the full-pool run,
    # and the two would overwrite each other (their pool hashes differ, so
    # the resume check would reject the second one).  The suffix keeps the
    # two experiments in separate directories while leaving the alpha/beta
    # values of the cell itself untouched.
    cell_tag = getattr(args, "cell_tag", None)
    if cell_tag:
        tag = f"{cell_tag}_{tag}"
    if args.mode == "build" and not args.limit:
        # The repaired modulation writes into its own directory, so that
        # its manifest can never be confused with the original method's.
        #
        # Every *other* cell tag also needs a directory of its own.  A
        # build that carries an unrecognised tag used to fall through to
        # the shared ``aug_sp`` directory and replace the original
        # (Sobel) dataset there -- silently, because the tag was only
        # used to name the sweep directories.  The shared directory now
        # belongs to the original method alone, so a tagged build gets
        # its own directory and the original method keeps the shared
        # one.
        if cell_tag is None:
            root = data_root / "aug_sp"
            return root, root / "manifest_ours.txt"
        root = data_root / "aug_sp" / cell_tag
        return root, root / "manifest_ours.txt"
    category = "sweeps" if args.mode == "sweep" else "previews"
    selection = f"{args.mode}_n{args.limit}" if args.limit else f"{args.mode}_full"
    root = data_root / "aug_sp" / category / selection / tag
    return root, root / "manifest_ours.txt"


def _output_files(root, job, diagnostics, with_baseline, with_matched=False):
    relative = Path(job.original)
    # The multi-style protocol generates several images per original, so the
    # style index has to be part of the name.  The one-style protocol keeps
    # its historical name, so the images already on disk stay valid.
    if job.style_index:
        name = f"{relative.stem}__{job.style_index:02d}_{job.style}"
    else:
        name = f"{relative.stem}__{job.style}"
    # Keep class subdirectories: identical stems in different classes cannot collide.
    result = {"ours": root / "images" / relative.parent / f"{name}.jpg"}
    if with_baseline:
        result["adain"] = root / "adain" / relative.parent / f"{name}.jpg"
    if with_matched:
        result["adain_matched"] = root / "adain_matched" / relative.parent / f"{name}.jpg"
    if diagnostics:
        base = root / "figures" / relative.parent
        result.update({"structure": base / f"{name}__M.png",
                       "alpha": base / f"{name}__alpha.png",
                       "comparison": base / f"{name}__compare.png"})
    return result


def _relative(path, root):
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mean_matched_alpha(outputs):
    """Mean of the per-image uniform AdaIN strength used for the matched control.

    This is the number that makes the control comparable: the uniform AdaIN
    receives the same average style strength as the spatial map, so the two
    differ only in whether the strength is modulated in space.
    """
    values = [entry.get("matched_alpha") for entry in outputs.values()
              if entry.get("matched_alpha") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fingerprint_sources(weights_dir):
    own = Path(__file__).resolve().parent
    paths = list(own.glob("*.py")) + [config.PROJECT_ROOT / "member2_adain" / "adain.py"]
    code = {path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sorted(paths)}
    return {"code": code,
            "weights": {name: sha256(weights_dir / name)
                        for name in ("decoder.pth", "vgg_normalised.pth")}}


def run(args):
    if args.limit < 0 or args.threads < 1:
        raise ValueError("limit must be non-negative; threads must be positive")
    torch.set_num_threads(args.threads)
    structure = StructureConfig(args.preblur_sigma, args.edge_quantile,
                                args.dilation_radius, args.map_blur_sigma)
    if args.content_size < 32 or args.style_size < 32:
        raise ValueError("content-size and style-size must be at least 32")
    alphas = args.alphas if args.mode == "sweep" else [args.alpha]
    betas = args.betas if args.mode == "sweep" else [args.beta]
    if len(set(alphas)) != len(alphas) or len(set(betas)) != len(betas):
        raise ValueError("duplicate alpha/beta sweep values")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in alphas + betas):
        raise ValueError("alpha and beta must be finite and in [0,1]")
    if args.lpips and not args.metrics:
        raise ValueError("--lpips requires --metrics")
    protocol = FrozenProtocol(args.data_root)
    jobs = protocol.jobs(args.limit, balanced=args.mode == "samples" or args.balanced)
    # Validate every source before loading models or writing any run files.
    input_hashes = {job.original: sha256(job.content_path) for job in jobs}
    style_hashes = {meta["sha256"] for meta in protocol.manifest["pool"]["styles"].values()}
    if style_hashes.intersection(input_hashes.values()):
        raise ValueError("a training content image is byte-identical to a frozen style")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metrics = StructureMetrics(args.lpips, device) if args.metrics else None
    foreground = ForegroundConfig(
        spectral_size=args.spectral_size, center_sigma=args.center_sigma,
        center_weight=args.center_weight, smooth_sigma=args.foreground_blur_sigma)
    generator = StructurePreservingStylizer(
        structure_config=structure, foreground_config=foreground,
        modulation=args.modulation, foreground_kind=args.foreground_kind,
        content_size=args.content_size,
        style_size=args.style_size, device=device, weights_dir=args.weights_dir)
    fingerprints = _fingerprint_sources(args.weights_dir)
    diagnostics = args.mode == "samples" or args.diagnostics
    with_baseline = args.mode in ("samples", "sweep") or args.with_adain
    with_matched = args.with_matched
    common = {"schema": 2,
              "method": ("sobel-spatial-adain-v1" if args.modulation == "edge"
                         else f"foreground-spatial-adain-v1/{args.foreground_kind}"),
              "mode": args.mode,
              "partial": len(jobs) != len(protocol.rows), "count": len(jobs),
              "pool_hash": protocol.pool_hash,
              "styles_per_image": protocol.styles_per_image,
              "inputs_sha256": input_hashes,
              "modulation": args.modulation,
              "foreground_kind": args.foreground_kind if args.modulation == "foreground" else None,
              "foreground": foreground.to_dict() if args.modulation == "foreground" else None,
              "pool_manifest_sha256_lf": hashlib.sha256(
                  config.STYLE_POOL_MANIFEST.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "split_sha256_lf": hashlib.sha256(protocol.split_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "structure": structure.to_dict(), "content_size": args.content_size,
              "style_size": args.style_size, "jpeg_quality": 95, "device": str(generator.model.device),
              "torch": torch.__version__, "threads": args.threads,
              "dependencies": {name: version(name) for name in ("torchvision", "Pillow", "numpy")},
              "opencv": metrics.cv2.__version__ if metrics else None,
              "diagnostics": diagnostics,
              "with_adain": with_baseline, "with_matched": with_matched,
              "metrics": args.metrics, "lpips": args.lpips,
              **fingerprints}
    runs = []
    for alpha in alphas:
        for beta in betas:
            root, manifest = _locations(args, protocol.data_root, alpha, beta)
            paths = [_output_files(root, job, diagnostics, with_baseline)["ours"] for job in jobs]
            if len(paths) != len(set(paths)):
                raise ValueError("duplicate output names in the training split")
            settings = dict(common, alpha=alpha, beta=beta)
            state = Run(root, manifest, alpha, beta, settings, {})
            if state.record_path.exists():
                previous = json.loads(state.record_path.read_text(encoding="utf-8"))
                if previous["settings"] != settings and not args.overwrite:
                    raise ValueError(f"existing generation settings differ: {state.record_path}; use another parameter set or --overwrite")
                if previous["settings"] == settings:
                    state.outputs = previous["outputs"]
            elif root.exists() and not args.overwrite:
                # Official aug_sp also contains independently recorded preview
                # and sweep runs. They must not block or be replaced by build.
                allowed = {"previews", "sweeps"} if args.mode == "build" and not args.limit else set()
                unexpected = [path for path in root.iterdir()
                              if not (path.name in allowed and path.is_dir())]
                if unexpected:
                    raise ValueError(f"unrecorded files already exist in {root}; use --overwrite only if replacing them")
            runs.append(state)
    for state in runs:
        # Invalidate manifests before modifying their images. Only publish a
        # complete manifest after the entire run succeeds (also on resume).
        pending = [state.manifest, state.root / "manifest_adain.txt", state.root / "analysis.json",
                   state.root / "photo_train.txt", state.root / "photo_train_partial.txt", state.root / "pairs.txt",
                   state.root / "manifest_adain_matched.txt"]
        if args.mode == "build" and not args.limit:
            # Only the original method's build owns the shared manifest
            # under ``augmented/``; a tagged build must leave it alone.
            if getattr(args, "cell_tag", None) is None:
                pending.append(protocol.data_root / "augmented" / "manifest_ours.txt")
        for path in pending:
            if path.exists():
                path.unlink()
        state.save()
    generated = skipped = 0
    for index, job in enumerate(jobs, 1):
        prepared = style_features = None
        for state in runs:
            files = _output_files(state.root, job, diagnostics, with_baseline, with_matched)
            previous = state.outputs.get(job.key, {})
            recorded = previous.get("sha256", {})
            valid = bool(recorded) and all(path.is_file() and sha256(path) == recorded.get(key)
                                           for key, path in files.items())
            if valid:
                skipped += 1
                continue
            if prepared is None:
                prepared = generator.prepare_content(job.content_path)
                style_features = generator.encode_style(job.style_path)
            result = generator.render(prepared, style_features, state.alpha, state.beta,
                                      with_baseline, with_matched)
            _save_image(files["ours"], result.image)
            if with_baseline:
                _save_image(files["adain"], result.baseline)
            if with_matched:
                _save_image(files["adain_matched"], result.matched_baseline)
            if diagnostics:
                _save_image(files["structure"], map_image_2d(result.structure))
                _save_image(files["alpha"], map_image_2d(result.alpha_map))
                _save_image(files["comparison"], comparison_panel(result, job.style_path, state.alpha, state.beta))
            measurement = {"mean_alpha": float(result.alpha_map.mean())}
            baseline_measurement = {}
            matched_measurement = {}
            if metrics:
                measurement.update(metrics.compare(job.content_path, files["ours"]))
                if with_baseline:
                    baseline_measurement = metrics.compare(job.content_path, files["adain"])
                if with_matched:
                    matched_measurement = metrics.compare(job.content_path, files["adain_matched"])
            state.outputs[job.key] = {
                "files": {key: _relative(path, protocol.data_root) for key, path in files.items()},
                "sha256": {key: sha256(path) for key, path in files.items()},
                "metrics": measurement, "adain_metrics": baseline_measurement,
                "adain_matched_metrics": matched_measurement,
                "matched_alpha": result.matched_alpha}
            state.save()
            generated += 1
        print(f"[{index}/{len(jobs)}] {job.original} [{job.style_index}]", flush=True)
    summaries = []
    for state in runs:
        ours_lines, adain_lines, matched_lines = [], [], []
        dataset_lines, pair_lines = [], []
        for job in jobs:
            entry = state.outputs[job.key]
            provenance = f"{job.label} {job.original} {job.style} {state.alpha}"
            ours_lines.append(f"{entry['files']['ours']} {provenance} {state.beta}")
            image_path = protocol.data_root / entry["files"]["ours"]
            dataset_lines.append(f"{_relative(image_path, state.root / 'images')} {job.label}")
            pair_lines.append(f"{_relative(image_path, state.root)}\t{job.original}\t{job.style}\t{job.label}")
            if with_baseline:
                adain_lines.append(f"{entry['files']['adain']} {provenance}")
            if with_matched:
                matched_lines.append(f"{entry['files']['adain_matched']} {provenance}")
        _atomic_text(state.manifest, "\n".join(ours_lines) + "\n")
        split_name = "photo_train_partial.txt" if common["partial"] else "photo_train.txt"
        _atomic_text(state.root / split_name, "\n".join(dataset_lines) + "\n")
        _atomic_text(state.root / "pairs.txt", "\n".join(pair_lines) + "\n")
        if args.mode == "build" and not args.limit:
            # The original method's build also publishes the manifest under
            # ``augmented/`` for the older members' scripts.  A tagged
            # build must not: it would overwrite the original method's
            # manifest and make the two indistinguishable.
            if getattr(args, "cell_tag", None) is None:
                _atomic_text(protocol.data_root / "augmented" / "manifest_ours.txt",
                             "\n".join(ours_lines) + "\n")
        if adain_lines:
            _atomic_text(state.root / "manifest_adain.txt", "\n".join(adain_lines) + "\n")
        if matched_lines:
            _atomic_text(state.root / "manifest_adain_matched.txt", "\n".join(matched_lines) + "\n")
        summary = {"alpha": state.alpha, "beta": state.beta,
                   "manifest": str(state.manifest), "count": len(jobs),
                   "partial": common["partial"], "pool_hash": protocol.pool_hash,
                   "ours": summarize([r["metrics"] for r in state.outputs.values()]),
                   "adain": summarize([r["adain_metrics"] for r in state.outputs.values()]),
                   "adain_matched": summarize([r["adain_matched_metrics"] for r in state.outputs.values()]),
                   "matched_alpha_mean": _mean_matched_alpha(state.outputs),
                   "lpips_enabled": args.lpips,
                   "note": "Image diagnostics only; cross-domain classification gains have not been measured."}
        _json(state.root / "analysis.json", summary)
        state.save(complete=True)
        summaries.append(summary)
        print(f"manifest: {state.manifest}")
    if args.mode == "sweep":
        index_path = runs[0].root.parent / "sweep_summary.json"
        _json(index_path, {"runs": summaries, "automatic_selection": False})
    print(f"done: {generated} generated, {skipped} verified/resumed")
    return summaries


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="mode", required=True)
    for mode in ("build", "samples", "sweep"):
        command = commands.add_parser(mode)
        command.add_argument("--data-root", type=Path, default=None, help="PACS parent; defaults to member 4 config")
        command.add_argument("--limit", type=int, default=7 if mode == "samples" else 0,
                             help="0 = full training set; partial builds go into previews/")
        command.add_argument("--balanced", action="store_true",
                             help="take the limited subset round-robin over the seven "
                                  "classes instead of the first N rows, which are all "
                                  "of one class because the split is ordered by class")
        if mode == "sweep":
            command.add_argument("--alphas", type=float, nargs="+", default=[0.5])
            command.add_argument("--betas", type=float, nargs="+", default=[0, 0.3, 0.5, 0.7, 1])
        else:
            command.add_argument("--alpha", type=float, default=0.5)
            command.add_argument("--beta", type=float, default=0.7)
        command.add_argument("--content-size", type=int, default=512, help="resize shorter side, same as member 2")
        command.add_argument("--style-size", type=int, default=512)
        command.add_argument("--weights-dir", type=Path, default=config.PROJECT_ROOT / "member2_adain" / "models")
        command.add_argument("--device", default=None)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--preblur-sigma", type=float, default=1)
        command.add_argument("--edge-quantile", type=float, default=0.95)
        command.add_argument("--dilation-radius", type=int, default=1)
        command.add_argument("--map-blur-sigma", type=float, default=2)
        command.add_argument("--diagnostics", action="store_true", help="save M, alpha and labeled comparisons")
        command.add_argument("--with-adain", action="store_true", help="also export a matched standard AdaIN reference")
        command.add_argument("--with-matched", action="store_true",
                             help="also export the strength-matched uniform AdaIN control: "
                                  "a plain uniform AdaIN whose strength equals the mean of the "
                                  "spatial alpha map, so that 'structure preservation' is not "
                                  "confounded with 'weaker stylization'")
        command.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=mode != "build",
                             help="fixed Canny EdgeScore after JPEG encoding; OpenCV required")
        command.add_argument("--lpips", action="store_true", help="also compute LPIPS (optional package/weights)")
        command.add_argument("--overwrite", action="store_true", help="allow replacing an incompatible existing run")
        command.add_argument("--cell-tag", default=None,
                             help="extra prefix for the output directory of this cell. "
                                  "The style-pool sensitivity experiment uses it to "
                                  "keep the half-pool runs apart from the full-pool "
                                  "run of the same alpha/beta values.")
        command.add_argument("--modulation", choices=("edge", "foreground"), default="edge",
                             help="what modulates the style strength. 'edge' is the "
                                  "original soft Sobel response; 'foreground' is the "
                                  "repaired variant, which uses an unsupervised "
                                  "foreground estimate instead. The blending formula "
                                  "is identical in both cases, so the comparison "
                                  "isolates the map.")
        command.add_argument("--foreground-kind", choices=("spectral", "center", "spectral_center"),
                             default="spectral_center",
                             help="which foreground estimator to use with --modulation foreground")
        command.add_argument("--center-sigma", type=float, default=0.35,
                             help="width of the Gaussian centre prior, as a fraction of the image")
        command.add_argument("--center-weight", type=float, default=0.5,
                             help="weight of the centre prior in the combined foreground map")
        command.add_argument("--spectral-size", type=int, default=64,
                             help="the size the image is reduced to before its spectrum is taken")
        command.add_argument("--foreground-blur-sigma", type=float, default=4.0,
                             help="blur applied to the foreground map, in pixels")
    return result


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
