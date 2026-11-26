"""Read member 4's frozen protocol without regenerating splits or pairings."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from member4_training_eval import config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    """Reject traversal, absolute paths and manifest fields with whitespace."""
    if not relative or Path(relative).is_absolute() or any(c.isspace() for c in relative) or "\\" in relative:
        raise ValueError(f"invalid protocol path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"path outside expected root: {relative}")
    return path


@dataclass(frozen=True)
class GenerationJob:
    original: str
    label: int  # 1..7, as required by the shared manifest
    style: str
    content_path: Path
    style_path: Path
    #: Which of the styles of this original this job is.  It is 0 for the
    #: frozen one-style protocol and 0..k-1 for the multi-style one.
    style_index: int = 0

    @property
    def key(self) -> str:
        """A key that is unique per (original, style) pair.

        The one-style protocol can key its outputs by the original alone,
        but the multi-style protocol generates several images per original,
        so the outputs have to be keyed by the pair.  ``key`` is that key in
        both cases, so the recording code does not need a branch.
        """
        return f"{self.original}#{self.style_index}"


class FrozenProtocol:
    def __init__(self, data_root: Path | None = None):
        self.data_root = Path(data_root or config.DATA_ROOT).resolve()
        self.image_root = self.data_root / "PACS"
        self.split_file = config.SPLIT_FILES["photo_train"]
        self.manifest = json.loads(config.STYLE_POOL_MANIFEST.read_text(encoding="utf-8"))
        pool = self.manifest["pool"]
        payload = {k: v for k, v in pool.items() if k != "pool_hash"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        if digest != pool["pool_hash"]:
            raise ValueError("frozen style-pool hash does not match its content")
        # The pool normally holds the ten frozen paintings.  The
        # sensitivity experiment runs the same pipeline with a half-pool of
        # five, so the count is read from the manifest instead of being
        # hard-coded; the count is recorded in the manifest and therefore
        # travels with the run.
        self.n_styles = len(pool["styles"])
        if self.n_styles < 1:
            raise ValueError("the style pool is empty")
        self.pool_hash = digest
        self.styles = {}
        for name, meta in pool["styles"].items():
            path = contained(config.STYLE_POOL_DIR, meta["file"])
            if sha256(path) != meta["sha256"]:
                raise ValueError(f"style image hash mismatch: {name}")
            self.styles[name] = path
        # The multi-style protocol stores the full content->style assignment
        # next to the manifest, because an original that carries several
        # styles cannot be described by a single mapping.  When it is absent
        # the pool is the frozen one-style pool and the assignment is the
        # frozen mapping.
        multi = self.manifest.get("pairing_multi")
        self.styles_per_image = int(self.manifest.get("styles_per_image", 1))
        if multi is None:
            if self.styles_per_image != 1:
                raise ValueError(
                    "the manifest declares a multi-style protocol but carries no "
                    "pairing_multi assignment")
            self.pairings: dict[str, list[str]] = {}
        else:
            if self.styles_per_image < 2:
                raise ValueError("pairing_multi is present but styles_per_image is not")
            self.pairings = {rel: list(styles) for rel, styles in multi.items()}
            for rel, styles in self.pairings.items():
                if len(styles) != self.styles_per_image:
                    raise ValueError(
                        f"original {rel} carries {len(styles)} styles, expected "
                        f"{self.styles_per_image}")
                unknown = set(styles) - set(self.styles)
                if unknown:
                    raise ValueError(f"unknown frozen style: {sorted(unknown)}")
        rows = config.read_split(self.split_file)
        self.rows = rows
        originals = [rel for rel, _ in rows]
        if len(originals) != len(set(originals)) or set(originals) != set(self.manifest["pairing"]):
            raise ValueError("photo_train and frozen pairing coverage differ")
        protocol = json.loads((config.SPLIT_DIR / "protocol.json").read_text(encoding="utf-8"))
        spec = next(row for row in protocol["splits"] if row["split"] == "photo_train")
        raw = self.split_file.read_bytes().replace(b"\r\n", b"\n")
        if len(rows) != spec["count"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("photo_train differs from the shared protocol")

    def jobs(self, limit: int = 0, balanced: bool = False) -> list[GenerationJob]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = self.rows
        if balanced:
            # Deterministic round-robin over classes, without changing pairing.
            # This is the same selection member 4 uses for the reduced
            # protocol, so the generator and the trainer see the same images.
            buckets = [[row for row in rows if row[1] == label] for label in range(len(config.CLASSES))]
            rows = [bucket[i] for i in range(max(map(len, buckets))) for bucket in buckets if i < len(bucket)]
        if limit:
            rows = rows[:limit]
        jobs = []
        for rel, label in rows:
            if not 0 <= label < len(config.CLASSES) or Path(rel).parts[:2] != ("photo", config.CLASSES[label]):
                raise ValueError(f"invalid Photo training row: {rel} {label}")
            path = contained(self.image_root, rel)
            if not path.is_file():
                raise FileNotFoundError(f"missing Photo Training image: {path}; set NNDL_DATA_ROOT or --data-root")
            # The multi-style protocol assigns several styles to the same
            # original, and each of them becomes its own job.  The one-style
            # protocol assigns one, and produces one job per original.
            styles = self.pairings.get(rel)
            if styles is None:
                style = self.manifest["pairing"][rel]
                if style not in self.styles:
                    raise ValueError(f"unknown frozen style: {style}")
                styles = [style]
            for style_index, style in enumerate(styles):
                if style not in self.styles:
                    raise ValueError(f"unknown frozen style: {style}")
                jobs.append(GenerationJob(rel, label + 1, style, path,
                                          self.styles[style], style_index))
        return jobs
