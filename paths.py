"""Repository-wide path resolution.

Every member works on a separate machine, so no absolute path may be baked
into the code.  The three locations the project needs are resolved in one
place, in this order:

1. the ``NNDL_*`` environment variable, if it is set;
2. a sensible default relative to the repository root.

The repository root is found by walking upwards from this file until a
directory that contains the ``repo`` marker is reached.  The marker is
``member1_data/`` and ``member4_training_eval/`` -- both are always present,
so the root is unambiguous even if the project is checked out under any name
or at any depth.

Layout assumed under the data root::

    <DATA_ROOT>/PACS/<domain>/<class>/<image>     original PACS images
    <DATA_ROOT>/augmented/<group>/<image>         stylized images
    <DATA_ROOT>/aug_sp/images/<class>/<image>     structure-preserving images

Layout assumed under the runs root::

    <RUNS_ROOT>/<tag>/                             one run directory
    <RUNS_ROOT>/summary_<tag>.json                one result summary
"""
from __future__ import annotations

import os
from pathlib import Path

# Markers that identify the repository root.  At least one must exist in the
# parent directory of the directory that holds this file.
_ROOT_MARKERS = ("member1_data", "member4_training_eval", "member3_structure_preserving")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root (the directory that holds the member folders)."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if candidate.is_dir() and any((candidate / marker).is_dir() for marker in _ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root. Expected a parent directory "
        f"containing one of {_ROOT_MARKERS}. Run this script from inside the "
        "repository instead of copying it elsewhere."
    )


REPO_ROOT = find_repo_root()


def _resolve(env_name: str, default: Path) -> Path:
    """Environment variable wins; otherwise use the default, resolved absolutely."""
    raw = os.environ.get(env_name)
    chosen = Path(raw).expanduser() if raw else default
    return chosen.resolve()


#: Root that contains ``PACS/`` and the generated augmentation folders.
DATA_ROOT = _resolve("NNDL_DATA_ROOT", REPO_ROOT / "data")

#: Root that contains the run directories and the summary JSON files.
RUNS_ROOT = _resolve("NNDL_RUNS_ROOT", REPO_ROOT.parent / "runs")

#: Root of the PACS images themselves.
IMAGE_ROOT = DATA_ROOT / "PACS"

#: Where members 2 and 3 write their generated images.
AUGMENTED_ROOT = DATA_ROOT / "augmented"

#: The one frozen split directory.  Member 1 owns it; everyone else reads it.
SPLIT_DIR = REPO_ROOT / "member4_training_eval" / "project_splits"

#: The frozen style pool (10 external paintings, never from PACS).
STYLE_POOL_DIR = REPO_ROOT / "member4_training_eval" / "style_pool"
STYLE_POOL_IMAGES = STYLE_POOL_DIR / "images"

#: The style-pool manifest that the generators and the checkers read.  It is
#: the frozen ten-painting pool by default; the ``NNDL_STYLE_POOL``
#: environment variable points the whole pipeline at one of the half-pools
#: of the sensitivity experiment, without any code change.
STYLE_POOL_MANIFEST = Path(
    os.environ.get("NNDL_STYLE_POOL")
    or (STYLE_POOL_DIR / "style_pool.json")
).resolve()

#: Where the LaTeX sources and figures live.
REPORT_DIR = REPO_ROOT / "report"


def ensure_data_root() -> Path:
    """Return the data root, or explain how to point the project at a PACS copy."""
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(
            f"No PACS images at {IMAGE_ROOT}.\n"
            "Put the dataset at <repo>/data/PACS, or set the environment "
            "variable NNDL_DATA_ROOT to the folder that contains PACS/:\n"
            "    $env:NNDL_DATA_ROOT = '<path>\\to\\data'"
        )
    return DATA_ROOT


def relative_to_repo(path: Path) -> str:
    """Render a path for display: relative to the repo when possible."""
    path = Path(path)
    for base in (REPO_ROOT, REPO_ROOT.parent):
        try:
            return path.resolve().relative_to(base).as_posix()
        except ValueError:
            continue
    return str(path.resolve())
