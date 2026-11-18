"""Paths and Photo-training manifest helpers for the AdaIN augmentation set."""
import os
from pathlib import Path
import sys

# Allow both `python member2_adain/build_adain_aug.py` and
# `python -m member2_adain.build_adain_aug` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The repository root, so that ``import paths`` works from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths

from pacs_dataset import CLASSES, KNOWN_EXCLUDED

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
AUG_ROOT = Path('data') / 'aug_adain'
STYLE_ALPHAS = (0.25, 0.5, 0.75, 1.0)
# Frozen protocol of the project: the main comparison uses alpha = 0.5.
# The 0.25 / 0.75 / 1.0 strengths are produced only by the explicit alpha sweep.
DEFAULT_ALPHA = 0.5


def project_root():
    """The repository root, not this member's folder.

    Member 2's scripts look for ``<repo>/data`` and ``<repo>/project_splits``.
    Returning ``member2_adain/`` itself made those lookups fail on a fresh
    checkout; the repository root is found by walking upwards from this file.
    """
    return paths.find_repo_root(Path(__file__).resolve())


def resolve_data_root(root, data_root):
    """Return (data root, PACS image root).

    The project-wide layout is ``<data root>/PACS/<domain>/<class>/<image>``,
    which is what members 1, 3 and 4 use.  Member 2's original scripts looked
    for ``<data root>/images/``, so both layouts are accepted and the image
    root is returned alongside the data root.

    ``NNDL_DATA_ROOT`` overrides the default, matching member 4's config.py.
    """
    if data_root is not None:
        chosen = Path(data_root)
    else:
        env_root = os.environ.get('NNDL_DATA_ROOT')
        if env_root:
            chosen = Path(env_root)
        elif (root / 'data' / 'PACS').is_dir():
            chosen = root / 'data'
        elif (root / 'data' / 'images').is_dir():
            chosen = root / 'data'
        elif (root / 'pacs' / 'images').is_dir():
            chosen = root / 'pacs'
        else:
            chosen = root / 'data'
    chosen = chosen.resolve()
    for candidate in (chosen / 'PACS', chosen / 'images'):
        if candidate.is_dir():
            return chosen, candidate.resolve()
    raise FileNotFoundError(
        f'PACS image folder not found under {chosen}. Expected {chosen / "PACS"} '
        f'or {chosen / "images"}. Set NNDL_DATA_ROOT or pass --data-root.'
    )


def resolve_image_root(root, data_root):
    """Return only the PACS image root (see resolve_data_root)."""
    return resolve_data_root(root, data_root)[1]


def resolve_split_file(root, split_file, allow_other_split=False):
    """Locate the split manifest.

    The canonical location is the single copy under
    ``member4_training_eval/project_splits/``.  Member 2 keeps a byte-identical
    copy in its own folder as a fallback, so the scripts still work if the
    canonical folder has not been checked out yet.
    """
    if split_file is not None:
        path = Path(split_file)
    else:
        path = paths.SPLIT_DIR / 'photo_train.txt'
        if not path.is_file():
            fallback = Path(__file__).resolve().parent / 'project_splits' / 'photo_train.txt'
            if fallback.is_file():
                path = fallback
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'Split file not found: {path}. Expected the project splits under '
            f'{paths.SPLIT_DIR}.'
        )
    if path.name != 'photo_train.txt' and not allow_other_split:
        raise ValueError(
            f'Refusing split {path.name}. Member 2 augments photo_train.txt only.'
        )
    return path


def read_split(split_file, image_root):
    """Return (relative posix path, label 1-7, absolute image path) in manifest order."""
    image_root = Path(image_root).resolve()
    samples = []
    seen = set()
    for number, line in enumerate(Path(split_file).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            relative, raw_label = line.rsplit(maxsplit=1)
            label = int(raw_label)
        except ValueError:
            raise ValueError(f'Invalid split row {number}: {line}') from None
        relative = relative.replace('\\', '/').removeprefix('./')
        if relative in KNOWN_EXCLUDED:
            continue
        path = (image_root / relative).resolve()
        if image_root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid image path at row {number}: {relative}')
        if not 1 <= label <= len(CLASSES) or path.parent.name != CLASSES[label - 1]:
            raise ValueError(f'Invalid class label at row {number}: {line}')
        if relative in seen:
            raise ValueError(f'Duplicate sample: {relative}')
        seen.add(relative)
        samples.append((relative, label, path))
    if not samples:
        raise ValueError('The split contains no usable samples.')
    return samples


def resolve_alphas(values, default=None):
    """Return style strengths. None selects the frozen default (alpha = 0.5).

    The project protocol fixes the main comparison at alpha = 0.5; the
    0.25 / 0.75 / 1.0 strengths are produced only by the explicit alpha sweep.
    """
    if values is None:
        values = (DEFAULT_ALPHA,) if default is None else default
    if len(values) == 0:
        raise ValueError(
            'Pass at least one alpha, or omit --alpha to use the frozen default 0.5.'
        )
    alphas = []
    for value in values:
        alpha = round(float(value), 4)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be between 0 and 1, got {value}')
        if any(abs(alpha - earlier) < 1e-8 for earlier in alphas):
            raise ValueError(f'Duplicate alpha: {alpha}')
        alphas.append(alpha)
    return alphas


def alpha_output_name(alpha):
    return f'alpha_{alpha:.2f}'


def output_relative(relative, style_path, style_index, multiple_styles):
    """Manifest path under the augmented images/ folder.

    The name always carries the style stem.  Returning the bare original path
    (as an earlier version did when ``styles_per_image == 1``) would make the
    generated file collide with the original image, and the two augmented
    groups would write into the same directory under the same name.
    """
    source = Path(relative)
    style_stem = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in style_path.stem)[:40]
    if multiple_styles:
        name = f'{source.stem}__{style_index:02d}_{style_stem}.jpg'
    else:
        name = f'{source.stem}__{style_stem}.jpg'
    return (source.parent / name).as_posix()
