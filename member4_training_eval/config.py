"""Central configuration for all training and evaluation runs.

Every quantity that the three experimental groups (Original / +AdaIN / +Ours)
must share lives here.  The config hash computed by ``config_hash()`` is
stored in every result file and checked by ``fairness_check.py`` so that a
run with a mismatched configuration cannot silently enter the results table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torchvision import transforms

# ---------------------------------------------------------------- paths ----
CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
import os
import sys
# The repository-wide path resolver.  It is the single place that knows how
# the data root, the runs root and the split directory are located, so that
# every member can work on a separate machine without hard-coded paths.
sys.path.insert(0, str(PROJECT_ROOT))
import paths

# Shared data root.  Every member resolves it the same way:
#   $NNDL_DATA_ROOT  >  <repo>/data
# The tree under it is
#   <DATA_ROOT>/PACS/<domain>/<class>/<image>   original images
#   <DATA_ROOT>/augmented/<group>/<image>       stylized images
#   <DATA_ROOT>/augmented/manifest_<group>.txt  training manifests
DATA_ROOT = paths.DATA_ROOT
# Shared local layout: <project>/data/PACS. NNDL_DATA_ROOT overrides data/.
IMAGE_ROOT = paths.IMAGE_ROOT
SPLIT_DIR = paths.SPLIT_DIR
# Run artefacts (checkpoints, summaries) are kept outside the repository so
# that no .pt file can ever be committed by accident.
OUTPUT_ROOT = paths.RUNS_ROOT
STYLE_POOL_DIR = paths.STYLE_POOL_DIR
STYLE_POOL_IMAGES = paths.STYLE_POOL_IMAGES
# The manifest of the active style pool.  It is the frozen ten-painting
# pool by default; the ``NNDL_STYLE_POOL`` environment variable points the
# whole pipeline at one of the half-pools of the sensitivity experiment.
STYLE_POOL_MANIFEST = paths.STYLE_POOL_MANIFEST
AUG_ROOT = DATA_ROOT / "augmented"       # stylized images written by members 2/3

# ------------------------------------------------------------- protocol ----
# Member 1's label mapping, 0-based: dog=0, elephant=1, giraffe=2, guitar=3,
# horse=4, house=5, person=6.  verify_splits.py enforces folder/label
# consistency, so an order change here fails loud.
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
# The experimental groups.  ``adain_matched`` is the strength-matched
# uniform AdaIN control: it receives exactly the same average stylization
# strength as the spatial map of ``ours``, so that a comparison between
# ``ours`` and ``adain_matched`` isolates the effect of the *spatial*
# modulation from the effect of the overall style strength.
#
# ``duplicate`` and ``nonstyle`` are the two data-volume controls.  Both
# have exactly the same training-set *shape* as ``adain`` (1,349 originals
# plus 1,349 manifest rows) but differ in what the second copy contains:
# ``duplicate`` is a byte-identical copy, ``nonstyle`` is a generic
# non-stylistic transform.  They separate "twice the rows" from "twice the
# rows with a real appearance change" from "twice the rows with a style".
# --------------------------------------------------------- repaired map ----
# The repaired variant of the spatial modulation.  The blending formula is
# unchanged -- the strength still falls where the modulation map is large --
# but the map is an unsupervised foreground estimate instead of a soft Sobel
# edge response.  The measurement in Section IV.E showed that the Sobel map
# responds to background texture rather than to the object's outline, so
# the original rule put the strongest style exactly on the class cue.
#
# This group is reported separately and never mixed into the main table: it
# is a different method, so it carries its own fingerprint, exactly like
# the MixStyle baseline.
FOREGROUND_GROUP = "ours_fg"
#: The foreground estimator that the repaired variant uses.
FOREGROUND_KIND = "spectral_center"
#: The manifest of the repaired variant, relative to the data root.
FOREGROUND_MANIFEST = "aug_sp/foreground/manifest_ours.txt"

#: The variant that won the map selection: the centre prior on its own.
#:
#: It is a *different* map from :data:`FOREGROUND_KIND`, so it is a
#: different method and gets its own fingerprint, exactly like every
#: other group here.  It is reported in its own section, never in the
#: main table.
FOREGROUND_CENTER_GROUP = "ours_fg_center"
FOREGROUND_CENTER_KIND = "center"
FOREGROUND_CENTER_MANIFEST = "aug_sp/fgcenter/manifest_ours.txt"

#: Every group that is a repaired modulation of the style strength.  Each
#: one is a different map and therefore a different method, with its own
#: fingerprint, manifest and section in the report.
FOREGROUND_GROUPS = (FOREGROUND_GROUP, FOREGROUND_CENTER_GROUP)

GROUPS = ("original", "adain", "adain_matched", "ours", "ours_equal",
          "duplicate", "nonstyle", "mixstyle", FOREGROUND_GROUP,
          FOREGROUND_CENTER_GROUP)
# Groups whose training set is the original Photo images plus one styled
# (or controlled) image per original, taken from that group's manifest.
AUGMENTED_GROUPS = ("adain", "adain_matched", "ours", "ours_equal",
                    "duplicate", "nonstyle")# Groups whose manifest has no style image.  Their manifests carry the
# placeholder ``none`` in the style column, so the fairness checker must not
# compare them against the frozen style pool or the content-style pairing.
NO_STYLE_GROUPS = ("duplicate", "nonstyle")
# The manifest of every augmented group, relative to the data root.  The
# sweep uses this table; ``run_all.py`` can still override it with
# ``--aug-manifest``.
GROUP_MANIFESTS = {
    "adain": "aug_sp/manifest_adain.txt",
    "adain_matched": "aug_sp/manifest_adain_matched.txt",
    "ours": "aug_sp/manifest_ours.txt",
    "ours_equal": "aug_sp/sweeps/sweep_full/a0p669215_b0p7/manifest_ours.txt",
    "duplicate": "augmented/manifest_duplicate.txt",
    "nonstyle": "augmented/manifest_nonstyle.txt",
    FOREGROUND_GROUP: FOREGROUND_MANIFEST,
    FOREGROUND_CENTER_GROUP: FOREGROUND_CENTER_MANIFEST,
}

SPLIT_FILES = {
    "photo_train": SPLIT_DIR / "photo_train.txt",
    "photo_val": SPLIT_DIR / "photo_val.txt",
    "photo_test": SPLIT_DIR / "photo_test.txt",
    "art_painting_test": SPLIT_DIR / "art_painting_test.txt",
    "cartoon_test": SPLIT_DIR / "cartoon_test.txt",
    "sketch_test": SPLIT_DIR / "sketch_test.txt",
}

SEEDS = (0, 1, 2)

# ----------------------------------------------------------- optimization ---
BATCH_SIZE = 64
#: The number of DataLoader workers.  Each worker holds a copy of the
#: training set in its own process, and the stylised images are large, so
#: four workers of a stylised set need about a gigabyte of resident memory
#: each.  The count is therefore an environment variable: on a machine with
#: little RAM the batch can be run with fewer workers, which slows the
#: loader down but keeps the machine alive.
NUM_WORKERS = int(os.environ.get("NNDL_NUM_WORKERS", "4") or 4)
# Same optimizer-step budget for every group: 30 nominal epochs over the
# 1,349-image Original training set at batch size 64 (30 * ceil(1349/64)).
ORIGINAL_EPOCHS = 30
TOTAL_STEPS = 660
# Optional scaled-down protocol for the pipeline smoke test.  When
# ``NNDL_SUBSET`` is set to the number of training images to use, every group
# trains on the same subset and the step budget is scaled by the same factor,
# so the number of passes over the data stays identical for all groups and
# the comparison remains fair.  It is never used for a reported result.
SUBSET_SIZE = int(os.environ.get("NNDL_SUBSET", "0") or 0)
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
EVAL_EVERY = 60            # photo-val evaluation cadence in optimizer steps
# Checkpoint policy.  Photo Validation accuracy saturates at 97-98% within the
# first 60 steps (150 images => 1 image = 0.67%), so selecting the "best
# validation" checkpoint picks noise and more than doubles the seed-to-seed
# spread of cross-domain accuracy (std 7.22 vs 2.71 in our baseline).  We
# therefore use the final checkpoint of the fixed step budget: it introduces no
# selection bias at all and is the protocol reported in Sections 3.3 and 3.7.
CHECKPOINT = "last"        # "last" | "best"
# Whether to keep ``best.pt`` next to ``last.pt``.  The reported checkpoint is
# always the final one of the fixed step budget, so ``best.pt`` is never used
# for selection -- it exists only as a reference.  Each checkpoint is 43 MB,
# and the full experiment sweep produces dozens of runs, so it is off by
# default and can be switched on per run.
#
# This flag is deliberately NOT part of ``config_hash()``: it does not change
# a single number that is reported, only which extra file ends up on disk.
SAVE_BEST_CHECKPOINT = bool(int(os.environ.get("NNDL_SAVE_BEST", "0") or 0))

# ---------------------------------------------------------------- MixStyle ----
# MixStyle is a real domain-generalisation baseline that mixes the instance
# statistics of the feature maps.  It is the closest published method to
# what this project does -- it removes the style of the input -- but it does
# it in feature space instead of by generating images.
#
# It is OFF by default.  Switching it on changes the architecture, and an
# architecture change must never be hidden inside the fingerprint of the
# other groups: the MixStyle runs therefore carry the group name
# ``mixstyle`` and a *separate* hash, computed by ``mixstyle_config_hash()``
# below.  The main ``config_hash()`` stays ``ecc6090e5b29`` for every other
# group, so the reported results keep their comparability.
MIXSTYLE = bool(int(os.environ.get("NNDL_MIXSTYLE", "0") or 0))
MIXSTYLE_P = float(os.environ.get("NNDL_MIXSTYLE_P", "0.5"))
MIXSTYLE_ALPHA = float(os.environ.get("NNDL_MIXSTYLE_ALPHA", "0.1"))
#: The group that carries the MixStyle baseline.
MIXSTYLE_GROUP = "mixstyle"
# Human-readable description of the checkpoint policy, recorded in every run
# summary.  It must agree with CHECKPOINT: "last" means the final checkpoint
# of the fixed step budget, with no validation-based selection.
CHECKPOINT_SELECTION = "final checkpoint of the fixed step budget"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ aug ----
# One stylized image per original training image, for both augmented groups.
AUG_PER_IMAGE = 1
# Size of the frozen external style pool.  The pool holds paintings that are
# NOT part of PACS (see style_pool/SOURCES.txt); PACS Art Painting, Cartoon
# and Sketch are evaluation domains and must never be used as style sources.
N_STYLE_IMAGES = 10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Geometric augmentation only: photometric augmentation is intentionally
# excluded because style statistics are the variable under study.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def read_split(path: Path) -> list[tuple[Path, int]]:
    """Parse a member-1 split manifest: ``relative/path label_1_to_7``."""
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rel, raw = line.rsplit(maxsplit=1)
        rows.append((rel.replace("\\", "/").removeprefix("./"), int(raw) - 1))
    return rows


def balanced_train_subset(size: int) -> list[str]:
    """The first ``size`` Photo Training paths, round-robin over the classes.

    ``photo_train.txt`` is ordered by class, so the first N rows are all of
    one class (the first 153 rows are all ``dog``).  A subset taken that way
    would train a single-class classifier and score at chance on the
    validation set, which says nothing about the method under test.  This
    helper is the single source of truth for the reduced protocol: members 2
    and 3 generate exactly these images and member 4 trains on exactly these
    images, so all three groups see the same content.
    """
    if size <= 0:
        return []
    rows = read_split(SPLIT_FILES["photo_train"])
    buckets = [[rel for rel, label in rows if label == index]
               for index in range(len(CLASSES))]
    chosen = []
    position = 0
    while len(chosen) < size:
        added = False
        for bucket in buckets:
            if position < len(bucket):
                chosen.append(bucket[position])
                added = True
                if len(chosen) == size:
                    break
        if not added:
            break
        position += 1
    return chosen


def training_config() -> dict:
    """The fairness-relevant subset of the configuration.

    ``checkpoint_selection`` is documentation of the policy, not a behaviour:
    it is recorded in every run summary but excluded from ``config_hash()`` so
    that rewording it never invalidates runs that are already on disk.
    """
    return {
        "batch_size": BATCH_SIZE,
        "total_steps": TOTAL_STEPS,
        "lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "eval_every": EVAL_EVERY,
        "checkpoint": CHECKPOINT,
        "train_transform": repr(TRAIN_TRANSFORM),
        "eval_transform": repr(EVAL_TRANSFORM),
    "arch": "resnet18",
    "pretrained": "imagenet",
    "checkpoint_selection": CHECKPOINT_SELECTION,
    "subset_size": SUBSET_SIZE,
}


# The fields that decide what the trained model is.  This is an explicit
# allow-list rather than "everything except", so that adding a descriptive
# field to training_config() can never silently change the fingerprint.
#
# ``checkpoint`` and ``checkpoint_selection`` are deliberately NOT in it:
# they describe *which saved checkpoint we report*, not how the model is
# trained, and the choice is applied identically to every group by the same
# code path.  They are still recorded in every run summary, so the reported
# policy stays auditable.
_BEHAVIOURAL_FIELDS = (
    "batch_size",
    "total_steps",
    "lr",
    "momentum",
    "weight_decay",
    "label_smoothing",
    "eval_every",
    "train_transform",
    "eval_transform",
    "arch",
    "pretrained",
)

# ``subset_size`` is not part of the frozen protocol: it is only set for the
# reduced smoke-test protocol, where it changes both the data and the step
# budget.  It is folded into the fingerprint *only* when it is actually in
# effect, so that the reported baseline runs keep their original hash and stay
# comparable with the new runs.
_SUBSET_FIELD = "subset_size"


def config_hash_of(training_config_dict: dict) -> str:
    """Canonical fingerprint of a recorded training configuration.

    Computed from the allow-list above, so it can be recomputed from the
    ``training_config`` block stored inside any run summary -- including runs
    recorded before this definition existed -- and compared with the current
    one.

    The reduced smoke-test protocol (``subset_size``) is included only when it
    is non-zero, so that switching the subset on or off changes the
    fingerprint: a subset run must never be compared with a full run.

    The ``config_hash`` string stored in a result file is a *snapshot* of this
    value at write time.  Earlier versions of this module used a different
    field set, so some archived files carry a stale string.  The stored
    ``training_config`` block is authoritative: recomputing from it always
    yields the canonical hash, which is why ``fairness_check.py`` recomputes
    instead of trusting the stored string.
    """
    behavioural = {k: training_config_dict.get(k) for k in _BEHAVIOURAL_FIELDS}
    subset = training_config_dict.get(_SUBSET_FIELD) or 0
    if subset:
        behavioural[_SUBSET_FIELD] = subset
    payload = json.dumps(behavioural, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def config_hash() -> str:
    """Fingerprint of the behaviourally relevant configuration.

    Two runs may be compared only if this matches.
    """
    return config_hash_of(training_config())


def mixstyle_config_hash() -> str:
    """The fingerprint of a MixStyle run.

    The MixStyle baseline changes the architecture, so it must not share
    the fingerprint of the other groups: a table that put them side by side
    would compare two different models under one protocol.  The hash is
    derived from the main one plus the MixStyle parameters, so the two are
    visibly related but cannot be confused.
    """
    payload = {
        "base": config_hash(),
        "mixstyle": True,
        "mixstyle_p": MIXSTYLE_P,
        "mixstyle_alpha": MIXSTYLE_ALPHA,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def foreground_config_hash() -> str:
    """The fingerprint of a repaired-modulation run.

    The repaired variant changes the *augmentation*, not the training
    protocol: the network, the optimiser and the step budget are the same
    as for every other group.  It nevertheless gets its own fingerprint,
    because it is a different method and must never be pooled with the
    main groups.  The hash records which foreground estimator was used, so
    two different repairs cannot be confused either.
    """
    payload = {
        "base": config_hash(),
        "modulation": "foreground",
        "foreground_kind": FOREGROUND_KIND,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def foreground_center_config_hash() -> str:
    """The fingerprint of the centre-prior repair.

    The map selection chose the centre prior over the spectral residual
    on the mechanism criterion, by a factor of thirty.  It is a
    different map and therefore a different method, so it carries its
    own fingerprint: pooling it with the spectral variant would hide
    exactly the comparison that decided the choice.
    """
    payload = {
        "base": config_hash(),
        "modulation": "foreground",
        "foreground_kind": FOREGROUND_CENTER_KIND,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def config_hash_for_group(group: str) -> str:
    """The fingerprint that applies to a group."""
    if group == MIXSTYLE_GROUP:
        return mixstyle_config_hash()
    if group == FOREGROUND_GROUP:
        return foreground_config_hash()
    if group == FOREGROUND_CENTER_GROUP:
        return foreground_center_config_hash()
    return config_hash()


def canonical_hash_of_run(run_dir: Path) -> None | str:
    """Recompute the canonical hash of a run directory, or None if unknown.

    The fingerprint depends on the *group*: the MixStyle baseline changes
    the architecture and therefore has its own fingerprint.  The group is
    read from the recorded summary, so a MixStyle run is compared against
    the MixStyle hash and every other run against the main one.
    """
    summary = Path(run_dir) / "train_summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text(encoding="utf-8"))
    group = payload.get("group")
    recorded = payload.get("training_config")
    if isinstance(recorded, dict):
        return config_hash_of(recorded)
    if group is not None:
        return config.config_hash_for_group(group)
    return payload.get("config_hash")
