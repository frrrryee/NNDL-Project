"""Central configuration for all training and evaluation runs.

Every quantity that the three experimental groups (Original / +AdaIN / +Ours)
must share lives here.  The config hash computed by ``config_hash()`` is
stored in every result file and checked by ``fairness_check.py`` so that a
run with a mismatched configuration cannot silently enter the results table.
"""
from __future__ import annotations

import hashlib
import json
import math
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

#: The matched control *of* the centre prior.
#:
#: It is the same pipeline with a uniform map whose strength is the mean
#: of the centre-prior alpha field, so comparing against it isolates what
#: the spatial modulation contributes and nothing else.  It is required
#: rather than optional: the centre prior covers 34% of the pixels, so
#: its alpha field (0.3068) is well below the Sobel one (0.3736) and a
#: comparison across groups would otherwise be confounded with the
#: amount of style applied.
FOREGROUND_CENTER_MATCHED_GROUP = "ours_fg_center_matched"
FOREGROUND_CENTER_MATCHED_MANIFEST = \
    "aug_sp/fgcenter/manifest_adain_matched.txt"

# ----------------------------------------------------------- line art ----
#: The line-art family: four units built by ``build_line_art_sets.py``.
#:
#: Two are the method and two are its controls.  The method applies the
#: attention gate to the line width, so the strokes on the object are
#: strong and the strokes on the background are faint.  The controls
#: carry the *same ink* but distribute it differently, and the two of
#: them differ in which quantity they hold fixed:
#:
#:   ``line_attn``           the attention gate
#:   ``line_attn_vote``      the attention gate plus the superpixel soft vote
#:   ``line_plain_total``    uniform strokes, total ink matched to the gate
#:   ``line_plain_foreground`` uniform strokes, foreground ink matched to the gate
#:
#: The two controls are matched against *different* members of the pair,
#: and that is the point: matching the total ink isolates where the ink
#: sits, and matching the foreground ink isolates how much of the ink
#: falls on the object.  Each control therefore answers a different
#: question, and together they separate "the gate moves the ink onto the
#: object" from "the gate merely adds ink".
LINE_ART_UNITS = ("line_attn", "line_attn_vote",
                  "line_plain_total", "line_plain_foreground")

#: The group name of each line-art unit, with its manifest.
LINE_ART_GROUPS = {
    "ours_line_attn": "aug_sp/lineart/line_attn/manifest_ours.txt",
    "ours_line_vote": "aug_sp/lineart/line_attn_vote/manifest_ours.txt",
    "ours_line_plain_total": "aug_sp/lineart/line_plain_total/manifest_ours.txt",
    "ours_line_plain_fg": "aug_sp/lineart/line_plain_foreground/manifest_ours.txt",
}

#: Every group that is a repaired modulation of the style strength.  Each
#: one is a different map and therefore a different method, with its own
#: fingerprint, manifest and section in the report.
FOREGROUND_GROUPS = (FOREGROUND_GROUP, FOREGROUND_CENTER_GROUP)

GROUPS = ("original", "adain", "adain_matched", "ours", "ours_equal",
          "duplicate", "nonstyle", "mixstyle", FOREGROUND_GROUP,
          FOREGROUND_CENTER_GROUP, FOREGROUND_CENTER_MATCHED_GROUP,
          *LINE_ART_GROUPS)
# Groups whose training set is the original Photo images plus one styled
# (or controlled) image per original, taken from that group's manifest.
AUGMENTED_GROUPS = ("adain", "adain_matched", "ours", "ours_equal",
                    "duplicate", "nonstyle", FOREGROUND_CENTER_MATCHED_GROUP,
                    *LINE_ART_GROUPS)# Groups whose manifest has no style image.  Their manifests carry the
# placeholder ``none`` in the style column, so the fairness checker must not
# compare them against the frozen style pool or the content-style pairing.
NO_STYLE_GROUPS = ("duplicate", "nonstyle", *LINE_ART_GROUPS)
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
    FOREGROUND_CENTER_MATCHED_GROUP: FOREGROUND_CENTER_MATCHED_MANIFEST,
    **LINE_ART_GROUPS,
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
#: The number of passes over the training set that every group gets.
#:
#: The step budget is *derived* from this number and from the size of the
#: group's own training set, so that every group is trained for the same
#: number of epochs instead of the same number of optimizer steps.
#:
#: It used to be the other way round: the budget was a fixed 660 steps
#: (30 * ceil(1349/64), i.e. 30 epochs over the Original training set), and
#: every group received exactly those 660 steps whatever its dataset size.
#: The Original group has 1,349 images and therefore saw 31.3 epochs, while
#: every augmented group has 2,698 images (the originals plus one styled
#: image each) and therefore saw only 15.7 epochs.  The augmented groups
#: were therefore under-trained by half, and the comparison between them
#: and the Original one was not a comparison of methods but of methods *and*
#: training budgets.
#:
#: The duplicated group is the clean proof that this was a real handicap
#: and not a theoretical one: it holds the *original* images and their
#: byte-identical copies, so nothing about stylization can explain a loss,
#: yet it scored 3.14 points below the Original group (37.69% vs 40.83%).
ORIGINAL_EPOCHS = 30
#: The step budget of the Original group, kept for the runs that are
#: already on disk: 30 epochs over 1,349 images at batch size 64 is
#: ``30 * ceil(1349/64) = 660`` steps.  It is the *reference* the
#: epoch-based budget is checked against, not a constant that is handed out
#: to every group.
TOTAL_STEPS = 660

def total_steps_for(n_images: int) -> int:
    """The step budget of a group with ``n_images`` training images.

    The budget is the number of epochs every group is entitled to, turned
    into optimizer steps for that group's own dataset size::

        total_steps = ORIGINAL_EPOCHS * ceil(n_images / BATCH_SIZE)

    With this rule every group performs the same number of *passes over
    its own data*, which is the property the comparison between the
    groups needs.  The Original group (1,349 images) still gets exactly
    660 steps, so its budget -- and therefore its fingerprint and every
    result already on disk -- is unchanged.  An augmented group (2,698
    images) now gets 1,290 steps instead of 660, which is the half of its
    training that it was missing.

    The epoch count is the one that is held fixed, not the step count, and
    the rounding is up to a whole batch, so the last epoch of every group
    is complete.  This is what makes the duplicated group a meaningful
    control again: it can no longer lose to the Original group for
    reasons that have nothing to do with its content.

    The budget of the Original group is 30 * ceil(1349/64) = 660 steps and
    the budget of an augmented group (2,698 images) is 30 * ceil(2698/64) =
    1,290 steps.  The Original budget is unchanged by this rule, so every
    result already recorded for it stays valid; the augmented groups get
    the half of their training that they were missing.
    """
    if n_images <= 0:
        return 0
    steps_per_epoch = math.ceil(n_images / BATCH_SIZE)
    return ORIGINAL_EPOCHS * steps_per_epoch


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
DEVICE = os.environ.get("NNDL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

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


def train_images_of_group(group: str, manifest: Path | None = None) -> int:
    """The number of training images of a group.

    The Original group and the MixStyle baseline train on the Photo
    Training split alone.  Every other group trains on that split *plus*
    one row per manifest entry, so its training set is twice as large.

    The count is computed from the manifest rather than assumed, so that a
    group whose manifest has a different number of rows than the split --
    a subset protocol, or a generator that was run with a different
    setting -- gets the budget that matches the data it actually has.

    A manifest that cannot be read is a hard error rather than a guess:
    the budget must follow the data, and a budget guessed from a missing
    manifest is exactly the kind of silent mismatch this protocol is
    meant to prevent.
    """
    originals = len(read_split(SPLIT_FILES["photo_train"]))
    if group in ("original", MIXSTYLE_GROUP):
        return originals
    if group not in AUGMENTED_GROUPS and group not in FOREGROUND_GROUPS:
        raise ValueError(f"unknown group: {group}")
    if manifest is None:
        raise ValueError(f"group '{group}' requires an augmentation manifest")
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"group '{group}': the manifest {manifest} does not exist, so the "
            f"size of its training set -- and therefore its step budget -- "
            f"cannot be established")
    rows = 0
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows += 1
    return originals + rows


def total_steps_of_group(group: str, manifest: Path | None = None) -> int:
    """The step budget of a group under the epoch-based protocol.

    Every group gets ``ORIGINAL_EPOCHS`` passes over *its own* training
    set.  The Original group therefore keeps its 660 steps and its
    fingerprint, and an augmented group gets 1,290 steps for its 2,698
    images instead of the 660 that left half of its data unseen.

    A subset run keeps the reduced budget it was given: the subset is a
    separate protocol with its own fingerprint, and its budget is scaled
    by the same factor as the data, so the number of passes over the data
    stays the same there too.
    """
    images = train_images_of_group(group, manifest)
    if SUBSET_SIZE:
        factor = SUBSET_SIZE / len(read_split(SPLIT_FILES["photo_train"]))
        return max(1, round(TOTAL_STEPS * factor))
    return total_steps_for(images)


def training_config(group: str | None = None,
                    manifest: Path | None = None) -> dict:
    """The fairness-relevant subset of the configuration.

    ``checkpoint_selection`` is documentation of the policy, not a behaviour:
    it is recorded in every run summary but excluded from ``config_hash()`` so
    that rewording it never invalidates runs that are already on disk.

    ``group`` selects the step budget.  Under the epoch-based protocol the
    budget is a property of the *data* and not a constant, so the training
    block of a group records the budget that group actually received.
    Without a group the Original reference budget (660 steps) is used,
    which is what the recorded baseline runs carry.
    """
    if group is None:
        total_steps = TOTAL_STEPS
    else:
        total_steps = total_steps_of_group(group, manifest)
    return {
        "batch_size": BATCH_SIZE,
        "total_steps": total_steps,
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


def config_hash(group: str | None = None,
                manifest: Path | None = None) -> str:
    """Fingerprint of the behaviourally relevant configuration.

    Two runs may be compared only if this matches.

    ``group`` selects the step budget that goes into the fingerprint.
    Under the epoch-based protocol the budget is a property of the data,
    so two groups with the same hyper-parameters but a different number of
    training images have *different* fingerprints -- which is correct,
    because they were trained with a different budget.  The Original group
    keeps its historical fingerprint, because its budget is unchanged.
    """
    return config_hash_of(training_config(group, manifest))


def mixstyle_config_hash() -> str:
    """The fingerprint of a MixStyle run.

    The MixStyle baseline changes the architecture, so it must not share
    the fingerprint of the other groups: a table that put them side by side
    would compare two different models under one protocol.  The hash is
    derived from the main one plus the MixStyle parameters, so the two are
    visibly related but cannot be confused.

    The MixStyle baseline trains on the Photo Training split alone, so its
    budget is the Original one (660 steps) and its base hash is unchanged.
    """
    payload = {
        "base": config_hash(MIXSTYLE_GROUP),
        "mixstyle": True,
        "mixstyle_p": MIXSTYLE_P,
        "mixstyle_alpha": MIXSTYLE_ALPHA,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def foreground_config_hash(kind: str | None = None,
                           group: str | None = None,
                           manifest: Path | None = None) -> str:
    """The fingerprint of a repaired-modulation run.

    The repaired variant changes the *augmentation*, not the training
    protocol: the network, the optimiser and the step budget are the same
    as for every other group.  It nevertheless gets its own fingerprint,
    because it is a different method and must never be pooled with the
    main groups.  The hash records which foreground estimator was used, so
    two different repairs cannot be confused either.

    The estimator must be part of the fingerprint, and not merely the
    group name: the map selection trains several candidates *under one
    group name*, and a fingerprint that ignored the map would make all of
    them indistinguishable.  That is exactly the comparison the selection
    turns on, so it must not be hidden.
    """
    payload = {
        "base": config_hash(group, manifest),
        "modulation": "foreground",
        "foreground_kind": kind or FOREGROUND_KIND,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def foreground_center_config_hash(group: str | None = None,
                                  manifest: Path | None = None) -> str:
    """The fingerprint of the centre-prior repair.

    The map selection chose the centre prior over the spectral residual
    on the mechanism criterion, by a factor of thirty.  It is a
    different map and therefore a different method, so it carries its
   own fingerprint: pooling it with the spectral variant would hide
   exactly the comparison that decided the choice.
    """
    payload = {
        "base": config_hash(group, manifest),
        "modulation": "foreground",
        "foreground_kind": FOREGROUND_CENTER_KIND,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def foreground_center_matched_config_hash(group: str | None = None,
                                          manifest: Path | None = None) -> str:
    """The fingerprint of the matched control *of* the centre prior.

    It is a uniform map, so the map alone does not distinguish it from
    the centre prior it controls -- and it must not be given the main
    fingerprint either, or it would be indistinguishable from the Sobel
    group.  The record therefore names both the map it belongs to and
    the fact that the map was flattened.
    """
    payload = {
        "base": config_hash(group, manifest),
        "modulation": "foreground",
        "foreground_kind": FOREGROUND_CENTER_KIND,
        "uniform": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def line_art_config_hash(unit: str, manifest: Path | None = None,
                          group: str | None = None) -> str:
    """The fingerprint of one line-art unit.

    The line-art family is a different method from the AdaIN style
    transfer: the images are line drawings rather than stylised
    photographs, so a line-art run must never be pooled with the main
    groups.  The unit is part of the fingerprint because the four units
    differ in exactly the thing the experiment turns on -- where the
    ink is and how much of it there is.

    The *data* is part of the fingerprint as well.  The line-art units
    were rebuilt after the matched controls were found to be degenerate,
    and a run that trained on the old images must not be reused for the
    new ones: the fingerprint of the training protocol alone would be
    identical, and the reuse check would happily keep the stale result.
    The data fingerprint is the hash of the manifest, which names every
    image the run saw.
    """
    payload = {
        "base": config_hash(group, manifest),
        "method": "line-art",
        "unit": unit,
        "data": manifest_fingerprint(manifest),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def manifest_fingerprint(manifest: Path | None) -> str | None:
    """The hash of a manifest's contents.

    The manifest names every image the run trains on, so its hash
    changes whenever the data does.  The file's own modification time
    is deliberately not used: it changes when a file is rewritten with
    the same content, which would invalidate every run for no reason.
    """
    if manifest is None:
        return None
    manifest = Path(manifest)
    if not manifest.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(manifest.read_bytes())
    return digest.hexdigest()[:12]


def line_art_unit_of_group(group: str) -> str | None:
    """The line-art unit that a group name stands for.

    The mapping is by unit name, so the group name and the unit it
    refers to cannot drift apart.
    """
    units = {
        "ours_line_attn": "line_attn",
        "ours_line_vote": "line_attn_vote",
        "ours_line_plain_total": "line_plain_total",
        "ours_line_plain_fg": "line_plain_foreground",
    }
    return units.get(group)


def foreground_kind_of_manifest(manifest: Path | None) -> str | None:
    """The foreground estimator that a manifest was built with.

    The map selection trains several candidates under one group name, and
    the candidates differ only in the estimator that produced their
    images.  The estimator is recorded in the generator's own
    ``generation.json`` next to the manifest, so the map can be recovered
    from the manifest alone.

    Returns ``None`` when the map cannot be determined -- a manifest
    without a generator record is not a foreground dataset, and guessing
    would be worse than admitting the uncertainty.
    """
    if manifest is None:
        return None
    manifest = Path(manifest)
    for directory in (manifest.parent, manifest.parent.parent):
        record = directory / "generation.json"
        if not record.exists():
            continue
        try:
            settings = json.loads(record.read_text(encoding="utf-8")).get("settings", {})
        except (OSError, ValueError):
            continue
        kind = settings.get("foreground_kind")
        if kind:
            return kind
    return None


def config_hash_for_group(group: str, manifest: Path | None = None) -> str:
    """The fingerprint that applies to a group.

    For a repaired modulation the fingerprint depends on the *map*, and
    the map is read from the manifest that the run actually uses.  The
    group name alone is not enough: the map selection trains several
    candidates under one group name, and a fingerprint that ignored the
    map would make them indistinguishable -- hiding precisely the
    comparison the selection turns on.

    The group's *budget* is part of the fingerprint as well.  Under the
    epoch-based protocol the step budget follows the size of the group's
    own training set, so the fingerprint must carry it: two runs that
    differ in their budget are not comparable, and the fingerprint is what
    says so.
    """
    if group == MIXSTYLE_GROUP:
        return mixstyle_config_hash()
    if group == FOREGROUND_CENTER_MATCHED_GROUP:
        return foreground_center_matched_config_hash(group, manifest)
    unit = line_art_unit_of_group(group)
    if unit is not None:
        return line_art_config_hash(unit, manifest, group)
    if group in FOREGROUND_GROUPS:
        kind = foreground_kind_of_manifest(manifest)
        if kind is None:
            # The map could not be read from the manifest.  Falling back
            # to the group's declared kind would silently claim a map
            # the run did not use, so the run is refused instead: a
            # fingerprint that cannot be established must not be
            # invented.
            raise ValueError(
                f"group '{group}': the foreground estimator of manifest "
                f"'{manifest}' could not be read, so no fingerprint can be "
                f"established; pass the map explicitly")
        return foreground_config_hash(kind, group, manifest)
    return config_hash(group, manifest)


def run_fingerprint(run_dir: Path) -> None | str:
    """The fingerprint a run recorded for itself.

    This is the string the run wrote into its own summary, which is the
    fingerprint that includes everything that distinguishes the method --
    the map, for a repaired modulation.  It is the value that a reuse
    check must compare, because it answers "was this run produced by the
    method I am about to run?".
    """
    summary = Path(run_dir) / "train_summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("config_hash")


def run_matches_protocol(run_dir: Path, group: str,
                         manifest: Path | None = None) -> bool:
    """Whether a run was trained under the protocol that is current now.

    This is the *protocol* half of the reuse check: the hyper-parameters
    and the step budget.  Two things are checked, and the first one is
    self-contained so that it cannot be skipped by a missing manifest:

    1. the *epoch invariant*.  The protocol is "30 passes over the group's
       own training set", so a run is only of the current protocol if its
       recorded budget is the budget that its own recorded data size
       implies.  A run of the old fixed-budget protocol fails this: 660
       steps over 2,698 images is 15.7 epochs, not 30.

    2. the *hyper-parameters*.  The recorded training block is
       authoritative, so its canonical hash is compared with the canonical
       hash of the training block the group gets now.  This needs the
       manifest, because the budget is part of the block.

    A run whose group cannot be identified is judged on the epoch
    invariant alone, which is the part that decides whether the run has to
    be retrained.
    """
    summary = Path(run_dir) / "train_summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    recorded = payload.get("training_config")
    if not isinstance(recorded, dict):
        return True

    # The reduced smoke-test protocol has a budget of its own, so it is
    # not judged against the epoch invariant.
    subset = recorded.get("subset_size") or 0
    if subset:
        expected = training_config(group, manifest)
        return config_hash_of(recorded) == config_hash_of(expected)

    # 1. the epoch invariant, computed from the run's own record
    images = payload.get("train_images")
    steps = recorded.get("total_steps")
    if images and steps:
        if steps != total_steps_for(int(images)):
            return False

    # 2. the hyper-parameters
    try:
        expected = training_config(group, manifest)
    except (OSError, ValueError):
        # The manifest is missing, so the budget of the group cannot be
        # established.  The epoch invariant above has already decided
        # whether the run has to be retrained.
        return True
    return config_hash_of(recorded) == config_hash_of(expected)


def canonical_hash_of_run(run_dir: Path) -> None | str:
    """Recompute the canonical hash of a run directory, or None if unknown.

    The fingerprint depends on the *group*: the MixStyle baseline changes
    the architecture and therefore has its own fingerprint.  The group is
    read from the recorded summary, so a MixStyle run is compared against
    the MixStyle hash and every other run against the main one.

    A repaired modulation depends on the *map* as well, and the map is
    read from the manifest the run recorded.  A run whose manifest is
    missing cannot have its map established, so no fingerprint is
    returned for it rather than a guessed one.
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
        manifest = payload.get("aug_manifest")
        if group == FOREGROUND_CENTER_MATCHED_GROUP:
            return foreground_center_matched_config_hash(group, manifest)
        unit = line_art_unit_of_group(group)
        if unit is not None:
            return line_art_config_hash(unit, manifest, group)
        if group in FOREGROUND_GROUPS:
            kind = foreground_kind_of_manifest(
                Path(manifest) if manifest else None)
            if kind is None:
                return None
            return foreground_config_hash(kind, group, manifest)
        return config_hash_for_group(group)
    return payload.get("config_hash")


def snapshot_is_consistent(stored: str | None,
                           recorded_config: dict | None) -> bool:
    """Whether a stored fingerprint snapshot matches the run's own record.

    The ``config_hash`` string written into a run summary is a *snapshot* of
    the hash definition that was in force when the run was written.  Two
    definitions have been in force over the project's history:

    * the current one, which hashes an explicit allow-list of the fields
      that decide what the trained model is, and

    * an earlier one, which hashed the recorded ``training_config`` block
      in its entirety -- including descriptive fields such as
      ``checkpoint_selection`` that do not change the training.

    A run recorded under the earlier definition therefore carries a
    different string even when its training behaviour is identical to
    today's.  Rejecting it would retire a valid baseline for a
    bookkeeping reason.

    The snapshot is accepted when it is the hash of the run's own recorded
    training block under *either* definition.  Both hashes are recomputed
    from the recorded block, so nothing is taken on trust: a snapshot that
    matches neither describes a different training and is refused.

    The recorded block must itself be the one of the current protocol for
    the *caller* to accept the run; this function only answers whether the
    snapshot and the record agree with each other.
    """
    if stored is None:
        return False
    if not isinstance(recorded_config, dict):
        # Nothing to recompute from: the snapshot cannot be checked.
        return False
    if stored == config_hash_of(recorded_config):
        return True
    # The earlier definition: the whole recorded block, verbatim.
    payload = json.dumps(recorded_config, sort_keys=True).encode()
    return stored == hashlib.sha256(payload).hexdigest()[:12]
