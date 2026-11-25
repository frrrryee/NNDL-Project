# Structure-Preserving Style Augmentation for Cross-Domain Image Classification (PACS)

NNDL 课程小组项目仓库。Topic: single-source (Photo-only) cross-domain image
classification on PACS with structure-preserving style augmentation.

## Member 3 implementation

Member 3 uses member 2's AdaIN core from commit
`416bdc9811d67aae80fd79e8fac1ae0f37d5f339`. See
[`member3_structure_preserving/README.md`](member3_structure_preserving/README.md)
for setup and [`member2_adain/IMPORT_NOTES.md`](member2_adain/IMPORT_NOTES.md)
for the imported files and their provenance.

Member 3 and member 4 share `data/PACS/` for original images. Member 3 writes
its deliverable to `data/aug_sp/` and exports a member-4-compatible manifest
at `data/augmented/manifest_ours.txt`, pointing to the same generated images.
Set `NNDL_DATA_ROOT` to use another data directory. The dataset itself is not
included in the repository.

Member 3's checked-in [results](member3_structure_preserving/results/README.md)
contain generation manifests, audit records, structural metrics, and the beta
pilot. Seven real PACS comparison panels are in `member3_structure_preserving/figures/`.

The [one-time handoff to member 5](member3_structure_preserving/handoff_member5/README.md)
collects report drafts for 2.4, 3.5, 4.3 and 5.3, formulas, standalone figures,
the frozen interface contract, BibTeX, evidence, and a compiled PDF preview.

## Working journals

`docs/` holds the dated working journals. They record what was tried, what the
numbers were, and -- just as importantly -- which directions were ruled out
and why, so that the next day does not repeat them.

| Journal | Contents |
|---|---|
| [`docs/journal_2026-11-25_photo_val_gate_softvote_and_what_next.md`](docs/journal_2026-11-25_photo_val_gate_softvote_and_what_next.md) | The Photo Validation safety gate and the mechanism ranking; the implementation of the attention gate and the superpixel soft vote; the full result table and an analysis of which directions remain open |
| [`docs/line_art_attention_project.md`](docs/line_art_attention_project.md) | The line-art proposal: motivation, literature, design, and its falsifiable prediction |
| [`docs/incident_2026-11-24_center_overwrite.md`](docs/incident_2026-11-24_center_overwrite.md) | Incident: the foreground build overwrote the original method's dataset |
| [`docs/incident_2026-11-24_lineart_controls.md`](docs/incident_2026-11-24_lineart_controls.md) | Incident: three bugs that degenerated the line-art controls |

## Repository layout — one folder per member

| Folder | Owner | Contents |
|---|---|---|
| `member1_data/` | Member 1 (data) | PACS splits (`project_splits/`), `PACSDataset` loader, data README |
| `member2_adain/` | Member 2 | AdaIN generator, style-pool reader, sample generation |
| `member3_structure_preserving/` | Member 3 | Structure-preserving stylization, structure map, β experiments |
| `member4_training_eval/` | Member 4 | Unified training/evaluation/fairness code, frozen style pool, run results |
| `report/` | Member 5 (integration) | LaTeX progress report, section drafts, figures |

## Where things are (member 4's part)

- **Shared experiment protocol**: `member4_training_eval/config.py`
  (step budget 660, batch 64, SGD, cosine, label smoothing, geometric-only
  augmentation, seeds {0,1,2}, final-checkpoint reporting).
- **Frozen style pool**: `member4_training_eval/style_pool/`
  (10 external paintings + sha256 manifest + frozen content→style pairing,
  pool_hash `72cdbadb8970`). Verified byte-disjoint from PACS — Art/Cartoon/
  Sketch are test domains and must never supply style images.
- **Training / evaluation**: `train.py`, `run_all.py`, `evaluate.py`,
  `summarize_results.py`.
- **Fairness checks**: `fairness_check.py` (manifest constraints, style-pool
  leak guard, config-hash verification), `verify_splits.py`.
- **Generator interface** (for members 2/3): `GENERATOR_INTERFACE.md`.
- **Results so far**: `results/` at the repository root — the JSON summaries
  of every reported run, the paired statistics and the structural metrics.
  Checkpoints `.pt` are 43 MB each and are not committed; rerun
  `run_all.py --group original` to reproduce. See `results/README.md`.

## Baseline result (experiment 1, complete)

Original group, 3 seeds, 660 steps, final checkpoint:

| | Photo | Art | Cartoon | Sketch | Avg |
|---|---|---|---|---|---|
| Original | 99.0±0.3 | 63.2±3.7 | 25.1±1.8 | 31.5±4.2 | **39.9±2.7** |

## Environment

Every script runs through `nn.ps1`, which picks a usable interpreter for you:

```powershell
.\nn.ps1 -c "import torch, torchvision; print(torch.__version__)"
```

Resolution order: `$env:NNDL_PYTHON` → `.\.venv\Scripts\python.exe` →
conda env `$env:NNDL_ENV` (default `NN`) → `~\anaconda3` → any conda env under
`~\.conda\envs` → `python` on `PATH`.

An interpreter is only accepted if it has **torch and torchvision** *and* runs a
real CUDA convolution on the local GPU. The last condition matters: an old
`cu113` build reports `torch.cuda.is_available() == True` on a newer card and
then dies at the first kernel launch, so the probe executes a convolution
rather than reading the version string. The chosen interpreter is cached in
`.nndl_python` (git-ignored).

The environment the reported baseline was produced in is
`~\anaconda3\python.exe`: torch 2.11.0+cu128, torchvision 0.26.0, with
`lpips` and `opencv-python` installed.

> **Note for members 2 and 3:** `torchvision < 0.13` is not usable here —
> `member4_training_eval/model.py` uses the modern
> `models.resnet18(weights=...)` API, which older torchvision does not have.

## Paths

All paths are resolved in one place, `repo/paths.py`. Every member works on a
separate machine, so **no absolute path is stored in any script, config, or
manifest**. Three optional environment variables control the layout:

| Variable | Meaning | Default |
|---|---|---|
| `NNDL_DATA_ROOT` | the folder containing `PACS/` | `<repo>/data` |
| `NNDL_RUNS_ROOT` | the folder for run artefacts | `<repo>/../runs` |
| `NNDL_PYTHON` | the Python interpreter | probed by `nn.ps1` |

Set them per shell, or leave them alone and put the data at `<repo>/data/PACS`:

```powershell
$env:NNDL_DATA_ROOT = 'D:\datasets\pacs\data'
$env:NNDL_RUNS_ROOT = 'D:\nndl-runs'
```

`repo/ENVIRONMENT.md` has the full setup instructions and the rules for
writing new code. The short version: import `paths` (or `config`, which
re-exports it) instead of building paths by hand, and store paths in manifests
**relative to the data root** so they stay portable between machines.

## Before the real experiments: a lightweight smoke test

The full protocol is 3 groups x 3 seeds x 660 steps. Before spending that,
run the reduced protocol once to prove every seam works. See
[`member4_training_eval/README.md` §2.5](member4_training_eval/README.md)
for the step-by-step runbook.

```powershell
$env:NNDL_DATA_ROOT = '<path>\to\data'
$env:NNDL_SUBSET = '150'
$env:NNDL_RUNS_ROOT = '<path>\to\runs_smoke'
```

`NNDL_SUBSET` restricts every group to the same class-balanced subset and
scales the step budget by the same factor, so the three groups stay comparable.
The subset selection lives in one place, `config.balanced_train_subset()`,
and members 2/3 use it to generate exactly the matching images.

**The smoke test produces numbers that must never enter the report.** Its
purpose is to find pipeline bugs, not to measure the method.

## Data setup

The PACS images live outside this repo (9,990 usable images ≈ hundreds of MB).
Download PACS, then point the code at your copy:

```
set NNDL_DATA_ROOT=<path>\to\data     # parent of the PACS folder
```

`member1_data/README.md` documents the dataset interface and the six splits.

## Report

`report/progress_report.tex` (IEEEtran). Member 4's sections live in
`report/sections/member4_*.tex`; regenerate the concatenated file with
`member4_training_eval/_sync_sections.py` if you edit them.
