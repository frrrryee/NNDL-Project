# Environment and Path Setup

Every member works on a separate machine.  No absolute path is stored in the
code, and no path from one member's computer appears in another member's
scripts.

## 1. Where things are resolved

The single place that knows the layout is `repo/paths.py`.  Every module
imports it instead of building paths by hand:

| Name | Meaning | Environment variable | Default |
|---|---|---|---|
| `paths.REPO_ROOT` | the folder holding `member1_data/`, `member2_adain/`, ... | — | found by walking upwards from `paths.py` |
| `paths.DATA_ROOT` | the folder holding `PACS/` and the generated image folders | `NNDL_DATA_ROOT` | `<repo>/data` |
| `paths.RUNS_ROOT` | the folder holding the run directories and summaries | `NNDL_RUNS_ROOT` | `<repo>/../runs` |
| `paths.IMAGE_ROOT` | `DATA_ROOT/PACS` | — | — |
| `paths.SPLIT_DIR` | the one frozen split directory | — | `<repo>/member4_training_eval/project_splits` |
| `paths.STYLE_POOL_DIR` | the frozen style pool | — | `<repo>/member4_training_eval/style_pool` |
| `paths.REPORT_DIR` | LaTeX sources and figures | — | `<repo>/report` |

`paths.py` finds the repository root by walking upwards from its own location
until it sees a directory that contains `member1_data/` or
`member4_training_eval/`.  The repository may therefore be cloned under any
name, at any depth, on any drive.

## 2. Setting up on a new machine

### 2.1 Get the code and the data

```powershell
git clone <repository-url> repo
```

Place the PACS images so that the layout is:

```text
<repo>/data/PACS/photo/dog/...
<repo>/data/PACS/art_painting/dog/...
<repo>/data/PACS/cartoon/dog/...
<repo>/data/PACS/sketch/dog/...
```

or, if the copy of PACS lives elsewhere, point the environment variable at it
(section 2.3).  The dataset itself is not part of the repository.

### 2.2 Python environment

`repo/nn.ps1` picks a usable interpreter for you.  It tries, in order:

1. `$env:NNDL_PYTHON`
2. `.\.venv\Scripts\python.exe`
3. the conda environment named `$env:NNDL_ENV` (default `NN`)
4. any conda environment under `$env:USERPROFILE\.conda\envs`
5. `~\anaconda3\python.exe` and `~\miniconda3\python.exe`
6. `python` on `PATH`

A candidate is accepted only if it has **torch and torchvision** installed and
runs a real CUDA convolution on the local GPU.  The convolution matters: an
old `cu113` build reports `torch.cuda.is_available() == True` on a newer card
and then dies at the first kernel launch.  The choice is cached in
`.nndl_python` (git-ignored).

```powershell
.\nn.ps1 -c "import torch, torchvision; print(torch.__version__)"
```

Requirements: `torch >= 2.0`, `torchvision >= 0.13` (member 4's `model.py`
uses `models.resnet18(weights=...)`), plus `numpy`, `scipy`, `Pillow`,
`matplotlib`, `opencv-python` and `lpips`.  The recorded results were produced
with torch 2.11.0+cu128 and torchvision 0.26.0.

### 2.3 Point the project at your data

Nothing is required when the data sits at `<repo>/data/PACS`.  Otherwise set
one variable, in the shell you are working in:

```powershell
$env:NNDL_DATA_ROOT = 'D:\datasets\pacs\data'
```

`NNDL_DATA_ROOT` must point at the **parent** of the `PACS` folder.

For the run artefacts, which are kept out of the repository so that no
checkpoint can be committed by accident:

```powershell
$env:NNDL_RUNS_ROOT = 'D:\nndl-runs'
```

To make the setting permanent for your account:

```powershell
[Environment]::SetEnvironmentVariable('NNDL_DATA_ROOT', 'D:\datasets\pacs\data', 'User')
```

If the variable is not set, the defaults `<repo>/data` and `<repo>/../runs`
are used.  Nothing else has to be edited.

## 3. Verifying the setup

```powershell
.\nn.ps1 member4_training_eval\verify_splits.py
```

Expected output ends with:

```text
PASS: all six splits are consistent with protocol.json and the local PACS copy.
```

Then the quick pipeline check:

```powershell
$env:NNDL_SUBSET = '150'
$env:NNDL_RUNS_ROOT = 'D:\nndl-runs-smoke'
.\nn.ps1 member4_training_eval\run_all.py --group original
```

`NNDL_SUBSET` restricts every group to the same class-balanced subset and
scales the step budget by the same factor.  **The numbers from the smoke test
must never enter the report** — its purpose is to find pipeline bugs, not to
measure the method.

## 4. Rules for writing new code

1. **Never build a path by hand.**  Import `paths` (or `config`, which re-exports
   it) and use `paths.DATA_ROOT`, `paths.RUNS_ROOT`, `paths.SPLIT_DIR`,
   `paths.STYLE_POOL_DIR`, `paths.IMAGE_ROOT`.
2. **Never hard-code an absolute path** in a script, a default argument, a
   config file, or a manifest.  Manifests store paths **relative to the data
   root**, which is what makes them portable between machines.
3. **A CLI override is always welcome**, but its default must come from
   `paths.py`, not from a literal.
4. **Do not add a second copy of the splits.**  The canonical location is
   `member4_training_eval/project_splits/`.  Member 1 owns the content;
   member 2 keeps a byte-identical copy only as a fallback for the case where
   the canonical folder has not been checked out yet.
5. **Use `paths.relative_to_repo(path)`** when a path has to be printed or
   written into a summary, so that the output is machine-independent.

## 5. Paths that must not be shared through the repository

| What | Where it lives | Why |
|---|---|---|
| PACS images | outside the repository | ~1 GB, never committed |
| Generated augmented images | `DATA_ROOT/augmented/`, `DATA_ROOT/aug_sp/` | ~120 MB per set, regenerated from a seed |
| Checkpoints (`*.pt`) | `RUNS_ROOT/<tag>/` | 43–86 MB each, git-ignored |
| Summaries and figures | `RUNS_ROOT/*.json`, `repo/report/figures/` | small, safe to commit |

Everything needed to regenerate the results is determined by the code plus
the recorded `generation.json` and `config_hash` of each run, so the artefacts
can be rebuilt on any machine from the frozen protocol.
