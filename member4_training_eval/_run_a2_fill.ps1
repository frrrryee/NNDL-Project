# Fill in the two seeds that failed because the disk filled up.
#
# Both failures were the same Windows error -- "Couldn't open shared file
# mapping ... error code 1455" -- which is the page file running out of room
# because the free space on the volume had dropped below 4 GB.  The conda
# package cache has been cleaned since, and the runner only picks up the
# seeds whose evaluation is still missing.
#
# The checkpoint is removed after every evaluation, so the free space does
# not shrink again while this runs.

$ErrorActionPreference = "Continue"

$repo     = "C:\Users\qth02\Desktop\NNDL-Project\repo"
$py       = "$env:USERPROFILE\anaconda3\python.exe"
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"
$env:NNDL_STYLE_POOL_DIR = "$repo\member4_training_eval\style_pool_a2"
$env:NNDL_STYLE_POOL = "$env:NNDL_STYLE_POOL_DIR\style_pool.json"
$env:NNDL_DEVICE = "cuda"
$outRoot  = "C:\Users\qth02\Desktop\NNDL-Project\runs"

$manifest = "C:\Users\qth02\Desktop\nndl_pacs\data\aug_sp\poolab_a2\manifest_ours.txt"

Set-Location "$repo\member4_training_eval"

foreach ($seed in 2, 3) {
    $tag = "a2_ours_s$seed"
    $run = Join-Path $outRoot $tag

    if (Test-Path (Join-Path $run "eval_last.json")) {
        Write-Output "[skip] $tag already evaluated"
        continue
    }

    Write-Output "=== $tag : training ==="
    & $py train.py --group ours --seed $seed --aug-manifest $manifest `
        --tag $tag --out-root $outRoot --keep-checkpoint 2>&1 | Select-Object -Last 2

    if (-not (Test-Path (Join-Path $run "last.pt"))) {
        Write-Output "=== $tag : training produced no checkpoint, skipping ==="
        continue
    }

    Write-Output "=== $tag : evaluating ==="
    & $py evaluate.py --run $run --checkpoint last.pt 2>&1 | Select-Object -Last 7

    $ckpt = Join-Path $run "last.pt"
    if (Test-Path $ckpt) { Remove-Item $ckpt -Force }

    Write-Output "=== $tag : done ==="
}

Write-Output "=== all seeds done ==="
