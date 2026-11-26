# Train the uniform-AdaIN group on the twenty-painting style pool, so that
# the pool-size ablation covers both methods.
#
# The images already exist: data/aug_sp/poolab_p20/manifest_adain.txt points
# at the 1,349 AdaIN images that were generated with the twenty-painting pool,
# using the same content-style pairing as the Ours arm of the same pool.
#
# This is the "10 + the ten remaining paintings of the upstream demo set"
# pool, so it includes the black-and-white pencil sketch.  The leak check
# passes for all twenty images against the whole local PACS copy.
#
# The step budget is derived from the data, so the run gets 1,290 steps over
# 2,698 images, the same protocol as every other augmented group.

$ErrorActionPreference = "Stop"

$repo     = "C:\Users\qth02\Desktop\NNDL-Project\repo"
$py       = "$env:USERPROFILE\anaconda3\python.exe"
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"
$manifest = "C:\Users\qth02\Desktop\nndl_pacs\data\aug_sp\poolab_p20\manifest_adain.txt"
$outRoot  = "C:\Users\qth02\Desktop\NNDL-Project\runs"

Set-Location "$repo\member4_training_eval"

foreach ($seed in 0, 1, 2, 3, 4) {
    $tag = "poolab_adain_p20_s$seed"
    $run = Join-Path $outRoot $tag

    if (Test-Path (Join-Path $run "eval_last.json")) {
        Write-Output "[skip] $tag already evaluated"
        continue
    }

    Write-Output "=== $tag : training ==="
    & $py train.py --group adain --seed $seed --aug-manifest $manifest `
        --tag $tag --out-root $outRoot --keep-checkpoint 2>&1 | Select-Object -Last 3

    Write-Output "=== $tag : evaluating ==="
    & $py evaluate.py --run $run --checkpoint last.pt 2>&1 | Select-Object -Last 7

    # The checkpoint has served its purpose; the numbers live in the JSON.
    $ckpt = Join-Path $run "last.pt"
    if (Test-Path $ckpt) { Remove-Item $ckpt -Force }

    Write-Output "=== $tag : done ==="
}

Write-Output "=== all seeds done ==="
