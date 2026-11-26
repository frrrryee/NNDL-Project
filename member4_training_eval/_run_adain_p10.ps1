# Train the uniform-AdaIN group on the frozen ten-painting pool under the
# epoch-based protocol, so that the style-pool ablation has all four cells:
# {ours, adain} x {10 paintings, 20 paintings}.
#
# The existing adain_s0..s4 runs use the old fixed 660-step budget, so they
# are not comparable with the new runs and cannot serve as the fourth cell.
# The images already exist -- data/augmented/adain/manifest_training.txt
# points at the 1,349 AdaIN images of the frozen ten-painting pool, the same
# pool and the same content-style pairing that the Ours arm of the ablation
# uses -- so only the training has to be redone.
#
# The step budget is derived from the data: 1,290 steps over 2,698 images.

$ErrorActionPreference = "Continue"

$repo     = "C:\Users\qth02\Desktop\NNDL-Project\repo"
$py       = "$env:USERPROFILE\anaconda3\python.exe"
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"
# Pin the device explicitly.  A previous script in this shell session had set
# NNDL_DEVICE=cpu (for the CPU-only LPIPS/edge metrics) and the variable
# survives into every later script run in the same session, which silently
# turns this training run into a CPU run -- roughly nine times slower.
$env:NNDL_DEVICE = "cuda"
$manifest = "C:\Users\qth02\Desktop\nndl_pacs\data\augmented\adain\manifest_training.txt"
$outRoot  = "C:\Users\qth02\Desktop\NNDL-Project\runs"

Set-Location "$repo\member4_training_eval"

foreach ($seed in 0, 1, 2, 3, 4) {
    $tag = "poolab_adain_p10_s$seed"
    $run = Join-Path $outRoot $tag

    if (Test-Path (Join-Path $run "eval_last.json")) {
        Write-Output "[skip] $tag already evaluated"
        continue
    }

    Write-Output "=== $tag : training ==="
    & $py train.py --group adain --seed $seed --aug-manifest $manifest `
        --tag $tag --out-root $outRoot --keep-checkpoint 2>$null | Select-Object -Last 2

    Write-Output "=== $tag : evaluating ==="
    & $py evaluate.py --run $run --checkpoint last.pt 2>$null | Select-Object -Last 7

    $ckpt = Join-Path $run "last.pt"
    if (Test-Path $ckpt) { Remove-Item $ckpt -Force }

    Write-Output "=== $tag : done ==="
}

Write-Output "=== all seeds done ==="
