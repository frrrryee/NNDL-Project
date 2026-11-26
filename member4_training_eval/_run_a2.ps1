# Train the two-style arms: Ours and AdaIN, five seeds each.
#
# The multi-style protocol gives every original two style references, so the
# training set grows from 2,698 to 4,047 images and the epoch-based budget
# grows from 1,290 to 1,920 steps with it.  The two arms share the same
# manifest pairing and the same step budget, so the only difference between
# them is the stylization algorithm -- exactly as in every other comparison
# of this project.
#
# The dataset itself is data/aug_sp/poolab_a2/, generated with the two-style
# pool in member4_training_eval/style_pool_a2/.

$ErrorActionPreference = "Continue"

$repo     = "C:\Users\qth02\Desktop\NNDL-Project\repo"
$py       = "$env:USERPROFILE\anaconda3\python.exe"
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"
$env:NNDL_STYLE_POOL_DIR = "$repo\member4_training_eval\style_pool_a2"
$env:NNDL_STYLE_POOL = "$env:NNDL_STYLE_POOL_DIR\style_pool.json"
# Pin the device.  A leftover NNDL_DEVICE=cpu in the session -- the
# structure-metrics script sets one -- turns this into a CPU run, which is
# about eight times slower.
$env:NNDL_DEVICE = "cuda"
$outRoot  = "C:\Users\qth02\Desktop\NNDL-Project\runs"

$arms = @(
    @{ group = "ours";  manifest = "C:\Users\qth02\Desktop\nndl_pacs\data\aug_sp\poolab_a2\manifest_ours.txt" },
    @{ group = "adain"; manifest = "C:\Users\qth02\Desktop\nndl_pacs\data\aug_sp\poolab_a2\manifest_adain.txt" }
)

Set-Location "$repo\member4_training_eval"

foreach ($arm in $arms) {
    $group = $arm.group
    $manifest = $arm.manifest
    foreach ($seed in 0, 1, 2, 3, 4) {
        $tag = "a2_${group}_s$seed"
        $run = Join-Path $outRoot $tag

        if (Test-Path (Join-Path $run "eval_last.json")) {
            Write-Output "[skip] $tag already evaluated"
            continue
        }

        Write-Output "=== $tag : training ==="
        & $py train.py --group $group --seed $seed --aug-manifest $manifest `
            --tag $tag --out-root $outRoot --keep-checkpoint 2>&1 | Select-Object -Last 2

        Write-Output "=== $tag : evaluating ==="
        & $py evaluate.py --run $run --checkpoint last.pt 2>&1 | Select-Object -Last 7

        $ckpt = Join-Path $run "last.pt"
        if (Test-Path $ckpt) { Remove-Item $ckpt -Force }

        Write-Output "=== $tag : done ==="
    }
}

Write-Output "=== all seeds done ==="
