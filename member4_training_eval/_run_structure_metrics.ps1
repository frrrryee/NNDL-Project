# Structure-preservation metrics for the style-pool ablation.
#
# The ablation needs the structural numbers for both arms and for both
# methods, because the cross-domain accuracy alone does not say whether the
# structure-preserving variant is doing its job: the report reads the two
# kinds of number separately.
#
# Four manifests: {ours, adain} x {10-painting pool, 20-painting pool}.
# Each has 1,349 rows.  LPIPS dominates the runtime.

$ErrorActionPreference = "Continue"

$repo = "C:\Users\qth02\Desktop\NNDL-Project\repo"
$py   = "$env:USERPROFILE\anaconda3\python.exe"
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"
# The training sweep is using the GPU at the same time, and the LPIPS model
# plus the ResNet-18 training run do not both fit in 8 GB.  The edge score
# does not need a GPU, and LPIPS on the CPU is slower but exact, so the
# structural metrics are computed on the CPU.
$env:NNDL_DEVICE = "cpu"
$data = "C:\Users\qth02\Desktop\nndl_pacs\data\aug_sp"

Set-Location "$repo\member4_training_eval"

foreach ($arm in @("poolab_p10", "poolab_p20")) {
    foreach ($method in @("ours", "adain")) {
        $manifest = "$data\$arm\manifest_$method.txt"
        $out = "$data\$arm\structure_metrics_$method.json"

        if (-not (Test-Path $manifest)) {
            Write-Output "[skip] $arm/$method : manifest not found"
            continue
        }
        if (Test-Path $out) {
            Write-Output "[skip] $arm/$method : metrics already there"
            continue
        }

        Write-Output "=== $arm / $method ==="
        # The warnings that torchvision and lpips print are not errors; the
        # script's own output is the last few lines it writes.
        & $py structure_metrics.py --manifest $manifest --out $out 2>$null |
            Select-Object -Last 4
    }
}

Write-Output "=== structure metrics done ==="
