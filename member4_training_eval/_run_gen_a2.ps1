# Generate the multi-style augmentation set: two styles per original.
#
# The frozen one-style protocol gives every Photo training image a single
# style reference.  This run gives every original *two*, so the training set
# of every augmented group grows from 1,349 + 1,349 = 2,698 images to
# 1,349 + 2,698 = 4,047 images, and the epoch-based step budget follows it
# from 1,290 to 1,890 steps.
#
# The two styles of an original are the frozen pairing and the next entry of
# the same cyclic walk, so the multi-style set is a strict superset of the
# one-style one: every stylization the one-style arm trains on is also here.
# That is what makes the two arms comparable.
#
# Both arms come out of member 3's generator in one pass, because its
# --with-adain baseline is byte-identical to member 2's plain AdaIN
# (verified by hashing the images of a preview run).

$ErrorActionPreference = "Continue"

$repo      = "C:\Users\qth02\Desktop\NNDL-Project\repo"
$py        = "$env:USERPROFILE\anaconda3\python.exe"
$env:NNDL_DATA_ROOT = "C:\Users\qth02\Desktop\nndl_pacs\data"
$env:NNDL_STYLE_POOL_DIR = "$repo\member4_training_eval\style_pool_a2"
$env:NNDL_STYLE_POOL = "$env:NNDL_STYLE_POOL_DIR\style_pool.json"
# Pin the device: the LPIPS/edge metrics need the CPU, and a leftover
# NNDL_DEVICE=cpu in the session would silently turn the generation into a
# CPU run as well.
$env:NNDL_DEVICE = "cuda"

Set-Location $repo

Write-Output "=== generating the two-style set (2,698 images per arm) ==="
& $py -m member3_structure_preserving.generate build `
    --alpha 0.5 --beta 0.7 --cell-tag poolab_a2 --with-adain --no-metrics 2>&1 |
    Select-Object -Last 6

Write-Output "=== done ==="
