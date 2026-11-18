# Run this project with a working PyTorch environment, from any directory.
#
# Resolution order:
#   1. $env:NNDL_PYTHON          (explicit override, e.g. a full path to python.exe)
#   2. .\.venv\Scripts\python.exe (local project virtualenv)
#   3. conda env named $env:NNDL_ENV (default: 'NN')
#   4. any conda env under $env:USERPROFILE\.conda\envs
#   5. the base conda install under $env:USERPROFILE\anaconda3
#   6. the 'python' on PATH
#
# Among the candidates the script picks the best one: it must have torch and
# torchvision, it must run a real CUDA convolution on this GPU (a CPU-only
# build, or one without kernels for this card, is rejected), and among those
# the newest torchvision wins, because member 4's model.py uses the modern
# `models.resnet18(weights=...)` API that needs torchvision >= 0.13.
#
# The decision is cached in .nndl_python so repeated calls are cheap.
$ErrorActionPreference = 'Stop'

$cacheFile = Join-Path $PSScriptRoot '.nndl_python'
$candidates = New-Object System.Collections.Generic.List[string]

function Add-Candidate([string]$path) {
    if ($path -and (Test-Path -LiteralPath $path) -and -not $candidates.Contains($path)) {
        $candidates.Add($path) | Out-Null
    }
}

if ($env:NNDL_PYTHON) { Add-Candidate $env:NNDL_PYTHON }
Add-Candidate (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')

$condaEnvRoot = Join-Path $env:USERPROFILE '.conda\envs'
$preferredEnv = if ($env:NNDL_ENV) { $env:NNDL_ENV } else { 'NN' }
Add-Candidate (Join-Path $condaEnvRoot "$preferredEnv\python.exe")
# The base conda install is the environment the recorded baseline results were
# produced in (torch 2.11 + cu128, torchvision 0.26, lpips, OpenCV), so it is
# preferred over the task-specific envs.
Add-Candidate (Join-Path $env:USERPROFILE 'anaconda3\python.exe')
Add-Candidate (Join-Path $env:USERPROFILE 'miniconda3\python.exe')
if (Test-Path -LiteralPath $condaEnvRoot) {
    foreach ($envDir in (Get-ChildItem -LiteralPath $condaEnvRoot -Directory | Sort-Object Name)) {
        Add-Candidate (Join-Path $envDir.FullName 'python.exe')
    }
}

# The probe runs as a file: PowerShell mangles quotes inside a multi-line
# `python -c` argument, which would silently report every env as broken.
$probeFile = Join-Path ([System.IO.Path]::GetTempPath()) 'nndl_env_probe.py'
@'
import torch, torch.nn as nn, torchvision
try:
    assert torch.cuda.is_available(), "no cuda"
    c = nn.Conv2d(3, 8, 3, padding=1).cuda()
    y = c(torch.randn(2, 3, 64, 64, device="cuda"))
    torch.cuda.synchronize()
    cuda = "1"
except Exception:
    cuda = "0"
print("%s|%s" % (torchvision.__version__, cuda))
'@ | Set-Content -LiteralPath $probeFile -Encoding utf8

# Probe one interpreter; returns a hashtable or $null.
function Probe-Env([string]$pythonPath) {
    $root = Split-Path -Parent $pythonPath
    $sitePackages = Join-Path $root 'Lib\site-packages'
    if (-not (Test-Path -LiteralPath (Join-Path $sitePackages 'torch'))) { return $null }
    if (-not (Test-Path -LiteralPath (Join-Path $sitePackages 'torchvision'))) { return $null }

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = & $pythonPath $probeFile 2>$null
    $ErrorActionPreference = $previous
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    $line = (($out | Select-Object -Last 1) -replace "`r", '').Trim()
    if ($line -notmatch '^([^|]+)\|(0|1)$') { return $null }
    return @{ Cuda = [int]$Matches[2]; Tv = $Matches[1] }
}

function Get-TvSortKey([string]$version) {
    $parts = ($version -replace '[^0-9.].*$', '') -split '\.'
    $major = if ($parts.Count -ge 1) { [int]$parts[0] } else { 0 }
    $minor = if ($parts.Count -ge 2) { [int]$parts[1] } else { 0 }
    return ($major * 1000) + $minor
}

$resolved = $null
if (Test-Path -LiteralPath $cacheFile) {
    $cached = (Get-Content -LiteralPath $cacheFile -Raw).Trim()
    if ($cached -and (Test-Path -LiteralPath $cached)) {
        $probe = Probe-Env $cached
        if ($probe -and $probe.Cuda -eq 1) { $resolved = $cached }
    }
}

if (-not $resolved) {
    # Candidates are ordered by preference (explicit override, local venv,
    # named env, the base install the baseline was produced in, then the rest).
    # The first one whose CUDA kernels actually run wins; if none has a usable
    # GPU, fall back to the first one that merely has torch.
    $scored = @()
    foreach ($candidate in $candidates) {
        $probe = Probe-Env $candidate
        if (-not $probe) { continue }
        $scored += [pscustomobject]@{
            Path = $candidate
            Cuda = $probe.Cuda
            Sort = (Get-TvSortKey $probe.Tv)
        }
    }
    $withCuda = $scored | Where-Object { $_.Cuda -eq 1 }
    if ($withCuda) {
        $resolved = ($withCuda | Select-Object -First 1).Path
    } elseif ($scored.Count -gt 0) {
        $resolved = ($scored | Sort-Object -Property @{Expression = 'Sort'; Descending = $true} |
                     Select-Object -First 1).Path
        Write-Warning ("No conda env with working CUDA was found; falling back to the CPU " +
                       "environment $resolved. Training will be much slower.")
    }
}

if (-not $resolved) {
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $resolved = $onPath.Source }
}

if (-not $resolved) {
    throw ("No usable Python interpreter found. Set `$env:NNDL_PYTHON to a python.exe " +
           "that has torch and torchvision installed, or create .\.venv with them.")
}

Set-Content -LiteralPath $cacheFile -Value $resolved -Encoding utf8

Push-Location -LiteralPath $PSScriptRoot
try {
    & $resolved @args
    $nnExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $nnExitCode
