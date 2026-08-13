<#
.SYNOPSIS
    Sequential Step 2 sweep over datasets x split seeds, one log file per run.

.DESCRIPTION
    Runs scripts/run_step2.py once per (dataset, split seed), each initialised
    from that pair's own Step 1 checkpoint. Designed to be launched detached and
    left overnight.

    Everything that could waste the night is checked up front, before the first
    run starts: the self-test, the processed split for every job, and the Step 1
    checkpoint for every job. A job whose checkpoint is missing is skipped
    loudly rather than silently falling back to the pretrained weights, because
    starting ES from pretrained is a materially different (and much harder)
    experiment whose numbers must not be mixed in with the rest.

    -Split picks the split family and must match how preprocess_data.py and
    run_step1.py were run. It is also the top path component for the splits it
    reads, the runs it writes and its own logs -- data/processed/{split}/...,
    experiments/{split}/step2/... and logs/{split}/step2/... -- so the two
    families never share a directory.

.EXAMPLE
    # foreground, all five scaffold seeds of ESOL on GPU 0
    .\scripts\run_all_seeds.ps1

.EXAMPLE
    # detached, survives closing the SSH session
    Start-Process powershell -WindowStyle Hidden -ArgumentList `
        '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\run_all_seeds.ps1'

.EXAMPLE
    # the full random-split matrix, split across the two GPUs. lipo is roughly
    # as expensive as the other three together, so it gets a card to itself.
    .\scripts\run_all_seeds.ps1 -Split random -Dataset 'lipo' -GpuId 0
    .\scripts\run_all_seeds.ps1 -Split random -Dataset 'esol,freesolv,bace' -GpuId 1

.EXAMPLE
    # both GPUs on one dataset: launch twice, odd and even seeds
    .\scripts\run_all_seeds.ps1 -Seeds '0,2,4' -GpuId 0
    .\scripts\run_all_seeds.ps1 -Seeds '1,3'   -GpuId 1

.EXAMPLE
    # pass options through to run_step2.py
    .\scripts\run_all_seeds.ps1 -ExtraArgs '--no-probe --es-pop 256'
#>
param(
    # Comma-separated dataset keys, in the order they should run.
    [string]$Dataset = 'esol',
    [string]$Seeds = '0,1,2,3,4',
    [ValidateSet('scaffold', 'random')]
    [string]$Split = 'scaffold',
    [int]$GpuId = 0,
    # Defaults to logs/{split}/step2/{timestamp} -- see below.
    [string]$LogDir = '',
    [switch]$SkipSelfTest,
    # Extra run_step2.py flags as one space-separated string. A single string
    # rather than remaining-arguments: PowerShell's `--` token would otherwise
    # reach argparse, which treats everything after it as positional and then
    # rejects the flags.
    [string]$ExtraArgs = ''
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$datasetList = $Dataset.Split(',') | ForEach-Object { $_.Trim() } |
    Where-Object { $_ }
$seedList = $Seeds.Split(',') | ForEach-Object { [int]$_.Trim() }
# An explicit -LogDir is used verbatim, so the log paths are predictable enough
# to tail by name. The default gets a timestamp so successive sweeps don't
# overwrite each other.
if (-not $LogDir) {
    $LogDir = "logs/$Split/step2/" + (Get-Date -Format 'yyyyMMdd_HHmmss')
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$summaryPath = Join-Path $LogDir 'sweep.log'

function Get-SeedPath {
    <#
      Directory for one seed under an already split-qualified root, mirroring
      split_dir() / experiment_name() in src/data/datasets.py. Keep the two in
      step.
    #>
    param([string]$Root, [string]$DatasetName, [int]$Seed)
    return "$Root/$DatasetName/seed_$Seed"
}

function Invoke-Logged {
    <#
      Run python with every stream going to one file.

      Not `python ... *>&1 | Out-File`: PowerShell wraps each stderr line from a
      native command in a NativeCommandError, and with $ErrorActionPreference =
      'Stop' the first one terminates the script. unimol_tools logs to stderr,
      so that fires on its very first line. Handing the redirection to cmd keeps
      PowerShell's error machinery out of the way entirely.
    #>
    param([string[]]$Argv, [string]$LogPath)
    $quoted = ($Argv | ForEach-Object {
        if ("$_" -match '[\s"]') { '"' + ("$_" -replace '"', '\"') + '"' } else { "$_" }
    }) -join ' '
    cmd /c "python $quoted > `"$LogPath`" 2>&1"
    return $LASTEXITCODE
}

function Write-Log($message) {
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $message
    Write-Host $line
    Add-Content -Path $summaryPath -Value $line
}

Write-Log "step 2 sweep: split=$Split datasets=$Dataset seeds=$Seeds gpu=$GpuId"
Write-Log "logs -> $LogDir"

# --- pre-flight ------------------------------------------------------------
# Resolve every checkpoint and split now, so a missing file fails in the first
# minute rather than at 3am after four hours of GPU time.
$plan = @()
$skipped = @()
foreach ($ds in $datasetList) {
    foreach ($seed in $seedList) {
        $splitDir = Get-SeedPath "data/processed/$Split" $ds $seed
        if (-not (Test-Path "$splitDir/${ds}_train.csv")) {
            $skipped += "$ds seed $seed - no split at $splitDir (run preprocess_data.py --split $Split)"
            continue
        }
        $ckptRoot = Get-SeedPath "experiments/$Split/step1" $ds $seed
        $ckpt = $null
        if (Test-Path $ckptRoot) {
            $ckpt = Get-ChildItem -Path $ckptRoot -Filter 'model_0.pth' -Recurse `
                -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        }
        if ($null -eq $ckpt) {
            $skipped += "$ds seed $seed - no step 1 checkpoint under $ckptRoot"
            continue
        }
        $plan += [pscustomobject]@{
            Dataset = $ds; Seed = $seed; Checkpoint = $ckpt.FullName
        }
    }
}

foreach ($reason in $skipped) { Write-Log "SKIP  $reason" }
if ($plan.Count -eq 0) {
    Write-Log 'nothing to run - aborting'
    exit 1
}
foreach ($job in $plan) {
    Write-Log ("plan  {0} seed {1} <- {2}" -f $job.Dataset, $job.Seed, $job.Checkpoint)
}

if (-not $SkipSelfTest) {
    Write-Log 'running self-test'
    $selfTestLog = Join-Path $LogDir 'selftest.log'
    $code = Invoke-Logged -Argv @('-u', 'scripts/selftest.py') -LogPath $selfTestLog
    if ($code -ne 0) {
        Write-Log "self-test FAILED (see $selfTestLog) - aborting before any training"
        exit 1
    }
    Write-Log 'self-test passed'
}

# --- sweep -----------------------------------------------------------------
$results = @()
foreach ($job in $plan) {
    $ds = $job.Dataset
    $seed = $job.Seed
    $log = Join-Path $LogDir "${ds}_seed${seed}.log"
    Write-Log "start $ds seed $seed -> $log"
    $t0 = Get-Date

    $argv = @(
        '-u', 'scripts/run_step2.py',
        '--dataset', $ds,
        '--split', $Split,
        '--split-seed', $seed,
        '--gpu-id', $GpuId,
        '--init-checkpoint', $job.Checkpoint
    )
    if ($ExtraArgs) {
        $argv += $ExtraArgs.Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
    }
    $code = Invoke-Logged -Argv $argv -LogPath $log

    $mins = ((Get-Date) - $t0).TotalMinutes
    $status = if ($code -eq 0) { 'ok' } else { "FAILED (exit $code)" }
    Write-Log ("done  {0} seed {1}: {2} in {3:N1} min" -f $ds, $seed, $status, $mins)
    $results += [pscustomobject]@{
        Dataset = $ds; Seed = $seed; Status = $status; Minutes = $mins
    }
}

# --- summary ---------------------------------------------------------------
Write-Log '--- sweep finished ---'
foreach ($r in $results) {
    Write-Log ("{0} seed {1}: {2} ({3:N1} min)" -f `
        $r.Dataset, $r.Seed, $r.Status, $r.Minutes)
}

# Pull the best validation/test numbers straight out of each results.json so the
# morning check is one file, not a directory hunt per run.
Write-Log '--- best per run (valid-selected) ---'
foreach ($r in $results) {
    if ($r.Status -ne 'ok') { continue }
    $root = Get-SeedPath "experiments/$Split/step2" $r.Dataset $r.Seed
    $json = Get-ChildItem -Path $root -Filter 'results.json' -Recurse `
        -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $json) {
        Write-Log ("{0} seed {1}: no results.json" -f $r.Dataset, $r.Seed)
        continue
    }
    $res = Get-Content $json.FullName -Raw | ConvertFrom-Json
    $best = $res.best
    # valid_score/test_score are metric-agnostic; fall back to the rmse keys so
    # runs made before classification support still summarise.
    $v = if ($null -ne $best.valid_score) { $best.valid_score } else { $best.valid_rmse }
    $t = if ($null -ne $best.test_score)  { $best.test_score }  else { $best.test_rmse }
    $m = if ($res.metric) { $res.metric } else { 'rmse' }
    Write-Log ("{0} seed {1}: valid={2:N4} test={3:N4} [{4}] (phase {5})" -f `
        $r.Dataset, $r.Seed, $v, $t, $m, $best.phase)
}
Write-Log "summary written to $summaryPath"
