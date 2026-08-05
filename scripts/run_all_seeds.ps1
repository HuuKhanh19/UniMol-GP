<#
.SYNOPSIS
    Sequential Step 2 sweep over scaffold split seeds, one log file per seed.

.DESCRIPTION
    Runs scripts/run_step2.py once per split seed, each initialised from that
    seed's own Step 1 checkpoint. Designed to be launched detached and left
    overnight.

    Everything that could waste the night is checked up front, before the first
    run starts: the self-test, the processed split for every seed, and the
    Step 1 checkpoint for every seed. A seed whose checkpoint is missing is
    skipped loudly rather than silently falling back to the pretrained weights,
    because starting ES from pretrained is a materially different (and much
    harder) experiment whose numbers must not be mixed in with the rest.

.EXAMPLE
    # foreground, all five seeds on GPU 0
    .\scripts\run_all_seeds.ps1

.EXAMPLE
    # detached, survives closing the SSH session
    Start-Process powershell -WindowStyle Hidden -ArgumentList `
        '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\run_all_seeds.ps1'

.EXAMPLE
    # both GPUs: launch twice, odd and even seeds
    .\scripts\run_all_seeds.ps1 -Seeds '0,2,4' -GpuId 0
    .\scripts\run_all_seeds.ps1 -Seeds '1,3'   -GpuId 1

.EXAMPLE
    # pass options through to run_step2.py
    .\scripts\run_all_seeds.ps1 -ExtraArgs '--no-probe --es-pop 256'
#>
param(
    [string]$Dataset = 'esol',
    [string]$Seeds = '0,1,2,3,4',
    [int]$GpuId = 0,
    [string]$LogDir = 'logs/step2',
    [switch]$SkipSelfTest,
    # Extra run_step2.py flags as one space-separated string. A single string
    # rather than remaining-arguments: PowerShell's `--` token would otherwise
    # reach argparse, which treats everything after it as positional and then
    # rejects the flags.
    [string]$ExtraArgs = ''
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$seedList = $Seeds.Split(',') | ForEach-Object { [int]$_.Trim() }
# An explicit -LogDir is used verbatim, so the log paths are predictable enough
# to tail by name. The default gets a timestamp so successive sweeps don't
# overwrite each other.
if (-not $PSBoundParameters.ContainsKey('LogDir')) {
    $LogDir = Join-Path $LogDir (Get-Date -Format 'yyyyMMdd_HHmmss')
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$summaryPath = Join-Path $LogDir 'sweep.log'

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

Write-Log "step 2 sweep: dataset=$Dataset seeds=$Seeds gpu=$GpuId"
Write-Log "logs -> $LogDir"

# --- pre-flight ------------------------------------------------------------
# Resolve every checkpoint and split now, so a missing file fails in the first
# minute rather than at 3am after four hours of GPU time.
$plan = @()
$skipped = @()
foreach ($seed in $seedList) {
    $splitDir = "data/processed/$Dataset/seed_$seed"
    if (-not (Test-Path "$splitDir/${Dataset}_train.csv")) {
        $skipped += "seed $seed - no split at $splitDir (run preprocess_data.py)"
        continue
    }
    $ckptRoot = "experiments/step1/$Dataset/seed_$seed"
    $ckpt = $null
    if (Test-Path $ckptRoot) {
        $ckpt = Get-ChildItem -Path $ckptRoot -Filter 'model_0.pth' -Recurse `
            -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    if ($null -eq $ckpt) {
        $skipped += "seed $seed - no step 1 checkpoint under $ckptRoot"
        continue
    }
    $plan += [pscustomobject]@{ Seed = $seed; Checkpoint = $ckpt.FullName }
}

foreach ($reason in $skipped) { Write-Log "SKIP  $reason" }
if ($plan.Count -eq 0) {
    Write-Log 'nothing to run - aborting'
    exit 1
}
foreach ($job in $plan) {
    Write-Log ("plan  seed {0} <- {1}" -f $job.Seed, $job.Checkpoint)
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
    $seed = $job.Seed
    $log = Join-Path $LogDir "${Dataset}_seed${seed}.log"
    Write-Log "start seed $seed -> $log"
    $t0 = Get-Date

    $argv = @(
        '-u', 'scripts/run_step2.py',
        '--dataset', $Dataset,
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
    Write-Log ("done  seed {0}: {1} in {2:N1} min" -f $seed, $status, $mins)
    $results += [pscustomobject]@{ Seed = $seed; Status = $status; Minutes = $mins }
}

# --- summary ---------------------------------------------------------------
Write-Log '--- sweep finished ---'
foreach ($r in $results) {
    Write-Log ("seed {0}: {1} ({2:N1} min)" -f $r.Seed, $r.Status, $r.Minutes)
}

# Pull the best validation/test numbers straight out of each results.json so the
# morning check is one file, not five directory hunts.
Write-Log '--- best per seed (valid-selected) ---'
foreach ($r in $results) {
    if ($r.Status -ne 'ok') { continue }
    $json = Get-ChildItem -Path "experiments/step2/$Dataset/seed_$($r.Seed)" `
        -Filter 'results.json' -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $json) { Write-Log ("seed {0}: no results.json" -f $r.Seed); continue }
    $best = (Get-Content $json.FullName -Raw | ConvertFrom-Json).best
    Write-Log ("seed {0}: valid={1:N4} test={2:N4} (phase {3})" -f `
        $r.Seed, $best.valid_rmse, $best.test_rmse, $best.phase)
}
Write-Log "summary written to $summaryPath"
