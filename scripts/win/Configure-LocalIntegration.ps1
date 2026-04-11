<#
.SYNOPSIS
    One-shot interactive bootstrap for the local-integration CI rig on
    grumpy. Prompts for WAV cache + BEQ working dir paths and persists
    them so the GitHub Actions self-hosted runner can read them later.

.DESCRIPTION
    The local-integration workflow (.github/workflows/local-integration.yml)
    runs the full spike test + experiment suite inside a Docker container.
    The container needs two host-side volume mounts at build time:
      - WAV_CACHE_DIR : path to the BEQ WAV cache (fixtures for experiment tests)
      - BEQ_DIR       : path to the BEQ working dir (catalogue + saved model)

    Paths differ per machine (grumpy maps its NAS share to a different
    drive letter than the dev laptop), so we prompt interactively once
    and persist the answers to a shared-writable JSON file under
    C:\ProgramData\beqdesigner-ci\ so the runner service account can
    read them non-interactively later via Assert-LocalIntegrationConfig.ps1.

    Fail-fast contract: if this script is run without a usable TTY
    (pipeline, CI, scheduled task), each prompt times out after
    PromptTimeoutSec seconds (default 30) and the script exits 2
    rather than hanging. That's the "if we accidentally run it on a
    CI server with no human, it won't just hang" guarantee.

.PARAMETER Force
    Re-prompt even if config already exists. Use when paths change
    (e.g. NAS got remapped to a different drive letter).

.PARAMETER WavCacheDir
    Non-interactive override for WAV_CACHE_DIR (skip the prompt).

.PARAMETER BeqDir
    Non-interactive override for BEQ_DIR (skip the prompt).

.PARAMETER PromptTimeoutSec
    Seconds to wait for each prompt before giving up. Default 30.

.EXAMPLE
    pwsh scripts/win/Configure-LocalIntegration.ps1
        Interactive — prompts for both paths with a 30 s timeout each.

.EXAMPLE
    pwsh scripts/win/Configure-LocalIntegration.ps1 -Force
        Same, but re-prompts even if config already exists.

.EXAMPLE
    pwsh scripts/win/Configure-LocalIntegration.ps1 `
        -WavCacheDir 'Z:\media\wav-cache' `
        -BeqDir 'Z:\jetspeed\beqdesigner'
        Non-interactive write. Useful for provisioning scripts.

.NOTES
    See docs/local_integration.md for the full grumpy setup walkthrough.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [string]$WavCacheDir,
    [string]$BeqDir,
    [int]$PromptTimeoutSec = 30
)

$ErrorActionPreference = 'Stop'

$ConfigDir  = 'C:\ProgramData\beqdesigner-ci'
$ConfigPath = Join-Path $ConfigDir 'config.json'

function Write-Info($msg)  { Write-Host "[configure] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[configure] $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "[configure] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[configure] $msg" -ForegroundColor Red }

function Test-Interactive {
    # A host is "interactive" if it has a real console we can Read-Host
    # from. On the GH Actions runner service, stdin is redirected and
    # $Host.UI.RawUI reports sensibly that we can't prompt.
    try {
        return [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    } catch {
        return $false
    }
}

function Read-HostWithTimeout {
    <#
    .SYNOPSIS
        Read-Host that returns $null if the user doesn't respond
        within $TimeoutSec seconds.
    #>
    param(
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][int]$TimeoutSec
    )

    if (-not (Test-Interactive)) {
        Write-Err "Non-interactive session detected — cannot prompt for '$Prompt'."
        Write-Err "Supply -WavCacheDir / -BeqDir on the command line, or run"
        Write-Err "this script from an interactive PowerShell window on grumpy."
        return $null
    }

    Write-Host -NoNewline "${Prompt}: "
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $buffer = New-Object System.Text.StringBuilder

    while ((Get-Date) -lt $deadline) {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq 'Enter') {
                Write-Host ""
                return $buffer.ToString()
            } elseif ($key.Key -eq 'Backspace') {
                if ($buffer.Length -gt 0) {
                    [void]$buffer.Remove($buffer.Length - 1, 1)
                    [Console]::Write("`b `b")
                }
            } elseif (-not [char]::IsControl($key.KeyChar)) {
                [void]$buffer.Append($key.KeyChar)
                [Console]::Write($key.KeyChar)
            }
        } else {
            Start-Sleep -Milliseconds 100
        }
    }

    Write-Host ""
    Write-Err "No response within $TimeoutSec s — aborting."
    return $null
}

function Test-PathReadable {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Err "Path does not exist: $Path"
        return $false
    }
    try {
        [void](Get-ChildItem -LiteralPath $Path -ErrorAction Stop | Select-Object -First 1)
    } catch {
        Write-Err "Path exists but cannot be read (permissions?): $Path"
        return $false
    }
    return $true
}

# --- 1. Load existing config, short-circuit if -Force isn't passed ---

$existing = $null
if ((Test-Path -LiteralPath $ConfigPath) -and -not $Force) {
    try {
        $existing = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
        Write-Info "Config already exists at $ConfigPath."
        Write-Info "  WAV_CACHE_DIR = $($existing.WavCacheDir)"
        Write-Info "  BEQ_DIR       = $($existing.BeqDir)"
        Write-Info "Pass -Force to re-prompt."
        exit 0
    } catch {
        Write-Warn2 "Existing config at $ConfigPath is unreadable — treating as missing."
        $existing = $null
    }
}

# --- 2. Resolve each path: explicit param > prompt with timeout ---

if (-not $WavCacheDir) {
    $WavCacheDir = Read-HostWithTimeout `
        -Prompt 'WAV cache dir (e.g. Z:\media\wav-cache)' `
        -TimeoutSec $PromptTimeoutSec
    if ($null -eq $WavCacheDir -or [string]::IsNullOrWhiteSpace($WavCacheDir)) {
        exit 2
    }
}

if (-not $BeqDir) {
    $BeqDir = Read-HostWithTimeout `
        -Prompt 'BEQ working dir (e.g. Z:\jetspeed\beqdesigner)' `
        -TimeoutSec $PromptTimeoutSec
    if ($null -eq $BeqDir -or [string]::IsNullOrWhiteSpace($BeqDir)) {
        exit 2
    }
}

# --- 3. Validate both paths are reachable before we commit them ---

$WavCacheDir = $WavCacheDir.Trim().Trim('"').Trim("'")
$BeqDir      = $BeqDir.Trim().Trim('"').Trim("'")

$valid = $true
if (-not (Test-PathReadable -Path $WavCacheDir)) { $valid = $false }
if (-not (Test-PathReadable -Path $BeqDir))      { $valid = $false }
if (-not $valid) {
    Write-Err "Validation failed — config not written. Fix paths and re-run."
    exit 2
}

# --- 4. Persist to C:\ProgramData\beqdesigner-ci\config.json ---

if (-not (Test-Path -LiteralPath $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
}

$payload = [pscustomobject]@{
    WavCacheDir    = $WavCacheDir
    BeqDir         = $BeqDir
    ConfiguredAt   = (Get-Date).ToString('o')
    ConfiguredBy   = "$env:USERDOMAIN\$env:USERNAME"
    ConfiguredOn   = $env:COMPUTERNAME
}
$payload | ConvertTo-Json | Set-Content -LiteralPath $ConfigPath -Encoding UTF8

Write-Ok "Wrote $ConfigPath"
Write-Ok "  WAV_CACHE_DIR = $WavCacheDir"
Write-Ok "  BEQ_DIR       = $BeqDir"
Write-Info "Next: push a commit to div/local-integration to trigger the workflow,"
Write-Info "or run 'pwsh scripts/win/Run-ExperimentBuild.ps1 -Scope unit' to smoke-test."
exit 0
