<#
.SYNOPSIS
    Non-interactive config reader used by the local-integration GitHub
    Actions workflow. Fails fast if the config is missing or its
    referenced paths are unreachable.

.DESCRIPTION
    Runs as the first step of the `full-suite` job in
    .github/workflows/local-integration.yml, before the Docker build.
    Reads C:\ProgramData\beqdesigner-ci\config.json (written by
    Configure-LocalIntegration.ps1) and exports WAV_CACHE_DIR /
    BEQ_DIR into the GitHub Actions job environment via $env:GITHUB_ENV
    so subsequent steps (e.g. Run-ExperimentBuild.ps1) can pass them to
    docker compose.

    Unlike Configure-LocalIntegration.ps1, this script NEVER prompts —
    the runner service account is always non-interactive. If the
    config is missing, the script exits with a loud error telling the
    operator to run the configure script on grumpy directly.

    This is the "fail fast if nobody's home" half of the user's spec:
    Configure is interactive-with-timeout, Assert is read-only and
    crashes immediately on missing state.

.NOTES
    See docs/local_integration.md for the full grumpy setup walkthrough.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ConfigPath = 'C:\ProgramData\beqdesigner-ci\config.json'

function Write-Err($msg) { Write-Host "[assert-config] $msg" -ForegroundColor Red }
function Write-Ok($msg)  { Write-Host "[assert-config] $msg" -ForegroundColor Green }

# --- 1. Config file exists ---

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Err "Config file missing: $ConfigPath"
    Write-Err ""
    Write-Err "Run this on grumpy interactively (as the runner's owning user):"
    Write-Err "    pwsh scripts/win/Configure-LocalIntegration.ps1"
    Write-Err ""
    Write-Err "Then re-run this workflow (Actions tab → Re-run failed jobs)."
    exit 2
}

# --- 2. Config file parses ---

try {
    $config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
} catch {
    Write-Err "Config file is not valid JSON: $ConfigPath"
    Write-Err "Delete it and re-run Configure-LocalIntegration.ps1."
    exit 2
}

if (-not $config.WavCacheDir -or -not $config.BeqDir) {
    Write-Err "Config is missing required keys WavCacheDir / BeqDir."
    Write-Err "Re-run: pwsh scripts/win/Configure-LocalIntegration.ps1 -Force"
    exit 2
}

# --- 3. Both paths are reachable from the runner service account ---

$failures = @()
if (-not (Test-Path -LiteralPath $config.WavCacheDir)) {
    $failures += "WAV_CACHE_DIR not reachable: $($config.WavCacheDir)"
}
if (-not (Test-Path -LiteralPath $config.BeqDir)) {
    $failures += "BEQ_DIR not reachable: $($config.BeqDir)"
}

if ($failures.Count -gt 0) {
    foreach ($f in $failures) { Write-Err $f }
    Write-Err ""
    Write-Err "Common causes on grumpy:"
    Write-Err "  - NAS drive mapping was not persisted (use 'net use Z: \\nas\\share /persistent:yes')"
    Write-Err "  - Runner service account cannot see the user's mapped drives (run the"
    Write-Err "    runner under a named account that has the share mounted, not LocalSystem)"
    Write-Err "  - Share is temporarily unreachable — try 'Test-Path $($config.WavCacheDir)' manually"
    exit 2
}

# --- 4. Export into the GH Actions job env so downstream steps see them ---

if ($env:GITHUB_ENV) {
    "WAV_CACHE_DIR=$($config.WavCacheDir)" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    "BEQ_DIR=$($config.BeqDir)"            | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
} else {
    # Not running inside GH Actions — set the process env vars so a
    # manual invocation from a dev PowerShell session still works.
    $env:WAV_CACHE_DIR = $config.WavCacheDir
    $env:BEQ_DIR       = $config.BeqDir
}

Write-Ok "Config OK (configured $($config.ConfiguredAt) on $($config.ConfiguredOn))"
Write-Ok "  WAV_CACHE_DIR = $($config.WavCacheDir)"
Write-Ok "  BEQ_DIR       = $($config.BeqDir)"
exit 0
