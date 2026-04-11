<#
.SYNOPSIS
    Build the beq-test Docker image and run the spike test + experiment
    suite inside it. Invoked by the local-integration GitHub Actions
    workflow; also runnable from a dev PowerShell session for a manual
    dry run before pushing.

.DESCRIPTION
    Thin wrapper around `docker compose`. The workflow job has already
    checked out the right SHA via actions/checkout@v4, so this script
    just has to:
      1. Make sure WAV_CACHE_DIR / BEQ_DIR are set (Assert-LocalIntegrationConfig.ps1
         fills these in when running in GH Actions; otherwise we read
         the persisted config ourselves).
      2. Build the beq-test image (layer-cached so only poetry.lock +
         src/ churn trigger a slow rebuild).
      3. Run the image with the requested test scope, capturing its
         exit code so the workflow job fails loudly on any non-zero.

    The actual test dispatch (which of the three run-spike-*.sh scripts
    to invoke) lives in docker/test-entrypoint.sh, not here — so this
    script stays short and focused on the docker side of the bridge.

.PARAMETER Scope
    Test scope forwarded to test-entrypoint.sh via the TEST_SCOPE env
    var. Valid values: `unit`, `integration`, `all`. Default `all`
    matches the workflow's default (the user's chosen "run everything"
    policy from the planning session).

.EXAMPLE
    pwsh scripts/win/Run-ExperimentBuild.ps1
        Full suite (unit + integration + experiment). Takes minutes to
        hours, per AGENTS.md:173.

.EXAMPLE
    pwsh scripts/win/Run-ExperimentBuild.ps1 -Scope unit
        Only the fast ~1 min unit spike. Useful for smoke-testing the
        Docker image builds cleanly before kicking off a full run.

.NOTES
    See docs/local_integration.md for the full grumpy setup walkthrough.
#>
[CmdletBinding()]
param(
    [ValidateSet('unit', 'integration', 'all')]
    [string]$Scope = 'all'
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[run-build] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[run-build] $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "[run-build] $msg" -ForegroundColor Red }

# --- 1. Resolve the repo root relative to this script ---

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot
Write-Info "Repo root: $repoRoot"

$composeFile = 'docker/docker-compose.test.yml'
if (-not (Test-Path -LiteralPath $composeFile)) {
    Write-Err "Compose file not found: $composeFile (wrong working dir?)"
    exit 1
}

# --- 2. Make sure WAV_CACHE_DIR / BEQ_DIR are present ---

if (-not $env:WAV_CACHE_DIR -or -not $env:BEQ_DIR) {
    Write-Info "WAV_CACHE_DIR / BEQ_DIR not in env — loading persisted config."
    $configPath = 'C:\ProgramData\beqdesigner-ci\config.json'
    if (-not (Test-Path -LiteralPath $configPath)) {
        Write-Err "No config at $configPath."
        Write-Err "Run: pwsh scripts/win/Configure-LocalIntegration.ps1"
        exit 2
    }
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $env:WAV_CACHE_DIR = $config.WavCacheDir
    $env:BEQ_DIR       = $config.BeqDir
}
Write-Info "WAV_CACHE_DIR = $env:WAV_CACHE_DIR"
Write-Info "BEQ_DIR       = $env:BEQ_DIR"
Write-Info "Scope         = $Scope"

# --- 3. Build the image (cached on poetry.lock + src/ churn) ---

Write-Info "Building beq-test image..."
docker compose -f $composeFile build beq-test-runner
if ($LASTEXITCODE -ne 0) {
    Write-Err "docker compose build failed (exit=$LASTEXITCODE)"
    exit $LASTEXITCODE
}

# --- 4. Run the image with TEST_SCOPE passed through ---

Write-Info "Running beq-test-runner (TEST_SCOPE=$Scope)..."
$env:TEST_SCOPE = $Scope
docker compose -f $composeFile run --rm beq-test-runner
$runExit = $LASTEXITCODE

if ($runExit -eq 0) {
    Write-Ok "Test run succeeded."
} else {
    Write-Err "Test run failed (exit=$runExit)."
    Write-Err "Check .pytest_cache/spike_*.log for per-stage output."
}
exit $runExit
