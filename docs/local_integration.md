# Local integration CI on a Windows build box

This page documents the optional push-triggered CI rig that runs the
**full** spike test + experiment suite — unit + integration +
experiment, in sequence — on a dedicated Windows machine whenever a
commit lands on the `div/local-integration` branch.

The reference hardware is a Windows + RTX box called `grumpy`. Any
Windows machine with Docker Desktop + WSL2 will work the same way —
the `grumpy` label is just the convention we use to target the runner
from the GitHub Actions workflow.

## What this is

- **Trigger**: GitHub Actions `push` event on `div/local-integration`
  (plus `workflow_dispatch` for manual retriggers).
- **Runtime**: a Docker image (`docker/Dockerfile.test`) built from a
  `python:3.13-slim` base, carrying the full Poetry dev dependency
  closure (scipy, xgboost, scikit-learn, pytest, PyQt6 in offscreen
  mode) and ffmpeg. Distinct from the lean NAS-only
  [`docker/Dockerfile`](../docker/Dockerfile) used by
  [`lfe_extractor.md`](lfe_extractor.md).
- **Scope**: everything. [`docker/test-entrypoint.sh`](../docker/test-entrypoint.sh)
  runs `scripts/run-spike-tests.sh` → `scripts/run-spike-integration.sh`
  → `scripts/run-spike-experiments.sh` in order, collecting all three
  log files and propagating the worst exit code so a unit-test
  failure doesn't mask whether the experiment suite also broke.
- **Concurrency**: the workflow uses
  `concurrency.cancel-in-progress: true`, so a rapid burst of pushes
  cancels older runs and only the newest SHA gets built. Cleaner than
  a 30-second client-side debounce because in-flight builds don't run
  to completion on superseded SHAs.

## Prerequisites on the build box

- **Docker Desktop** with the WSL2 backend enabled (required so
  `network_mode: host` and bind-mounting Windows drives Just Work).
- **Git for Windows** (Git Bash is fine — the runner uses PowerShell
  for orchestration and bash only inside the container).
- **PowerShell 7+** (`pwsh`). Windows PowerShell 5.1 is *not* enough —
  some cmdlets used by the helper scripts are PS7-only.
- **GitHub account with Admin on `astubbs/beqdesigner`**. Needed once,
  to register the self-hosted runner.
- **Enough free disk**: first build pulls `python:3.13-slim` (~50 MB),
  runs `poetry install --with dev` (adds ~1.5 GB of wheels for
  scipy/xgboost/PyQt6), and keeps the layer cache around for next time.
  Plan on ~5 GB for Docker alone, plus the existing disk footprint of
  the WAV cache if you're hosting it locally.

## Step-by-step setup

### 1. Map the NAS share to a drive letter

Docker Desktop on Windows can't bind-mount raw UNC paths
(`\\nas\share\...`). Map the share to a drive letter first and make
the mapping persistent across reboots:

```powershell
net use Z: \\nas\share /persistent:yes
```

Verify from PowerShell:

```powershell
Test-Path 'Z:\media\wav-cache'       # should return True
Test-Path 'Z:\jetspeed\beqdesigner'  # should return True
```

**Gotcha**: if you later install the GitHub runner as a Windows
service under the default `NT AUTHORITY\NetworkService` account, that
account does **not** see drives mapped by your interactive user. Two
options:

1. Register the runner under a named account that has the mapping
   (GitHub's `svc.cmd install <domain\user>` supports this).
2. Create the mapping inside the runner's session via a startup
   script.

Option 1 is usually simpler. See the troubleshooting section below.

### 2. Clone the repo and check out the branch

```powershell
git clone https://github.com/astubbs/beqdesigner.git
cd beqdesigner
git checkout div/local-integration
```

You can put the clone anywhere — `C:\actions-runner\_work\beqdesigner\`
(the default runner workspace) is fine, as is `C:\dev\beqdesigner\`.

### 3. Persist WAV cache + BEQ dir paths

Run the interactive bootstrap from a normal PowerShell window (not a
runner session):

```powershell
pwsh scripts/win/Configure-LocalIntegration.ps1
```

It asks two questions, each with a **30-second timeout**:

```
[configure] ...
WAV cache dir (e.g. Z:\media\wav-cache): Z:\media\wav-cache
BEQ working dir (e.g. Z:\jetspeed\beqdesigner): Z:\jetspeed\beqdesigner
[configure] Wrote C:\ProgramData\beqdesigner-ci\config.json
```

The script validates both paths are reachable before writing. If you
run it headless (CI agent, piped stdin, scheduled task) it **fails
fast with exit code 2** instead of hanging — that's the "no human,
crash fast" guarantee.

Re-run with `-Force` if the paths need to change later (e.g. NAS
remapped to a different drive letter).

### 4. Register the self-hosted GitHub Actions runner

On GitHub: **repo Settings → Actions → Runners → New self-hosted
runner**. Pick **Windows x64**. GitHub generates a copy-pasteable
block. Run it from an Administrator PowerShell on grumpy:

```powershell
# (GitHub copies these exact lines for you; the token is one-time-use)
mkdir C:\actions-runner; cd C:\actions-runner
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.xxx.x/actions-runner-win-x64-2.xxx.x.zip `
    -OutFile actions-runner-win-x64.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64.zip", "$PWD")

# IMPORTANT: add --labels grumpy so the workflow's
# `runs-on: [self-hosted, windows, grumpy]` selector matches.
./config.cmd --url https://github.com/astubbs/beqdesigner `
    --token <GENERATED_TOKEN> `
    --labels grumpy
```

Install as a Windows service so the runner auto-starts after reboot:

```powershell
./svc.cmd install
./svc.cmd start
./svc.cmd status   # should print 'Active (running)'
```

Back on GitHub, the runner should now appear as **Idle** with labels
`self-hosted`, `windows`, `X64`, `grumpy`.

### 5. Smoke test the Docker rig before pushing

Before pushing a commit that would trigger a real workflow run, test
the Docker plumbing locally:

```powershell
pwsh scripts/win/Run-ExperimentBuild.ps1 -Scope unit
```

This builds `docker/Dockerfile.test` and runs only the ~1 minute
fast unit spike. Watch for:

- `[run-build] Building beq-test image...` — first run is slow (~5
  min) because Poetry downloads wheels. Subsequent runs reuse the
  layer cache and finish in seconds unless `poetry.lock` changed.
- `[run-build] Running beq-test-runner (TEST_SCOPE=unit)...`
- `[test-entrypoint] spike-unit passed`
- `[run-build] Test run succeeded.`

If you see errors about missing volume mounts, re-check your
`Configure-LocalIntegration.ps1` output and confirm `Test-Path` sees
the paths.

### 6. Trigger a real workflow run

Push a trivial commit (or open a PR from a feature branch into
`div/local-integration`):

```powershell
git commit --allow-empty -m "ci: trigger local-integration run"
git push origin div/local-integration
```

Watch the **Actions** tab on GitHub. You should see a new
`local-integration full suite` run queue up, then transition to
in-progress on your grumpy runner. Logs stream live to the Actions
UI. The `.pytest_cache/spike_*.log` files get uploaded as a
`spike-logs-<sha>` artifact on completion, regardless of pass/fail.

To confirm real-audio fixtures are reachable, look for a line like
`test_real_audio_roundtrip PASSED` in the `spike-experiments` stage
output — if the WAV cache mount didn't work, those tests would skip
instead of run.

## Environment variables

| Var | Where set | Purpose |
|---|---|---|
| `WAV_CACHE_DIR` | `Configure-LocalIntegration.ps1` → `C:\ProgramData\beqdesigner-ci\config.json` → loaded by `Assert-LocalIntegrationConfig.ps1` | Host path to the WAV cache; bind-mounted at `/wav-cache` inside the container |
| `BEQ_DIR` | same | Host path to the BEQ working dir; bind-mounted at `/beq` |
| `TEST_SCOPE` | workflow `workflow_dispatch` input, default `all` | Which spike runners to execute: `unit`, `integration`, `all` |
| `AUTO_BEQ_ADVISOR` | `docker/docker-compose.test.yml` default `measurement` | Advisor implementation. Override to `ollama` / `heuristic` for specific experiments. |
| `SPIKE_VERBOSE` | same, default `0` | `1` enables pytest `-s` (no capture) inside the container |

The container already sets `PYTHONPATH`, `QT_QPA_PLATFORM=offscreen`,
and the advisor default, matching `scripts/run-spike-experiments.sh`.

## What gets run inside the container

In order, per [`docker/test-entrypoint.sh`](../docker/test-entrypoint.sh):

1. **`bash scripts/run-spike-tests.sh`** — ~1 min, hermetic unit
   tests. No media scans, no network, no model training.
2. **`bash scripts/run-spike-integration.sh`** — minutes, needs real
   media files / TMDb / Ollama / populated sweep config. Tests skip
   gracefully when resources are absent.
3. **`bash scripts/run-spike-experiments.sh`** — **minutes to hours**,
   retrains models from scratch for the F/G/H/I experiment batches,
   the real-audio training regime (E77/E82), chunked-strategy
   comparison (E62+), and E31 extract+validate. Warn-tagged in
   `AGENTS.md:173`.

The dispatcher runs every stage even if an earlier one failed, so a
single build produces all three log files. The workflow job's final
exit code is the worst exit code seen — a unit failure does not mask
an experiment failure.

## Troubleshooting

### Runner shows "Offline" in GitHub

```powershell
cd C:\actions-runner
./svc.cmd status
```

If the service is stopped, `./svc.cmd start` it. If it's running but
GitHub still shows offline, check Windows Event Viewer →
`Applications and Services Logs → GitHubActionsRunner`.

### Docker build cache balloons

```powershell
docker system df        # see how much space layers + volumes use
docker system prune     # remove dangling images/layers
docker builder prune    # remove build cache specifically
```

You can also delete the `beq-test:latest` image and rebuild from
scratch: `docker image rm beq-test:latest`.

### Ollama unreachable from inside the container

The compose file uses `network_mode: host`, so the container sees
Ollama exactly the way the host does. If integration tests report
connection refused:

- Check Ollama is bound to `0.0.0.0`, not `127.0.0.1` (WSL2 has its
  own loopback that's distinct from the Windows host's).
- From inside the container:
  `docker compose -f docker/docker-compose.test.yml run --rm beq-test-runner bash -c 'curl http://localhost:11434/api/tags'`

### TMDb rate-limit errors

TMDb gives generous free tier, but the experiment suite can hit it
hard on a fresh run. Set a TMDb API token in your environment and the
requests library will pick it up. Retries are built into the
integration tests — one blip usually doesn't fail the whole run.

### `Assert-LocalIntegrationConfig.ps1` says a path isn't reachable

Most common cause: the runner service account can't see drives that
your interactive user mapped. Either:

1. Re-register the runner under a named account:
   `./svc.cmd uninstall && ./svc.cmd install MYDOMAIN\runner-user`,
   enter the password, start the service.
2. Or mount the share inside the runner's session via a startup
   script that runs `net use` before the runner picks up a job.

Option 1 is recommended — one-time config, no moving parts.

### WAV cache exists but tests skip instead of running

The fixture-discovery code walks the cache looking for files matching
specific naming patterns. Confirm the mount is non-empty from inside
the container:

```powershell
docker compose -f docker/docker-compose.test.yml run --rm beq-test-runner `
    bash -c 'ls -la /wav-cache | head'
```

If `/wav-cache` is empty but the host path has files, the issue is
usually a drive-letter mapping that didn't persist.

## Uninstall / teardown

On grumpy:

```powershell
cd C:\actions-runner
./svc.cmd stop
./svc.cmd uninstall
./config.cmd remove --token <GENERATED_REMOVE_TOKEN>

# Drop the persisted config
Remove-Item -LiteralPath 'C:\ProgramData\beqdesigner-ci' -Recurse

# Optionally, reclaim Docker space
docker image rm beq-test:latest
docker system prune
```

On GitHub, the runner should disappear from the repo's runners list
within a few seconds of `./config.cmd remove`.
