# Branch plan: `claude/max-effort-2YyNi`

## Goal

Set up a self-hosted local-integration CI rig on **grumpy** (the
Windows + RTX box introduced in `docs/design/auto_beq_experiments.md`
at line 796) so that pushes to `div/local-integration` automatically
run the **full** spike test + experiment suite — unit + integration +
experiment, the three `scripts/run-spike-*.sh` runners in sequence —
inside a Docker container, without the dev laptop's involvement.

The purpose of the `div/local-integration` branch (which this branch
*enables* but does not become) is to be a dedicated "try this
experiment on grumpy's GPU / proper hardware before promoting it to a
real feature branch" sandbox. Having a push-triggered runner on grumpy
means iteration doesn't depend on the dev laptop's ~4× slower CPU.

## Parent branches

- `feats/neural-net-strat` — E82 production model + `train_production_model.py`
  + `generate_beq_profile.py`. The local-integration rig is the piece
  of infra that lets future neural-net experiments iterate faster.

## Current state

### Completed

- **W1**: Docker test image
  - `docker/Dockerfile.test` — `python:3.13-slim` + ffmpeg + libegl1 +
    full dev deps (scipy, xgboost, scikit-learn, pytest, PyQt6 in
    offscreen mode). Deliberately distinct from the lean stdlib-only
    `docker/Dockerfile` used by the NAS LFE extractor.
  - `docker/docker-compose.test.yml` — `beq-test-runner` service,
    `network_mode: host` (so integration tests can reach Ollama /
    TMDb without host.docker.internal gymnastics), volume mounts for
    WAV cache (read-only) + BEQ working dir + the repo work tree.
  - `docker/test-entrypoint.sh` — dispatcher that runs all three
    spike runners sequentially, collects the worst exit code, honours
    `TEST_SCOPE=unit|integration|all` (default `all`).

- **W2**: GitHub Actions workflow
  - `.github/workflows/local-integration.yml` — push-triggered on
    `div/local-integration`, `runs-on: [self-hosted, windows, grumpy]`,
    `concurrency.cancel-in-progress: true` (replaces the 30 s
    quiet-period debounce from the rejected polling-watcher design),
    `timeout-minutes: 360`, uploads `.pytest_cache/spike_*.log` as a
    retained artifact on every run.

- **W3**: PowerShell helpers on grumpy
  - `scripts/win/Configure-LocalIntegration.ps1` — one-shot
    interactive bootstrap; prompts for `WAV_CACHE_DIR` + `BEQ_DIR`
    with a **30 s timeout per prompt** via a custom
    `Read-HostWithTimeout` helper that polls `[Console]::KeyAvailable`;
    validates both paths are reachable; persists to
    `C:\ProgramData\beqdesigner-ci\config.json`. Exits 2 on timeout,
    non-interactive stdin, or path validation failure.
  - `scripts/win/Assert-LocalIntegrationConfig.ps1` — runs as the
    first step of the workflow job; read-only, never prompts, fails
    fast with a pointer to the Configure script if the file is
    missing or paths are unreachable. Exports `WAV_CACHE_DIR` /
    `BEQ_DIR` into `$env:GITHUB_ENV` for downstream steps.
  - `scripts/win/Run-ExperimentBuild.ps1` — `docker compose build` +
    `run` wrapper accepting `-Scope unit|integration|all`. Propagates
    the container's exit code so the workflow job fails loudly.

- **W4**: Documentation
  - `readme.md` — new "Quick start (local CI on a Windows build box)"
    subsection in the Auto-BEQ spike section, 6 numbered steps, links
    to the full doc.
  - `docs/local_integration.md` — full walkthrough: prerequisites,
    NAS-drive-mapping gotcha (`net use Z: \\nas\share /persistent:yes`),
    Configure-LocalIntegration.ps1 run, GitHub self-hosted runner
    registration + Windows service install, smoke test, environment
    variables table, troubleshooting (runner offline, docker cache
    full, Ollama unreachable, service account permissions), teardown.
  - `mkdocs.yml` — nav entry under the Auto-BEQ section.
  - `AGENTS.md` — scripts table extended with the new Dockerfile,
    compose file, entrypoint, and three PowerShell helpers. Branch
    plan pointer updated.
  - `docs/design/auto_beq_experiments.md` — new "Infrastructure
    improvements (2026-04-11)" grouped entry capturing the rig.

### In progress

_None — all W1–W4 deliverables are code-complete on this branch._

### Pending

- **W5**: End-to-end smoke test on grumpy itself. Requires someone
  with physical access to grumpy to:
  1. Install Docker Desktop (WSL2 backend), Git for Windows, PowerShell 7
  2. Clone the repo, `git checkout div/local-integration`
  3. Run `pwsh scripts/win/Configure-LocalIntegration.ps1`
  4. Register the self-hosted runner with label `grumpy` via GitHub
     repo Settings → Actions → Runners
  5. Install the runner as a Windows service (`.\svc.cmd install && .\svc.cmd start`)
  6. Push a trivial commit to `div/local-integration` and watch the
     Actions tab
  7. Confirm `test_auto_beq_nn_real.py::test_real_audio_roundtrip`
     actually runs (is NOT skipped for missing fixtures), proving the
     `WAV_CACHE_DIR` → `/wav-cache` mount wired through correctly
  See `docs/local_integration.md` §"Step-by-step setup" for the full
  checklist.

- **Follow-up** (not blocking): switch the runner to a named Windows
  account if LocalSystem can't see the mapped NAS drive. Documented as
  a troubleshooting bullet in `docs/local_integration.md`, but may
  need to become the default if mapping survives reboot less reliably
  than hoped.

## Key files

| File | Role |
|---|---|
| `docker/Dockerfile.test` | Full-deps test image (poetry + scipy + pytest + PyQt6 offscreen) |
| `docker/docker-compose.test.yml` | `beq-test-runner` service + volume mounts + host networking |
| `docker/test-entrypoint.sh` | Dispatcher: runs all three `scripts/run-spike-*.sh` sequentially, propagates worst exit code |
| `.github/workflows/local-integration.yml` | Push-triggered GH Actions workflow targeting `runs-on: [self-hosted, windows, grumpy]` |
| `scripts/win/Configure-LocalIntegration.ps1` | Interactive one-shot bootstrap with 30 s prompt timeout |
| `scripts/win/Assert-LocalIntegrationConfig.ps1` | Non-interactive workflow-time config check |
| `scripts/win/Run-ExperimentBuild.ps1` | `docker compose build` + `run` wrapper |
| `docs/local_integration.md` | End-user setup + troubleshooting doc |

## How to run it (short form)

On grumpy, one-time setup:

```powershell
# 1. Prereqs: Docker Desktop (WSL2) + Git for Windows + PowerShell 7
# 2. Clone the repo
git clone https://github.com/astubbs/beqdesigner.git
cd beqdesigner
git checkout div/local-integration

# 3. Persist WAV cache + BEQ dir paths (interactive, 30 s timeout)
pwsh scripts/win/Configure-LocalIntegration.ps1

# 4. Register the self-hosted runner
#    → GitHub → repo Settings → Actions → Runners → New self-hosted runner
#    → follow the copy-pasteable block; add `grumpy` as an extra label
#    → .\svc.cmd install && .\svc.cmd start

# 5. Smoke-test locally before pushing
pwsh scripts/win/Run-ExperimentBuild.ps1 -Scope unit
```

After that, every push to `div/local-integration` triggers the full
suite automatically via the GitHub Actions workflow. See
[`docs/local_integration.md`](../docs/local_integration.md) for the
long-form walkthrough and troubleshooting.

## Removed before merging

Per `AGENTS.md:86`, `branch-plans/plan-*.md` is working-context only
and gets deleted before the PR merges upstream.
