# Agent instructions

## Mandatory rules

### Experiment logging

**Every experiment MUST be logged in
`docs/design/auto_beq_experiments.md` before committing.** This is
non-negotiable. The experiment log is the source of truth for "what
we tried and what happened". If you ran it, log it.

Each entry must include:
- Experiment ID (E1, E2, ... sequential)
- What was changed and why
- Quantitative results (per-title mean/max error, verdict counts)
- Lesson learned
- Whether the change was kept or reverted

Infrastructure changes that aren't algorithm experiments (new scripts,
cache dir, parallelism, etc) get a grouped entry under
"Infrastructure improvements" with the date range.

### Preserving experiment code

**Never delete experimental code paths.** Each experiment (E1, E2,
etc) should be preserved as a selectable alternative, not overwritten
by the next attempt.

In practice this means:
- New algorithms go in new classes/functions (e.g.
  `MeasurementAdvisor`, `HeuristicAdvisor`, `OllamaAdvisor` are all
  kept — not replaced one-for-one).
- The `Advisor` protocol + `get_advisor(name)` factory lets us select
  between approaches at runtime via `AUTO_BEQ_ADVISOR` env var.
- When iterating on a formula within an advisor, keep the old formula
  accessible (e.g. as a flag, a subclass, or a clearly-named private
  method) rather than editing it in place.
- Tests should be written so they can run against ANY advisor
  implementation by changing the env var, not hardcoded to one.

The goal: at any point we can re-run any prior experiment's code
path to compare results. The experiment log says what happened; the
code lets us reproduce it.

Keep experimental code DRY and modular. Shared logic (fitter,
feature extraction, smoothing, caching) lives in common modules.
Only the decision-making (gain formula, knee detection, topology
classification) differs between experiments.

### Shared infrastructure — no forking

**Never duplicate audio extraction, probing, or configuration loading.**
Use the single shared implementations in `spike/_auto_beq_helpers.py`:

- **LFE extraction**: `extract_lfe_wav()` — handles caching, LFE detection,
  atomic writes, WAV validation. Never inline ffmpeg extraction.
- **Audio probing**: `probe_audio_stream()` — single ffprobe wrapper.
- **Configuration**: `load_settings()` / `save_settings()` — reads/writes
  `~/.config/beqdesigner/settings.json`. Never create separate config files.
- **Cache directories**: `audio_cache_dir()`, `wav_cache_dir()`,
  `beq_config_dir()` — single source of truth for paths.

If a shared function is missing a feature you need, extend it rather
than writing a parallel implementation.

### Documentation

**README and design docs must be updated alongside code changes.**

- `README.md` quickstart section must reflect current scripts and
  workflow. If you add/rename/remove a script, update the README.
- `docs/design/auto_beq.md` must reflect current architecture.
- `AGENTS.md` must reflect current scripts table and env vars.
- `branch-plans/plan-*.md` must reflect current branch state.

Do NOT commit code changes without checking whether the docs need
updating. Stale docs are worse than no docs.

## Branch plans

**Rule**: when working on a feature branch, save the branch's current
plan/intent to `branch-plans/plan-<branch-name>.md` (project root).

- Captures the goal, current state, and next steps for the branch.
- Lets other agent sessions pick up the branch mid-flight.
- Is committed alongside code changes and experiment logs.
- Is **removed before merging PRs upstream** (working context only).

Also keep experiment logs and living design docs updated and committed
alongside code — these are gold for resuming work across sessions.

Current branch plan: [`branch-plans/plan-audio-chunks-strat.md`](branch-plans/plan-audio-chunks-strat.md)
Parent branch plan: [`branch-plans/plan-sharp-goldberg.md`](branch-plans/plan-sharp-goldberg.md)

## Ollama model usage

**Use small fast models for integration tests, proper models for
accuracy tests.**

- **Integration tests** (testing plumbing works, multi-host round-
  robin, JSON parsing, etc): use `llama3.2:latest` or any small
  model that responds in <5s. Set `OLLAMA_MODEL=llama3.2:latest`.
- **Accuracy/profile-quality tests** (sweep, real-media roundtrip):
  use `qwen:14b` or larger. Set `OLLAMA_MODEL=qwen:14b`.
- **MeasurementAdvisor** (default advisor, no LLM): no Ollama needed.
  Set `AUTO_BEQ_ADVISOR=measurement`.

When running accuracy tests, process **one media file at a time**
so the user can see results incrementally: `AUTO_BEQ_SWEEP_LIMIT=1`.

## CLI

All operations are accessed through a single entry point: `bin/beq-designer`.
Run with no arguments for an interactive menu, or use subcommands directly.

```bash
bin/beq-designer                        # interactive menu
bin/beq-designer profile movie.mkv      # generate BEQ profile
bin/beq-designer extract --media-root . # extract LFE to cache
bin/beq-designer dev test               # run spike tests
bin/beq-designer --help                 # list all subcommands
```

### CLI module layout (`src/main/python/cli/`)

| Module | Purpose |
|---|---|
| `cli/main.py` | Unified typer app — menu + all subcommands |
| `cli/common.py` | Shared utilities (filterable_select, config, banner) |
| `cli/profile.py` | Profile generation (interactive directory browser, progress) |
| `cli/generate.py` | Profile pipeline (LFE → NN → biquads → spectrographs → JSON) |
| `cli/extract.py` | LFE extraction to portable WAV cache |
| `cli/cache_status.py` | WAV cache summary report |
| `cli/verify_cache.py` | WAV cache integrity check |
| `cli/nn_report.py` | NN vs catalogue comparison report |
| `cli/sweep_report.py` | Experiment sweep comparison report |
| `cli/spike_playground.py` | Dev: single-title filter proposal test |

### Other files

| Path | Purpose |
|---|---|
| `bin/beq-designer` | Entry point (executable, auto-enters poetry venv) |
| `build/regen_ui.py` | Build tool: regenerate Python from Qt `.ui` files |
| `docker/Dockerfile` | Docker image for NAS deployment |
| `docker/docker-compose.example.yml` | Example compose config |

## Docker (NAS LFE extraction)

The extraction pipeline runs in Docker for NAS deployment — no code
duplication, no scp of scripts, no version drift.

**Setup (one-time):**
```bash
# Build image
docker build -f docker/Dockerfile -t beq-extract .

# Deploy to NAS
docker save beq-extract | ssh nas docker load

# Copy and edit compose config on NAS
scp docker/docker-compose.example.yml nas:/path/to/beqdesigner/docker-compose.yml
# Edit volume paths to match NAS media layout
```

**Usage on NAS:**
```bash
cd /path/to/beqdesigner

# Extract LFE WAVs (resumes from cache, breadth-first ordering)
docker compose run extract

# Verify WAV cache integrity
docker compose run verify

# Check training set status (run on dev machine, not Docker — needs numpy)
bin/beq-designer cache-status
```

**After code changes:** rebuild image and redeploy:
```bash
docker build -f docker/Dockerfile -t beq-extract .
docker save beq-extract | ssh nas docker load
```

## Running spike tests

**Use `bin/beq-designer dev test` to run spike tests.**

```bash
# All spike tests
bin/beq-designer dev test

# Specific test file
bin/beq-designer dev test --file src/test/python/spike/test_auto_beq.py

# Single test
bin/beq-designer dev test --file 'src/test/python/spike/test_auto_beq.py::test_synthetic_roundtrip'

# With specific advisor and verbose output
bin/beq-designer dev test --advisor measurement --verbose

# Run sweep pipeline
bin/beq-designer dev sweep --limit 3 --advisor measurement
```

## Verifying behaviour

**Always verify behaviour with a test, not ad-hoc code.** When checking
that something works (edge cases, parsing, formatting, etc.), write a
test that captures the expectation. Don't run throwaway Python snippets
or inline assertions — if it's worth verifying, it's worth a test.
