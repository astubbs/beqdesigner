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

## Scripts

| Script | Purpose | Executable | Standalone? |
|---|---|---|---|
| `scripts/run-sweep-discover.sh` | Discover media + match catalogue | yes | no |
| `scripts/run-sweep-tests.sh` | Run auto-BEQ pipeline on discovered media | yes | no |
| `scripts/run-spike-tests.sh` | Full spike test suite (unit + integration) | yes | no |
| `scripts/spike_auto_beq.py` | Interactive single-title CLI playground | no (run via poetry) | no |
| `scripts/sweep_report.py` | Generate unified sweep report + session summary | yes | no |
| `scripts/extract_lfe.py` | Extract LFE WAVs to portable cache. **Standalone** — no project deps, scp to NAS | yes | **yes** |
| `scripts/verify_wav_cache.py` | Verify WAV cache integrity, delete corrupt files | yes | no (imports wav_integrity) |
| `scripts/wav_cache_status.py` | Summarise WAV cache: counts, titles, author breakdown, growth | yes | no (imports helpers) |
| `scripts/nn_comparison_report.py` | Compare NN-predicted vs hand-coded BEQ filters, markdown output | yes | no (imports model + helpers) |
| `scripts/nn_f_experiment_report.py` | Generate markdown comparison report from F-experiment CSV | yes | no (reads CSV) |
| `scripts/nn_author_pattern_report.py` | Per-author distribution analysis from BEQ catalogue | yes | no (reads JSON) |
| `scripts/nn_cache_bias_report.py` | WAV cache vs catalogue distribution bias report | yes | no (imports helpers) |
| `scripts/nn_acquisition_recommender.py` | Recommend N missing catalogue titles to acquire (greedy bias correction) | yes | no (imports helpers) |
| `docker/Dockerfile` | Docker image: Python 3.13-slim + ffmpeg + project source | — | — |
| `docker/docker-compose.example.yml` | Example compose config — copy, edit paths, run | — | — |

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
poetry run python3 scripts/wav_cache_status.py /path/to/wav-cache
```

**After code changes:** rebuild image and redeploy:
```bash
docker build -f docker/Dockerfile -t beq-extract .
docker save beq-extract | ssh nas docker load
```

All `.sh` scripts must have the executable flag set (`chmod +x`).

## Running spike tests

**Always use `bash scripts/run-spike-tests.sh` to run spike tests.** Never
invoke `poetry run pytest` directly for spike tests. The wrapper exists
so that repeated runs share a single permission approval — each
distinct `poetry run pytest ...` command line requires a fresh
approval, which is disruptive during iteration.

The wrapper honours these env vars (set them inline on the same line):

| Var | Purpose | Default |
|---|---|---|
| `SPIKE_TEST` | pytest selector (file path or nodeid) | `src/test/python/spike/` (all) |
| `AUTO_BEQ_ADVISOR` | advisor impl: heuristic / mock / ollama / measurement | `mock` |
| `SPIKE_VERBOSE` | `1` enables `-s` (no capture) | `0` |
| Any test-specific env var | passed through to pytest | — |

Canonical invocation pattern (works from the repo root):

```bash
SPIKE_TEST=src/test/python/spike/test_auto_beq.py \
  bash scripts/run-spike-tests.sh
```

For a single test within a file, use `::`:

```bash
SPIKE_TEST='src/test/python/spike/test_auto_beq.py::test_synthetic_roundtrip' \
  bash scripts/run-spike-tests.sh
```

Stack env vars for configuration:

```bash
AUTO_BEQ_SWEEP_CONFIG=/tmp/sweep.json \
AUTO_BEQ_SWEEP_LIMIT=3 \
AUTO_BEQ_ADVISOR=measurement \
SPIKE_TEST=src/test/python/spike/test_auto_beq_library_sweep.py \
  bash scripts/run-spike-tests.sh
```

**Do not** invoke the venv python, `poetry run python`, or `poetry run
pytest` directly for spike tests. If a new script/runner is needed, add
it under `scripts/` with a fixed command line.

## Verifying behaviour

**Always verify behaviour with a test, not ad-hoc code.** When checking
that something works (edge cases, parsing, formatting, etc.), write a
test that captures the expectation. Don't run throwaway Python snippets
or inline assertions — if it's worth verifying, it's worth a test.
