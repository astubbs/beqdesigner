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

**User-facing docs must be updated alongside code changes.** Updating
AGENTS.md is not enough — humans don't read it.

End-user docs live in three places, in priority order:

1. **`readme.md`** — developer quickstart (Quick start sections for
   Docker and local Python). The first thing anyone reads.
2. **`docs/lfe_extractor.md`** + other `docs/*.md` pages — the
   readthedocs site (`mkdocs.yml` controls the nav).
3. **`docs/design/*.md`** — internal design + experiment logs. Less
   visible to end users, but the source of truth for "why is the
   model the way it is".

Rules:

- If you add/rename/remove a script, update the **scripts tables in
  both `readme.md` and `AGENTS.md`**.
- If you change the Docker workflow (build steps, compose service
  names, output files, deploy steps), update **`readme.md` Quick
  start AND `docs/lfe_extractor.md`** together. AGENTS.md should not
  duplicate this — it should just reference the canonical doc.
- If you add a new user-facing CLI script or workflow, add a section
  to `readme.md` and consider whether it deserves its own
  `docs/*.md` page (and entry in `mkdocs.yml`).
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

Current branch plan: [`branch-plans/plan-neural-net-strat.md`](branch-plans/plan-neural-net-strat.md)
Parent branch plans: [`branch-plans/plan-audio-chunks-strat.md`](branch-plans/plan-audio-chunks-strat.md), [`branch-plans/plan-sharp-goldberg.md`](branch-plans/plan-sharp-goldberg.md)

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
| `scripts/run-spike-tests.sh` | Spike test suite — default excludes `integration` + `experiment` markers (fast unit only, ~1 min) | yes | no |
| `scripts/run-spike-integration.sh` | Spike tests marked `integration` — needs real media / TMDb / Ollama. Opt-in | yes | no |
| `scripts/run-spike-experiments.sh` | Spike tests marked `experiment` — F/G/H/I batches, real-audio training (minutes-to-hours). Opt-in | yes | no |
| `scripts/spike_auto_beq.py` | Interactive single-title CLI playground | no (run via poetry) | no |
| `scripts/sweep_report.py` | Generate unified sweep report + session summary | yes | no |
| `scripts/extract_lfe.py` | Extract LFE WAVs to portable cache. **Standalone** — no project deps, scp to NAS | yes | **yes** |
| `scripts/verify_wav_cache.py` | Verify WAV cache integrity, delete corrupt files | yes | no (imports wav_integrity) |
| `scripts/wav_cache_status.py` | Summarise WAV cache: counts, titles, author breakdown, growth | yes | no (imports helpers) |
| `scripts/nn_comparison_report.py` | Compare NN-predicted vs hand-coded BEQ filters, markdown output | yes | no (imports model + helpers) |
| `scripts/nn_f_experiment_report.py` | Generate markdown comparison report from F-experiment CSV | yes | no (reads CSV) |
| `scripts/nn_author_pattern_report.py` | Per-author distribution analysis from BEQ catalogue | yes | no (reads JSON) |
| `scripts/nn_cache_bias_report.py` | WAV cache vs catalogue distribution bias report | yes | no (imports helpers) |
| `scripts/nn_acquisition_recommender.py` | Recommend N missing catalogue titles to acquire (greedy bias correction). Reads `media_inventory.json` to exclude already-owned titles. | yes | no (imports helpers) |
| `scripts/train_production_model.py` | Train + save the E82 50:1 weighted hybrid production model to `{beq-dir}/production_model.joblib` (+ metadata sidecar) | yes | no (imports model + helpers) |
| `scripts/train_torch_model.py` | Train + save the E85 differentiable-DSP production model to `{beq-dir}/e85_torch_filter.pt`. Uses E82 XGBoost as warm-start teacher, then fine-tunes on acoustic loss. **New champion (1.49 dB)** | yes | no (imports model + torch) |
| `scripts/run_tier1_comparison.py` | Run all Tier 1 experiments (E82/E83/E84/E85) on the same split, produce a unified leaderboard + CSV | yes | no (imports all models) |
| `scripts/run_e85_experiment.py` | Standalone E85 experiment runner (train + eval on the E82 test split) | yes | no |
| `scripts/generate_beq_profile.py` | Generate complete BEQ profiles for uncatalogued media. Supports `AUTO_BEQ_ADVISOR=trained_model` (E82 XGBoost) or `AUTO_BEQ_ADVISOR=torch_differentiable` (E85 diff-DSP champion) | yes | no (imports model + helpers) |
| `docker/Dockerfile` | Docker image: Python 3.13-slim + ffmpeg + project source | — | — |
| `docker/docker-compose.example.yml` | Example compose config — copy, edit paths, run | — | — |

## Docker (NAS LFE extraction)

**Canonical docs**: end-user instructions live in
[`docs/lfe_extractor.md`](docs/lfe_extractor.md) (published on
readthedocs) and the **Quick start (Docker)** section of `readme.md`.
Both cover the build → deploy → run loop, the volume mount layout,
the output files, and how to pull results back to the dev machine.

When making changes that affect end users (new compose service, new
output file, new flag, new deploy step), **update both `readme.md`
and `docs/lfe_extractor.md` together** — AGENTS.md should not
duplicate the workflow.

Service names in `docker-compose.example.yml` are deliberately
prefixed with `beq-` (`beq-lfe-extract`, `beq-wav-verify`) so they
don't collide with other compose stacks on the same host.

All `.sh` scripts must have the executable flag set (`chmod +x`).

## Running spike tests

**Always use the `scripts/run-spike-*.sh` wrappers to run spike tests.** Never
invoke `poetry run pytest` directly for spike tests. The wrappers exist
so that repeated runs share a single permission approval — each
distinct `poetry run pytest ...` command line requires a fresh
approval, which is disruptive during iteration.

### Three runner scripts, three test groups

The spike suite is segregated by pytest markers. **CI runs only the
default group**; the other two are opt-in.

| Runner | Marker filter | Runtime | When to use |
|---|---|---|---|
| `bash scripts/run-spike-tests.sh` | `not integration and not experiment` (default) | ~1 min | Every iteration, CI, pre-commit. Hermetic — no media scans, no network, no model retraining. |
| `bash scripts/run-spike-integration.sh` | `integration` | minutes | Verifying code that touches real external resources (media files, TMDb API, Ollama hosts, library sweep config). Tests skip if their resources aren't configured locally. |
| `bash scripts/run-spike-experiments.sh` | `experiment` | **minutes to hours** | Reproducing or iterating on F/G/H/I experiment batches, real-audio training (E77/E82), chunked-strategy comparison. Not for CI. |

### Marker rules

- **Default** (no marker): unit tests. Must be hermetic, no filesystem/network/model-training side effects. The 500 MB media-size filter in `inventory_root` is automatically bypassed for tests via the `_allow_zero_byte_fixtures` autouse fixture in `test_sweep_discover.py`.
- **`@pytest.mark.integration`**: needs real media files, TMDb, Ollama, or a populated `~/.config/beqdesigner/auto_beq_sweep.json`. Apply via file-level `pytestmark = pytest.mark.integration` when every test in the file needs external resources, or decorate individual tests when the file is mixed (e.g. `test_auto_beq.py::test_real_media_roundtrip`).
- **`@pytest.mark.experiment`**: retrains one or more models from scratch. F/G/H/I/real/chunked/extract test files all carry `pytestmark = pytest.mark.experiment` at the top.

Markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`. Adding a new marker needs both a `pytestmark = ...` in the test file and a registration entry in `pyproject.toml`.

### Env vars honoured by all three wrappers

| Var | Purpose | Default |
|---|---|---|
| `SPIKE_TEST` | pytest selector (file path or nodeid) | `src/test/python/spike/` (all) |
| `SPIKE_MARKERS` | override marker filter (`run-spike-tests.sh` only) | `not integration and not experiment` |
| `AUTO_BEQ_ADVISOR` | advisor impl: heuristic / mock / ollama / measurement | `measurement` |
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

Run a specific experiment (opt-in):

```bash
SPIKE_TEST='src/test/python/spike/test_auto_beq_nn_experiments.py::test_g_experiment_comparison' \
  bash scripts/run-spike-experiments.sh
```

Run the library sweep with a custom config (opt-in integration):

```bash
AUTO_BEQ_SWEEP_CONFIG=/tmp/sweep.json \
AUTO_BEQ_SWEEP_LIMIT=3 \
SPIKE_TEST=src/test/python/spike/test_auto_beq_library_sweep.py \
  bash scripts/run-spike-integration.sh
```

**Do not** invoke the venv python, `poetry run python`, or `poetry run
pytest` directly for spike tests. If a new script/runner is needed, add
it under `scripts/` with a fixed command line.

## Verifying behaviour

**Always verify behaviour with a test, not ad-hoc code.** When checking
that something works (edge cases, parsing, formatting, etc.), write a
test that captures the expectation. Don't run throwaway Python snippets
or inline assertions — if it's worth verifying, it's worth a test.
