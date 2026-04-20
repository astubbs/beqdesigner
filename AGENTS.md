# Agent instructions

## Mandatory rules

### Git safety

**NEVER commit or push without explicitly asking the user first.**
Wait for approval. This is the #1 rule. The user almost always spots
a bug during validation, so committing prematurely creates noise in
the git log. Let them test first.

**User acceptance before every commit.** Do not ask to commit until
the user has tested the implementation as an end user and confirmed it
looks right. Implement, let them verify, fix anything they flag, then
ask to commit. No exceptions.

**PR discipline:** after opening a PR, follow up on the duplication
reports. Duplicate-code and file-similarity tools post comments
flagging new clones and similarity warnings. Read them, identify
duplication introduced by *this* PR, and refactor to remove it
before the PR merges.

**Run the full default test suite before every commit.** Use
`poetry run pytest -m "not integration and not experiment"` (~1 min).
This catches cross-module breakage that targeted test runs miss. Do
not commit if any test fails.

### Development discipline

- **Skateboard first.** Build the simplest end-to-end thing that
  works, then improve it. Don't go deep into interesting features
  before shipping. Before starting any feature, ask: "Is this
  blocking the next public milestone?" If not, flag it and move on.
- **Never paper over the real problem** -- make the proper fix. No
  band-aids, no lazy imports to avoid a refactor, no duplicated code
  to work around a dependency. If the architecture is wrong, fix
  the architecture.
- **Don't propose workarounds that require user action** when the
  software can solve it. If the software has enough information to
  derive the right answer, it should just do it.
- **Never hardcode local paths.** All paths must be configurable
  (env var, settings file, or CLI flag). Never commit paths to
  specific machines, volume mounts, or user directories. Use
  `beq_shared_dir()`, `beq_config_dir()`, or env vars instead.
- **Be DRY.** Reuse existing functions. Don't copy code -- refactor
  where necessary. Extract common patterns into shared utilities.
- **Use shared libraries, refactor where needed.** When a pattern
  exists in one place and is needed in another, move it to a shared
  module (e.g. `model/wav_cache.py`, `model/media_utils.py`). Do NOT
  copy-paste or write a parallel implementation. Code that drifts out
  of sync between modules is a recurring source of bugs.
- **Give things meaningful names** that describe what they do. Never
  use random or generic names.
- **Never weaken test assertions** -- classify exceptions instead of
  ignoring them.
- When you fix something or finish implementing something, record
  what lessons you learnt.
- Don't write with em dash characters.
- **If constructing data in memory** that is eventually going to be
  saved, save it as soon as it's created. Don't delay in case the
  programme crashes or the user exits.

### Architecture and state

- **Collapse parallel state when bugs recur.** If a subsystem keeps
  sprouting new bugs and each fix adds a Map/Set/flag or a "when X
  changes, update Y" sync hook, stop. Those are symptoms of too many
  caches holding the same information in different shapes. Draw the
  state graph, identify the minimal keying, and collapse.
- **Pick the full cache key up front.** Key a cache by everything it
  needs to distinguish. When you later need an additional dimension,
  re-key the original; do not add a parallel cache.
- **Reducer pattern for concurrent mutations.** When multiple async
  operations write to one reactive container in parallel, the mutation
  API must take a mutation function that reads latest state --
  `update(prev => next)`, not `apply(new Map(snapshot))`.

### UI discipline

- **Don't mutate UI in response to toggle state.** When a
  checkbox/switch is turned on, don't append "(active)" to its label,
  don't reveal helper text that was hidden when off, don't change
  button text. The control itself (checked state, color, focus) already
  shows whether it's on. Only change UI across genuinely different
  modes (e.g., "+ Add" vs "In Library"), not on/off status of the
  same feature.
- **Verify UI with your own eyes.** When building or changing any
  interface (visual or text), use your tools to verify the result
  looks correct and is high quality. For CLI output, run the command
  and read the output. Do not assume a UI change is correct just
  because it compiles. Look at it.
- **Show progress for any wait over 200ms.** When the user won't get
  an instant response (I/O, network, builds, long computations),
  start rendering progress as soon as the delay exceeds ~200ms. Show
  as much information as possible: percentage complete, ETA, time
  elapsed, rate, and x/y counts. Do a small amount of upfront work
  to discover the total (y) so progress bars can be meaningful. Never
  leave the user staring at a blank screen wondering if something is
  happening. Use `ProgressLogger` from `model/media_utils.py` (see
  shared infrastructure section) - it handles time-throttling, ETA,
  and first-update-always-fires.

### UI/CLI layer separation

**CLI and service code must never depend on Qt.** The `model/` layer
and `cli/` layer must be importable without PyQt6. If a `model/`
module imports from `model/preferences.py` or any Qt widget, that
dependency must be refactored out so headless Docker containers and
CLI scripts work without a display server.

Current known violation: `model/auto_beq.py` -> `model/iir.py` ->
`model/xy.py` -> `model/preferences.py` -> PyQt6. This blocks
profile generation in Docker. Tracked for refactoring.

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
  `beq_config_dir()`, `beq_shared_dir()` — single source of truth for paths.
  `beq_shared_dir()` returns the *shared* (portable) directory; `beq_config_dir()`
  returns the *local* machine-specific config directory (`~/.config/beqdesigner/`).
  Machine-specific config (like `extract_config.json` with media root paths)
  lives in the local dir; portable data (wav-cache, catalogue, inventory) lives
  in the shared dir. **Canonical reference**: see the "Configuration storage"
  section in [`docs/architecture.md`](docs/architecture.md) for the full
  schema of both files and the split rationale.
- **Extract config service**: `get_configured_media_roots()` (read) and
  `save_extract_config()` (write) in `cli/extract.py` — single code path
  for media root persistence. Tests must use this service, not write JSON
  directly.
- **Media depth detection**: `find_media_dirs()` in `model/media_utils.py` —
  finds the directory depth where media files live by sampling one file,
  then lists all directories at that depth. Adapts to any library
  layout (flat, one-level, genre-grouped). Use for progress bars and
  directory counting.
- **Progress logging**: `ProgressLogger` in `model/media_utils.py` —
  time-throttled progress with ETA. Only logs when `min_interval_s`
  has elapsed (default 5s) or on the final item. Use for any loop
  that may take >5s: directory walks, file scans, extraction batches.

If a shared function is missing a feature you need, extend it rather
than writing a parallel implementation.

### Save results incrementally

**Never defer saving results to the end of a long operation.** Any
loop that builds up data (directory scanning, extraction, inventory
construction) must save its results to disk as each logical unit
completes — not in a single batch at the end. Users will Ctrl+C
long-running operations on NAS/NFS volumes, and losing 20 minutes
of work because the save was deferred to the end is unacceptable.

Rules:
- Save after each media root finishes scanning.
- Save periodically during walks within a single root (every ~30s).
- Save after each stage boundary (parent check → rescan → walk).
- Extraction is inherently incremental (each WAV is an atomic file).

### Log before I/O

**Any I/O operation that could pause must be logged before it starts.**
This includes filesystem scans (`rglob`, `exists()` on network paths),
HTTP requests, and database queries. The user must always have an
explanation for any delay — a CLI that pauses silently is broken.

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
4. **`docs/solutions/`** — documented solutions to past problems
   (bugs, best practices, workflow patterns), organized by category
   with YAML frontmatter (`module`, `tags`, `problem_type`).
   Relevant when implementing or debugging in documented areas.

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

### Accessible writing rule

**All documentation must be accessible to developers who have NO
background in machine learning or audio engineering.** This is a hard
project rule, not a guideline.

Why: "vibe coding" risk — AI-assisted development can produce
systems whose internals nobody understands when returning cold. If
the docs use jargon without explanation, future readers (including
the original authors) can't reason about the system.

Mandatory practices:

- **Define every abbreviation on first use.** MSE → "Mean Squared
  Error (MSE) — the average of (prediction minus target) squared".
  XGBoost → "XGBoost (eXtreme Gradient Boosting) — a machine
  learning algorithm that builds ensembles of decision trees".
- **Link to external explanations** for non-trivial concepts: the
  Audio EQ Cookbook for biquad maths, PyTorch docs for tensors,
  3Blue1Brown for neural network intuition, Wikipedia for DSP terms.
- **Teach concepts progressively**: start with "why does this matter
  for our project" before "how does the maths work". Use analogies
  and concrete examples over abstract definitions.
- **Target reader**: a competent Python developer who has never
  touched machine learning, digital signal processing, or audio
  engineering. They should be able to read
  `docs/design/auto_beq_how_it_works.md` and understand the full
  system.
- **When in doubt, over-explain.** A developer who already knows can
  skim; a developer who doesn't know can't guess.

The canonical "plain language" overview lives at
`docs/design/auto_beq_how_it_works.md`. Any new technique added to
the system must be explained there in the same accessible style.

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
| `cli/train_production_model.py` | Train E82 production XGBoost model |
| `cli/train_torch_model.py` | Train E85 differentiable-DSP model |
| `cli/nn_acquisition_recommender.py` | Acquisition recommendation report |
| `cli/nn_author_pattern_report.py` | Per-author distribution report |
| `cli/nn_cache_bias_report.py` | Cache vs catalogue bias report |
| `cli/nn_f_experiment_report.py` | F-experiment comparison report |

### Other files

| Path | Purpose |
|---|---|
| `bin/beq-designer` | Entry point (executable, auto-enters poetry venv) |
| `scripts/` | Developer helper scripts (convenience wrappers, not end-user programs) |
| `build/regen_ui.py` | Build tool: regenerate Python from Qt `.ui` files |
| `docker/Dockerfile` | Docker image for NAS deployment |
| `docker/docker-compose.example.yml` | Example compose config |
| `experiments/*.py` | One-off experiment runners (E85-E87, tier1 comparison) |

**`bin/` vs `scripts/`**: `bin/` is exclusively for end-user programs.
Developer convenience scripts (test runners, code generators, CI
helpers) go in `scripts/`. Do not add developer-only executables to
`bin/`.

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

```bash
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

## Running tests

**Use `poetry run pytest` to run tests.** The `pythonpath` setting in
`pyproject.toml` ensures both `src/main/python` and `src/test/python`
are on the import path automatically.

### Three test groups (pytest markers)

The spike suite is segregated by pytest markers. **CI runs only the
default group**; the other two are opt-in.

| Command | Marker filter | Runtime | When to use |
|---|---|---|---|
| `poetry run pytest -m "not integration and not experiment"` | default | ~1 min | Every iteration, CI, pre-commit. Hermetic - no media scans, no network, no model retraining. |
| `poetry run pytest -m integration` | integration | minutes | Verifying code that touches real external resources (media files, TMDb API, Ollama hosts, library sweep config). Tests skip if their resources aren't configured locally. |
| `poetry run pytest -m experiment` | experiment | **minutes to hours** | Reproducing or iterating on F/G/H/I experiment batches, real-audio training (E77/E82), chunked-strategy comparison. Not for CI. |

**Note:** torch tests (`test_auto_beq_torch.py`) should run in a
separate pytest invocation to avoid a segfault caused by torch + PyQt6
in the same process on macOS (MPS conflict):
```bash
poetry run pytest --ignore=src/test/python/spike/test_auto_beq_torch.py -m "not integration and not experiment"
poetry run pytest src/test/python/spike/test_auto_beq_torch.py
```

### Marker rules

- **Default** (no marker): unit tests. Must be hermetic, no filesystem/network/model-training side effects. The 500 MB media-size filter in `inventory_root` is automatically bypassed for tests via the `_allow_zero_byte_fixtures` autouse fixture in `test_sweep_discover.py`.
- **`@pytest.mark.integration`**: needs real media files, TMDb, Ollama, or a populated `~/.config/beqdesigner/auto_beq_sweep.json`. Apply via file-level `pytestmark = pytest.mark.integration` when every test in the file needs external resources, or decorate individual tests when the file is mixed (e.g. `test_auto_beq.py::test_real_media_roundtrip`).
- **`@pytest.mark.experiment`**: retrains one or more models from scratch. F/G/H/I/real/chunked/extract test files all carry `pytestmark = pytest.mark.experiment` at the top.

Markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`. Adding a new marker needs both a `pytestmark = ...` in the test file and a registration entry in `pyproject.toml`.

### Env vars

| Var | Purpose | Default |
|---|---|---|
| `AUTO_BEQ_ADVISOR` | advisor impl: heuristic / mock / ollama / measurement | `measurement` |
| `AUTO_BEQ_MODEL_PATH` | path to production model file | auto-detected |
| `BEQ_SHARED_DIR` | shared BEQ directory (wav-cache, catalogue, inventory) - **required** | from `shared_beq_dir` in settings.json |

```bash
# Default test suite (unit tests only)
poetry run pytest -m "not integration and not experiment"

# Specific test file
poetry run pytest src/test/python/spike/test_auto_beq.py -v

# Single test
poetry run pytest 'src/test/python/spike/test_auto_beq.py::test_synthetic_roundtrip' -v

# Integration tests (opt-in, needs real media/TMDb/Ollama)
poetry run pytest -m integration

# Experiment tests (opt-in, minutes to hours)
poetry run pytest -m experiment src/test/python/spike/test_auto_beq_nn_experiments.py
```


## Verifying behaviour

**Always verify behaviour with a test, not ad-hoc code.** When checking
that something works (edge cases, parsing, formatting, etc.), write a
test that captures the expectation. Don't run throwaway Python snippets
or inline assertions -- if it's worth verifying, it's worth a test.

Additional test discipline:
- Search for existing test harnesses and utilities before creating new
  ones.
- Run the complete test suite periodically, not just targeted tests.
- Maintain good high-level test coverage. Only get detailed on
  particularly complex functions that benefit from fine-grained testing.
- **Evaluate every diagnostic script for becoming a test.** If you
  write a one-off script to investigate a bug or verify behavior,
  assess whether it should be a permanent test before moving on.
  Keep it if it verifies behavior that could regress. Drop it if
  it was a one-time check (file exists, print a value).
