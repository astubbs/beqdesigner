---
title: Reassess/Training Pipeline UX Improvements
status: complete
created: 2026-04-20
scope: standard
---

# Reassess/Training Pipeline UX Improvements

## Problem Frame

The `bin/beq-designer reassess` pipeline (which runs `experiments/run_tier1_comparison.py`) produces results that lack critical context:

1. **No model metadata** - the output shows experiment names and dB scores but not which model was used, when it was trained, or how old the training data is. A developer returning to this after a week can't tell if the results are fresh or stale.
2. **No champion tracking** - the only "champion" concept is a hardcoded verdict string. There's no persistent record of which model was the previous best, so the comparison is always against E82 baseline, not against the last best model.
3. **No staleness detection** - `production_model.meta.json` records `wav_cache_mtime` at training time, but nothing reads it back to compare against current cache state. The developer has no way to know if retraining is warranted.
4. **Whisper loading spam** - E83 logs "loading foundation model: whisper-tiny (one-time cost)" repeatedly because the `.npy` disk cache files aren't being found (the cache directory exists but the lookup key doesn't match what was written). The in-memory cache (`_FOUNDATION_MODEL_CACHE`) works correctly, so the model only loads once per process, but the log message fires on every `extract_foundation_embedding()` call that misses the disk cache.
5. **WAV count confusion** - log messages say "563 WAV-catalogue pairs" without explaining that this is the number of unique catalogue-matched titles (not total WAVs on disk), and separately "448 training entries" without explaining the 80/20 split. Users ask "where did the rest go?"
6. **One-episode-per-title unexplained** - the deduplication in `deduplicate_by_title()` selects one episode per title for training, but the output doesn't explain why. Users see fewer training samples than WAVs and assume data is being lost.

## Scope

### In scope

- U0: Fix test infrastructure - `poetry run pytest` should work without wrappers; remove `dev test` from end-user CLI
- U1: Model metadata banner in tier1 comparison and train-production outputs
- U2: Champion history file (`champion_history.json`) for tracking progression
- U3: Auto-suggest retraining when WAV cache has changed since last training
- U4: Fix Whisper `.npy` cache miss (the disk cache key mismatch)
- U5: Clarify WAV count and split messages in tier1 comparison output
- U6: Explain one-episode-per-title deduplication in output

### Out of scope

- Changing the actual training algorithms or experiment configurations
- Auto-retraining (only auto-*suggesting*; the user triggers training)
- Vendoring the Whisper encoder (separate task)
- ~~Unifying library_roots and media_roots config~~ (done - media roots now stored in extract_config.json only)
- Full CLI/UI menu redesign (but see "Future: CLI and UI menu consistency" note below)

## Requirements Trace

| # | Requirement | Source |
|---|---|---|
| R1 | Every model-using output shows: model type, training date (human-readable), age, feature count, foundation model (if any) | User: "whenever we're using our models, we need to output which model is being used" |
| R2 | Tier1 comparison shows which experiment is the new champion AND which was the previous champion | User: "the comparison should show us which model is the best and which model used to be the best" |
| R3 | Pipeline auto-detects when training data has changed and suggests retraining | User: "can we have it dynamically suggest to retrain the models if they become older or if change has been detected" |
| R4 | Whisper model loading message appears at most once per process, not per embedding call | User: "E83 is still causing the loading foundation model whispered tiny to be output many times" |
| R5 | WAV count messages explain what the numbers mean (unique titles vs total files, train vs test split) | User: "the 563 wave cache number still isn't explained" |
| R6 | Output explains why one episode per title is used for training | User: implied by confusion about 448 vs 563 counts |

## Implementation Units

### U0: Fix test infrastructure and clean up dev subcommand

**Goal:** Make `poetry run pytest` work out of the box (no wrapper script needed) and remove developer-only test commands from the end-user CLI. Keep end-user-relevant advanced commands (train, reassess) in the CLI.

**Files:**
- Modify: `pyproject.toml` - add `pythonpath = ["src/main/python", "src/test/python"]` to `[tool.pytest.ini_options]`
- Modify: `src/main/python/cli/main.py` - remove `dev test` subcommand; keep `dev reassess`, `dev train`, `dev train-torch` (these are end-user operations)
- Modify: `AGENTS.md` - update test runner instructions from `bin/beq-designer dev test` to `poetry run pytest`
- Modify: `docs/plans/reassess-pipeline-ux.md` - update all verification steps that reference `bin/beq-designer dev test`

**Approach:**
- The only reason `poetry run pytest` fails today is that `pyproject.toml` lacks `pythonpath`. One line fixes it.
- `dev test` is a pytest wrapper that adds sys.path and invokes pytest with marker filters. Once `pythonpath` is configured, the standard pytest invocations are equivalent:
  - `poetry run pytest src/test/python/spike/` (default suite)
  - `poetry run pytest -m integration` (integration tests)
  - `poetry run pytest -m experiment` (experiment tests)
- The `dev reassess` and `dev train` / `dev train-torch` commands stay - these are end-user operations (running experiments, training production models), not developer tooling.
- Consider renaming `dev` to something clearer (e.g. `train` group or promoting commands to top level), but that's optional for this unit.

**Test scenarios:**
- `poetry run pytest src/test/python/spike/test_model_metadata.py -v` runs successfully
- `poetry run pytest -m "not integration and not experiment"` runs the default suite
- Existing `bin/beq-designer reassess` still works

**Verification:** `poetry run pytest` runs the full default suite without errors.

---

### U1: Model metadata banner

**Goal:** Add a `format_model_banner()` utility that reads `production_model.meta.json` and formats a human-readable block showing model provenance. Display it at the top of tier1 comparison output and train-production summary.

**Files:**
- Create: `src/main/python/model/model_metadata.py` - `load_model_metadata(model_path)`, `format_model_banner(metadata)`, `format_age(trained_at_unix)`
- Modify: `experiments/run_tier1_comparison.py` - call `format_model_banner()` in the header
- Modify: `src/main/python/cli/train_production_model.py` - use `format_model_banner()` in summary output
- Test: `src/test/python/spike/test_model_metadata.py`

**Approach:**
- `load_model_metadata(model_path)` takes a Path to the `.joblib` file and reads the `.meta.json` sidecar (same stem, `.meta.json` suffix). Returns `None` if the sidecar doesn't exist.
- `format_model_banner(metadata)` returns a multi-line string like:
  ```
  Model: E82 XGBoost (production_model.joblib)
  Trained: 2026-04-18 14:23 (2 days ago)
  Features: 102 (base, no foundation model)
  Training data: 448 real (50:1 weight) + 7102 synthetic
  WAV cache age: current (mtime matches)
  ```
- `format_age(unix_ts)` returns human-readable relative time: "2 hours ago", "3 days ago", "2 weeks ago"
- When metadata is missing (no sidecar), output: "Model: production_model.joblib (no metadata - retrain to generate)"

**Patterns to follow:** `model/media_utils.py` for utility module structure. The `format_duration()` function there is the closest analogue for `format_age()`.

**Test scenarios:**
- Happy path: metadata with all fields present, formats correctly
- Missing sidecar: returns None / prints fallback message
- Edge: `trained_at` in the future (clock skew) - shows "just now"
- Edge: very old model (months) - shows "3 months ago"
- Foundation model present vs absent changes the "Features" line

**Verification:** `poetry run pytest src/test/python/spike/test_model_metadata.py -v` passes; tier1 comparison output shows the banner.

---

### U2: Champion history tracking

**Goal:** Persist a `champion_history.json` in the BEQ shared directory that records each time a new champion is crowned. The tier1 comparison reads the previous champion from this file and shows "Previous champion: E82 (2.82 dB) -> New champion: E85 (2.27 dB)".

**Files:**
- Modify: `src/main/python/model/model_metadata.py` - add `load_champion_history()`, `record_champion()`, `format_champion_comparison()`
- Modify: `experiments/run_tier1_comparison.py` - after computing results, call `record_champion()` if a new champion is found; display comparison in the report
- Test: `src/test/python/spike/test_model_metadata.py` - champion history tests

**Approach:**
- `champion_history.json` lives in `beq_shared_dir()` (the shared directory). Schema:
  ```json
  {
    "current": {"experiment": "E85 diff-DSP", "mean_db": 2.27, "date": "2026-04-20", "wav_count": 1279},
    "history": [
      {"experiment": "E82 baseline", "mean_db": 2.82, "date": "2026-04-18", "wav_count": 1091},
      {"experiment": "E85 diff-DSP", "mean_db": 2.27, "date": "2026-04-20", "wav_count": 1279}
    ]
  }
  ```
- `record_champion(experiment, mean_db, wav_count)` appends to history and updates current. Only records if mean_db is lower than current champion (or no current exists).
- `format_champion_comparison(history, new_results)` returns lines like:
  ```
  Previous champion: E82 baseline (2.82 dB mean, 2026-04-18)
  New champion:      E85 diff-DSP (2.27 dB mean, -0.55 dB improvement)
  ```
- The dynamic verdict logic in `run_tier1_comparison.py` (line 287) compares against E82 baseline. Change it to compare against the previous champion from history instead.

**Patterns to follow:** `_discovery_cache_signature()` pattern in `_auto_beq_helpers.py` for JSON persistence with atomic writes.

**Test scenarios:**
- First run (no history file): all experiments recorded, best becomes champion
- Subsequent run with improvement: new champion recorded, previous shown
- Subsequent run with regression: no new champion, current champion unchanged
- Corrupt/missing history file: treated as first run

**Verification:** Two consecutive tier1 runs show champion progression.

---

### U3: Auto-suggest retraining on data change

**Goal:** At the start of the tier1 comparison (and train-production), compare the current WAV cache mtime against what's stored in `production_model.meta.json`. If they differ, print a prominent suggestion to retrain.

**Files:**
- Modify: `src/main/python/model/model_metadata.py` - add `check_training_staleness(metadata)`
- Modify: `experiments/run_tier1_comparison.py` - call staleness check at startup, print suggestion
- Modify: `src/main/python/cli/train_production_model.py` - same staleness check before training
- Test: `src/test/python/spike/test_model_metadata.py`

**Approach:**
- `check_training_staleness(metadata)` compares `metadata["wav_cache_mtime"]` against `_latest_wav_mtime(wav_cache_dir())`. Returns a dict: `{"stale": bool, "reason": str, "model_mtime": float, "current_mtime": float}`.
- Reasons: "WAV cache has changed since training (N new/modified files)", "model metadata missing (retrain to enable staleness detection)", "up to date"
- Output in tier1 comparison header:
  ```
  ⚠ Training data has changed since the production model was trained.
    Model trained at: 2026-04-18 (WAV cache mtime: 1713456000)
    Current WAV cache mtime: 1713542400
    Consider retraining: bin/beq-designer train-production
  ```
- The staleness check is informational only - it never blocks execution.

**Patterns to follow:** The `wav_cache_mtime` field already exists in meta.json (written at `train_production_model.py:308`). `_latest_wav_mtime()` already exists in `_auto_beq_helpers.py`.

**Test scenarios:**
- Matching mtimes: "up to date" message
- Different mtimes: staleness warning with retrain suggestion
- No metadata file: graceful "retrain to enable" message
- No WAV cache configured: skip check silently

**Verification:** Modify a WAV file's mtime, run reassess, see the suggestion.

---

### U4: Fix Whisper `.npy` cache miss

**Goal:** Fix the disk cache key mismatch that causes every `extract_foundation_embedding()` call to log "loading foundation model" even though the model is already in memory.

**Files:**
- Modify: `src/main/python/model/auto_beq_advisor.py` - fix the cache key or lookup logic in `extract_foundation_embedding()`
- Test: `src/test/python/spike/test_model_metadata.py` or existing Whisper tests

**Approach:**

The root cause needs confirmation during implementation. The investigation so far found:
- The `.npy` cache directory exists but is empty
- The in-memory cache (`_FOUNDATION_MODEL_CACHE`) works - model loads once per process
- The "loading foundation model" log at line 472 fires because the disk cache miss at line 446 (`cache_file.exists()`) falls through to the Whisper path

**Likely root cause candidates** (resolve during implementation):
1. The `cache_dir` being computed differently between calls (the fallback logic at lines 430-436 may resolve to different directories depending on whether `beq_shared_dir()` import succeeds)
2. The `cache_key` format (`{stem}-{size}-{mtime}`) may produce a different key between the write (line 482 `np.save(cache_file, embedding)`) and subsequent lookups if the WAV file's mtime changes between extraction and embedding
3. Permission issue writing to the cache directory (silent failure from `np.save`)

**Implementation-time investigation:**
1. Add a `log.debug()` showing the resolved `cache_dir` and `cache_file` path on each call
2. Check whether `np.save()` at line 482 is actually succeeding (file exists after write)
3. If the cache key is the issue, ensure `cache_dir` resolution is deterministic

The fix should also ensure the "loading foundation model" log message only fires when the model is actually being loaded from disk/network, not just when the in-memory cache is being used. Currently the log is inside the double-checked lock (line 472), so it only fires on actual loads - but if the disk cache always misses, the outer `if` at line 469 always enters the lock path and the log fires on the first call per process. This is correct behavior (one log per process), but the disk cache miss means embeddings are recomputed on every call instead of loaded from `.npy` files.

**Execution note:** Investigation-first. Reproduce the cache miss, confirm the root cause, then fix.

**Test scenarios:**
- Extract embedding, verify `.npy` file written to expected path
- Extract same embedding again, verify it's loaded from `.npy` (no model load log)
- Different WAV files get different cache keys
- Cache directory auto-created if missing

**Verification:** Run E83 in tier1 comparison; "loading foundation model" appears exactly once, not per-sample.

---

### U5: Clarify WAV count and split messages

**Goal:** Make the tier1 comparison output self-explanatory about what the numbers mean.

**Files:**
- Modify: `experiments/run_tier1_comparison.py` - expand the header messages and add explanatory lines after the split

**Approach:**

Currently the output says:
```
split: 448 train / 113 test (stratified by rolloff severity)
```

Change to:
```
WAV cache: 1279 WAV files matched to BEQ catalogue entries
  (563 unique titles - some titles have multiple episodes/versions)
  Deduplicated: 1 episode per title selected for training (see below)

Split: 448 train / 113 test (80/20 stratified by rolloff severity)
  - 80% of unique titles used for training
  - 20% held out for testing (never seen during training)
  - 'Stratified' = severity distribution preserved in both sets
```

The key numbers to explain:
- 1279 = total WAV-catalogue pairs (raw count from `discover_wav_catalogue_pairs_cached()`)
- 563 = unique titles (after `deduplicate_by_title()`)
- 448/113 = train/test split of unique titles
- Why 563 not 1279: one episode per title for training fairness

**Patterns to follow:** The existing `log.info` block at lines 64-76 already explains the stratified split concept. Extend this pattern.

**Test scenarios:**
- N/A (output formatting change, verified visually)

**Verification:** Run `bin/beq-designer reassess`, verify the output is self-explanatory.

---

### U6: Explain one-episode-per-title in output

**Goal:** Add an explanation of why deduplication happens, so users understand that training on multiple episodes of the same title would bias the model.

**Files:**
- Modify: `experiments/run_tier1_comparison.py` - add explanation after the deduplication step

**Approach:**

After `deduplicate_by_title()` runs, add:
```python
log.info("")
log.info("  Why one episode per title: training on multiple episodes of")
log.info("  the same show would let the model memorise title-specific")
log.info("  patterns instead of learning general audio correction rules.")
log.info("  Deduplication ensures the model generalises across titles.")
log.info("  (Full cache still used for feature extraction and evaluation)")
```

Also add the count: "Deduplicated: 563 -> 561 unique titles with filters (2 titles have no BEQ filters)"

This can be combined with U5's output - the explanation should flow naturally after the WAV count clarification.

**Verification:** Visual inspection of reassess output.

---

## Dependencies

```
U0 (test infra)        ── first: unblocks running tests for all other units
U1 (metadata banner)  ──┐
U2 (champion history) ──┤── can run in parallel, all modify run_tier1_comparison.py
U3 (staleness check)  ──┘   but in distinct sections (header, results, startup)
U4 (Whisper cache fix)  ── independent (auto_beq_advisor.py only)
U5 (WAV count clarity)  ── independent (output formatting in run_tier1_comparison.py)
U6 (one-episode explain) ── combine with U5 (same output section)
```

**Suggested execution order:**
1. U0 first - fixes test infrastructure so `poetry run pytest` works
2. U1 - creates `model/model_metadata.py` that U2 and U3 build on
3. U2 + U3 in parallel (both extend `model_metadata.py` but different functions)
4. U4 independently (different file entirely)
5. U5 + U6 together (same output section in `run_tier1_comparison.py`)

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| U4 root cause is not what we expect | Medium | Investigation-first approach; don't commit a fix until the cache miss is reproduced and understood |
| `beq_shared_dir()` import fails in experiment scripts (no spike on sys.path) | Low | `run_tier1_comparison.py` already imports from spike; same sys.path setup works |
| Champion history file corruption on concurrent writes | Very low | Only one process writes at a time (reassess is single-threaded) |

## Test Strategy

All new code goes in `model/model_metadata.py` with tests in `src/test/python/spike/test_model_metadata.py`. The test file uses synthetic metadata dicts (no actual model training needed) to verify formatting, age calculation, champion tracking, and staleness detection.

U4 (Whisper cache fix) tested by verifying `.npy` files are written and read back successfully.

U5/U6 are output formatting changes verified visually by running `bin/beq-designer reassess`.

---

### U7: Logging unification - all CLI output to log file

**Goal:** Every `bin/beq-designer` command should write logs to the log file, not just interactive mode. Currently `setup_log_file()` only runs for interactive mode. Also fix duplicate log handler accumulation that causes messages to repeat 30+ times.

**Root cause of duplicate messages:** Multiple `logging.basicConfig()` calls and `addHandler()` calls across CLI modules (`extract.py`, `train_production_model.py`, `verify_cache.py`, `generate.py`, `profile.py`, `cache_status.py`, `nn_report.py`) each add handlers. When the CLI imports these modules, handlers stack up on the root logger.

**Files:**
- Modify: `src/main/python/cli/main.py` - move `setup_log_file()` before the interactive-mode check; clear stale handlers in the callback
- Modify: All CLI modules with module-level `basicConfig` - guard with `if not logging.getLogger().handlers`
- Modify: `experiments/run_tier1_comparison.py` - same guard

**Approach:**
- `main_callback()` should be the ONE place that configures logging: clear all handlers, add exactly stderr + file handler
- All other modules should NOT call `basicConfig` or `addHandler` when imported as part of the CLI. Guard with `if __name__ == "__main__"` or handler-exists checks.

**Verification:** Run `bin/beq-designer dev reassess`, check that the log file contains all output and no message is duplicated.

---

### U8: Ollama fail-fast check

**Goal:** Commands that use Ollama (`dev evaluate --advisor ollama`, `dev benchmark`) should check host availability at startup. Graceful degradation: if at least one configured host responds, proceed and warn about unreachable hosts. If NO hosts respond, fail fast with a clear error instead of hanging mid-run.

**Files:**
- Modify: `src/main/python/model/auto_beq_advisor.py` - add `check_ollama_hosts(hosts)` utility
- Modify: `src/main/python/cli/main.py` - call check in `dev_evaluate` and `dev_benchmark` before proceeding

**Approach:**
- `check_ollama_hosts(hosts)` pings each configured host via HTTP GET to `{host}/api/version` with a 5s timeout
- Returns `{"available": [...], "unavailable": [...]}`
- If some hosts unavailable: log warning "Ollama host {host} is not reachable - skipping"
- If ALL hosts unavailable: raise with "No Ollama hosts are reachable. Configure hosts or use --advisor measurement."
- If at least one available: proceed normally, load balancer will skip the dead ones
- Only check when the selected advisor requires Ollama (OllamaAdvisor)

**Test scenarios:**
- All hosts down: clear error message, no hang
- One host down, one up: warning for dead host, proceeds with live one
- All hosts up: proceeds normally, no warnings
- Non-Ollama advisor selected: no check at all

---

### U9: FAQ, glossary, and documentation cleanup

**Goal:** Create a FAQ/glossary that explains the key concepts a developer encounters when working with this system. Document the advisor system, experiment naming, WAV cache, and training pipeline in accessible language.

**Key terms to define:**
- **Advisor** - the strategy pattern for generating BEQ filter proposals. Each experiment tried a different approach:
  - `MeasurementAdvisor` - heuristic from audio frequency analysis (no ML, no Ollama)
  - `TrainedModelAdvisor` - XGBoost model trained on real audio (E82, production default)
  - `OllamaAdvisor` - asks an LLM about the film's likely bass profile (needs Ollama)
  - `TopologyAdvisor` / `SlopeExtensionAdvisor` - other heuristic approaches
- **Experiment (E82, E83, etc.)** - numbered iterations of the training approach
- **WAV cache** - extracted LFE audio from media files, stored in the shared directory
- **Tier1 comparison / reassess** - runs all experiment approaches on the same data split
- **Champion** - the experiment with the lowest mean error on the test set
- **dev sweep** - runs a selected advisor across the whole media library
- **dev compare-advisors** - benchmarks all advisors side by side
- **dev test-advisor** - tests one advisor's proposal on a single title interactively

**Files:**
- Create: `docs/faq.md` - glossary + FAQ for developers
- Modify: `src/main/python/cli/main.py` - improve help text for dev commands
- Modify: `docs/design/auto_beq.md` - cross-reference the FAQ
- Modify: `mkdocs.yml` - add FAQ to nav

---

### U10: Rename dev commands for clarity

**Goal:** Rename the dev subcommands so their purpose is obvious.

| Current | New | Why |
|---|---|---|
| `dev sweep` | `dev evaluate` | "sweep" is meaningless; this evaluates an advisor across the library |
| `dev compare-advisors` | `dev benchmark` | benchmarking is what it does |
| `dev playground` | `dev test-advisor` | already renamed in code |

**Files:**
- Modify: `src/main/python/cli/main.py` - rename commands
- Modify: `AGENTS.md` - update CLI module table
- Modify: test assertions that check help text

---

### U11: Full-library evaluation against ground truth

**Goal:** Replace the current `dev evaluate` pytest wrapper with a proper evaluation pipeline that loads the production model, runs it on ALL catalogue-matched titles, compares predictions against ground truth, and produces a comprehensive report.

**Requirements:**
- R7: Evaluate production model against all catalogue-matched WAVs, not just the test split
- R8: Report per-title, per-author, and aggregate dB error with train/held-out labels
- R9: Persist results as CSV + JSON for cross-run comparison
- R10: Prompt user to retrain before evaluating if model is stale

**The gap today:**
- `dev reassess` only evaluates on 113 held-out titles (research metric)
- The old `dev evaluate` (was `dev sweep`) shells out to a pytest that runs an advisor on the media library but doesn't use `downstream_loss()` or compare against catalogue filters systematically
- No way to know how the production model performs across the full 563-title matched set

**Key difference from reassess:**
- Reassess: trains from scratch, evaluates on held-out split. "Which approach generalises best?" (research)
- Evaluate: loads saved production model, evaluates on ALL matched titles. "How does the deployed model perform on my library?" (QA)

**Dependencies:** U10 (command already renamed to `dev evaluate`)

**Files:**
- Create: `src/main/python/cli/evaluate.py` - full-library evaluation pipeline
- Modify: `src/main/python/cli/main.py` - replace the pytest-wrapper `dev_evaluate` with a call to the new pipeline
- Modify: `AGENTS.md` - document the command in CLI module table
- Test: `src/test/python/spike/test_evaluate.py`

**Approach:**

The pipeline has five stages:

1. **Load model** - Load `production_model.joblib` (XGBoost/E82) from `beq_shared_dir()`. Check staleness via `check_training_staleness()` from `model/model_metadata.py`. If stale, prompt user: "Training data has changed since the model was trained. Retrain first? [y/N]". Default No - proceed with saved model. Also attempt to load `e85_torch_filter.pt` if present (for E85 comparison column).

2. **Discover WAV-catalogue pairs** - Use `discover_wav_catalogue_pairs_cached()` from `_auto_beq_helpers.py` (same as reassess). Filter to pairs where `catalogue_entry` has filters.

3. **Extract features + predict** - Use `_extract_features_parallel()` for feature extraction (ThreadPoolExecutor, same as reassess). Then for each title, call `model.predict(feature_vector)` to get predicted filters. Also run E85 predictor if loaded. Use `ProgressLogger` for both extraction and prediction.

4. **Score against ground truth** - For each title, call `downstream_loss(predicted_filters, catalogue_filters, DEFAULT_GRID)` to get the dB error. This is the same metric reassess uses.

5. **Reconstruct train/held-out membership** - Reproduce the same stratified split logic from `run_tier1_comparison.py` (same `random_state=42`, same severity classification, same `train_test_split` call). This determines which titles the model saw during training. Label each result row as "train" or "held-out".

**Report output (three formats):**

Console (Rich table):
```
PRODUCTION MODEL EVALUATION - 563 titles (448 train, 113 held-out)
Model: production_model.joblib (trained 2026-04-18, 2 days ago)

  Title                     Author      Split     E82 dB   E85 dB
  ---------------------------------------------------------------
  Dune (2021)               aron7awol   held-out    1.23     0.95
  Mad Max: Fury Road        mobe1969    train       2.45     1.80
  ...

  Summary:
                  E82 mean    E85 mean
  All (563)         2.82        2.27
  Train (448)       2.65        2.10
  Held-out (113)    3.14        2.65
  Gap (train-held)  0.49        0.55   <-- overfitting indicator

  Per-author:
  ...

  Worst 10 titles:
  ...
```

CSV: `{beq_shared_dir}/evaluation_results.csv` - one row per title with columns: title, author, tmdb_id, split, e82_loss_db, e85_loss_db, catalogue_filter_count, catalogue_summed_gain_db

JSON: `{beq_shared_dir}/evaluation_results.json` - full structured output including metadata, per-title results, and summary statistics

**Patterns to follow:**
- `experiments/run_tier1_comparison.py` for the feature extraction + evaluation loop
- `model/model_metadata.py` for `load_model_metadata()` and `check_training_staleness()`
- `cli/nn_report.py` for CLI report module structure
- `model/media_utils.py` `ProgressLogger` for progress reporting

**Test scenarios:**
- Happy path: mock model + 5 synthetic WAV pairs -> produces report with correct columns
- Happy path: train/held-out split labels match the stratified split logic
- Edge case: no production model found -> clear error with retrain instruction
- Edge case: E85 model not found -> runs E82 only, notes "E85 not available"
- Edge case: zero catalogue-matched pairs -> clear error "no matched pairs found"
- Error path: corrupt model file -> clear error, not a crash
- Integration: summary statistics (mean, max) match manual calculation from per-title losses

**Verification:** `bin/beq-designer dev evaluate` produces per-title CSV + JSON + console table across all matched titles, with train/held-out labels and overfitting gap metric.

---

### U12: Document reassess vs evaluate clearly everywhere

**Goal:** Make the distinction between reassess (research) and evaluate (QA) unmissable in all user-facing surfaces: CLI help, docs, FAQ, and the output of each command itself.

**Requirements:** R7-R10 (from U11), plus user request for clear documentation

**Dependencies:** U9 (FAQ exists), U10 (command names finalised), U11 (evaluate pipeline exists)

**Files:**
- Modify: `src/main/python/cli/main.py` - help text for `dev reassess` and `dev evaluate`
- Modify: `experiments/run_tier1_comparison.py` - header banner clarification
- Modify: `src/main/python/cli/evaluate.py` - header banner in evaluate output
- Modify: `docs/faq.md` - reassess vs evaluate section (already partially written)
- Modify: `docs/design/auto_beq.md` - evaluation modes section

**Approach:**

CLI help text (one sentence each):
- `dev reassess`: "Train all experiment approaches from scratch and compare on a held-out test split. Measures generalisation - which approach learns best."
- `dev evaluate`: "Run the saved production model on all catalogue-matched titles. Measures real-world performance - how the deployed model performs on your library."

Output banners:
- Reassess header: "Training and evaluating on an 80/20 held-out split (seed=42). This measures how well each approach generalises to titles it has never seen during training."
- Evaluate header: "Evaluating the production model on all N catalogue-matched titles. Titles marked (train) were used during model training; (held-out) titles were not. A large gap between train and held-out performance indicates overfitting."

FAQ update: the `dev reassess vs dev evaluate` section already has the conceptual explanation. Add concrete examples of when to use each:
- "Just extracted 50 new WAVs" -> retrain, then reassess, then evaluate
- "Want to check if model is good enough for my library" -> evaluate
- "Trying a new experiment approach" -> reassess

**Test scenarios:**
- `dev reassess --help` contains "held-out" and "generalisation"
- `dev evaluate --help` contains "production model" and "catalogue-matched"
- FAQ contains both terms with distinct definitions

**Verification:** A developer reading only the CLI help or only the FAQ can explain the difference between reassess and evaluate.

## Dependencies and Status

```
[x] U0  (test infra)        -- DONE
[x] U1  (metadata banner)   -- DONE
[x] U2  (champion history)  -- DONE
[x] U3  (staleness check)   -- DONE
[x] U4  (Whisper spam fix)  -- DONE (root cause: ImportError caught per-sample)
[x] U5  (WAV count clarity) -- DONE
[x] U6  (one-episode explain) -- DONE
[x] U7  (logging unification) -- DONE
[x] U8  (Ollama fail-fast)  -- DONE
[x] U9  (FAQ/docs)          -- DONE
[x] U10 (rename commands)   -- DONE
[ ] U11 (full evaluation)   -- depends on U10
[ ] U12 (document reassess vs evaluate) -- depends on U9, U10, U11
```

## Future: CLI and UI menu consistency

The CLI subcommands (`bin/beq-designer`) and the Qt UI menus should present
a consistent set of operations. Currently they can diverge independently
since neither references a shared specification.

Recommended follow-up (out of scope for this plan):
- Create a **command/menu specification** document that defines all available
  operations, their grouping, and which surfaces expose them (CLI, UI, or both)
- Both the CLI (`cli/main.py`) and UI menu code should reference this spec
- When adding a new operation, add it to the spec first, then implement in
  the appropriate surfaces
