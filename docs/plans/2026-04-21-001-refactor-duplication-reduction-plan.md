---
title: "refactor: Reduce code duplication flagged by CI"
type: refactor
status: active
date: 2026-04-21
---

# Reduce code duplication flagged by CI

## Overview

The CI duplicate-code-cross-check reports 44 new clones (+1.28%
duplication increase) and the file-similarity tool flags 20 file pairs
above 50% similarity. Most duplication falls into 4 clusters that can
be collapsed by extracting shared infrastructure.

## Problem Frame

Code duplication creates drift risk - when a pattern exists in 5 files
and one gets fixed, the other 4 don't. The CI report identified these
clusters from PR #3:

1. **Report scripts** (5 files, 57-73% similar) - same argparse/
   catalogue loading/markdown rendering boilerplate
2. **Experiment test files** (3 files, 51-54% similar) - same feature
   extraction + training setup
3. **Training scripts** (2 files, 51% similar) - same discovery +
   feature extraction preamble
4. **Experiment runners** (3 files, 91 lines identical) - same
   experiment boilerplate

Plus the user's notes:
- `_find_media_dirs()` still in `sweep_discover.py` instead of shared
  utils (was supposed to move to `_auto_beq_helpers.py`)
- Reusable functions should be in common areas so LLMs find them

## Requirements Trace

- R1. PMD CPD duplication increase drops below +0.1% threshold
- R2. No file pair above 70% similarity in the report scripts
- R3. Shared infrastructure functions discoverable in documented
  shared modules (not buried in test/experiment files)

## Scope Boundaries

- Only deduplicate patterns identified by CI reports + user notes
- Do not refactor experiment algorithms or change behavior
- Do not consolidate test files that are intentionally separate
  (different pytest markers)

## Key Technical Decisions

- **Report scripts get a shared base module**: Extract common
  argparse, catalogue loading, and markdown rendering into
  `cli/report_base.py`. Each report script becomes a thin wrapper
  calling the shared infrastructure.

- **Training preamble goes into a shared function**: The discovery +
  feature extraction pattern used by `train_production_model.py`,
  `train_torch_model.py`, and `evaluate.py` should be a single
  `prepare_training_data()` function in `_auto_beq_helpers.py`.

- **Experiment runners use a shared harness**: `run_e85`, `run_e86`,
  `run_e87` share 91 lines of identical setup. Extract into a
  `run_single_experiment()` function or shared base.

- **Test helpers stay as shared fixtures**: The duplicated feature
  extraction in test files should use shared pytest fixtures or
  helper functions in conftest.py, not copy-paste.

## Implementation Units

- [ ] **Unit 1: Extract report script base module**

**Goal:** Eliminate 57-73% similarity across 5 nn_report scripts.

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Create: `src/main/python/cli/report_base.py`
- Modify: `src/main/python/cli/nn_report.py`
- Modify: `src/main/python/cli/nn_cache_bias_report.py`
- Modify: `src/main/python/cli/nn_author_pattern_report.py`
- Modify: `src/main/python/cli/nn_acquisition_recommender.py`
- Modify: `src/main/python/cli/nn_f_experiment_report.py`

**Approach:**
- Extract common patterns into `report_base.py`:
  - `create_report_argparser(description)` - shared argparse setup
    with `-o/--output` flag
  - `load_report_data()` - catalogue loading, WAV pair discovery,
    feature extraction (the preamble every report does)
  - `render_and_output(markdown, args)` - write to file or print
    with Rich rendering
- Each report script becomes: import base, define its unique analysis
  function, call `render_and_output()`

**Patterns to follow:**
- The existing report scripts for the common pattern to extract
- `cli/common.py` for shared CLI utility structure

**Test scenarios:**
- Happy path: each report produces same output before and after
- Edge case: `-o` flag writes to file correctly
- Integration: `bin/beq-designer report cache-bias` still works

**Verification:** All 5 report commands produce identical output.
File similarity between any two drops below 50%.

---

- [ ] **Unit 2: Extract training data preparation function**

**Goal:** Eliminate duplication between train_production_model.py,
train_torch_model.py, and evaluate.py.

**Requirements:** R1, R3

**Dependencies:** None

**Files:**
- Modify: `src/test/python/spike/_auto_beq_helpers.py` - add
  `prepare_training_data()` function
- Modify: `src/main/python/cli/train_production_model.py`
- Modify: `src/main/python/cli/train_torch_model.py`
- Modify: `src/main/python/cli/evaluate.py`

**Approach:**
- The common preamble: discover WAV pairs, extract features, load
  TMDb cache, build feature vectors, split into train/test. All 3
  files do this nearly identically.
- Extract `prepare_training_data(config)` that returns a structured
  dict: `{"train_samples": [...], "synth_entries": [...],
  "tmdb_cache": {...}, "X_train": ndarray, "X_test": ndarray, ...}`
- Each caller then does its own model-specific training on the
  prepared data.

**Execution note:** Characterization-first. Capture the output of
each training script before refactoring, verify identical after.

**Patterns to follow:**
- `run_tier1_comparison.py` lines 78-140 for the canonical data
  preparation sequence

**Test scenarios:**
- Happy path: `prepare_training_data()` returns same structure as
  inline code produced
- Edge case: empty WAV cache -> clear error
- Integration: `dev train` produces same model quality metrics

**Verification:** Training scripts produce identical models (same
n_real, n_synth, same evaluation metrics).

---

- [ ] **Unit 3: Consolidate experiment runner boilerplate**

**Goal:** Eliminate 91 lines of identical code across run_e85, run_e86,
run_e87.

**Requirements:** R1

**Dependencies:** Unit 2 (uses shared data preparation)

**Files:**
- Create: `experiments/experiment_base.py` - shared experiment setup
- Modify: `experiments/run_e85_experiment.py`
- Modify: `experiments/run_e86_experiment.py`
- Modify: `experiments/run_e87_experiment.py`

**Approach:**
- The identical code is: sys.path setup, logging config, WAV pair
  discovery, feature extraction, train/test split, evaluation loop.
- Extract into `experiment_base.py`:
  - `setup_experiment()` - sys.path, logging
  - Use `prepare_training_data()` from Unit 2 for data setup
  - `evaluate_and_report(model, X_test, entries, label)` - the eval
    loop + reporting
- Each experiment script becomes: import base, define its unique
  model config, call train + evaluate.

**Test scenarios:**
- Test expectation: none - experiment scripts verified by running
  them (experiment marker tests)

**Verification:** Each experiment script produces same results.

---

- [ ] **Unit 4: Extract shared test fixtures for experiment tests**

**Goal:** Reduce 51-54% similarity across test_auto_beq_nn_real,
test_auto_beq_nn_extract, test_auto_beq_nn_chunked.

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `src/test/python/conftest.py` or create
  `src/test/python/spike/conftest_nn.py`
- Modify: `src/test/python/spike/test_auto_beq_nn_real.py`
- Modify: `src/test/python/spike/test_auto_beq_nn_extract.py`
- Modify: `src/test/python/spike/test_auto_beq_nn_chunked.py`

**Approach:**
- The duplicated code is: synthetic data generation, feature
  extraction setup, model training boilerplate, evaluation helpers.
- Extract into shared fixtures:
  - `@pytest.fixture` for synthetic training data
  - `@pytest.fixture` for pre-trained baseline model
  - Helper function for the train-and-evaluate pattern
- Each test file keeps its unique test cases but uses shared
  fixtures for setup.

**Execution note:** Characterization-first. Run all 3 test files
before and after, verify identical pass/fail results.

**Test scenarios:**
- All existing tests pass unchanged after fixture extraction

**Verification:** `poetry run pytest -m experiment` on all 3 files
passes with identical results.

---

- [ ] **Unit 5: Move _find_media_dirs to shared utils**

**Goal:** Move `_find_media_dirs()` from `sweep_discover.py` to
`_auto_beq_helpers.py` (or `model/media_utils.py`).

**Requirements:** R3

**Dependencies:** None

**Files:**
- Modify: `src/main/python/model/media_utils.py` (preferred
  location - already has shared media utilities)
- Modify: `src/test/python/spike/sweep_discover.py` - import from
  new location
- Modify: Any other files that reference `_find_media_dirs`

**Approach:**
- `_find_media_dirs()` is already documented in AGENTS.md as a
  shared utility in `model/media_utils.py` as `find_media_dirs()`.
  Check if it's already there or if the AGENTS.md documentation is
  aspirational.
- If it's only in `sweep_discover.py`, move it. Rename from
  `_find_media_dirs` (private) to `find_media_dirs` (public).
- Update all imports.

**Test scenarios:**
- Happy path: `find_media_dirs()` returns same results from new
  location
- Integration: sweep_discover works with import from new location

**Verification:** `sweep_discover` tests pass. `find_media_dirs`
is importable from `model.media_utils`.

---

- [ ] **Unit 6: Deduplicate _auto_beq_helpers discovery functions**

**Goal:** Reduce 21-26 line clones between
`discover_wav_catalogue_pairs()` and `discover_unmatched_wavs()`.

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `src/test/python/spike/_auto_beq_helpers.py`

**Approach:**
- Both functions walk the WAV cache (now via `iter_cached_wavs()`),
  then match each WAV against the catalogue using the same regex
  and lookup logic. The difference is: one returns matched pairs,
  the other returns unmatched WAVs.
- Extract the common matching loop into a shared function:
  `_match_wavs_to_catalogue(wav_files, catalogue) -> (matched, unmatched)`
- Both public functions become thin wrappers calling the shared
  matcher.

**Execution note:** Characterization-first. Run discovery tests
before and after.

**Test scenarios:**
- Same matched pairs returned by `discover_wav_catalogue_pairs()`
- Same unmatched WAVs returned by `discover_unmatched_wavs()`

**Verification:** All existing discovery tests pass unchanged.

## Dependencies

```
Unit 1 (report base)           -- independent
Unit 2 (training prep)         -- independent
Unit 3 (experiment runners)    -- depends on Unit 2
Unit 4 (test fixtures)         -- independent
Unit 5 (find_media_dirs move)  -- independent
Unit 6 (discovery dedup)       -- independent
```

Suggested execution:
1. Units 1, 2, 4, 5, 6 in parallel (independent)
2. Unit 3 after Unit 2

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Report output changes after extraction | Characterization: capture output before, compare after |
| Training metrics change after shared prep function | Compare n_real, n_synth, evaluation metrics before/after |
| Test fixtures break experiment isolation | Keep experiment markers on test files, verify each runs independently |

## Verification

After all units: re-run the duplication scan locally with jscpd and
verify the new-clone count drops significantly. The PMD CPD increase
should be well under the +0.1% threshold.
