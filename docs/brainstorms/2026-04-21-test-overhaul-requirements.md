# Test Overhaul - E2E Coverage, MPS Fix, Spike Restructure

**Created:** 2026-04-21

## Problem

The primary user feature (profile generation) crashes with SIGSEGV on
macOS, and there is no test that catches it. The test suite hides
critical failures behind markers and skips, and production code lives
in a test directory called "spike" that confuses both humans and LLMs.

## Requirements

### R1. End-to-end profile generation test (CI-ready)

A test that exercises the full pipeline: synthetic WAV -> feature
extraction -> model training (tiny, ~5 synthetic samples) -> model
prediction -> filter output. Must:

- Run in CI without real media files, Ollama, NAS, or network
- Train a minimal XGBoost model during the test (~2s) to exercise
  the real model loading/prediction code path
- Assert that coherent filters come out (non-empty list of dicts with
  freq, gain, Q keys)
- Catch the SIGSEGV that currently crashes profile generation on macOS
  (the test itself will crash, proving the bug exists)
- Run as a default test (no `integration` or `experiment` marker)

### R2. Fix xgboost MPS SIGSEGV

The same fix applied for torch (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
before import) needs to be applied for xgboost. On macOS with Apple
Silicon, xgboost's Metal backend conflicts with other Metal users
(scipy, Qt) in the same process.

The guard must be set before `import xgboost` or `joblib.load()` of
an xgboost model. Places that load models:
- `model/auto_beq_nn.py:load_model()`
- `cli/evaluate.py:_load_production_model()`
- `cli/generate.py` (profile generation path)

### R3. Rename and restructure the spike directory

`src/test/python/spike/` contains three kinds of code that should be
separated:

**Production code (move to `src/main/python/`):**
- `_auto_beq_helpers.py` (1683 lines) - audio cache management,
  LFE extraction, feature extraction, WAV-catalogue discovery,
  settings persistence. Imported by `cli/` modules.
- `sweep_discover.py` (906 lines) - sweep discovery CLI. Used by
  `cli/main.py` dispatch.
- `conftest_nn.py` - shared test fixtures (stays in test/)

**Unit tests (rename directory):**
- All `test_*.py` files with no markers - legitimate unit tests

**Experiment/integration tests:**
- Files with `integration` or `experiment` markers - need external
  resources

The name "spike" signals throwaway exploratory work. The directory
contains production infrastructure and serious tests. Rename to
something meaningful.

### R4. Reduce skipped/deselected test count

Document what external resources each test group needs. Tests that
can be made CI-ready with small changes should be fixed. Tests that
genuinely need external resources should have clear documentation
about what to configure.

Specific fixes:
- `test_ollama_multihost.py` skips should be `integration` marker
  (not runtime skip)
- `test_auto_beq_helpers.py` has a hardcoded path to a specific
  developer's media file - remove it
- `test_auto_beq_metadata.py` pings TMDb at collection time - make
  it skip gracefully, not block collection

### R5. Developer setup documentation

The README or a developer guide must document:
- How to run the default test suite (`poetry run pytest`)
- What's needed for integration tests (ffmpeg, Ollama, TMDb key)
- What's needed for experiment tests (WAV cache, production model)
- How to set up the shared directory and config

## Scope Boundaries

- Do NOT rewrite the experiment test suite (E1-E87 tests)
- Do NOT change any ML algorithms or training logic
- Do NOT restructure `src/main/python/model/` (separate effort)

## Success Criteria

1. `bin/beq-designer profile` successfully generates a BEQ profile
   on macOS without SIGSEGV
2. A CI-ready test catches the profile generation crash (would have
   caught the SIGSEGV before this session)
3. Production code no longer lives in `src/test/python/spike/`
4. `poetry run pytest` with no markers runs all CI-safe tests with
   zero skips (deselected by marker is OK, runtime skips are not)

## Execution Order

1. **E2E test first** (R1) - write the test, watch it crash (proves
   the bug). This is test-first development.
2. **Fix the crash** (R2) - add MPS guard for xgboost. The E2E test
   now passes.
3. **Restructure spike/** (R3) - move production code, rename
   directory
4. **Reduce skips, add docs** (R4, R5)

## Resolved Questions

- **Spike directory rename:** `src/test/python/auto_beq/` - matches
  the feature area name
- **Helpers split:** Split `_auto_beq_helpers.py` by concern when
  moving to `src/main/python/`: `model/audio_extraction.py` (LFE/
  ffmpeg), `model/wav_discovery.py` (pair matching), `model/
  training_data.py` (feature prep). Each under 500 lines.
