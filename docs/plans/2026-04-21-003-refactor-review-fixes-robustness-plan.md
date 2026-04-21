---
title: "refactor: Review fixes - robustness, consolidation, test coverage"
type: refactor
status: active
date: 2026-04-21
---

# Review fixes - robustness, consolidation, test coverage

## Overview

Address 9 remaining findings from the code review that need design
decisions. These range from consolidating duplicate catalogue-fetch
implementations to adding missing test coverage and fixing cache
invalidation. All fixes are bounded refactors within existing modules.

## Problem Frame

The code review flagged issues in three clusters:

1. **Duplication that survived the reduction pass** - three catalogue-fetch
   implementations, hardcoded `Path.home()` in 5+ modules
2. **NFS/reliability gaps** - rglob on user paths, no subprocess timeouts,
   double WAV reads, stale discovery cache
3. **Architecture and safety** - circular imports between advisor modules,
   unvalidated LLM responses, missing test coverage

## Requirements Trace

- R1. Single catalogue-fetch implementation used by all callers
- R2. No rglob on user-configurable paths (NFS safety)
- R3. Blended extraction reads WAV data once, not twice
- R4. All config paths go through `beq_config_dir()`, never hardcode `Path.home()`
- R5. subprocess.run calls in CLI have timeouts
- R6. Test coverage for `sweep_report.py` and `wav_discovery` matching
- R7. Advisor protocol classes live in `auto_beq_advisor.py`
- R8. LLM response fields validated before `float()` conversion
- R9. Discovery cache invalidates when WAVs are added to existing subdirs

## Scope Boundaries

- Do not change the catalogue JSON format or URL
- Do not change the WAV cache directory layout
- Do not refactor the Advisor protocol itself (just move classes)
- Do not add new CLI commands or change command names
- GUI catalogue loading (`model/catalogue.py`) is a separate ecosystem, out of scope

## Context & Research

### Relevant Code and Patterns

- `model/wav_cache.py` - single source of truth pattern for cache paths
- `model/xy_data.py` extraction from `model/xy.py` - pattern for breaking
  circular imports (see origin: `docs/solutions/runtime-errors/qt-dependency-chain-blocks-cli-docker-extract-xy-data.md`)
- `model/wav_discovery.py:beq_config_dir()` - canonical config dir function
- `model/media_constants.py:CATALOGUE_URL` - canonical URL constant

### Institutional Learnings

- `docs/solutions/performance-issues/nfs-rglob-replaced-with-scandir-bucket-walk.md` -
  use `os.scandir()` + `os.walk()` with `ProgressLogger`, question whether
  each scan is needed at all
- `docs/solutions/runtime-errors/qt-dependency-chain-blocks-cli-docker-extract-xy-data.md` -
  extract pure computation into dependency-free module, re-export for
  backward compat, lazy imports for cross-boundary runtime calls
- `docs/solutions/runtime-errors/wav-cache-filesystem-portability-id-based-layout.md` -
  all path computation through single source of truth module

## Key Technical Decisions

- **Catalogue consolidation target**: `model/auto_beq_catalogue.py` becomes
  the single fetch module. It already has the most callers (8+). Adopt the
  If-Modified-Since and atomic-write patterns from `cli/extract.py`'s
  implementation, which is the most robust. Cache location moves to
  `beq_config_dir() / "catalogue_cache.json"`.

- **rglob replacement strategy**: Use `os.walk()` with early termination
  where possible. For `cleanup_tmp()` and `find_media_dirs()`, the
  directory structure is known (shallow), so bounded `os.scandir()` is
  sufficient.

- **Blended extraction**: Add a `read_wav_data()` call at the top of
  `load_and_smooth_blended()` and pass the raw samples array to both
  `load_and_smooth()` and `load_and_smooth_chunked()` via a new
  `preloaded_data` parameter. This avoids changing all callers of
  the individual functions.

- **Circular import fix**: Move `TrainedModelAdvisor` and
  `LateFusionAdvisor` from `auto_beq_nn.py` to `auto_beq_advisor.py`.
  The `auto_beq_nn` imports they need (`train_xgboost`, model loading)
  become lazy imports inside method bodies - matching the existing
  pattern already used for the other direction.

- **Discovery cache invalidation**: Replace the top-level-mtime-only
  check with a lightweight content hash: count of WAV files from
  `os.scandir()` of each bucket dir. When the count changes, the
  cache invalidates. This is O(buckets) scandir calls (fast on NFS),
  not O(files) stat calls.

## Implementation Units

- [ ] **Unit 1: Consolidate catalogue fetch into `auto_beq_catalogue.py`**

**Goal:** Single catalogue-fetch implementation with If-Modified-Since,
atomic writes, and configurable cache path.

**Requirements:** R1, R4

**Dependencies:** None

**Files:**
- Modify: `model/auto_beq_catalogue.py` - add If-Modified-Since, atomic
  write, use `beq_config_dir()`, make `fetch_catalogue()` public
- Modify: `cli/extract.py` - remove `fetch_catalogue()`, import from
  `auto_beq_catalogue`
- Modify: `cli/sweep_discover.py` - remove `load_or_fetch_catalogue()`,
  `_DEFAULT_CATALOGUE_URL`, `_catalogue_cache_path()`, import from
  `auto_beq_catalogue`
- Modify: `model/media_constants.py` - keep `CATALOGUE_URL` as the
  single URL constant (already there)
- Test: `src/test/python/auto_beq/test_auto_beq_helpers.py` - add tests
  for consolidated fetch

**Approach:**
- Rename `_fetch_or_cache()` to `fetch_catalogue()` (public)
- Add `If-Modified-Since` header when cache exists (from extract.py pattern)
- Use atomic `.tmp` rename for cache writes (from extract.py pattern)
- Replace `_CACHE_DIR` with `beq_config_dir()`
- Remove `_CATALOGUE_URL`, import `CATALOGUE_URL` from `media_constants`
- Update all 8+ callers to use `auto_beq_catalogue.fetch_catalogue()`
- `cli/extract.py` callers switch to the shared function
- `cli/sweep_discover.py` callers switch to the shared function

**Patterns to follow:**
- `model/wav_cache.py` single source of truth pattern

**Test scenarios:**
- Happy path: fetch returns fresh JSON, cache written atomically
- Happy path: cache hit within TTL returns cached data without network call
- Edge case: 304 Not Modified returns cached data, updates cache mtime
- Error path: network failure with existing cache returns stale data
- Error path: network failure with no cache raises RuntimeError
- Edge case: corrupt cache file triggers fresh fetch

**Verification:**
- `grep -r "fetch_catalogue\|load_or_fetch_catalogue\|_fetch_or_cache" src/main/`
  shows only `auto_beq_catalogue.fetch_catalogue`
- `grep -r "_CATALOGUE_URL\|_DEFAULT_CATALOGUE_URL" src/main/` returns
  only `media_constants.CATALOGUE_URL`

---

- [ ] **Unit 2: Replace rglob with os.walk on user paths**

**Goal:** No rglob calls on paths that may be NFS-mounted.

**Requirements:** R2

**Dependencies:** None

**Files:**
- Modify: `cli/extract.py` - `discover_media()` line 214 and
  `cleanup_tmp()` line 1247
- Modify: `cli/sweep_discover.py` - line 390
- Modify: `model/media_utils.py` - `find_media_dirs()` line 188
- Test: `src/test/python/auto_beq/test_extract_lfe.py` - add test for
  discover_media with nested dirs
- Test: `src/test/python/auto_beq/test_sweep_discover.py` - verify
  existing tests still pass

**Approach:**
- `discover_media()` in extract.py: replace `root.rglob(f"*{ext}")` with
  `os.walk(root)` filtering by extension in the inner loop
- `cleanup_tmp()`: replace `wav_root.rglob("*.tmp")` with `os.walk(wav_root)`
  matching `.tmp` suffix - the WAV cache is 3 levels deep (bucket/shard/file)
  so this is bounded
- `sweep_discover.py`: replace `child.rglob(f"*{ext}")` with `os.walk(child)`
- `find_media_dirs()` in media_utils.py: replace `child.rglob(f"*{ext}")`
  with depth-limited `os.walk()` with `break` after first match (this
  function only probes for structure depth)
- Add `ProgressLogger` to any walk that processes 100+ entries

**Patterns to follow:**
- `model/wav_cache.py` uses `os.scandir` + `os.walk` for NFS safety
- `docs/solutions/performance-issues/nfs-rglob-replaced-with-scandir-bucket-walk.md`

**Test scenarios:**
- Happy path: discover_media finds .mkv files in nested directories
- Happy path: cleanup_tmp finds and removes .tmp files at various depths
- Edge case: empty directory tree returns empty list
- Edge case: find_media_dirs with media at depth 1 vs depth 2

**Verification:**
- `grep -rn "\.rglob\(" src/main/` returns zero hits

---

- [ ] **Unit 3: Eliminate double WAV read in blended extraction**

**Goal:** `load_and_smooth_blended()` reads the WAV file once.

**Requirements:** R3

**Dependencies:** None

**Files:**
- Modify: `model/audio_extraction.py` - `load_and_smooth()`,
  `load_and_smooth_chunked()`, `load_and_smooth_blended()`
- Test: `src/test/python/auto_beq/test_extract_lfe.py` - add blended
  strategy test

**Approach:**
- Add `preloaded_samples: np.ndarray | None = None` parameter to
  `load_and_smooth()` and `load_and_smooth_chunked()`
- When `preloaded_samples` is provided, skip `read_wav_data()` and
  `validate_wav()` calls, use the provided array directly
- In `load_and_smooth_blended()`: call `read_wav_data()` once at the top,
  then pass the samples to both sub-functions via `preloaded_samples`
- All existing callers that don't pass `preloaded_samples` continue
  to work unchanged (parameter defaults to None)

**Patterns to follow:**
- Existing `return_absolute` and `return_chunk_stats` optional parameter
  pattern in these functions

**Test scenarios:**
- Happy path: blended strategy produces same output as before (regression)
- Happy path: preloaded_samples parameter works for load_and_smooth
- Integration: blended reads from disk once (mock read_wav_data, assert
  called exactly once)

**Verification:**
- Blended extraction produces identical output before and after

---

- [ ] **Unit 4: Wire all config paths through `beq_config_dir()`**

**Goal:** No hardcoded `Path.home() / ".config" / "beqdesigner"` outside
`beq_config_dir()` itself.

**Requirements:** R4

**Dependencies:** Unit 1 (catalogue already migrated)

**Files:**
- Modify: `model/wav_discovery.py` - make `_SETTINGS_PATH` lazy (function
  call instead of module-level constant), update `audio_cache_dir()`,
  `beq_shared_dir()`, `wav_cache_dir()` to use `beq_config_dir()`
- Modify: `model/auto_beq_metadata.py` - replace `_CACHE_DIR` with
  `beq_config_dir()`
- Modify: `cli/sweep_discover.py` - line 484 use `beq_config_dir()`
- Modify: `cli/extract.py` - line 1330 use `beq_config_dir()`
- Modify: `model/auto_beq_advisor.py` - line 1624 use `beq_config_dir()`
- Modify: `cli/nn_author_pattern_report.py` - line 160 use `beq_config_dir()`
- Test: `src/test/python/auto_beq/test_extract_lfe.py` - verify
  monkeypatching still works

**Approach:**
- Replace module-level `_SETTINGS_PATH = Path.home() / ...` in
  `wav_discovery.py` with a function `_settings_path() -> Path` that calls
  `beq_config_dir() / "settings.json"`. This makes it testable and
  overridable.
- Replace all other `Path.home() / ".config" / "beqdesigner"` with
  `beq_config_dir()` calls
- Check that tests that monkeypatch `_SETTINGS_PATH` are updated to
  monkeypatch `beq_config_dir` or `_settings_path` instead

**Patterns to follow:**
- `docs/solutions/runtime-errors/wav-cache-filesystem-portability-id-based-layout.md` -
  all path computation through single source of truth

**Test scenarios:**
- Happy path: `beq_config_dir()` returns expected path
- Edge case: monkeypatching `beq_config_dir` affects all downstream paths
- Edge case: settings loaded from custom config dir

**Verification:**
- `grep -rn "Path.home().*\.config.*beqdesigner" src/main/` returns only
  `beq_config_dir()` definition itself

---

- [ ] **Unit 5: Add subprocess timeouts in CLI**

**Goal:** All subprocess.run calls in cli/main.py have timeouts.

**Requirements:** R5

**Dependencies:** None

**Files:**
- Modify: `cli/main.py` - add `timeout=` to all subprocess.run calls

**Approach:**
- `dev_test` Pass 1 (full suite): `timeout=600` (10 min)
- `dev_test` Pass 2 (torch only): `timeout=300` (5 min)
- `dev_benchmark`: `timeout=3600` (1 hour, runs multiple advisors)
- On `subprocess.TimeoutExpired`: log which command timed out and
  raise `typer.Exit(code=1)`

**Test scenarios:**
- Test expectation: none - these are thin subprocess wrappers. Verifying
  timeout behavior requires mocking subprocess.run which adds no value
  beyond reading the code.

**Verification:**
- `grep -n "subprocess.run" src/main/python/cli/main.py` shows all calls
  have `timeout=`

---

- [ ] **Unit 6: Add test coverage for sweep_report.py and wav_discovery matching**

**Goal:** Non-trivial aggregation logic in sweep_report.py and matching
logic in wav_discovery.py have unit tests.

**Requirements:** R6

**Dependencies:** None

**Files:**
- Create: `src/test/python/auto_beq/test_sweep_report.py`
- Create: `src/test/python/auto_beq/test_wav_discovery.py`
- Read: `cli/sweep_report.py` - understand ExperimentResult, grading,
  improvement/degradation counting
- Read: `model/wav_discovery.py` - understand `_match_wavs_to_catalogue()`

**Approach:**
- `test_sweep_report.py`: test with synthetic CSV data and ExperimentResult
  objects. Focus on grade comparison, improvement counting, and matrix output.
- `test_wav_discovery.py`: test `_match_wavs_to_catalogue()` with
  controlled WAV path fixtures and fake catalogue entries. Cover tmdb direct
  match, tvdb/imdb title+year fallback, and unmatched cases.

**Patterns to follow:**
- `test_auto_beq_helpers.py:TestPrepareTrainingData` - monkeypatching
  discovery and metadata functions for unit isolation

**Test scenarios:**
- sweep_report: Happy path: 3 experiments, compute grades and improvements
- sweep_report: Edge case: single experiment (no comparison possible)
- sweep_report: Edge case: all grades identical (no improvement/degradation)
- wav_discovery: Happy path: tmdb ID match finds catalogue entry
- wav_discovery: Happy path: tvdb fallback via title+year
- wav_discovery: Edge case: WAV with no matching catalogue entry -> unmatched
- wav_discovery: Edge case: duplicate WAV paths (same movie, different seasons)
- wav_discovery: Edge case: empty catalogue returns all as unmatched

**Verification:**
- Both test files pass with `poetry run pytest <file> -v`
- `_match_wavs_to_catalogue` has direct unit test coverage

---

- [ ] **Unit 7: Move Advisor classes to `auto_beq_advisor.py`**

**Goal:** `TrainedModelAdvisor` and `LateFusionAdvisor` live in
`auto_beq_advisor.py` alongside all other Advisor implementations.

**Requirements:** R7

**Dependencies:** None

**Files:**
- Modify: `model/auto_beq_advisor.py` - add both classes, remove
  deferred imports from `get_advisor()`
- Modify: `model/auto_beq_nn.py` - remove both classes, add backward-compat
  re-exports
- Test: `src/test/python/auto_beq/test_auto_beq_nn.py` - verify imports
  still work

**Approach:**
- Move `TrainedModelAdvisor` and `LateFusionAdvisor` class definitions
  from `auto_beq_nn.py` to `auto_beq_advisor.py`
- Their methods that import from `auto_beq_nn` (model loading, training
  functions) keep using deferred imports inside method bodies - this is
  the same pattern already used by `get_advisor()`
- `get_advisor()` no longer needs deferred imports for these two classes
  since they are now local
- Add re-exports in `auto_beq_nn.py` for backward compatibility:
  `from model.auto_beq_advisor import TrainedModelAdvisor, LateFusionAdvisor`
- Verify: `python -c "from model.auto_beq_advisor import TrainedModelAdvisor"`
  works without importing Qt or torch

**Patterns to follow:**
- `model/xy_data.py` extraction from `model/xy.py` with re-exports
  (see `docs/solutions/runtime-errors/qt-dependency-chain-blocks-cli-docker-extract-xy-data.md`)

**Test scenarios:**
- Happy path: `get_advisor("trained_model")` returns TrainedModelAdvisor
- Happy path: `get_advisor("late_fusion")` returns LateFusionAdvisor
- Integration: importing `auto_beq_advisor` does not trigger torch or
  xgboost imports (verified via `python -c` in a clean environment)
- Edge case: backward-compat import from `auto_beq_nn` still works

**Verification:**
- No deferred imports of `TrainedModelAdvisor` or `LateFusionAdvisor`
  remain in `get_advisor()`
- `auto_beq_nn.py` re-exports both classes

---

- [ ] **Unit 8: Validate LLM response fields before float conversion**

**Goal:** Ollama response parsing handles non-numeric string values
gracefully instead of raising ValueError.

**Requirements:** R8

**Dependencies:** None

**Files:**
- Modify: `model/auto_beq_advisor.py` - add `_safe_float()` helper,
  use it in `advise()` and `_apply_diff()`
- Test: `src/test/python/auto_beq/test_auto_beq_advisor.py` - add
  validation tests

**Approach:**
- Add a `_safe_float(value, default: float, field: str) -> float` helper
  that tries `float(value)`, catches `(ValueError, TypeError)`, logs a
  warning with the field name and raw value, and returns the default
- Replace all bare `float()` calls on LLM response fields with
  `_safe_float()`:
  - `max_gain_db` (default 0.0)
  - `knee_hz` (default 80.0)
  - `confidence` (default 0.5)
  - `factor` in `_apply_diff()` (default 1.0)
  - `delta_hz`, `freq_hz` in `_apply_diff()` (default 0.0)
  - Individual filter `freq`, `q`, `gain` in chain parsing

**Patterns to follow:**
- `_clamp_advice()` already validates ranges after conversion - this
  adds the missing pre-conversion safety

**Test scenarios:**
- Happy path: numeric string "12.5" converts correctly
- Edge case: string "about 15 dB" returns default, logs warning
- Edge case: None value returns default
- Edge case: empty string returns default
- Happy path: full advise() call with valid JSON succeeds
- Error path: advise() with non-numeric fields produces clamped defaults
  instead of ValueError

**Verification:**
- `grep -n "float(" model/auto_beq_advisor.py` shows all LLM response
  field conversions use `_safe_float()`

---

- [ ] **Unit 9: Fix discovery cache staleness**

**Goal:** Discovery cache invalidates when WAVs are added inside
existing subdirectories.

**Requirements:** R9

**Dependencies:** None

**Files:**
- Modify: `model/wav_discovery.py` - improve `_discovery_cache_signature()`
- Test: `src/test/python/auto_beq/test_wav_discovery.py` (from Unit 6)

**Approach:**
- Replace `_latest_wav_mtime()` with a `_wav_count_signature(cache_root)`
  that counts WAV files per top-level bucket via `os.scandir()` at each
  level. On NFS this is O(buckets * shards) scandir calls (fast) rather
  than O(files) stat calls.
- The signature becomes: `{"wav_count": total_wav_count, "catalogue_mtime": ...}`
- When a new WAV is added to an existing shard, the count changes,
  invalidating the cache.
- Keep the catalogue mtime check (catalogue updates also invalidate).
- The `AUTO_BEQ_DISCOVERY_CACHE=0` bypass remains for manual override.

**Patterns to follow:**
- `model/wav_cache.py` - uses `os.scandir` for NFS-safe directory enumeration

**Test scenarios:**
- Happy path: cache valid when WAV count unchanged
- Happy path: cache invalid when new WAV added to existing shard
- Happy path: cache invalid when WAV deleted
- Edge case: empty cache root returns count 0
- Edge case: `AUTO_BEQ_DISCOVERY_CACHE=0` always forces fresh discovery

**Verification:**
- Adding a WAV to an existing shard directory invalidates the cache
  without requiring `AUTO_BEQ_DISCOVERY_CACHE=0`

## System-Wide Impact

- **Interaction graph:** Unit 1 touches 8+ callers across cli/ and model/.
  All must switch to the consolidated function. Unit 4 changes how
  `_SETTINGS_PATH` is resolved, affecting `load_settings()`,
  `save_settings()`, and every function that reads config.
- **Error propagation:** Unit 8 converts hard ValueError crashes into
  logged warnings with defaults. This changes error behavior from
  crash-the-batch to degrade-gracefully for individual titles.
- **State lifecycle risks:** Unit 9 changes cache invalidation semantics.
  Existing caches will be invalidated on first run after the change
  (signature format changes). This is intentional and self-healing.
- **Unchanged invariants:** The catalogue JSON format, WAV cache layout,
  Advisor protocol interface, and CLI command names are all unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Catalogue consolidation breaks a caller's specific error handling | Each caller's error handling is preserved by the consolidated function's fallback behavior (stale cache on failure) |
| `_SETTINGS_PATH` change breaks monkeypatched tests | Update tests to monkeypatch `beq_config_dir` or `_settings_path()` instead |
| Discovery cache signature change forces full re-discovery for all users | This is a one-time cost (~30s). Document in commit message. |
| Moving Advisor classes triggers import-order issues | Verify with `python -c` imports in clean environment before committing |

## Sources & References

- Related code: `model/wav_cache.py` (NFS-safe patterns)
- Related code: `model/xy_data.py` (circular import resolution pattern)
- Solution docs: `docs/solutions/performance-issues/nfs-rglob-replaced-with-scandir-bucket-walk.md`
- Solution docs: `docs/solutions/runtime-errors/qt-dependency-chain-blocks-cli-docker-extract-xy-data.md`
- Solution docs: `docs/solutions/runtime-errors/wav-cache-filesystem-portability-id-based-layout.md`
