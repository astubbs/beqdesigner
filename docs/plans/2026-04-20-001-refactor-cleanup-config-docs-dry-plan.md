---
title: "refactor: Cleanup - config unification, DRY walk, WAV counts, stale docs"
type: refactor
status: active
date: 2026-04-20
---

# Cleanup - config unification, DRY walk, WAV counts, stale docs

## Overview

Four related cleanup items that reduce technical debt and improve
clarity: fix missing WAV count explanations in reassess output, rewrite
the stale auto_beq.md architecture doc, unify duplicate media root
config, and consolidate the WAV cache walk into a shared function.

## Problem Frame

1. **WAV count explanation not showing** -
   `experiments/run_tier1_comparison.py` still has a hardcoded
   "1279-WAV cache" string in the report output and lacks the
   one-episode-per-title explanation that was planned in U5/U6. The
   helpers file has the "unique titles vs total media files" log, but
   the tier1 comparison report itself doesn't explain the numbers.

2. **`docs/design/auto_beq.md` is stale** - still says "spike, not
   shipped" despite the system being in production. Has a staleness
   note but the actual content hasn't been rewritten.

3. **Duplicate media root config** - `library_roots` (in
   `settings.json`, used by sweep/discover) and `media_roots` (in
   `extract_config.json`, used by extract) store the same concept in
   two places. `docs/architecture.md` already flags this: "to be
   unified". AGENTS.md rule: "Collapse parallel state when bugs recur."

4. **DRY violation in WAV cache walk** -
   `discover_wav_catalogue_pairs()` in `_auto_beq_helpers.py` has its
   own `os.scandir` + `os.walk` loop instead of using
   `model/wav_cache.py`. The walk pattern is also used in
   `cli/cache_status.py` and `model/wav_integrity.py`.

## Requirements Trace

- R1. Reassess output explains all key numbers (WAV count, unique
  titles, one-episode rationale, train/test split) inline
- R2. `docs/design/auto_beq.md` accurately reflects the current system
- R3. Single config for media library roots, usable by both extract
  and sweep
- R4. WAV cache walking logic lives in `model/wav_cache.py`, not
  duplicated across 3+ files

## Scope Boundaries

- No changes to training algorithms or experiment logic
- No changes to the WAV cache layout (ID-based stays as-is)
- No changes to the `beq_shared_dir()` / `beq_config_dir()` split

## Key Technical Decisions

- **Config unification uses `extract_config.json` as the winner**:
  `extract_config.json` already has `get_configured_media_roots()` and
  `save_extract_config()` as a proper service layer (per AGENTS.md
  shared infrastructure). `library_roots` in `settings.json` is the
  one to remove - migrate sweep to use the extract config service.
  Rationale: extract was built with the single-code-path discipline;
  sweep predates it.

- **Walk function goes in `model/wav_cache.py`**: This module already
  owns cache path logic. Adding `iter_cached_wavs()` here keeps all
  cache filesystem operations in one place. The function returns
  `list[Path]` of WAV files, using the scandir+walk pattern from the
  NFS performance fix.

- **auto_beq.md rewrite is a docs-only unit**: No code changes. The
  doc should describe the current architecture (advisors, experiments,
  training pipeline, evaluation modes) rather than the original spike
  goals.

## Implementation Units

- [ ] **Unit 1: Add `iter_cached_wavs()` to `model/wav_cache.py`**

**Goal:** Centralize the WAV cache walk into a shared function.

**Requirements:** R4

**Dependencies:** None

**Files:**
- Modify: `src/main/python/model/wav_cache.py`
- Test: `src/test/python/spike/test_auto_beq_helpers.py` (or new
  `test_wav_cache.py` if one doesn't exist)

**Approach:**
- Add `iter_cached_wavs(cache_root: Path) -> list[Path]` that walks
  all bucket directories under `cache_root` using `os.scandir` +
  `os.walk`, collecting files ending in `WAV_SUFFIX`.
- Uses the same pattern as the existing walk in
  `discover_wav_catalogue_pairs()` and the NFS performance fix
  (scandir bucket dirs, walk each one).
- Add `ProgressLogger` for caches with many buckets.

**Patterns to follow:**
- `discover_wav_catalogue_pairs()` in `_auto_beq_helpers.py` lines
  1337-1349 for the current walk pattern
- `docs/solutions/performance-issues/nfs-rglob-replaced-with-scandir-bucket-walk.md`
  for the NFS-safe walk approach

**Test scenarios:**
- Happy path: tmp_path with 3 bucket dirs, 5 WAV files -> returns all 5
- Edge case: empty cache dir -> returns empty list
- Edge case: non-WAV files in buckets -> excluded from results
- Edge case: nested subdirectories (ID-based layout) -> WAVs found at
  any depth within buckets

**Verification:** `iter_cached_wavs()` returns the same WAV list as
the current inline walk in `discover_wav_catalogue_pairs()`.

---

- [ ] **Unit 2: Migrate `discover_wav_catalogue_pairs()` to use `iter_cached_wavs()`**

**Goal:** Replace the inline walk with the shared function.

**Requirements:** R4

**Dependencies:** Unit 1

**Files:**
- Modify: `src/test/python/spike/_auto_beq_helpers.py` -
  `discover_wav_catalogue_pairs()`
- Test: existing tests that exercise discovery (run them to verify
  no regression)

**Approach:**
- Replace the `os.scandir` + `os.walk` loop in
  `discover_wav_catalogue_pairs()` with a call to
  `iter_cached_wavs(cache_root)`.
- The matching logic (pairing WAVs with catalogue entries) stays
  unchanged - only the walk is replaced.

**Execution note:** Characterization-first. Run the existing discovery
tests before and after the change to verify identical behavior.

**Test scenarios:**
- Integration: `discover_wav_catalogue_pairs_cached()` returns same
  pairs as before the refactor (same count, same titles)

**Verification:** All existing discovery tests pass unchanged.

---

- [ ] **Unit 3: Migrate other WAV cache walks to `iter_cached_wavs()`**

**Goal:** Replace inline walks in `cli/cache_status.py` and
`model/wav_integrity.py` with the shared function.

**Requirements:** R4

**Dependencies:** Unit 1

**Files:**
- Modify: `src/main/python/cli/cache_status.py`
- Modify: `src/main/python/model/wav_integrity.py`

**Approach:**
- Find the `os.scandir`/`os.walk` loops in each file and replace
  with `iter_cached_wavs()`.
- These files may have slightly different walk patterns (e.g.
  cache_status counts per-bucket, wav_integrity checks file
  integrity). Adapt the shared function or use its output as the
  starting point.

**Execution note:** Characterization-first. Run existing cache_status
and verify_cache tests before and after.

**Test scenarios:**
- `cache-status` command produces same output before and after
- `verify` command produces same output before and after

**Verification:** All existing tests pass unchanged.

---

- [ ] **Unit 4: Unify media root config**

**Goal:** Remove `library_roots` from `settings.json`. Migrate sweep
to use `get_configured_media_roots()` from `cli/extract.py`.

**Requirements:** R3

**Dependencies:** None

**Files:**
- Modify: `src/test/python/spike/sweep_discover.py` -
  `_resolve_library_roots()` and `_save_library_roots()`
- Modify: `src/main/python/cli/main.py` - any menu items that
  reference library_roots
- Modify: `docs/architecture.md` - remove "to be unified" note,
  document single config
- Modify: `AGENTS.md` - update shared infrastructure section if
  needed

**Approach:**
- `_resolve_library_roots()` currently checks CLI args, env var
  `AUTO_BEQ_LIBRARY_ROOTS`, then `settings.json["library_roots"]`.
  Change it to: CLI args, env var, then
  `get_configured_media_roots()` (which reads
  `extract_config.json`).
- `_save_library_roots()` currently writes to `settings.json`.
  Change it to call `save_extract_config()`.
- Keep `AUTO_BEQ_LIBRARY_ROOTS` env var as an override for
  backwards compatibility.
- Remove the `library_roots` key handling from `settings.json` -
  if an old config has it, ignore it (the extract_config.json is
  the source of truth).
- Add a one-time migration: if `settings.json` has `library_roots`
  but `extract_config.json` has no `media_roots`, copy the roots
  over and log a migration notice.

**Patterns to follow:**
- `get_configured_media_roots()` / `save_extract_config()` in
  `cli/extract.py` for the target config service

**Test scenarios:**
- Happy path: `extract_config.json` has roots -> sweep uses them
- Migration: `settings.json` has library_roots, extract_config.json
  empty -> roots copied, sweep uses them
- Migration: both have roots -> extract_config.json wins
- Env var override: `AUTO_BEQ_LIBRARY_ROOTS` set -> overrides config
- Edge case: neither config has roots -> prompt user

**Verification:** `sweep_discover` reads from the same config as
`extract`. Removing `library_roots` from `settings.json` doesn't
break sweep.

---

- [ ] **Unit 5: Fix WAV count explanation in reassess output**

**Goal:** Make `run_tier1_comparison.py` output self-explanatory about
all key numbers.

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `experiments/run_tier1_comparison.py`

**Approach:**
- Replace the hardcoded "1279-WAV cache" string in the report with
  a dynamic count from `len(all_real)`.
- After the split, add log messages explaining: total WAV-catalogue
  pairs, unique titles, why one episode per title (deduplication
  prevents memorisation), and the train/test split meaning.
- These changes were planned in U5/U6 of the previous plan but the
  file was overwritten by external edits. Reapply them.
- Check the current file state first - some changes may have
  survived.

**Test scenarios:**
- Test expectation: none - output formatting verified visually

**Verification:** Run `bin/beq-designer dev reassess` and verify the
output explains all numbers inline.

---

- [ ] **Unit 6: Rewrite `docs/design/auto_beq.md`**

**Goal:** Replace the stale "spike, not shipped" content with an
accurate description of the current system.

**Requirements:** R2

**Dependencies:** Units 4 and 5 (so the config and output are
accurate when documented)

**Files:**
- Modify: `docs/design/auto_beq.md`

**Approach:**
- Keep the existing "Evaluation modes" section (added recently,
  still accurate).
- Rewrite the Overview to describe the current state: production
  model (E82 XGBoost), champion (E85 differentiable DSP), CLI
  commands, Docker deployment.
- Replace "spike, not shipped" status with the actual status.
- Keep the detailed optimizer/fitter sections as they are still
  accurate for the underlying algorithm.
- Remove references to future work that has been completed.
- Cross-reference `docs/faq.md` for terminology and
  `auto_beq_experiments.md` for the experiment log.
- Follow the accessible writing rule from AGENTS.md: define all
  terms, link external references, target developers with no
  ML/audio background.

**Test scenarios:**
- Test expectation: none - documentation change

**Verification:** Document accurately describes the current system.
No "spike" or "not shipped" language remains. Cross-references to
FAQ and experiment log are valid links.

## Dependencies

```
[x] Unit 1 (iter_cached_wavs) --+
[x] Unit 2 (migrate discovery)  |-- DONE
[x] Unit 3 (migrate other walks)|-- DONE
[x] Unit 4 (unify config)       -- DONE
[x] Unit 5 (WAV count fix)      -- DONE
[x] Unit 6 (rewrite docs)       -- DONE
[ ] Unit 7 (CI quality gates)   -- independent
```

---

- [ ] **Unit 7: CI quality gates - duplication, coverage, static analysis**

**Goal:** Add code duplication scanning, coverage gates, and enhanced
static analysis to the GitHub Actions CI pipeline. Match the quality
gates from the parallel-consumer project (adapted for Python).

**Requirements:** CI catches duplication introduced by PRs, coverage
regressions, and code quality issues before merge.

**Dependencies:** None (independent of Units 1-6)

**Files:**
- Modify: `.github/workflows/test.yaml` - add duplication and
  coverage gate jobs
- Create: `.jscpd.json` - jscpd configuration for Python
- Create: `codecov.yml` - coverage thresholds

**Approach:**

Add six new CI capabilities, modelled on parallel-consumer's
workflows:

**Quality gates (from maven.yml):**

1. **Code duplication scanning** - use
   `astubbs/duplicate-code-cross-check@v1` (already used in
   parallel-consumer). Supports Python via jscpd engine. Configure
   with Python-appropriate thresholds (5% max duplication). Compares
   base branch vs PR to catch new clones.

2. **File similarity detection** - use
   `astubbs/duplicate-code-detection-tool@feat/base-vs-pr-comparison`.
   Configure for `.py` files, warn at 50% similarity, fail at 80%,
   max 10% increase allowed, ignore files under 30 lines.

3. **Coverage gates** - add `codecov.yml` with thresholds:
   - Patch coverage: 80% minimum (new code must be tested)
   - Project coverage: no more than 1% drop allowed
   - This uses the existing Codecov integration (already uploads
     coverage in test.yaml)

4. **Ensure ruff + mypy run in CI** - the current test.yaml has a
   `lint` job but verify it's correctly configured and running on PRs.

**Claude and PR automation (from parallel-consumer workflows):**

5. **Claude Code Review** - automatic PR review on every PR using
   `anthropics/claude-code-action@v1` with the `code-review` plugin.
   Creates `.github/workflows/claude-code-review.yml`. Requires
   `CLAUDE_CODE_OAUTH_TOKEN` secret.

6. **Claude Code** - `@claude` mentions in issues and PR comments
   trigger Claude to respond. Creates
   `.github/workflows/claude.yml`. Same secret.

7. **PR Dependency Check** - use
   `astubbs/dependencies-action@feat/auto-unblock-children-on-merge`
   to block child PRs until parent merges (for stacked PRs). Creates
   `.github/workflows/check-dependencies.yml`. Uses GITHUB_TOKEN.

**Patterns to follow:**
- `parallel-consumer/.github/workflows/maven.yml` lines 121-162
  for duplication actions
- `parallel-consumer/.github/workflows/claude-code-review.yml` for
  Claude review
- `parallel-consumer/.github/workflows/claude.yml` for @claude
  mentions
- `parallel-consumer/.github/workflows/check-dependencies.yml` for
  PR dependencies
- `parallel-consumer/codecov.yml` for coverage gate config

**Test scenarios:**
- PR with duplicated code block -> CI reports duplication increase
- PR that drops test coverage -> CI fails with coverage gate
- PR with ruff/mypy violations -> CI fails in lint job
- PR opened -> Claude Code Review posts review comments
- @claude mentioned in PR comment -> Claude responds
- Stacked PR -> blocked until parent merges
- Clean PR -> all checks pass

**Verification:** Open a PR on this branch and verify all quality
gates run and post results.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Config migration loses roots | One-time copy from settings.json to extract_config.json; log the migration; never delete the old key, just stop reading it |
| Walk refactor changes discovery results | Characterization-first: run discovery tests before and after, verify identical output |
| auto_beq.md rewrite misses important content | Preserve all still-accurate algorithm sections; only rewrite the framing and status |

## System-Wide Impact

- **Config unification** touches sweep_discover.py and cli/extract.py
  but does NOT change the extract pipeline behavior - it only changes
  where sweep reads its roots from.
- **Walk refactor** touches the discovery path but does NOT change
  what pairs are discovered - only which function performs the walk.
- **Unchanged invariants:** WAV cache layout (ID-based), model
  training, evaluation pipeline, catalogue matching logic.
