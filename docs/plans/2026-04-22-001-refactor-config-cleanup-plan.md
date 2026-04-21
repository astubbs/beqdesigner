---
title: "refactor: Remove BEQ_WAV_CACHE and purge local-env paths"
type: refactor
status: active
date: 2026-04-22
origin: docs/brainstorms/2026-04-22-config-cleanup-requirements.md
---

# Config cleanup

## Overview

Remove the `BEQ_WAV_CACHE` env var (no real use case) and purge
local-environment path references from tracked files. Keep the
`wav_cache_dir` key in settings.json as the override escape hatch.

## Requirements Trace

- R1. Remove `BEQ_WAV_CACHE` env var entirely
- R2. Keep `wav_cache_dir` in settings.json
- R3. `cache_status --cache-dir` passes path directly, not via env var
- R4. Purge `/Volumes/`, `/Users/`, `/home/`, `/mnt/` prefixes from
  decorative references (examples, fixture paths, hardcoded scripts)
- R5. Scrub home directories from meta-references while keeping the
  learning intact (architecture.md, generate.py, solutions doc)

## Scope Boundaries

- Out of scope: test_media_discover.py (synthetic path inputs),
  media_discover.py docstring format examples, audio_extraction.py
  comment, readme.md `/mnt/media` CLI example, `bin/profiles/*.log`

## Key Technical Decisions

- **Keep the learning, scrub the environment.** Institutional knowledge
  about bug classes is valuable; specific home directories or NAS
  mounts add no value.
- **Override test deleted.** `test_explicit_wav_cache_takes_precedence`
  goes away with the feature. Keep tests for settings.json override and
  derivation from shared dir.

## Implementation Units

- [x] **Unit 1: Remove `BEQ_WAV_CACHE` env var** *(started during bare-prompt run)*

**Files:**
- Modify: `src/main/python/model/wav_discovery.py` ✅ (done)
- Modify: `src/main/python/cli/cache_status.py` ✅ (done)
- Modify: `src/main/python/cli/nn_report.py` ✅ (done)
- Modify: `src/main/python/cli/train_torch_model.py` ✅ (done)
- Modify: `src/main/python/cli/train_production_model.py` ✅ (done)
- Modify: `src/main/python/model/auto_beq_advisor.py` ✅ (done)
- Modify: `experiments/run_tier1_comparison.py` ✅ (done)
- Modify: `src/test/python/auto_beq/test_extract_lfe.py` (remove
  override test, clean up `delenv("BEQ_WAV_CACHE")` calls)
- Modify: `AGENTS.md` env var table
- Modify: `readme.md` env var table

**Test scenarios:**
- Happy path: `wav_cache_dir()` derives from `BEQ_SHARED_DIR`
- Happy path: `wav_cache_dir()` reads from settings.json
- Error path: `wav_cache_dir()` raises when nothing configured
- `cache_status --cache-dir /tmp/x` uses `/tmp/x` without env var

**Verification:**
- `grep -r BEQ_WAV_CACHE src/ docs/ AGENTS.md readme.md` returns zero hits
- Existing tests pass (minus the deleted override test)

---

- [ ] **Unit 2: Purge `/Volumes/NAS/` test fixtures**

**Files:**
- Modify: `src/test/python/auto_beq/test_auto_beq_helpers.py:58,76`

**Approach:**
- Replace hardcoded `Path("/Volumes/NAS/Movies/...")` with `tmp_path`
  fixtures or synthetic paths that don't look like real mount points.

**Test scenarios:**
- Happy path: tests continue to pass with the new fixture paths

**Verification:**
- `grep -n "/Volumes/" src/test/python/auto_beq/test_auto_beq_helpers.py`
  returns zero hits

---

- [ ] **Unit 3: Purge `/home/matt/` references**

**Files:**
- Modify: `src/test/python/beq_loader.py:70` - use tmp_path or env var
- Modify: `src/main/python/ui/convert.sh:5` - use `poetry run pyuic6`
  instead of hardcoded path

**Approach:**
- `beq_loader.py` is an upstream test loader for Matt's local BEQ
  database. Make the path configurable via env var or fallback to
  a generic location.
- `convert.sh` runs at Qt build time. Replace the absolute venv path
  with `poetry run pyuic6` which finds the binary portably.

**Test scenarios:**
- Test expectation: none - convert.sh is build tooling; beq_loader.py
  is an upstream test that doesn't run in our default suite

**Verification:**
- `grep -rn "/home/matt" src/` returns zero hits

---

- [ ] **Unit 4: Scrub home directories from meta-references**

**Files:**
- Modify: `docs/architecture.md:166` - replace `/Volumes/Batou/...`
- Modify: `src/main/python/cli/generate.py:309` - remove `wav-cache/Users/astubbs/...`
  example from defensive comment
- Modify: `docs/solutions/runtime-errors/wav-cache-filesystem-portability-id-based-layout.md:75`
  - remove `wav-cache/Users/astubbs/...` example

**Approach:**
- **architecture.md**: keep the "different machines have different
  mount points" point, replace specific examples with generic
  placeholders (`/Volumes/<name>/` etc.) or drop the example list.
- **generate.py**: keep the "never fall through to legacy mirrored-path
  layout" guidance, remove the specific `wav-cache/Users/astubbs/...`
  example.
- **solutions doc**: keep the bug-class description, remove the
  specific home directory from the example.

**Patterns to follow:**
- Principle: "keep the learning, scrub the environment"

**Test scenarios:**
- Test expectation: none - documentation and comment changes

**Verification:**
- `grep -rn "Batou\|astubbs" docs/ src/` returns zero hits

---

- [ ] **Unit 5: Update AGENTS.md and readme.md env var tables**

**Files:**
- Modify: `AGENTS.md` - remove `BEQ_WAV_CACHE` row from env vars table
- Modify: `readme.md` - remove `BEQ_WAV_CACHE` row from env vars table

**Test scenarios:**
- Test expectation: none - documentation

**Verification:**
- `grep -n BEQ_WAV_CACHE AGENTS.md readme.md` returns zero hits

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `beq_loader.py` is upstream code we don't actively run | Use env-var fallback so upstream contract is preserved |
| `convert.sh` with `poetry run pyuic6` might fail if poetry isn't on PATH | It's build tooling, developer-environment dependent anyway |
| `/home/matt` in convert.sh might be intentional for upstream maintainer | Use portable invocation; maintainer can still run locally via poetry |

## Sources & References

- **Origin document:** `docs/brainstorms/2026-04-22-config-cleanup-requirements.md`
- Related: earlier docs consolidation commit `ee5c22b` that added BEQ_WAV_CACHE to readme env var table
