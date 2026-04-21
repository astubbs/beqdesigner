---
title: "Config cleanup - remove BEQ_WAV_CACHE, purge /Volumes/ paths"
type: refactor
date: 2026-04-22
status: ready
---

# Config cleanup

## Problem

Two related cleanups surfaced during docs consolidation review:

1. **`BEQ_WAV_CACHE` env var has no real use case.** Having a separate
   override for the WAV cache directory (independent of `BEQ_SHARED_DIR`)
   adds configuration complexity without a realistic workflow. The shared
   dir already lives on the high-speed drive where the WAV cache belongs.

2. **Local-environment paths are still in tracked files** despite a
   prior rule to purge them. Any reference rooted at a machine-specific
   location (e.g. `/Volumes/`, `/Users/`, `/home/`, `/mnt/`) leaks the
   developer's local environment into the repo. The scan target is the
   *root prefix* itself, not specific subdirectories - `/Volumes/X/...`
   and `/Volumes/Y/...` are both machine-specific regardless of what
   comes after `/Volumes/`.

## Requirements

- R1. Remove `BEQ_WAV_CACHE` env var entirely from codebase and docs
- R2. Keep `wav_cache_dir` key in settings.json as the override escape
  hatch (rare but legitimate)
- R3. Update `cache_status --cache-dir` CLI flag to pass the path
  directly rather than via env var
- R4. Purge all references to local-environment path prefixes from
  tracked source files. The scan pattern is the prefix itself
  (`/Volumes/`, `/Users/`, `/home/`, `/mnt/`) regardless of what
  subdirectory follows. Test fixtures, CLI docstrings, and experiment
  runners should use portable paths (`tmp_path` fixture, `~` expansion,
  or generic placeholders like `/path/to/cache`).
- R5. Exempt legitimate meta-references: a doc that discusses the
  cross-platform mount-point problem (e.g. `docs/architecture.md`) may
  mention `/Volumes/` as an example of the problem itself. The test is
  whether removing the reference would lose meaning - if the reference
  is decorative (example command, fixture path), purge it; if it's
  structural (documenting the variability problem), keep it.

## Success Criteria

- `grep -r "BEQ_WAV_CACHE" src/ docs/ AGENTS.md readme.md` returns zero
  hits (besides the settings.json migration note if added)
- `grep -rE "^/Volumes/|^/Users/|^/home/|^/mnt/|[^a-zA-Z]/Volumes/|[^a-zA-Z]/Users/"
  src/ docs/ AGENTS.md readme.md scripts/ experiments/ bin/` returns
  only legitimate meta-references (e.g. architecture.md discussing the
  cross-platform problem itself). No decorative example paths.
- All existing tests pass
- `bin/beq-designer profile <file>` and `dev train` workflows still work
  for the default config (shared dir -> wav-cache/)

## Scope

### In scope

**Remove `BEQ_WAV_CACHE`:**
- `src/main/python/model/wav_discovery.py` - remove from `wav_cache_dir()`
  resolution order
- `src/main/python/cli/cache_status.py` - `--cache-dir` flag should
  pass path directly, not via env var
- `src/main/python/cli/nn_report.py` - docstring example
- `src/main/python/cli/train_torch_model.py` - docstring example
- `src/main/python/cli/train_production_model.py` - docstring example
- `src/main/python/model/auto_beq_advisor.py` - comment reference
- `experiments/run_tier1_comparison.py` - docstring example
- `src/test/python/auto_beq/test_extract_lfe.py` - test coverage (delete
  the override test since the feature is gone)
- `AGENTS.md` env var table
- `readme.md` env var table

**Purge decorative local-environment paths:**
- `src/test/python/auto_beq/test_auto_beq_helpers.py:58,76` - replace
  `/Volumes/NAS/Movies/...` fixture paths with `tmp_path`-based fixtures
- `src/test/python/beq_loader.py:70` - replace `/home/matt/.beq/...`
  (upstream maintainer's path) with env-based or tmp_path lookup
- `src/main/python/ui/convert.sh:5` - replace hardcoded
  `/home/matt/.cache/pypoetry/.../pyuic6` with `poetry run pyuic6` or
  portable PATH-based invocation
- `src/main/python/cli/train_torch_model.py:13` - `/Volumes/jetspeed/...`
  example removed with BEQ_WAV_CACHE cleanup
- `src/main/python/cli/train_production_model.py:25` - same
- `experiments/run_tier1_comparison.py:9` - same

**Drop home-directory references (keep the learning, scrub the env):**
- `docs/architecture.md:166` - replace `/Volumes/Batou/...` example
  with a generic placeholder (e.g. `/mnt/<mountname>/...`) OR drop the
  per-OS example list and keep only the general statement "different
  machines have different mount points". The config-split design is
  real and worth documenting; the specific per-OS examples are our
  dev context.
- `src/main/python/cli/generate.py:309` - rewrite the defensive comment
  to drop the `wav-cache/Users/astubbs/...` example. The guidance
  ("never fall through to the legacy mirrored-path layout") stays, but
  the example path leaks our dev history.
- `docs/solutions/runtime-errors/wav-cache-filesystem-portability-id-based-layout.md:75`
  - drop `wav-cache/Users/astubbs/...`; keep the description of the
  legacy mirrored-path bug class and the fix. The learning is valuable
  institutional knowledge; the specific home directory is not.

### Out of scope (legitimate meta-references - keep as-is)

- `src/test/python/auto_beq/test_media_discover.py` - synthetic path
  strings fed to path-parsing code under test. Fake inputs, not
  environment references.
- `src/main/python/cli/media_discover.py:17,681` - docstring examples
  showing `AUTO_BEQ_LIBRARY_ROOTS=/mnt/media1:/mnt/media2` format and
  escaping rules (generic placeholder, not a specific machine path)
- `src/main/python/model/audio_extraction.py:164` - comment illustrating
  cache-nesting pattern (`/mnt/media/Y.mkv -> cache_root/mnt/media/Y`)
- `readme.md:35` - `bin/beq-designer extract --media-root /mnt/media`
  as a generic CLI example (the `/mnt/media` is placeholder)
- `bin/profiles/*.log` - runtime output, already gitignored, not tracked

### Key principle: keep the learning, scrub the environment

Institutional knowledge about bug classes (docs/solutions/, defensive
comments) is valuable - it prevents the next developer from repeating
a mistake. But the *specific home directory* or *specific NAS mount*
used when the bug was found adds no value. Rewrite the example using
a generic placeholder while keeping the pattern description and the
guidance intact.

## Key Decisions

- **Remove env var, keep settings.json key.** The env var was
  convenience layered on top of settings.json support. Removing the
  env var removes surface area without losing capability for the rare
  override case.
- **`cache_status --cache-dir` stays but changes mechanism.** The
  flag itself is legitimate ("inspect a specific cache"); the
  implementation should pass the path directly to the status function
  instead of mutating the environment.
- **Delete the override test.** The test for "BEQ_WAV_CACHE overrides
  derivation" goes away with the feature. Keep tests for settings.json
  override and derivation from shared dir.

## Resolve Before Planning

None - all decisions made.
