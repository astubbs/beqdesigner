---
title: WAV cache paths differ across Mac/Linux filesystems - use ID-based layout
date: 2026-04-20
category: runtime-errors
module: wav-cache
problem_type: runtime_error
component: tooling
severity: high
symptoms:
  - "NAS Docker re-extracted all WAVs that were already cached on Mac"
  - "Duplicate bucket directories on NAS: '3/' and '3 /' (with trailing space)"
  - "Unicode normalization mismatch: 'BA/' vs 'BĀ/' for same title"
  - "Finder showed tilde-named short directories (AQZNE5~M) that were SMB aliases for space-suffixed dirs"
root_cause: config_error
resolution_type: code_fix
tags:
  - filesystem
  - nfs
  - portability
  - unicode
  - wav-cache
  - docker
  - nas
  - cross-platform
---

# WAV cache paths differ across Mac/Linux filesystems - use ID-based layout

## Problem

The WAV cache used title-based two-letter bucket directories (e.g. `wav-cache/AV/Avatar...`). When the same cache was accessed from both a Mac laptop (via NFS/SMB) and a Linux NAS (Docker), files were invisible to the other platform because Mac and Linux handle directory names differently. Every title was re-extracted on the NAS even though the WAV files already existed.

## Symptoms

- `bin/beq-designer extract` on NAS showed all titles as "Extracting" instead of "CACHED"
- `ls` on the NAS showed duplicate directories: `3/` (created by Mac) and `3 /` (created by Linux, with trailing space)
- Mac's Finder showed mysterious tilde-named directories (`AQZNE5~M`, `8KLWJ7~9`) - SMB short-name aliases for the space-suffixed dirs
- A `BĀ/` directory (with macron) existed alongside `BA/` - unicode normalization difference
- The `Users/` directory appeared inside wav-cache from the legacy mirrored-path fallback

## What Didn't Work

- **Title-based bucket sanitization**: Added `unicodedata.normalize("NFC")` and whitespace stripping to the bucket name function. This fixed NEW writes but didn't help existing files in the wrong-named directories.
- **Sanitization still fragile**: Any future encoding edge case (emoji, RTL text, surrogate pairs) could reopen the same wound.

## Solution

Switched to an ID-based cache layout using media database IDs (tmdb/tvdb/imdb) which are pure ASCII:

```
wav-cache/tmdb/19/19995/Avatar (2009) [tmdb-19995].lfe-1000hz.wav
wav-cache/tvdb/37/377543/Show [tvdb-377543]/Season 01/S01E01.lfe-1000hz.wav
```

Directory structure is `{id_type}/{shard}/{id_value}/{descriptive_filename}`. The shard (first 2 digits of the numeric ID) prevents 10000+ entries at one level.

Legacy title-bucket paths are checked as a fallback during reads (`find_cached_wav()`), so existing extracted WAVs are still found without migration.

All cache layout logic lives in `model/wav_cache.py` - single source of truth.

## Why This Works

Media DB IDs (`tmdb-19995`, `tvdb-377543`, `imdb-tt0133093`) are guaranteed ASCII - digits, dashes, and lowercase letters. No unicode normalization issues, no trailing spaces, no filesystem encoding differences. The same ID produces the same directory path on every filesystem.

The human-readable title is preserved in the FILENAME (not the directory structure), so `ls` and `find` are still useful for browsing.

## Prevention

- **Never use user-visible text (titles, names) as filesystem directory keys.** Use stable identifiers instead. Text encoding is a bottomless pit of cross-platform incompatibilities.
- **All cache path computation must go through `model/wav_cache.py`** - never construct paths ad-hoc. The `cache_path()` and `find_cached_wav()` functions are the single source of truth.
- **Test cache paths with problematic titles**: single-char titles ("A", "I"), digit-then-space ("3 Days"), unicode accents, IMDB `tt` prefix.

## Related Issues

- The legacy mirrored-path layout (`wav-cache/<source-media-path>/...`) was from `cli/generate.py` falling through to `extract_lfe_wav()` without a `target_path`. Fixed by making `generate.py` always pass the ID-based target and fail loudly if it can't compute one.
