---
title: rglob hangs on NFS-mounted WAV cache - replace with os.scandir bucket walk
date: 2026-04-20
category: performance-issues
module: wav-cache
problem_type: performance_issue
component: tooling
severity: high
symptoms:
  - "cache-status command hung for 10+ minutes on NFS-mounted WAV cache with 2000+ files"
  - "CLI startup hung at 'scanning WAV cache' with no progress output"
  - "discover_wav_catalogue_pairs took minutes on NFS, seconds on local SSD"
  - "verify_cache used rglob('*.wav') - too broad, even slower"
root_cause: wrong_api
resolution_type: code_fix
tags:
  - nfs
  - rglob
  - scandir
  - performance
  - wav-cache
  - progress
  - nas
---

# rglob hangs on NFS-mounted WAV cache - replace with os.scandir bucket walk

## Problem

Multiple code paths used `Path.rglob("*.lfe-1000hz.wav")` to enumerate WAV files in the cache. On a local SSD this completed in milliseconds, but on an NFS-mounted NAS volume (typical production deployment), it hung for 10+ minutes because `rglob` issues individual `stat()` calls for every file and directory in the tree.

## Symptoms

- `bin/beq-designer cache-status` printed "WAV cache:" then froze
- CLI startup hung at "scanning WAV cache" with no progress for minutes
- `discover_wav_catalogue_pairs()` cache miss path took 5+ minutes
- `verify_cache()` used `rglob("*.wav")` (even broader - all WAVs, not just lfe-1000hz)
- User had to Ctrl+C multiple times and lost patience

## What Didn't Work

- **WAV count disk cache with TTL**: Added a `.wav_count_cache` file with 1-hour expiry. Worked for repeat runs but the first scan still hung. Also, the TTL approach was wrong - should use mtime-based invalidation, not time-based.
- **WAV count scanning at startup for menu display**: The entire WAV counting feature was unnecessary - just cosmetic status info. Deleted entirely rather than optimizing.

## Solution

Replace every `rglob()` call with structured `os.scandir()` + `os.walk()` of known bucket directories, plus `ProgressLogger` for visibility:

```python
# Before (slow on NFS):
wav_files = sorted(cache_root.rglob("*.lfe-1000hz.wav"))

# After (fast - walks known structure):
from model.wav_cache import WAV_SUFFIX
bucket_dirs = sorted(
    e.path for e in os.scandir(cache_root)
    if e.is_dir()
)
wav_files = []
for bucket_path in bucket_dirs:
    for dirpath, _dirnames, filenames in os.walk(bucket_path):
        for f in filenames:
            if f.endswith(WAV_SUFFIX):
                wav_files.append(Path(dirpath) / f)
```

Sites fixed:
- `cli/cache_status.py` - 3 rglob calls replaced
- `model/wav_discovery.py` - `discover_wav_catalogue_pairs()` and `discover_unmatched_wavs()`
- `model/wav_integrity.py` - `verify_cache()` tightened to `.lfe-1000hz.wav` pattern
- `_latest_wav_mtime()` - replaced per-file stat with dir-level mtime scan

## Why This Works

`rglob` issues one `stat()` per file/directory in the entire tree. With 2000+ WAVs across 150+ bucket directories on NFS, that's 2000+ individual network round-trips. Each stat takes 5-50ms over NFS, totaling 10-100 seconds.

`os.scandir()` of the root returns directory entries with cached stat data from a single `readdir` syscall. `os.walk()` within each bucket is bounded to a small subtree. The total is ~150 readdir calls instead of 2000+ individual stats.

## Prevention

- **Never use `rglob()` on paths that might be NFS/SMB mounts.** Use `os.scandir()` + `os.walk()` with known directory structure instead.
- **Add `ProgressLogger` to any directory walk that might take >5 seconds.** The user must never stare at a silent terminal wondering if the program crashed.
- **Use `model/wav_cache.WAV_SUFFIX` constant** instead of hardcoding `"*.lfe-1000hz.wav"` patterns - ensures all code searches for the same file extension.
- **Question whether scanning is needed at all.** The WAV count at startup was purely cosmetic and was deleted entirely.

## Related Issues

- `_latest_wav_mtime()` was also rglob-based and was replaced with dir-level mtime scanning. However, with the new 3-level-deep ID-based layout, top-level dir mtimes don't update when files are added deep in the tree - this is a known advisory issue (ADV-006 from ce:review).
