#!/usr/bin/env python3
"""LFE extractor — portable WAV cache builder.

Scans media roots for .mkv files with a media DB ID tag ([tmdb-NNN],
[tvdb-NNN], [imdb-NNN]), extracts the LFE channel (or mono downmix) to
a portable WAV cache. Only extracts titles with a BEQ catalogue entry.

Run via Docker (recommended) or directly with PYTHONPATH set.

Usage:
    docker compose run extract                          # via Docker
    bin/beq-designer extract --beq-dir /path              # direct
    bin/beq-designer extract                              # uses saved config

Directory structure (managed by the script):
    beq-dir/                        # shared directory (e.g. NAS mount)
      beq_catalogue.json            # BEQ catalogue (auto-fetched from GitHub, freshness-checked)
      missing_ids.txt               # media files without DB ID tags (can't identify)
      media_inventory.json          # every media file with a DB ID, whether
                                    # catalogue-matched or not (used by the
                                    # acquisition recommender to dedupe).
                                    # Paths are stored *relative* to media roots
                                    # for portability across machines.
      wav-cache/
        AL/Alien (1979) [tmdb-348].lfe-1000hz.wav
        86/86 - Eighty Six [tvdb-378609]/Season 01/S01E02.lfe-1000hz.wav
    ~/.config/beqdesigner/          # local config dir (machine-specific)
      extract_config.json           # saved media roots (auto-created on first run)

Only media with a matching BEQ catalogue entry is extracted. The catalogue
is fetched from GitHub and cached locally — re-downloaded only when the
remote has been updated (HTTP If-Modified-Since check).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Import from project modules — no duplication.
# These are all stdlib-only (no scipy/numpy/PyQt).
from model.wav_integrity import validate_wav_header, verify_cache as _verify_cache_raw
from model.media_constants import (
    CATALOGUE_URL,
    EPISODE_RE,
    ID_RE,
    JUNK_SUBDIRS,
    MEDIA_EXTENSIONS,
    MIN_FEATURE_SIZE_BYTES,
    TITLE_YEAR_RE,
)
from model.media_utils import ProgressLogger, find_media_dirs, format_duration

log = logging.getLogger("extract_lfe")

# ---------------------------------------------------------------------------
# Constants (script-specific only — shared ones come from media_constants)
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 1000  # Hz — coupled to BEQ analysis algorithm, not configurable


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------



def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def extract_media_id(media_path: Path) -> tuple[str, str] | None:
    """Extract a media DB ID from a file path.

    Searches (in order): filename, parent dir, grandparent dir.
    First match wins. Returns (id_type, id_value) or None.
    """
    for source in (media_path.name, media_path.parent.name, media_path.parent.parent.name):
        m = ID_RE.search(source)
        if m:
            return m.group(1), m.group(2)
    return None


# ---------------------------------------------------------------------------
# BEQ catalogue fetch + cache
# ---------------------------------------------------------------------------


def fetch_catalogue(beq_dir: Path) -> list[dict]:
    """Fetch the BEQ catalogue, caching at {beq_dir}/beq_catalogue.json.

    Uses If-Modified-Since on the GET request. If the server returns 304
    Not Modified, we skip the download. Uses stdlib only.
    """
    import email.utils
    import urllib.error
    import urllib.request

    cache_path = beq_dir / "beq_catalogue.json"

    _CATALOGUE_TTL = 86400  # 24 hours

    req = urllib.request.Request(CATALOGUE_URL)
    if cache_path.exists():
        local_mtime = cache_path.stat().st_mtime
        age_hours = (time.time() - local_mtime) / 3600
        if age_hours < _CATALOGUE_TTL / 3600:
            log.info("catalogue cached (%.0fh old, <24h) — skipping freshness check", age_hours)
            data = cache_path.read_bytes()
            catalogue = json.loads(data)
            log.info("catalogue loaded: %d entries (%s)", len(catalogue), _human_size(len(data)))
            return catalogue
        mtime_str = email.utils.formatdate(local_mtime, usegmt=True)
        req.add_header("If-Modified-Since", mtime_str)
        log.info("checking catalogue freshness (cached: %s, %.0fh old)...", mtime_str, age_hours)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.rename(cache_path)
        log.info("catalogue updated: %s (%s)", cache_path, _human_size(len(data)))
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            log.info("catalogue is up to date (304 Not Modified)")
        elif cache_path.exists():
            log.warning("catalogue fetch failed (HTTP %d) — using cached copy", exc.code)
        else:
            raise RuntimeError(f"cannot fetch catalogue (HTTP {exc.code}) and no cache exists")
    except Exception as exc:
        if cache_path.exists():
            log.warning("catalogue fetch failed (%s) — using cached copy", exc)
        else:
            raise RuntimeError(f"cannot fetch catalogue and no cache exists: {exc}")

    catalogue = json.loads(cache_path.read_text())
    size = cache_path.stat().st_size
    log.info("catalogue loaded: %d entries (%s)", len(catalogue), _human_size(size))
    return catalogue


def build_catalogue_index(catalogue: list[dict]) -> dict:
    """Build in-memory lookup indices from the catalogue."""
    by_tmdb: dict[str, dict] = {}
    by_title_year: dict[tuple[str, str], dict] = {}
    for e in catalogue:
        tid = str(e.get("theMovieDB", "")).strip()
        if tid:
            by_tmdb.setdefault(tid, e)
        key = (e.get("title", "").lower().strip(), str(e.get("year", "")))
        by_title_year.setdefault(key, e)
    log.info("catalogue index: %d tmdb IDs, %d title+year keys",
             len(by_tmdb), len(by_title_year))
    return {"by_tmdb": by_tmdb, "by_title_year": by_title_year}


# ---------------------------------------------------------------------------
# Media discovery
# ---------------------------------------------------------------------------


def discover_media(
    roots: list[Path], catalogue_index: dict,
) -> tuple[list[dict], list[str], list[dict], list[dict]]:
    """Find .mkv files that have a DB ID tag AND a BEQ catalogue match.

    Returns (results, missing_ids, all_media_with_ids, no_catalogue_media).
    - results: only catalogue-matched media (extractable immediately)
    - missing_ids: media files without any DB ID tag (unusable)
    - all_media_with_ids: every media file that had a DB ID, whether
      catalogue-matched or not (used by acquisition recommender to know
      what's already in the library)
    - no_catalogue_media: media dicts (same shape as ``results``) for files
      that have a DB ID but no BEQ catalogue entry. These are candidates
      for the E84 self-training unlabelled pool; the user can opt into
      extracting a bias-corrected subset of them.

    Results sorted breadth-first.
    """
    by_tmdb = catalogue_index["by_tmdb"]
    by_title_year = catalogue_index["by_title_year"]

    results = []
    missing_ids = []
    no_catalogue_desc = []  # for log messages only
    no_catalogue_media: list[dict] = []  # full dicts for re-use
    all_with_ids: list[dict] = []
    seen_keys = set()

    for root in roots:
        if not root.exists():
            log.warning("media root does not exist: %s", root)
            continue

        log.info("scanning for media files in %s ...", root)
        media_files = []
        for ext in MEDIA_EXTENSIONS:
            media_files.extend(root.rglob(f"*{ext}"))

        for f in sorted(media_files):
            if any(part.lower() in JUNK_SUBDIRS for part in f.parts):
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size < MIN_FEATURE_SIZE_BYTES:
                continue

            id_result = extract_media_id(f)
            if not id_result:
                missing_ids.append(str(f))
                continue

            id_type, id_value = id_result
            media_id = f"{id_type}-{id_value}"

            # Extract title and year from directory name.
            title, year = None, None
            for dirname in (f.parent.name, f.parent.parent.name, f.parent.parent.parent.name):
                m2 = TITLE_YEAR_RE.match(dirname)
                if m2:
                    title = m2.group(1).strip()
                    year = m2.group(2)
                    break
            if not title:
                title = f"unknown-{media_id}"
                year = "0000"

            # Check BEQ catalogue.
            has_catalogue = False
            if id_type == "tmdb" and id_value in by_tmdb:
                has_catalogue = True
            elif title and year:
                if (title.lower().strip(), year) in by_title_year:
                    has_catalogue = True

            # Record every media file with a DB ID, whether catalogue-matched
            # or not.  Used by the acquisition recommender to know what's
            # already in the library.
            all_with_ids.append({
                "path": str(f),
                "media_id": media_id,
                "id_type": id_type,
                "id_value": id_value,
                "title": title,
                "year": year,
                "has_catalogue": has_catalogue,
            })

            # Detect TV episodes (done BEFORE the catalogue gate so
            # both the matched and no_catalogue buckets get populated
            # with the same shape).
            ep_match = EPISODE_RE.search(f.stem)
            season = int(ep_match.group(1)) if ep_match else None
            episode = int(ep_match.group(2)) if ep_match else None

            is_tv = ep_match is not None or any(
                "season" in p.lower() for p in f.parts
            )
            content_type = "TV" if is_tv else "film"

            dedup_key = (media_id, season, episode)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            media_entry = {
                "path": f,
                "media_id": media_id,
                "id_type": id_type,
                "id_value": id_value,
                "title": title,
                "year": year,
                "size_bytes": size,
                "content_type": content_type,
                "season": season,
                "episode": episode,
            }

            if not has_catalogue:
                no_catalogue_desc.append(
                    f"{media_id} {title} ({year}) — {f.name}",
                )
                no_catalogue_media.append(media_entry)
                continue

            results.append(media_entry)

    if missing_ids:
        log.warning("%d media files missing DB ID tag — skipped", len(missing_ids))
        for p in missing_ids[:10]:
            log.warning("  missing ID: %s", Path(p).name[:80])
        if len(missing_ids) > 10:
            log.warning("  ... and %d more", len(missing_ids) - 10)

    if no_catalogue_desc:
        log.info(
            "%d media files have DB ID but no BEQ catalogue entry "
            "(candidates for E84 self-training unlabelled pool)",
            len(no_catalogue_desc),
        )
        for desc in no_catalogue_desc[:10]:
            log.info("  no catalogue: %s", desc)
        if len(no_catalogue_desc) > 10:
            log.info("  ... and %d more", len(no_catalogue_desc) - 10)

    results = _breadth_first_sort(results)
    return results, missing_ids, all_with_ids, no_catalogue_media


# ---------------------------------------------------------------------------
# Incremental media discovery (directory-mtime caching)
# ---------------------------------------------------------------------------


def _scan_single_directory(
    dirpath: Path, catalogue_index: dict,
) -> tuple[list[dict], list[dict], list[str]]:
    """Scan one directory (non-recursively) for media files.

    Returns (results, all_with_ids, missing_ids) for this single directory.
    ``results`` are catalogue-matched media entries; ``all_with_ids`` is every
    media file with a DB ID; ``missing_ids`` is paths to files without any
    DB ID tag.

    Uses ``os.scandir`` for speed (avoids stat() per entry on the directory
    listing itself).
    """
    by_tmdb = catalogue_index["by_tmdb"]
    by_title_year = catalogue_index["by_title_year"]

    results: list[dict] = []
    all_with_ids: list[dict] = []
    missing_ids: list[str] = []

    try:
        entries = list(os.scandir(dirpath))
    except (OSError, PermissionError) as exc:
        log.debug("cannot scan directory %s: %s", dirpath, exc)
        return results, all_with_ids, missing_ids

    for entry in sorted(entries, key=lambda e: e.name):
        if not entry.is_file(follow_symlinks=True):
            continue
        f = Path(entry.path)

        # Extension filter.
        if f.suffix.lower() not in MEDIA_EXTENSIONS:
            continue

        # Junk subdirectory filter (check all path parts).
        if any(part.lower() in JUNK_SUBDIRS for part in f.parts):
            continue

        try:
            size = entry.stat(follow_symlinks=True).st_size
        except OSError:
            continue
        if size < MIN_FEATURE_SIZE_BYTES:
            continue

        id_result = extract_media_id(f)
        if not id_result:
            missing_ids.append(str(f))
            continue

        id_type, id_value = id_result
        media_id = f"{id_type}-{id_value}"

        # Extract title and year from directory name.
        title, year = None, None
        for dirname in (f.parent.name, f.parent.parent.name, f.parent.parent.parent.name):
            m2 = TITLE_YEAR_RE.match(dirname)
            if m2:
                title = m2.group(1).strip()
                year = m2.group(2)
                break
        if not title:
            title = f"unknown-{media_id}"
            year = "0000"

        # Check BEQ catalogue.
        has_catalogue = False
        if id_type == "tmdb" and id_value in by_tmdb:
            has_catalogue = True
        elif title and year:
            if (title.lower().strip(), year) in by_title_year:
                has_catalogue = True

        # Detect TV episodes.
        ep_match = EPISODE_RE.search(f.stem)
        season = int(ep_match.group(1)) if ep_match else None
        episode = int(ep_match.group(2)) if ep_match else None

        is_tv = ep_match is not None or any(
            "season" in p.lower() for p in f.parts
        )
        content_type = "TV" if is_tv else "film"

        all_with_ids.append({
            "path": str(f),
            "media_id": media_id,
            "id_type": id_type,
            "id_value": id_value,
            "title": title,
            "year": year,
            "size_bytes": size,
            "content_type": content_type,
            "season": season,
            "episode": episode,
            "has_catalogue": has_catalogue,
        })

        media_entry = {
            "path": f,
            "media_id": media_id,
            "id_type": id_type,
            "id_value": id_value,
            "title": title,
            "year": year,
            "size_bytes": size,
            "content_type": content_type,
            "season": season,
            "episode": episode,
        }

        if has_catalogue:
            results.append(media_entry)
        # Note: no_catalogue filtering is handled by the caller based on
        # has_catalogue in all_with_ids entries, not here.

    return results, all_with_ids, missing_ids



def _strip_root_prefix(abs_path: str, roots: list[Path]) -> str:
    """Strip the media root's *parent* prefix from an absolute path.

    Keeps the root directory's own name as a prefix so that multiple roots
    don't collide.  For example, with root ``/media/Movies``:
    - ``/media/Movies/Dune/Dune.mkv`` → ``Movies/Dune/Dune.mkv``
    - ``/media/Movies`` → ``Movies``

    If the path doesn't start with any root, returns it unchanged (defensive).
    """
    for root in roots:
        root_str = str(root)
        root_parent_str = str(root.parent)
        if abs_path.startswith(root_str + "/"):
            return abs_path[len(root_parent_str) + 1:]
        if abs_path == root_str:
            return root.name
    return abs_path


def _resolve_relative_path(rel_path: str, roots: list[Path]) -> str | None:
    """Resolve a relative path to an absolute path using root names.

    Relative paths start with the root's basename (e.g. ``media-library/Movies/...``).
    Match the first component against each root's name to reconstruct the
    absolute path. Does NOT stat the filesystem — just string manipulation.

    Returns the absolute path string, or None if no root matches.
    """
    rel = Path(rel_path)
    first_component = rel.parts[0] if rel.parts else ""

    for root in roots:
        if root.name == first_component:
            remaining = str(rel.relative_to(first_component)) if len(rel.parts) > 1 else ""
            if remaining and remaining != ".":
                return str(root / remaining)
            return str(root)

    # Fallback: try each root with the full relative path (no stat).
    if roots:
        return str(roots[0].parent / rel_path)

    # No roots — can't resolve.
    if roots:
        for root in roots:
            if root.name == first_component:
                remaining = str(rel.relative_to(first_component)) if len(rel.parts) > 1 else ""
                if remaining and remaining != ".":
                    return str(root / remaining)
                return str(root)
        return str(roots[0].parent / rel_path)
    return None


def _make_entries_relative(entries: list[dict], roots: list[Path]) -> list[dict]:
    """Convert absolute paths in entries to relative paths."""
    result = []
    for entry in entries:
        entry_copy = dict(entry)
        path_str = str(entry_copy.get("path", ""))
        entry_copy["path"] = _strip_root_prefix(path_str, roots)
        result.append(entry_copy)
    return result


def _make_entries_absolute(entries: list[dict], roots: list[Path]) -> list[dict]:
    """Convert relative paths in entries back to absolute paths."""
    result = []
    for entry in entries:
        entry_copy = dict(entry)
        path_str = entry_copy.get("path", "")
        # If already absolute, leave it (backward compat with v2 inventories).
        if not os.path.isabs(path_str):
            resolved = _resolve_relative_path(path_str, roots)
            if resolved:
                entry_copy["path"] = resolved
        result.append(entry_copy)
    return result


def _save_inventory(
    inventory_path: Path,
    directories: dict,
    all_with_ids: list,
    missing_ids: list,
    roots: list[Path],
) -> None:
    """Write the media inventory to disk.

    Paths are stored relative to media roots for portability across machines
    with different mount points (format_version 3).
    """
    # Make entries relative for storage.
    relative_media = _make_entries_relative(all_with_ids, roots)
    relative_missing = [_strip_root_prefix(p, roots) for p in missing_ids]

    # Make directory entries relative too.
    relative_dirs: dict = {}
    for dir_key, dir_data in directories.items():
        rel_key = _strip_root_prefix(dir_key, roots)
        dir_data_copy = dict(dir_data)
        if "entries" in dir_data_copy:
            dir_data_copy["entries"] = _make_entries_relative(
                dir_data_copy["entries"], roots,
            )
        if "missing_ids" in dir_data_copy:
            dir_data_copy["missing_ids"] = [
                _strip_root_prefix(p, roots) for p in dir_data_copy["missing_ids"]
            ]
        relative_dirs[rel_key] = dir_data_copy

    inventory_data = {
        "format_version": 3,
        "scanned_at": int(time.time()),
        "media_roots": [str(r) for r in roots],
        "directories": relative_dirs,
        "n_total_with_ids": len(relative_media),
        "n_catalogue_matched": sum(1 for m in relative_media if m.get("has_catalogue")),
        "n_missing_ids": len(relative_missing),
        "media": relative_media,
        "missing_ids": sorted(set(relative_missing)),
    }
    log.info("saving inventory (%d files, %d dirs) to %s ...",
             len(relative_media), len(relative_dirs), inventory_path)
    inventory_path.write_text(json.dumps(inventory_data, indent=2) + "\n")


def _process_cached_entry(
    entry: dict,
    by_tmdb: dict, by_title_year: dict,
    all_with_ids: list, results: list,
    no_catalogue_media: list, no_catalogue_desc: list,
    seen_keys: set,
) -> None:
    """Re-match a cached entry against the current catalogue and add to output lists."""
    id_type = entry.get("id_type")
    id_value = entry.get("id_value")
    title = entry.get("title")
    year = entry.get("year")

    has_catalogue = False
    if id_type == "tmdb" and id_value in by_tmdb:
        has_catalogue = True
    elif title and year:
        if (title.lower().strip(), year) in by_title_year:
            has_catalogue = True
    entry["has_catalogue"] = has_catalogue
    all_with_ids.append(entry)

    media_id = entry.get("media_id", f"{id_type}-{id_value}")
    season = entry.get("season")
    episode = entry.get("episode")
    dedup_key = (media_id, season, episode)
    if dedup_key in seen_keys:
        return
    seen_keys.add(dedup_key)

    if has_catalogue:
        results.append(entry)
    else:
        no_catalogue_desc.append(f"{media_id} {title} ({year})")
        no_catalogue_media.append(entry)


def _collect_cached_subtree(
    parent_key: str,
    cached_dirs: dict,
    new_directories: dict,
    *,
    all_with_ids_out: list,
    missing_ids_out: list,
    results_out: list,
    no_catalogue_media_out: list,
    no_catalogue_desc_out: list,
    seen_keys: set,
    by_tmdb: dict,
    by_title_year: dict,
    n_cached_counter,
) -> None:
    """Collect all cached entries from a subtree without filesystem access.

    Called when an entire directory subtree is unchanged (all mtimes match).
    Populates the output lists from cached data and copies directory entries
    into new_directories for the updated inventory.
    """
    prefix = parent_key + "/"
    for dir_key, dir_data in cached_dirs.items():
        if dir_key != parent_key and not dir_key.startswith(prefix):
            continue
        # Copy to new directories.
        new_directories[dir_key] = dir_data

        dir_entries = dir_data.get("entries", [])
        dir_missing = dir_data.get("missing_ids", [])
        missing_ids_out.extend(dir_missing)

        for entry in dir_entries:
            id_type = entry.get("id_type")
            id_value = entry.get("id_value")
            title = entry.get("title")
            year = entry.get("year")

            has_catalogue = False
            if id_type == "tmdb" and id_value in by_tmdb:
                has_catalogue = True
            elif title and year:
                if (title.lower().strip(), year) in by_title_year:
                    has_catalogue = True
            entry["has_catalogue"] = has_catalogue

            all_with_ids_out.append(entry)

            media_id = entry.get("media_id", f"{id_type}-{id_value}")
            season = entry.get("season")
            episode = entry.get("episode")
            dedup_key = (media_id, season, episode)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            if has_catalogue:
                results_out.append(entry)
            else:
                no_catalogue_desc_out.append(
                    f"{media_id} {title} ({year})"
                )
                no_catalogue_media_out.append(entry)


def _walk_root_with_progress(
    root: Path,
    catalogue_index: dict,
    by_tmdb: dict,
    by_title_year: dict,
    new_directories: dict[str, dict],
    all_with_ids: list[dict],
    results: list[dict],
    no_catalogue_media: list[dict],
    no_catalogue_desc: list[str],
    missing_ids: list[str],
    seen_keys: set[tuple],
    inventory_path: Path | None = None,
    roots: list[Path] | None = None,
) -> int:
    """Walk a media root with progress logging using adaptive depth detection.

    Saves inventory to disk after every title-level directory so that
    a Ctrl+C loses at most one title's worth of scanning.

    Returns the number of new directories scanned.
    """
    # Use find_media_dirs() to get total count for progress.
    log.info("detecting directory depth under %s ...", root)
    title_dirs = find_media_dirs(root)
    total = len(title_dirs)
    title_dirs_set = {str(d) for d in title_dirs}
    log.info("walking %s (%d title directories) ...", root, total)

    progress = ProgressLogger(total, logger=log, min_interval_s=5)
    n_new = 0
    n_title = 0
    _last_save_time = time.time()
    _SAVE_INTERVAL = 30  # save to disk at most every 30 seconds
    for dirpath_str, dirnames, _filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        dirnames[:] = [
            d for d in dirnames if d.lower() not in JUNK_SUBDIRS
        ]

        # Track and log progress at title-level directories.
        if total > 0 and dirpath_str in title_dirs_set:
            n_title += 1
            progress.update(n_title, label=dirpath.name)

            # Periodic save so Ctrl+C doesn't lose more than ~30s of work.
            now = time.time()
            if inventory_path and roots and (now - _last_save_time) >= _SAVE_INTERVAL:
                _save_inventory(inventory_path, new_directories,
                                all_with_ids, missing_ids, roots)
                _last_save_time = now

        dir_key = str(dirpath)
        try:
            current_mtime = dirpath.stat().st_mtime
        except OSError:
            continue

        _dir_results, dir_entries, dir_missing = _scan_single_directory(
            dirpath, catalogue_index,
        )
        n_new += 1

        new_directories[dir_key] = {
            "mtime": current_mtime,
            "entries": dir_entries,
            "missing_ids": dir_missing,
        }
        missing_ids.extend(dir_missing)
        for entry in dir_entries:
            _process_cached_entry(
                entry, by_tmdb, by_title_year,
                all_with_ids, results, no_catalogue_media,
                no_catalogue_desc, seen_keys,
            )

    progress.finish(f"walked {n_new} dirs ({n_title} titles)")
    return n_new


def discover_media_incremental(
    roots: list[Path], catalogue_index: dict, inventory_path: Path,
) -> tuple[list[dict], list[str], list[dict], list[dict]]:
    """Incremental media discovery with directory-mtime caching.

    Same return signature as ``discover_media()``:
    (results, missing_ids, all_media_with_ids, no_catalogue_media).

    Uses ``inventory_path`` as a cache keyed by directory mtime. Directories
    whose mtime has not changed since the last scan reuse cached entries;
    changed or new directories are rescanned. The catalogue is always
    re-checked (it may have been updated independently).
    """
    by_tmdb = catalogue_index["by_tmdb"]
    by_title_year = catalogue_index["by_title_year"]

    # Step 1: Load existing inventory (if present).
    cached_dirs: dict[str, dict] = {}
    if inventory_path.exists():
        inv_size = inventory_path.stat().st_size
        log.info("reading cached inventory (%s, %.1f MB) ...",
                 inventory_path, inv_size / 1e6)
        t0 = time.time()
        try:
            raw = inventory_path.read_bytes()
            t_read = time.time() - t0
            if t_read >= 5:
                log.info("read in %.1fs, parsing JSON...", t_read)
            old_data = json.loads(raw)
            t_total = time.time() - t0
            if t_total >= 5:
                log.info("inventory loaded in %.1fs", t_total)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to parse inventory %s: %s — rescanning all", inventory_path, exc)
            old_data = {}

        # Step 2: Check format — if old format, discard and rescan.
        fmt_version = old_data.get("format_version", 1)
        if "directories" not in old_data:
            log.info("old inventory format — discarding, will rescan")
            cached_dirs = {}
        elif fmt_version >= 3:
            # v3+: relative paths — convert back to absolute for processing.
            raw_dirs = old_data["directories"]
            n_unresolved = 0
            root_names = [r.name for r in roots]
            for rel_key, dir_data in raw_dirs.items():
                abs_key = _resolve_relative_path(rel_key, roots)
                if abs_key is None:
                    n_unresolved += 1
                    if n_unresolved == 1:
                        first_component = Path(rel_key).parts[0] if Path(rel_key).parts else "?"
                        log.warning(
                            "inventory path not resolved: %r "
                            "(first component %r does not match any root name: %s)",
                            rel_key, first_component, root_names,
                        )
                    continue
                dir_data_copy = dict(dir_data)
                if "entries" in dir_data_copy:
                    dir_data_copy["entries"] = _make_entries_absolute(
                        dir_data_copy["entries"], roots,
                    )
                if "missing_ids" in dir_data_copy:
                    dir_data_copy["missing_ids"] = [
                        _resolve_relative_path(p, roots) or p
                        for p in dir_data_copy["missing_ids"]
                    ]
                cached_dirs[abs_key] = dir_data_copy
            if n_unresolved:
                log.warning(
                    "%d of %d cached directories could not be resolved "
                    "— media root names may differ between machines",
                    n_unresolved, len(raw_dirs),
                )
            log.info("loaded cached inventory (v%d, relative paths): %d directories",
                     fmt_version, len(cached_dirs))
        else:
            # v2: absolute paths — use as-is.
            cached_dirs = old_data["directories"]
            log.info("loaded cached inventory (v%d): %d directories",
                     fmt_version, len(cached_dirs))
    else:
        log.info("no cached inventory at %s — full scan", inventory_path)

    # Step 3: Discover media — targeted stat or full walk.
    all_with_ids: list[dict] = []
    missing_ids: list[str] = []
    results: list[dict] = []
    no_catalogue_desc: list[str] = []
    no_catalogue_media: list[dict] = []
    seen_keys: set[tuple] = set()
    new_directories: dict[str, dict] = {}

    n_cached = 0
    n_rescanned = 0
    n_new = 0

    if cached_dirs:
        # FAST PATH: cache exists. Stat only the parent directories of
        # leaf dirs (dirs with media entries) to detect changes. This
        # replaces os.walk with ~200 targeted os.scandir calls instead
        # of 16000+ stat calls over NFS.

        # Check which roots have cached data. Roots with no cached dirs
        # need a full walk (they were never scanned or Ctrl+C interrupted).
        uncached_roots = []
        cached_roots = []
        for root in roots:
            if not root.exists():
                log.warning("media root does not exist: %s", root)
                continue
            root_str = str(root)
            has_cached = any(k.startswith(root_str) for k in cached_dirs)
            if has_cached:
                cached_roots.append(root)
            else:
                uncached_roots.append(root)
                log.info("root not in cache (needs full scan): %s", root)

        # Find ALL ancestor directories between roots and leaf dirs.
        # This ensures new directories at ANY level are detected.
        leaf_dirs = {k for k, v in cached_dirs.items() if v.get("entries")}
        check_parents: set[str] = set()
        root_strs = {str(r) for r in cached_roots}
        for leaf in leaf_dirs:
            p = Path(leaf).parent
            while str(p) not in check_parents:
                check_parents.add(str(p))
                if str(p) in root_strs or p == p.parent:
                    break
                p = p.parent
        # Include roots themselves.
        check_parents |= root_strs

        n_parents = len(check_parents)
        log.info("checking %d parent directories for changes...", n_parents)
        parent_progress = ProgressLogger(n_parents, logger=log, min_interval_s=5)

        changed_dirs: set[str] = set()
        sorted_parents = sorted(check_parents)
        for idx, parent_key in enumerate(sorted_parents):
            parent_path = Path(parent_key)
            parent_progress.update(idx + 1, label=parent_path.name)
            if not parent_path.exists():
                continue
            try:
                # One readdir per parent — gets all children's stats.
                with os.scandir(parent_path) as it:
                    for entry in it:
                        if not entry.is_dir():
                            continue
                        if entry.name.lower() in JUNK_SUBDIRS:
                            continue
                        child_key = entry.path
                        child_cached = cached_dirs.get(child_key)
                        try:
                            child_mtime = entry.stat().st_mtime
                        except OSError:
                            continue
                        if not child_cached:
                            # New directory — needs scanning.
                            changed_dirs.add(child_key)
                        elif child_cached.get("mtime") != child_mtime:
                            # Changed directory — needs rescanning.
                            changed_dirs.add(child_key)
            except OSError:
                continue

        if not changed_dirs:
            # Nothing changed — load everything from cache.
            log.info("all directories unchanged — loading from cache")
            for dir_key, dir_data in cached_dirs.items():
                new_directories[dir_key] = dir_data
                for entry in dir_data.get("entries", []):
                    _process_cached_entry(
                        entry, by_tmdb, by_title_year,
                        all_with_ids, results, no_catalogue_media,
                        no_catalogue_desc, seen_keys,
                    )
                missing_ids.extend(dir_data.get("missing_ids", []))
                n_cached += 1
        else:
            log.info("%d directories changed — rescanning those, caching rest",
                     len(changed_dirs))
            # Load unchanged dirs from cache.
            for dir_key, dir_data in cached_dirs.items():
                if dir_key in changed_dirs:
                    continue
                new_directories[dir_key] = dir_data
                for entry in dir_data.get("entries", []):
                    _process_cached_entry(
                        entry, by_tmdb, by_title_year,
                        all_with_ids, results, no_catalogue_media,
                        no_catalogue_desc, seen_keys,
                    )
                missing_ids.extend(dir_data.get("missing_ids", []))
                n_cached += 1

            # Rescan only changed directories.
            rescan_progress = ProgressLogger(
                len(changed_dirs), logger=log, min_interval_s=5,
            )
            for rescan_idx, dir_key in enumerate(sorted(changed_dirs)):
                dirpath = Path(dir_key)
                if not dirpath.exists():
                    continue
                rescan_progress.update(rescan_idx + 1, label=dirpath.name)
                try:
                    current_mtime = dirpath.stat().st_mtime
                except OSError:
                    continue
                dir_results, dir_entries, dir_missing = _scan_single_directory(
                    dirpath, catalogue_index,
                )
                n_rescanned += 1
                new_directories[dir_key] = {
                    "mtime": current_mtime,
                    "entries": dir_entries,
                    "missing_ids": dir_missing,
                }
                missing_ids.extend(dir_missing)
                for entry in dir_entries:
                    _process_cached_entry(
                        entry, by_tmdb, by_title_year,
                        all_with_ids, results, no_catalogue_media,
                        no_catalogue_desc, seen_keys,
                    )

        # Save after parent check + rescan so Ctrl+C doesn't lose that work.
        _save_inventory(inventory_path, new_directories, all_with_ids,
                        missing_ids, roots)

        # Walk any uncached roots (never scanned or interrupted).
        if uncached_roots:
            log.info("walking %d uncached root(s)...", len(uncached_roots))
            for root in uncached_roots:
                walked = _walk_root_with_progress(
                    root, catalogue_index, by_tmdb, by_title_year,
                    new_directories, all_with_ids, results,
                    no_catalogue_media, no_catalogue_desc,
                    missing_ids, seen_keys,
                    inventory_path=inventory_path, roots=roots,
                )
                n_new += walked
                # Save after each root (walk also saves every ~30s).
                _save_inventory(inventory_path, new_directories, all_with_ids,
                                missing_ids, roots)

    else:
        # SLOW PATH: no cache at all — full os.walk.
        for root in roots:
            if not root.exists():
                log.warning("media root does not exist: %s", root)
                continue
            walked = _walk_root_with_progress(
                root, catalogue_index, by_tmdb, by_title_year,
                new_directories, all_with_ids, results,
                no_catalogue_media, no_catalogue_desc,
                missing_ids, seen_keys,
                inventory_path=inventory_path, roots=roots,
            )
            n_new += walked
            # Save after each root (walk also saves every ~30s).
            _save_inventory(inventory_path, new_directories, all_with_ids,
                            missing_ids, roots)

    log.info(
        "directory scan complete: %d unchanged (cached), %d rescanned, %d new",
        n_cached, n_rescanned, n_new,
    )

    if missing_ids:
        log.warning("%d media files missing DB ID tag — skipped", len(missing_ids))
        for p in missing_ids[:10]:
            log.warning("  missing ID: %s", Path(p).name[:80])
        if len(missing_ids) > 10:
            log.warning("  ... and %d more", len(missing_ids) - 10)

    if no_catalogue_desc:
        log.info(
            "%d media files have DB ID but no BEQ catalogue entry "
            "(candidates for E84 self-training unlabelled pool)",
            len(no_catalogue_desc),
        )
        for desc in no_catalogue_desc[:10]:
            log.info("  no catalogue: %s", desc)
        if len(no_catalogue_desc) > 10:
            log.info("  ... and %d more", len(no_catalogue_desc) - 10)

    # Final save.
    _save_inventory(inventory_path, new_directories, all_with_ids,
                    missing_ids, roots)

    results = _breadth_first_sort(results)
    return results, missing_ids, all_with_ids, no_catalogue_media


def _breadth_first_sort(media: list[dict]) -> list[dict]:
    """Sort media: movies by size, TV round-robin across shows, interleaved."""
    movies = sorted(
        [m for m in media if m.get("content_type") != "TV"],
        key=lambda m: m.get("size_bytes", 0),
    )

    tv_by_show: dict[str, list[dict]] = defaultdict(list)
    for m in media:
        if m.get("content_type") == "TV":
            tv_by_show[m["media_id"]].append(m)
    for eps in tv_by_show.values():
        eps.sort(key=lambda m: (m.get("season") or 0, m.get("episode") or 0))

    tv_rounds: list[dict] = []
    show_ids = sorted(tv_by_show.keys())
    max_eps = max((len(eps) for eps in tv_by_show.values()), default=0)
    for round_idx in range(max_eps):
        for show_id in show_ids:
            eps = tv_by_show[show_id]
            if round_idx < len(eps):
                tv_rounds.append(eps[round_idx])

    result = []
    mi, ti = 0, 0
    while mi < len(movies) or ti < len(tv_rounds):
        if mi < len(movies):
            result.append(movies[mi])
            mi += 1
        if ti < len(tv_rounds):
            result.append(tv_rounds[ti])
            ti += 1

    log.info("extraction order: %d movies + %d TV episodes (%d shows) = %d total",
             len(movies), len(tv_rounds), len(tv_by_show), len(result))
    return result


# ---------------------------------------------------------------------------
# Portable cache path
# ---------------------------------------------------------------------------


# Cache layout helpers live in model/wav_cache.py — single source of truth
# used by extraction, profile generation, verification, and status reports.
# Re-exported here for backward compat.
from model.wav_cache import (  # noqa: F401, E402
    bucket_name,
    cache_path,
    find_cached_wav,
    legacy_title_cache_path,
)



# ---------------------------------------------------------------------------
# LFE extraction
# ---------------------------------------------------------------------------


def _probe_lfe(media_path: Path) -> bool:
    """Check if the media file has an LFE channel."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channel_layout",
             "-of", "csv=p=0", str(media_path)],
            capture_output=True, text=True, timeout=30,
        )
        layout = result.stdout.strip()
        return "LFE" in layout.upper() or any(
            layout.lower().startswith(p) for p in ("5.1", "6.1", "7.1")
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _failed_sentinel_path(wav_path: Path) -> Path:
    """Path to the sidecar sentinel marking a previously failed extraction."""
    return wav_path.with_suffix(".failed")


def _write_failed_sentinel(wav_path: Path, reason: str) -> None:
    """Record an extraction failure so we don't retry every run.

    Writes a JSON sidecar at ``{wav_path}.failed`` with the failure reason
    and timestamp. Delete the sentinel (or fix the source file) to retry.
    """
    sentinel = _failed_sentinel_path(wav_path)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    try:
        sentinel.write_text(json.dumps({
            "reason": reason,
            "timestamp": time.time(),
            "iso": datetime.now().isoformat(timespec="seconds"),
        }, indent=2) + "\n")
        log.info("  recorded failure: %s (delete to retry)", sentinel)
    except OSError as exc:
        log.warning("  could not write failure sentinel %s: %s", sentinel, exc)


def _check_failed_sentinel(wav_path: Path) -> str | None:
    """Return the prior failure reason if a .failed sentinel exists."""
    sentinel = _failed_sentinel_path(wav_path)
    if not sentinel.exists():
        return None
    try:
        data = json.loads(sentinel.read_text())
        return f"{data.get('reason', 'unknown')} (last attempt: {data.get('iso', '?')})"
    except (json.JSONDecodeError, OSError):
        return "previously failed (sentinel unreadable)"


def _check_mkv_header(media_path: Path) -> str | None:
    """Quick sanity check: file starts with EBML magic (1A 45 DF A3).

    Returns None if the file looks like a valid Matroska/WebM container,
    or a human-readable error string explaining why it doesn't. Avoids
    spinning up ffmpeg only to have it fail on a corrupt or truncated file.

    Only applied to .mkv / .webm files. Other extensions skip the check.
    """
    suffix = media_path.suffix.lower()
    if suffix not in (".mkv", ".webm"):
        return None  # not Matroska, can't pre-validate
    try:
        with media_path.open("rb") as f:
            header = f.read(4)
    except OSError as exc:
        return f"cannot read file: {exc}"
    if len(header) < 4:
        return f"file is too small ({len(header)} bytes) — likely truncated or empty"
    EBML_MAGIC = b"\x1a\x45\xdf\xa3"
    if header != EBML_MAGIC:
        actual = " ".join(f"{b:02x}" for b in header)
        expected = " ".join(f"{b:02x}" for b in EBML_MAGIC)
        return (
            f"not a valid Matroska file — first 4 bytes are {actual!r}, "
            f"expected EBML magic {expected!r}. "
            f"File is probably corrupt; try replacing it from source."
        )
    return None


def extract_one(media_path: Path, wav_path: Path) -> bool:
    """Extract LFE channel (or mono downmix) to a WAV file.

    Uses atomic write: extracts to .tmp, validates, renames on success.
    """
    tmp_path = wav_path.with_suffix(".tmp")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Quick MKV header check before invoking ffmpeg (which is slow to start
    # and produces a less helpful error message for corrupt containers).
    invalid_reason = _check_mkv_header(media_path)
    if invalid_reason is not None:
        log.warning("  SKIP — %s", invalid_reason)
        log.warning("  source: %s", media_path)
        return False

    has_lfe = _probe_lfe(media_path)
    if has_lfe:
        af_filter = "pan=mono|c0=LFE"
    else:
        af_filter = "aresample"
        log.info("  no LFE channel — falling back to mono downmix")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(media_path),
        "-af", af_filter,
        "-ac", "1",
        "-ar", str(_SAMPLE_RATE),
        "-sample_fmt", "s16",
        "-f", "wav",
        str(tmp_path),
    ]

    log.debug("  cmd: %s", " ".join(cmd))
    log.debug("  tmp: %s", tmp_path)
    log.debug("  out: %s", wav_path)

    # Scale ffmpeg timeout by file size: 5min base + 1min per GB.
    # Avoids timeouts on huge 4K Bluray rips (50GB+).
    try:
        size_gb = media_path.stat().st_size / (1024 ** 3)
    except OSError:
        size_gb = 0
    timeout_s = max(300, int(300 + size_gb * 60))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.error("  ffmpeg timed out after %ds", timeout_s)
        log.error("  source: %s", media_path)
        log.error("  target: %s", tmp_path)
        tmp_path.unlink(missing_ok=True)
        _write_failed_sentinel(wav_path, f"ffmpeg timeout after {timeout_s}s")
        return False

    if result.returncode != 0:
        log.error("  ffmpeg failed (code %d)", result.returncode)
        log.error("  stderr: %s", result.stderr.strip())
        log.error("  source: %s", media_path)
        log.error("  target: %s", tmp_path)
        log.error("  cmd: %s", " ".join(cmd))
        tmp_path.unlink(missing_ok=True)
        _write_failed_sentinel(
            wav_path, f"ffmpeg exit {result.returncode}: {result.stderr.strip()[:200]}",
        )
        return False

    # Validate before committing to cache.
    ok, reason = validate_wav_header(tmp_path)
    if not ok:
        log.error("  extracted WAV failed integrity check: %s", reason)
        tmp_path.unlink(missing_ok=True)
        _write_failed_sentinel(wav_path, f"WAV integrity check failed: {reason}")
        return False

    # Atomic rename — only on success.
    tmp_path.rename(wav_path)
    return True


# ---------------------------------------------------------------------------
# Verification (wraps model.wav_integrity)
# ---------------------------------------------------------------------------


def verify_cache(wav_root: Path) -> int:
    """Verify cache and delete corrupt WAVs. Returns count deleted."""
    valid, corrupt = _verify_cache_raw(wav_root)
    for p in corrupt:
        log.warning("corrupt — deleting: %s", p.name)
        p.unlink()
        try:
            p.parent.rmdir()
        except OSError:
            pass
    return len(corrupt)


def cleanup_tmp(wav_root: Path) -> int:
    """Delete leftover .tmp files from interrupted extractions."""
    cleaned = 0
    for tmp in wav_root.rglob("*.tmp"):
        log.info("cleaning up interrupted extraction: %s", tmp.name)
        tmp.unlink()
        cleaned += 1
    return cleaned


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

_CONFIG_NAME = "extract_config.json"


def get_configured_media_roots() -> list[Path]:
    """Return the current media roots without prompting or side effects.

    Resolution order:
    1. ``BEQ_MEDIA_DIR`` env var — auto-discover all dirs under it
    2. ``extract_config.json`` in local config dir (~/.config/beqdesigner/)
    3. Empty list (not configured)

    This is the single source of truth for "what media roots does this
    machine have?" — used by both the extract command and startup validation.
    """
    # 1. Auto-discovery (Docker).
    media_dir_env = os.environ.get("BEQ_MEDIA_DIR")
    if media_dir_env:
        media_dir = Path(media_dir_env)
        if media_dir.is_dir():
            return sorted(p for p in media_dir.iterdir() if p.is_dir())

    # 2. Local config file.
    try:
        from spike._auto_beq_helpers import beq_config_dir
        config_path = beq_config_dir() / _CONFIG_NAME
        if config_path.exists():
            data = json.loads(config_path.read_text())
            roots = data.get("media_roots", [])
            if roots:
                return [Path(p) for p in roots]
    except Exception:
        pass

    return []


def save_extract_config(media_roots: list[Path]) -> Path:
    """Save media roots to the local extract config file.

    Returns the path the config was written to. This is the write side
    of the config service -- get_configured_media_roots() is the read side.
    """
    from spike._auto_beq_helpers import beq_config_dir
    config_path = beq_config_dir() / _CONFIG_NAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(
        {"media_roots": [str(p) for p in media_roots]}, indent=2,
    ) + "\n")
    log.info("config saved to %s (%d media roots)", config_path, len(media_roots))
    return config_path


def _load_or_prompt_config(
    beq_dir_arg: Path | None,
    media_roots_arg: list[Path] | None,
) -> dict:
    """Load saved config, merge with CLI args, prompt if missing, save.

    extract_config.json is stored in the *local* config directory
    (~/.config/beqdesigner/) because it contains machine-specific paths
    (media roots). The shared BEQ directory (wav-cache, catalogue, inventory)
    is separate and portable across machines.
    """
    from spike._auto_beq_helpers import beq_config_dir

    beq_dir = beq_dir_arg
    if beq_dir is None:
        # Try to get default from shared config, fall back to ~/beqdesigner.
        try:
            from spike._auto_beq_helpers import beq_dir as _bd
            default_beq = str(_bd())
        except Exception:
            default_beq = str(Path.home() / ".config" / "beqdesigner")
        log.info("BEQ working directory not specified.")
        log.info("  Default: %s", default_beq)
        raw = input(f"BEQ working directory [{default_beq}]: ").strip()
        beq_dir = Path(raw) if raw else Path(default_beq)

    beq_dir = beq_dir.expanduser().resolve()
    beq_dir.mkdir(parents=True, exist_ok=True)

    wav_root = beq_dir / "wav-cache"
    wav_root.mkdir(parents=True, exist_ok=True)

    # Config lives in local config dir (machine-specific), not in the
    # shared BEQ directory.
    local_config_dir = beq_config_dir()
    config_path = local_config_dir / _CONFIG_NAME

    # Backward compat: migrate from old location (shared dir) if present.
    # Skip in auto-discovery mode (Docker) — config is rebuilt every run.
    old_config_path = beq_dir / _CONFIG_NAME
    auto_discovery = bool(os.environ.get("BEQ_MEDIA_DIR"))
    if not config_path.exists() and old_config_path.exists() and not auto_discovery:
        log.info("migrating %s from %s to %s", _CONFIG_NAME, old_config_path, config_path)
        import shutil
        shutil.copy2(old_config_path, config_path)

    saved: dict = {}
    if config_path.exists():
        try:
            saved = json.loads(config_path.read_text())
            log.info("loaded config from %s", config_path)
        except (json.JSONDecodeError, OSError):
            pass

    media_roots: list[Path] = []
    # Auto-discover from --media-dir if provided (e.g. /media in Docker).
    media_dir_env = os.environ.get("BEQ_MEDIA_DIR")
    if media_dir_env and not media_roots_arg:
        media_dir = Path(media_dir_env)
        if media_dir.is_dir():
            media_roots = sorted(
                p for p in media_dir.iterdir() if p.is_dir()
            )
            log.info("auto-discovered %d media roots under %s", len(media_roots), media_dir)
            for r in media_roots:
                log.info("  media root: %s", r)

    if media_roots_arg:
        media_roots = [p.expanduser().resolve() for p in media_roots_arg]
    elif media_roots:
        pass  # already set from auto-discovery above
    elif saved.get("media_roots"):
        media_roots = [Path(p) for p in saved["media_roots"]]
        log.info("using %d saved media root(s)", len(media_roots))
    else:
        print("Enter media library root paths (one per line, empty line to finish):")
        while True:
            line = input("  media root: ").strip()
            if not line:
                break
            p = Path(line).expanduser().resolve()
            if p.exists():
                media_roots.append(p)
            else:
                log.warning("path does not exist: %s", p)
                print(f"    ERROR: '{p}' does not exist. Please enter a valid path.")
                continue  # re-prompt

    save_extract_config(media_roots)

    return {"beq_dir": beq_dir, "wav_root": wav_root, "media_roots": media_roots}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unmatched media selection (E84 unlabelled-pool growth)
# ---------------------------------------------------------------------------
#
# Inverse of `bin/beq-designer report acquisitions`: that tool picks catalogue
# titles we DON'T have media for, to guide shopping. This picks media files
# we HAVE but the catalogue doesn't — to grow the E84 self-training
# unlabelled pool. Both use greedy bias-correction scoring against the
# BEQ catalogue distribution, but the dimensions differ because unmatched
# media has no catalogue entry (so no author/source/audioTypes fields
# directly — we classify from filename tags instead).


# Release-tag regexes for classifying format from filename. Matches
# common scene / Moozzi2 / Sonarr naming conventions.
_FORMAT_RE_ATMOS = re.compile(r"\b(?:atmos|truehd\s*atmos)\b", re.IGNORECASE)
_FORMAT_RE_TRUEHD = re.compile(r"\btruehd\b", re.IGNORECASE)
_FORMAT_RE_DTSHD = re.compile(r"\bdts[\s._-]*hd\b", re.IGNORECASE)
_FORMAT_RE_DDPLUS = re.compile(r"(?:\bddp|\beac3\b|\bdd\+|\bddplus\b)", re.IGNORECASE)
_FORMAT_RE_DTS = re.compile(r"\bdts\b", re.IGNORECASE)
_FORMAT_RE_FLAC = re.compile(r"\bflac\b", re.IGNORECASE)


def _classify_unmatched_format(path_str: str) -> str:
    """Classify a media file's audio format from filename tags.

    Mirrors the buckets used by ``nn_acquisition_recommender``'s
    ``_classify_format`` so the scoring is apples-to-apples. Checks in
    priority order (best format first). Returns "other" when no tag
    matches — still a usable bucket for scoring.
    """
    if _FORMAT_RE_ATMOS.search(path_str):
        return "atmos"
    if _FORMAT_RE_TRUEHD.search(path_str):
        return "truehd"
    if _FORMAT_RE_DTSHD.search(path_str):
        return "dts-hd"
    if _FORMAT_RE_DDPLUS.search(path_str):
        return "dd+"
    return "other"


def _classify_era(year) -> str:
    """Same buckets as nn_acquisition_recommender._classify_era."""
    try:
        y = int(year)
    except (ValueError, TypeError):
        return "unknown"
    if y < 1990:
        return "pre1990"
    if y < 2010:
        return "1990s-2000s"
    if y < 2020:
        return "2010s"
    return "2020s"


def _score_unmatched_candidate(
    candidate: dict,
    have_format: dict,
    have_era: dict,
    have_ct: dict,
    target_format: dict,
    target_era: dict,
    target_ct: dict,
    have_total: int,
    target_total: int,
) -> float:
    """Greedy bias-correction score for an unmatched media candidate.

    Mirrors ``nn_acquisition_recommender._score_candidate`` but drops
    the ``author`` and ``source`` dimensions (not derivable for
    unmatched media from filename alone). For each of (format, era,
    content_type) compute ``deficit = target_pct - current_pct``;
    squared, sum across dims. Positive deficits (we're underrepresented
    in that bucket) contribute to the score.
    """
    fmt = _classify_unmatched_format(str(candidate.get("path", "")))
    era = _classify_era(candidate.get("year"))
    ct = candidate.get("content_type", "film")

    score = 0.0
    for bucket, current, target in (
        (fmt, have_format, target_format),
        (era, have_era, target_era),
        (ct, have_ct, target_ct),
    ):
        target_pct = (
            100 * target.get(bucket, 0) / target_total if target_total > 0 else 0
        )
        current_pct = (
            100 * current.get(bucket, 0) / have_total if have_total > 0 else 0
        )
        deficit = target_pct - current_pct
        if deficit > 0:
            score += deficit ** 2
    return score


def select_unmatched_to_extract(
    no_catalogue_media: list[dict],
    have_entries: list[dict],
    catalogue: list[dict],
    n: int,
) -> list[dict]:
    """Pick N unmatched media to extract, greedily filling catalogue gaps.

    Uses the same bias-correction approach as
    ``nn_acquisition_recommender`` but with the unmatched media pool as
    candidates. At each step, re-scores the remaining candidates against
    the CURRENT (have + already-picked) distribution so each pick is
    evaluated relative to the most recently updated state.

    Parameters
    ----------
    no_catalogue_media
        Media dicts from ``discover_media``'s fourth return value. Each
        has ``path``, ``title``, ``year``, ``content_type``, ``size_bytes``.
    have_entries
        Catalogue entries for which we already have real WAVs. These
        set the starting "have" distribution we're trying to diversify.
    catalogue
        The full BEQ catalogue — provides the target distribution.
    n
        Number of candidates to return.

    Returns
    -------
    Top-N media dicts in pick order. Use with the same ``extract_one``
    path as the matched-media extraction.
    """
    if not no_catalogue_media or n <= 0:
        return []

    # Target distribution: the BEQ catalogue broken down by format / era /
    # content_type. Same derivation as nn_acquisition_recommender.
    target_format: dict[str, int] = defaultdict(int)
    target_era: dict[str, int] = defaultdict(int)
    target_ct: dict[str, int] = defaultdict(int)
    for e in catalogue:
        if not e.get("filters"):
            continue  # only count trainable entries
        # Format — use catalogue's audioTypes list.
        fmt = "other"
        j = " ".join(e.get("audioTypes", []) or []).lower()
        if "atmos" in j:
            fmt = "atmos"
        elif "truehd" in j:
            fmt = "truehd"
        elif "dts-hd" in j:
            fmt = "dts-hd"
        elif "dd+" in j or "eac3" in j:
            fmt = "dd+"
        target_format[fmt] += 1
        target_era[_classify_era(e.get("year"))] += 1
        target_ct[e.get("content_type", "film")] += 1
    target_total = sum(target_format.values())

    # Starting have distribution: current catalogue-matched WAVs.
    have_format: dict[str, int] = defaultdict(int)
    have_era: dict[str, int] = defaultdict(int)
    have_ct: dict[str, int] = defaultdict(int)
    for e in have_entries:
        # have_entries are catalogue entries (we matched WAV → catalogue),
        # so reuse the same classification as target.
        fmt = "other"
        j = " ".join(e.get("audioTypes", []) or []).lower()
        if "atmos" in j:
            fmt = "atmos"
        elif "truehd" in j:
            fmt = "truehd"
        elif "dts-hd" in j:
            fmt = "dts-hd"
        elif "dd+" in j or "eac3" in j:
            fmt = "dd+"
        have_format[fmt] += 1
        have_era[_classify_era(e.get("year"))] += 1
        have_ct[e.get("content_type", "film")] += 1
    have_total = sum(have_format.values())

    # --- Pre-filter: exclude already-cached media ---
    from model.wav_cache import find_cached_wav
    try:
        from spike._auto_beq_helpers import wav_cache_dir
        wav_root = wav_cache_dir()
    except Exception:
        wav_root = None

    uncached: list[dict] = []
    n_already_cached = 0
    for m in no_catalogue_media:
        if wav_root is not None:
            cached = find_cached_wav(
                wav_root, m["title"], m["year"], m["media_id"],
                content_type=m.get("content_type", "film"),
                season=m.get("season"), episode=m.get("episode"),
            )
            if cached is not None:
                n_already_cached += 1
                continue
        uncached.append(m)

    # --- Deduplicate by title (show-level): one entry per unique title ---
    # For training diversity, different shows teach more than multiple
    # episodes of the same show (same mixer, studio, codec config).
    by_title: dict[str, list[dict]] = {}
    for m in uncached:
        key = m["media_id"]  # group by show ID
        by_title.setdefault(key, []).append(m)
    # Pick one representative per title (first episode).
    representatives = [episodes[0] for episodes in by_title.values()]

    log.info(
        "unmatched selection: %d candidates, %d already cached (skipped), "
        "%d uncached, %d unique titles, %d in have-distribution, "
        "%d in catalogue target",
        len(no_catalogue_media), n_already_cached,
        len(uncached), len(representatives),
        have_total, target_total,
    )

    # Greedy loop: score, pick best, update have distribution, repeat.
    from model.media_utils import ProgressLogger
    progress = ProgressLogger(min(n, len(representatives)), logger=log, min_interval_s=5)
    selected: list[dict] = []
    remaining = list(representatives)

    for pick in range(n):
        if not remaining:
            break
        scored: list[tuple[float, int, dict]] = []
        for idx, cand in enumerate(remaining):
            s = _score_unmatched_candidate(
                cand,
                have_format, have_era, have_ct,
                target_format, target_era, target_ct,
                have_total, target_total,
            )
            scored.append((s, idx, cand))
        scored.sort(key=lambda x: -x[0])
        best_score, best_idx, best_cand = scored[0]
        if best_score <= 0:
            best_cand = min(
                remaining,
                key=lambda c: have_ct.get(c.get("content_type", "film"), 0),
            )
            best_idx = remaining.index(best_cand)

        # Commit the pick: update have-side distribution so the next
        # iteration sees it.
        fmt = _classify_unmatched_format(str(best_cand.get("path", "")))
        era = _classify_era(best_cand.get("year"))
        ct = best_cand.get("content_type", "film")
        have_format[fmt] += 1
        have_era[era] += 1
        have_ct[ct] += 1
        have_total += 1

        selected.append(best_cand)
        remaining.pop(best_idx)
        progress.update(pick + 1, label=best_cand.get("title", ""))

    progress.finish(f"selected {len(selected)} unique titles")
    return selected


def _prompt_unmatched_extraction(
    no_catalogue_count: int,
    default_n: int | None = None,
    assume_yes: bool = False,
) -> int:
    """Interactive prompt: extract N unmatched media? Returns chosen N or 0.

    Always prints the informational message so the user sees what's about
    to happen. If ``assume_yes`` is True (or stdin isn't a TTY), auto-answers
    yes with ``default_n`` items (or ALL if default_n is None) -- no prompt.
    Otherwise prompts interactively.

    Returns the chosen N, clamped to ``[0, no_catalogue_count]``.
    """
    print()
    print(
        f"Found {no_catalogue_count} media files with DB IDs but no "
        f"BEQ catalogue entry.",
        flush=True,
    )
    print(
        "These can grow the E84 self-training unlabelled pool. "
        "Selection uses bias-corrected diversity scoring against the "
        "catalogue distribution.",
        flush=True,
    )

    auto = assume_yes or not sys.stdin.isatty()
    if auto:
        # Default to ALL unmatched. Pass --extract-unmatched N to limit.
        if default_n is None:
            n = no_catalogue_count
            label = "all"
        else:
            n = min(default_n, no_catalogue_count)
            label = f"limit {default_n}"
        reason = "--yes" if assume_yes else "non-interactive (no TTY)"
        print(f"Extract WAVs for some of them? [y/N] y  (auto: {reason})",
              flush=True)
        print(f"How many? {n}  (auto: {label})", flush=True)
        return n

    try:
        resp = input(
            f"Extract WAVs for some of them? [y/N] ",
        ).strip().lower()
        if resp not in ("y", "yes"):
            return 0
        raw = input(
            f"How many? (default {default_n}, max {no_catalogue_count}): ",
        ).strip()
        if not raw:
            return min(default_n, no_catalogue_count)
        try:
            n = int(raw)
        except ValueError:
            print(f"not a number ({raw!r}) — skipping unmatched extraction")
            return 0
        return max(0, min(n, no_catalogue_count))
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Extract LFE audio from media files into a portable WAV cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--media-root", action="append", type=Path, dest="media_roots",
        help="Media library root(s) to scan. Can be specified multiple times.",
    )
    parser.add_argument(
        "--beq-dir", type=Path, default=None,
        help="BEQ working directory. WAVs go in {beq-dir}/wav-cache/. Saved to config on first use.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max titles to extract (0 = unlimited). Useful for testing.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing cache integrity. Deletes corrupt WAVs.",
    )
    parser.add_argument(
        "--extract-unmatched", type=int, default=None,
        help="Also extract WAVs for N media files that have DB IDs but "
             "no BEQ catalogue entry, selected by bias-corrected diversity "
             "scoring. These grow the E84 self-training unlabelled pool. "
             "If omitted and run interactively, the script prompts. Pass "
             "0 to force-skip the prompt in scripted runs.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Auto-answer yes to all prompts (non-interactive mode). "
             "The prompt is still printed so the user can see what's "
             "happening. Useful for Docker / scripted runs.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Log script identity.
    import hashlib
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    log.info("extract_lfe.py [%s]", script_hash)
    log.info("python %s on %s", sys.version.split()[0], sys.platform)

    # Config.
    config = _load_or_prompt_config(args.beq_dir, args.media_roots)
    beq_dir = config["beq_dir"]
    wav_root = config["wav_root"]
    media_roots = config["media_roots"]



    # Verify mode.
    if args.verify:
        log.info("verifying cache at %s ...", wav_root)
        deleted = verify_cache(wav_root)
        existing = len(list(wav_root.rglob("*.lfe-1000hz.wav")))
        log.info("verification complete: %d valid, %d corrupt (deleted)", existing, deleted)
        return

    # Extraction mode.
    if not media_roots:
        parser.error("no media roots configured — use --media-root or run interactively")

    # Validate media roots — any invalid root is a fatal config error.
    log.info("validating %d media root(s)...", len(media_roots))
    invalid_roots = [r for r in media_roots if not r.exists()]
    if invalid_roots:
        bad = "\n  ".join(str(r) for r in invalid_roots)
        raise RuntimeError(
            f"Invalid media roots:\n  {bad}\n\n"
            "Fix the paths or run `bin/beq-designer extract` to reconfigure.\n"
            "If these are NAS paths, make sure the drives are mounted."
        )

    # Fetch BEQ catalogue (may take a few seconds — HTTP check).
    log.info("fetching BEQ catalogue...")
    catalogue = fetch_catalogue(beq_dir)
    cat_index = build_catalogue_index(catalogue)

    # Discover media (incremental — uses media_inventory.json as a
    # directory-mtime cache, rescanning only changed/new directories).
    inventory_path = beq_dir / "media_inventory.json"
    media, missing_ids, all_with_ids, no_catalogue_media = discover_media_incremental(
        media_roots, cat_index, inventory_path,
    )
    if missing_ids:
        missing_file = beq_dir / "missing_ids.txt"
        log.info("writing %d missing IDs to %s ...", len(missing_ids), missing_file)
        missing_file.write_text("\n".join(sorted(set(missing_ids))) + "\n")

    if args.limit > 0:
        media = media[:args.limit]
        log.info("limited to %d titles", len(media))

    # --- Phase 1: catalogue-matched extraction ---
    matched_stats = _run_extraction_phase(
        media=media,
        wav_root=wav_root,
        phase_label="catalogue-matched",
    )

    # --- Phase 2: uncatalogued extraction (E84 unlabelled pool) ---
    unmatched_stats = {"total": 0, "extracted": 0, "skipped": 0, "errors": 0, "elapsed": 0.0}
    n_unmatched_picked = 0
    if no_catalogue_media:
        n_to_extract = args.extract_unmatched
        if n_to_extract is None:
            n_to_extract = _prompt_unmatched_extraction(
                len(no_catalogue_media), assume_yes=args.yes,
            )
        if n_to_extract > 0:
            log.info("")
            log.info("=" * 60)
            log.info(
                "  PHASE 2: selecting %d uncatalogued titles for E84 unlabelled pool",
                n_to_extract,
            )
            log.info("=" * 60)
            have_entries = [e for e in catalogue if e.get("filters") and (
                any(
                    m["has_catalogue"] and m.get("id_type") == "tmdb"
                    and m.get("id_value") == str(e.get("theMovieDB", "")).strip()
                    for m in all_with_ids
                )
            )]
            # Fallback if the above is empty (e.g. id_type mismatch): use
            # all trainable catalogue entries so target stats are non-zero.
            if not have_entries:
                have_entries = [e for e in catalogue if e.get("filters")]
            selected = select_unmatched_to_extract(
                no_catalogue_media=no_catalogue_media,
                have_entries=have_entries,
                catalogue=catalogue,
                n=n_to_extract,
            )
            n_unmatched_picked = len(selected)
            log.info(
                "unmatched selection picked %d titles (bias-corrected):",
                n_unmatched_picked,
            )
            for idx, m in enumerate(selected[:20]):
                log.info(
                    "  %2d. %s (%s) [%s] — format=%s era=%s %s",
                    idx + 1, m["title"], m["year"], m["media_id"],
                    _classify_unmatched_format(str(m["path"])),
                    _classify_era(m["year"]),
                    m["content_type"],
                )
            if n_unmatched_picked > 20:
                log.info("  ... and %d more", n_unmatched_picked - 20)
            unmatched_stats = _run_extraction_phase(
                media=selected,
                wav_root=wav_root,
                phase_label="uncatalogued",
            )

    # --- Summary ---
    total_extracted = matched_stats["extracted"] + unmatched_stats["extracted"]
    total_skipped = matched_stats["skipped"] + unmatched_stats["skipped"]
    total_errors = matched_stats["errors"] + unmatched_stats["errors"]
    total_time = matched_stats["elapsed"] + unmatched_stats["elapsed"]
    log.info("")
    log.info("=" * 60)
    log.info("  EXTRACTION COMPLETE")
    log.info("  Catalogue-matched: %d extracted, %d cached, %d errors",
             matched_stats["extracted"], matched_stats["skipped"], matched_stats["errors"])
    if n_unmatched_picked:
        log.info(
            "  Uncatalogued:      %d extracted, %d cached, %d errors",
            unmatched_stats["extracted"], unmatched_stats["skipped"], unmatched_stats["errors"],
        )
    log.info("  Total extracted:   %d", total_extracted)
    log.info("  Total cached:      %d", total_skipped)
    log.info("  Total errors:      %d", total_errors)
    log.info("  Missing IDs:       %d", len(missing_ids))
    log.info("  Time:              %s", format_duration(total_time))
    log.info("  WAV cache:         %s", wav_root)
    log.info("=" * 60)


def _run_extraction_phase(
    media: list[dict],
    wav_root: Path,
    phase_label: str,
) -> dict:
    """Run the shared extraction loop for a list of media dicts.

    Returns a stats dict with keys: ``total``, ``extracted``, ``skipped``,
    ``errors``, ``elapsed``. ETA tracking is independent per phase.
    Used by both the catalogue-matched main phase and the optional E84
    uncatalogued extraction phase so the actual ffmpeg path stays in
    one place.
    """
    total = len(media)
    if total == 0:
        return {"total": 0, "extracted": 0, "skipped": 0, "errors": 0, "elapsed": 0.0}

    log.info("")
    log.info("=" * 60)
    log.info("  %s extraction: %d titles", phase_label, total)
    log.info("=" * 60)

    extracted = 0
    skipped = 0
    errors = 0
    start_time = time.time()
    extract_times: list[float] = []
    extract_rates: list[float] = []

    for i, m in enumerate(media):
        title = m["title"]
        year = m["year"]
        media_id = m["media_id"]
        size_mb = m.get("size_bytes", 0) / 1e6
        content_type = m.get("content_type", "film")
        season = m.get("season")
        episode = m.get("episode")
        ep_label = f" S{season:02d}E{episode:02d}" if season is not None else ""

        # Check both new (ID-based) and legacy (title-bucket) cache paths.
        cached = find_cached_wav(
            wav_root, title, year, media_id,
            content_type=content_type, season=season, episode=episode,
        )
        # Always write to the canonical (ID-based) path on a fresh extraction.
        wav = cache_path(wav_root, title, year, media_id,
                         content_type=content_type, season=season, episode=episode)

        pct = (i + 1) * 100 // total

        if extract_times:
            avg_rate = sum(extract_rates) / len(extract_rates)
            remaining_mb = sum(
                mm.get("size_bytes", 0) / 1e6 for mm in media[i:]
                if find_cached_wav(
                    wav_root, mm["title"], mm["year"], mm["media_id"],
                    content_type=mm.get("content_type", "film"),
                    season=mm.get("season"), episode=mm.get("episode"),
                ) is None
            )
            eta_s = remaining_mb / avg_rate if avg_rate > 0 else 0
            eta_time = datetime.now() + timedelta(seconds=eta_s)
            eta_str = f" {format_duration(eta_s)} remaining, ETA {eta_time.strftime('%H:%M')}"
        else:
            eta_str = ""

        prefix = f"[{phase_label} {i + 1}/{total} {pct}%{eta_str}]"

        if cached is not None:
            log.info("%s CACHED: %s (%s)%s [%s]", prefix, title, year, ep_label, media_id)
            skipped += 1
            continue

        # Skip if we previously tried and failed (e.g. ffmpeg timeout, corrupt source).
        prior_failure = _check_failed_sentinel(wav)
        if prior_failure:
            log.warning("%s PREVIOUSLY FAILED: %s (%s)%s [%s] — %s",
                        prefix, title, year, ep_label, media_id, prior_failure)
            log.warning("  delete %s to retry",
                        _failed_sentinel_path(wav))
            skipped += 1
            continue

        if extract_rates:
            avg_rate = sum(extract_rates) / len(extract_rates)
            est_s = size_mb / avg_rate if avg_rate > 0 else 0
            est_str = f", est ~{format_duration(est_s)}"
        else:
            est_str = ""

        log.info("%s Extracting: %s (%s)%s [%s] — %.0f MB%s",
                 prefix, title, year, ep_label, media_id, size_mb, est_str)
        media_path = Path(m["path"]) if not isinstance(m["path"], Path) else m["path"]
        log.info("  source: %s", media_path)
        # Diagnostic: explain why this is being extracted (cache miss).
        legacy = legacy_title_cache_path(
            wav_root, title, year, media_id,
            content_type=content_type, season=season, episode=episode,
        )
        log.info("  cache miss — media_id=%s", media_id)
        log.info("    canonical path (not found): %s", wav)
        log.info("    legacy path (not found):    %s", legacy)

        t0 = time.time()
        ok = extract_one(media_path, wav)
        elapsed = time.time() - t0

        if ok:
            wav_size = wav.stat().st_size
            wav_duration = wav_size / (_SAMPLE_RATE * 2)
            rate = size_mb / elapsed if elapsed > 0 else 0
            log.info("  done in %.1fs (%.1f MB/s) — %d bytes (%.0fs audio)",
                     elapsed, rate, wav_size, wav_duration)
            extracted += 1
            extract_times.append(elapsed)
            extract_rates.append(rate)
        else:
            errors += 1

    return {
        "total": total,
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "elapsed": time.time() - start_time,
    }


if __name__ == "__main__":
    main()
