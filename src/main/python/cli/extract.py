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
    beq-dir/
      extract_config.json           # saved media roots (auto-created on first run)
      beq_catalogue.json            # BEQ catalogue (auto-fetched from GitHub, freshness-checked)
      missing_ids.txt               # media files without DB ID tags (can't identify)
      media_inventory.json          # every media file with a DB ID, whether
                                    # catalogue-matched or not (used by the
                                    # acquisition recommender to dedupe)
      wav-cache/
        AL/Alien (1979) [tmdb-348].lfe-1000hz.wav
        86/86 - Eighty Six [tvdb-378609]/Season 01/S01E02.lfe-1000hz.wav

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

    req = urllib.request.Request(CATALOGUE_URL)
    if cache_path.exists():
        local_mtime = cache_path.stat().st_mtime
        mtime_str = email.utils.formatdate(local_mtime, usegmt=True)
        req.add_header("If-Modified-Since", mtime_str)
        log.info("checking catalogue freshness (cached: %s)...", mtime_str)

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



def _save_inventory(
    inventory_path: Path,
    directories: dict,
    all_with_ids: list,
    missing_ids: list,
    roots: list[Path],
) -> None:
    """Write the media inventory to disk."""
    inventory_data = {
        "format_version": 2,
        "scanned_at": int(time.time()),
        "media_roots": [str(r) for r in roots],
        "directories": directories,
        "n_total_with_ids": len(all_with_ids),
        "n_catalogue_matched": sum(1 for m in all_with_ids if m.get("has_catalogue")),
        "n_missing_ids": len(missing_ids),
        "media": all_with_ids,
        "missing_ids": sorted(set(missing_ids)),
    }
    log.info("saving inventory (%d files, %d dirs) to %s ...",
             len(all_with_ids), len(directories), inventory_path)
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
        log.info("loading cached inventory from %s ...", inventory_path)
        try:
            old_data = json.loads(inventory_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to parse inventory %s: %s — rescanning all", inventory_path, exc)
            old_data = {}

        # Step 2: Check format — if old format, discard and rescan.
        if "directories" in old_data:
            cached_dirs = old_data["directories"]
            log.info("loaded cached inventory: %d directories", len(cached_dirs))
        else:
            log.info("old inventory format — discarding, will rescan")
            cached_dirs = {}
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

        log.info("checking %d parent directories for changes...", len(check_parents))

        changed_dirs: set[str] = set()
        for parent_key in sorted(check_parents):
            parent_path = Path(parent_key)
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
            for dir_key in sorted(changed_dirs):
                dirpath = Path(dir_key)
                if not dirpath.exists():
                    continue
                log.info("  rescanning: %s", dirpath.name)
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

        # Walk any uncached roots (never scanned or interrupted).
        if uncached_roots:
            log.info("walking %d uncached root(s)...", len(uncached_roots))
            for root in uncached_roots:
                log.info("walking directory tree under %s (not in cache)...", root)
                n_walked = 0
                for dirpath_str, dirnames, _filenames in os.walk(root):
                    dirpath = Path(dirpath_str)
                    n_walked += 1
                    if n_walked % 500 == 0:
                        log.info("  ... walked %d dirs so far", n_walked)
                    dirnames[:] = [
                        d for d in dirnames if d.lower() not in JUNK_SUBDIRS
                    ]
                    dir_key = str(dirpath)
                    try:
                        current_mtime = dirpath.stat().st_mtime
                    except OSError:
                        continue
                    dir_results, dir_entries, dir_missing = _scan_single_directory(
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
                log.info("  done: walked %d dirs", n_walked)
                # Save after each root.
                _save_inventory(inventory_path, new_directories, all_with_ids,
                                missing_ids, roots)

        # Save after fast path + any uncached roots.
        _save_inventory(inventory_path, new_directories, all_with_ids,
                        missing_ids, roots)

    else:
        # SLOW PATH: no cache at all — full os.walk.
        for root in roots:
            if not root.exists():
                log.warning("media root does not exist: %s", root)
                continue

            log.info("walking directory tree under %s (first run, no cache)...", root)
            n_walked = 0
            for dirpath_str, dirnames, _filenames in os.walk(root):
                dirpath = Path(dirpath_str)
                n_walked += 1
                if n_walked % 500 == 0:
                    log.info("  ... walked %d dirs so far", n_walked)

                dirnames[:] = [
                    d for d in dirnames if d.lower() not in JUNK_SUBDIRS
                ]

                dir_key = str(dirpath)
                try:
                    current_mtime = dirpath.stat().st_mtime
                except OSError:
                    continue

                dir_results, dir_entries, dir_missing = _scan_single_directory(
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

            log.info("  done: walked %d dirs", n_walked)

            # Save after each root so Ctrl+C doesn't lose everything.
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


def cache_path(
    wav_root: Path,
    title: str,
    year: str,
    media_id: str,
    content_type: str = "film",
    season: int | None = None,
    episode: int | None = None,
) -> Path:
    """Build the portable cache path for a title.

    Layout uses a two-letter bucket directory from the first two characters
    of the title (uppercased, padded with ``_`` if shorter than 2 chars).

    Films:  ``wav-cache/AV/Avatar (2009) [tmdb-19995].lfe-1000hz.wav``
    TV:     ``wav-cache/JU/Jujutsu Kaisen [tvdb-377543]/Season 01/S01E01.lfe-1000hz.wav``
    """
    bucket = (title[:2] if len(title) >= 2 else title.ljust(2, "_")).upper()

    if content_type.upper() == "TV" and season is not None and episode is not None:
        title_dir = f"{title} [{media_id}]"
        season_dir = f"Season {season:02d}"
        wav_name = f"S{season:02d}E{episode:02d}.lfe-1000hz.wav"
        return wav_root / bucket / title_dir / season_dir / wav_name

    wav_name = f"{title} ({year}) [{media_id}].lfe-1000hz.wav"
    return wav_root / bucket / wav_name



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


def extract_one(media_path: Path, wav_path: Path) -> bool:
    """Extract LFE channel (or mono downmix) to a WAV file.

    Uses atomic write: extracts to .tmp, validates, renames on success.
    """
    tmp_path = wav_path.with_suffix(".tmp")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.error("  ffmpeg timed out after 10 minutes")
        log.error("  source: %s", media_path)
        log.error("  target: %s", tmp_path)
        tmp_path.unlink(missing_ok=True)
        return False

    if result.returncode != 0:
        log.error("  ffmpeg failed (code %d)", result.returncode)
        log.error("  stderr: %s", result.stderr.strip())
        log.error("  source: %s", media_path)
        log.error("  target: %s", tmp_path)
        log.error("  cmd: %s", " ".join(cmd))
        tmp_path.unlink(missing_ok=True)
        return False

    # Validate before committing to cache.
    ok, reason = validate_wav_header(tmp_path)
    if not ok:
        log.error("  extracted WAV failed integrity check: %s", reason)
        tmp_path.unlink(missing_ok=True)
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


def _load_or_prompt_config(
    beq_dir_arg: Path | None,
    media_roots_arg: list[Path] | None,
) -> dict:
    """Load saved config, merge with CLI args, prompt if missing, save."""
    beq_dir = beq_dir_arg
    if beq_dir is None:
        for candidate in [Path.cwd(), Path.home() / "beqdesigner"]:
            cfg = candidate / _CONFIG_NAME
            if cfg.exists():
                beq_dir = candidate
                break
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

    config_path = beq_dir / _CONFIG_NAME
    saved: dict = {}
    if config_path.exists():
        try:
            saved = json.loads(config_path.read_text())
            log.info("loaded config from %s", config_path)
        except (json.JSONDecodeError, OSError):
            pass

    media_roots: list[Path] = []
    if media_roots_arg:
        media_roots = [p.expanduser().resolve() for p in media_roots_arg]
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

    config_data = {"media_roots": [str(p) for p in media_roots]}
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")
    log.info("config saved to %s (%d media roots)", config_path, len(media_roots))

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

    log.info(
        "unmatched selection: %d candidates, %d in have-distribution, "
        "%d in catalogue target",
        len(no_catalogue_media), have_total, target_total,
    )

    # Greedy loop: score, pick best, update have distribution, repeat.
    selected: list[dict] = []
    remaining = list(no_catalogue_media)

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
            # No positive deficits left — distribution is already over-
            # represented in every bucket. Fall back to "most diverse so
            # far by content_type" (smallest have_ct wins).
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

    return selected


def _prompt_unmatched_extraction(
    no_catalogue_count: int, default_n: int = 50,
) -> int:
    """Interactive prompt: extract N unmatched media? Returns 0 if no.

    Only prompts when stdin is a TTY (script invocation from a terminal).
    Returns 0 for non-TTY or "n"; otherwise returns the user's chosen N,
    clamped to ``[0, no_catalogue_count]``.
    """
    if not sys.stdin.isatty():
        return 0
    try:
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
            n_to_extract = _prompt_unmatched_extraction(len(no_catalogue_media))
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
    log.info("  Time:              %.0fs (%.1f min)", total_time, total_time / 60)
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

        wav = cache_path(wav_root, title, year, media_id,
                         content_type=content_type, season=season, episode=episode)

        pct = (i + 1) * 100 // total

        if extract_times:
            avg_rate = sum(extract_rates) / len(extract_rates)
            remaining = sum(mm.get("size_bytes", 0) / 1e6 for mm in media[i:] if not cache_path(
                wav_root, mm["title"], mm["year"], mm["media_id"],
                content_type=mm.get("content_type", "film"),
                season=mm.get("season"), episode=mm.get("episode"),
            ).exists())
            eta_s = remaining / avg_rate if avg_rate > 0 else 0
            import datetime as _dt
            eta_time = _dt.datetime.now() + _dt.timedelta(seconds=eta_s)
            eta_str = f" ETA {eta_time.strftime('%H:%M')}"
        else:
            eta_str = ""

        prefix = f"[{phase_label} {i + 1}/{total} {pct}%{eta_str}]"

        if wav.exists():
            log.info("%s CACHED: %s (%s)%s [%s]", prefix, title, year, ep_label, media_id)
            skipped += 1
            continue

        if extract_rates:
            avg_rate = sum(extract_rates) / len(extract_rates)
            est_s = size_mb / avg_rate if avg_rate > 0 else 0
            est_str = f", est ~{est_s / 60:.1f}m" if est_s > 60 else f", est ~{est_s:.0f}s"
        else:
            est_str = ""

        log.info("%s Extracting: %s (%s)%s [%s] — %.0f MB%s",
                 prefix, title, year, ep_label, media_id, size_mb, est_str)
        log.info("  source: %s", m["path"])

        t0 = time.time()
        ok = extract_one(m["path"], wav)
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
