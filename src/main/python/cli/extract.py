#!/usr/bin/env python3
"""LFE extractor — portable WAV cache builder.

Scans media roots for .mkv files with a media DB ID tag ([tmdb-NNN],
[tvdb-NNN], [imdb-NNN]), extracts the LFE channel (or mono downmix) to
a portable WAV cache. Only extracts titles with a BEQ catalogue entry.

Run via Docker (recommended) or directly with PYTHONPATH set.

Usage:
    docker compose run extract                          # via Docker
    python3 scripts/extract_lfe.py --beq-dir /path      # direct
    python3 scripts/extract_lfe.py                       # uses saved config

Directory structure (managed by the script):
    beq-dir/
      .extract_config.json          # saved media roots (auto-created on first run)
      beq_catalogue.json            # BEQ catalogue (auto-fetched from GitHub, freshness-checked)
      missing_ids.txt               # media files without DB ID tags
      wav-cache/
        Movies/A/Alien (1979) [tmdb-348]/Alien (1979) [tmdb-348].lfe-1000hz.wav
        TV/E/86 - Eighty Six (2021) [tvdb-378609]/Season 01/86 - Eighty Six S01E02 [tvdb-378609].lfe-1000hz.wav

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


def discover_media(roots: list[Path], catalogue_index: dict) -> tuple[list[dict], list[str]]:
    """Find .mkv files that have a DB ID tag AND a BEQ catalogue match.

    Returns (results, missing_ids). Results sorted breadth-first.
    """
    by_tmdb = catalogue_index["by_tmdb"]
    by_title_year = catalogue_index["by_title_year"]

    results = []
    missing_ids = []
    no_catalogue = []
    seen_keys = set()

    for root in roots:
        if not root.exists():
            log.warning("media root does not exist: %s", root)
            continue

        log.info("scanning %s ...", root)
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
            if not has_catalogue:
                no_catalogue.append(f"{media_id} {title} ({year}) — {f.name}")
                continue

            # Detect TV episodes.
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

            results.append({
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
            })

    if missing_ids:
        log.warning("%d media files missing DB ID tag — skipped", len(missing_ids))
        for p in missing_ids[:10]:
            log.warning("  missing ID: %s", Path(p).name[:80])
        if len(missing_ids) > 10:
            log.warning("  ... and %d more", len(missing_ids) - 10)

    if no_catalogue:
        log.info("%d media files have DB ID but no BEQ catalogue entry — skipped", len(no_catalogue))
        for desc in no_catalogue[:10]:
            log.info("  no catalogue: %s", desc)
        if len(no_catalogue) > 10:
            log.info("  ... and %d more", len(no_catalogue) - 10)

    results = _breadth_first_sort(results)
    return results, missing_ids


def _breadth_first_sort(media: list[dict]) -> list[dict]:
    """Sort media: movies by size, TV round-robin across shows, interleaved."""
    movies = sorted(
        [m for m in media if m.get("content_type") != "TV"],
        key=lambda m: m["size_bytes"],
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
    """Build the portable cache path for a title."""
    content_dir = "TV" if content_type.upper() == "TV" else "Movies"
    letter = title[0].upper() if title and title[0].isalpha() else "#"
    title_dir = f"{title} ({year}) [{media_id}]"

    if content_type.upper() == "TV" and season is not None and episode is not None:
        season_dir = f"Season {season:02d}"
        wav_name = f"{title} S{season:02d}E{episode:02d} [{media_id}].lfe-1000hz.wav"
        return wav_root / content_dir / letter / title_dir / season_dir / wav_name

    wav_name = f"{title} ({year}) [{media_id}].lfe-1000hz.wav"
    return wav_root / content_dir / letter / title_dir / wav_name


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

_CONFIG_NAME = ".extract_config.json"


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
        beq_dir = Path(input("BEQ working directory (will be created if needed): ").strip())

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
                print(f"    WARNING: {p} does not exist, skipping")

    config_data = {"media_roots": [str(p) for p in media_roots]}
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")
    log.info("config saved to %s (%d media roots)", config_path, len(media_roots))

    return {"beq_dir": beq_dir, "wav_root": wav_root, "media_roots": media_roots}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # Clean up interrupted extractions.
    cleaned = cleanup_tmp(wav_root)
    if cleaned:
        log.info("cleaned up %d interrupted extraction(s)", cleaned)

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

    # Fetch BEQ catalogue.
    catalogue = fetch_catalogue(beq_dir)
    cat_index = build_catalogue_index(catalogue)

    # Discover media.
    media, missing_ids = discover_media(media_roots, cat_index)
    log.info("found %d extractable titles (sorted by size, smallest first)", len(media))

    if missing_ids:
        missing_file = beq_dir / "missing_ids.txt"
        missing_file.write_text("\n".join(sorted(set(missing_ids))) + "\n")
        log.info("missing TMDb IDs written to %s", missing_file)

    if args.limit > 0:
        media = media[:args.limit]
        log.info("limited to %d titles", len(media))

    # Extract with ETA tracking.
    total = len(media)
    extracted = 0
    skipped = 0
    errors = 0
    start_time = time.time()
    extract_times: list[float] = []  # seconds per extraction (for ETA)
    extract_rates: list[float] = []  # MB/s per extraction (for per-file estimates)

    for i, m in enumerate(media):
        title = m["title"]
        year = m["year"]
        media_id = m["media_id"]
        size_mb = m["size_bytes"] / 1e6
        content_type = m.get("content_type", "film")
        season = m.get("season")
        episode = m.get("episode")
        ep_label = f" S{season:02d}E{episode:02d}" if season is not None else ""

        wav = cache_path(wav_root, title, year, media_id,
                         content_type=content_type, season=season, episode=episode)

        pct = (i + 1) * 100 // total

        # ETA calculation.
        if extract_times:
            avg_rate = sum(extract_rates) / len(extract_rates)  # MB/s
            remaining = sum(mm["size_bytes"] / 1e6 for mm in media[i:] if not cache_path(
                wav_root, mm["title"], mm["year"], mm["media_id"],
                content_type=mm.get("content_type", "film"),
                season=mm.get("season"), episode=mm.get("episode"),
            ).exists())
            eta_s = remaining / avg_rate if avg_rate > 0 else 0
            eta_str = f" ETA {eta_s / 60:.0f}m" if eta_s > 60 else f" ETA {eta_s:.0f}s"
        else:
            eta_str = ""

        prefix = f"[{i + 1}/{total} {pct}%{eta_str}]"

        if wav.exists():
            log.info("%s CACHED: %s (%s)%s [%s]", prefix, title, year, ep_label, media_id)
            skipped += 1
            continue

        # Estimate extraction time for this file based on average MB/s rate.
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

    # Summary.
    total_time = time.time() - start_time
    log.info("")
    log.info("=" * 60)
    log.info("  EXTRACTION COMPLETE")
    log.info("  Total titles:   %d", total)
    log.info("  Extracted:      %d", extracted)
    log.info("  Already cached: %d", skipped)
    log.info("  Errors:         %d", errors)
    log.info("  Missing IDs:    %d", len(missing_ids))
    log.info("  Time:           %.0fs (%.1f min)", total_time, total_time / 60)
    log.info("  WAV cache:      %s", wav_root)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
