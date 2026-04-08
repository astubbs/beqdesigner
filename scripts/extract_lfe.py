#!/usr/bin/env python3
"""Standalone LFE extractor — portable WAV cache builder.

**STANDALONE BY DESIGN** — this script intentionally has ZERO project
dependencies. It uses only Python stdlib + ffmpeg/ffprobe. This is so it
can be scp'd to a NAS and run directly without checking out the project.
Do not add imports from model.* or spike.* — any shared logic that exists
in those modules is deliberately duplicated here for portability.

Scans media roots for .mkv files with [tmdb-NNN] in their path, extracts
the LFE channel (or mono downmix) to a portable WAV cache keyed by TMDb ID.

Requires: Python 3.8+, ffmpeg, ffprobe on PATH. No other dependencies.

Usage:
    python3 extract_lfe.py --beq-dir /volume1/beqdesigner --media-root /volume1/media/Movies
    python3 extract_lfe.py --beq-dir /volume1/beqdesigner --verify
    python3 extract_lfe.py   # uses saved config from previous run

Directory structure (managed by the script):
    beq-dir/
      .extract_config.json          # saved media roots
      missing_ids.txt               # media files without DB IDs
      wav-cache/
        Movies/A/Alien (1979) [tmdb-348]/Alien (1979) [tmdb-348].lfe-1000hz.wav
        TV/E/86 - Eighty Six (2021) [tvdb-378609]/Season 01/86 - Eighty Six S01E02 [tvdb-378609].lfe-1000hz.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

log = logging.getLogger("extract_lfe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 1000  # Hz — coupled to BEQ analysis algorithm, not configurable
_MEDIA_EXTENSIONS = {".mkv"}
# Matches [tmdb-NNN], [tvdb-NNN], or [imdb-ttNNN] — any media DB ID tag.
_ID_RE = re.compile(r"\[(tmdb|tvdb|imdb)-([^\]]+)\]")
_TITLE_YEAR_RE = re.compile(r"^(.+?)\s*\((\d{4})\)")

# Files under 500 MB are samples/trailers, not features.
_MIN_FEATURE_SIZE = 500_000_000

# Subdirectories containing non-feature content.
_JUNK_SUBDIRS = {
    "sample", "samples", "featurettes", "featurette", "extras", "extra",
    "backdrops", "behind the scenes", "deleted scenes", "trailers", "trailer",
    "interviews", "shorts",
}


def extract_media_id(media_path: Path) -> tuple[str, str] | None:
    """Extract a media DB ID from a file path.

    Searches (in order): filename, parent dir, grandparent dir.
    First match wins. Returns (id_type, id_value) e.g. ("tmdb", "348")
    or ("tvdb", "378609"), or None if no ID found.
    """
    for source in (media_path.name, media_path.parent.name, media_path.parent.parent.name):
        m = _ID_RE.search(source)
        if m:
            return m.group(1), m.group(2)
    return None


# ---------------------------------------------------------------------------
# Media discovery
# ---------------------------------------------------------------------------


def discover_media(roots: list[Path]) -> list[dict]:
    """Find all .mkv files with [tmdb-NNN] in their path.

    Returns list of dicts: {path, media_id, id_type, id_value, title, year, size_bytes, ...}.
    Files without TMDb IDs are logged and added to the missing list.
    """
    results = []
    missing_ids = []
    seen_keys = set()  # (tmdb_id, season, episode) for dedup

    for root in roots:
        if not root.exists():
            log.warning("media root does not exist: %s", root)
            continue

        log.info("scanning %s ...", root)
        media_files = []
        for ext in _MEDIA_EXTENSIONS:
            media_files.extend(root.rglob(f"*{ext}"))

        for f in sorted(media_files):
            # Filter junk dirs and small files.
            if any(part.lower() in _JUNK_SUBDIRS for part in f.parts):
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size < _MIN_FEATURE_SIZE:
                continue

            # Extract media DB ID — required. Checks filename, parent, grandparent.
            id_result = extract_media_id(f)
            if not id_result:
                missing_ids.append(str(f))
                continue

            id_type, id_value = id_result
            media_id = f"{id_type}-{id_value}"

            # Extract title and year from directory name.
            title, year = None, None
            for dirname in (f.parent.name, f.parent.parent.name, f.parent.parent.parent.name):
                m2 = _TITLE_YEAR_RE.match(dirname)
                if m2:
                    title = m2.group(1).strip()
                    year = m2.group(2)
                    break
            if not title:
                title = f"unknown-{media_id}"
                year = "0000"

            # Detect TV episodes.
            ep_match = _EPISODE_RE.search(f.stem)
            season = int(ep_match.group(1)) if ep_match else None
            episode = int(ep_match.group(2)) if ep_match else None

            # Detect content type from path (TV dirs often contain "Season").
            is_tv = ep_match is not None or any(
                "season" in p.lower() for p in f.parts
            )
            content_type = "TV" if is_tv else "film"

            # Deduplicate: movies by media_id, TV by (media_id, season, episode).
            dedup_key = (media_id, season, episode)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            results.append({
                "path": f,
                "media_id": media_id,      # e.g. "tmdb-348", "tvdb-378609"
                "id_type": id_type,         # "tmdb", "tvdb", or "imdb"
                "id_value": id_value,       # "348", "378609", etc.
                "title": title,
                "year": year,
                "size_bytes": size,
                "content_type": content_type,
                "season": season,
                "episode": episode,
            })

    if missing_ids:
        log.warning("%d media files missing [tmdb-NNN] ID — skipped", len(missing_ids))
        for p in missing_ids[:10]:
            log.warning("  missing ID: %s", Path(p).name[:80])
        if len(missing_ids) > 10:
            log.warning("  ... and %d more", len(missing_ids) - 10)

    # Sort by size (smallest first = fastest extraction).
    results.sort(key=lambda r: r["size_bytes"])
    return results, missing_ids


# ---------------------------------------------------------------------------
# Portable cache path
# ---------------------------------------------------------------------------


_EPISODE_RE = re.compile(r"S(\d+)E(\d+)", re.IGNORECASE)


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

    ``media_id`` is e.g. "tmdb-78", "tvdb-378609" — included in dir and filename.

    Movies:
        wav_root/Movies/B/Blade Runner (1982) [tmdb-78]/Blade Runner (1982) [tmdb-78].lfe-1000hz.wav
    TV:
        wav_root/TV/E/86 - Eighty Six (2021) [tvdb-378609]/Season 01/86 - Eighty Six S01E02 [tvdb-378609].lfe-1000hz.wav
    """
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

    Uses atomic write: extracts to .tmp, renames on success.
    Returns True on success, False on failure.
    """
    tmp_path = wav_path.with_suffix(".tmp")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    has_lfe = _probe_lfe(media_path)
    if has_lfe:
        af_filter = "pan=mono|c0=LFE"
    else:
        af_filter = "aresample"  # mono downmix fallback
        log.info("  no LFE channel — falling back to mono downmix")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(media_path),
        "-af", af_filter,
        "-ac", "1",
        "-ar", str(_SAMPLE_RATE),
        "-sample_fmt", "s16",
        "-f", "wav",   # force WAV format — .tmp extension confuses ffmpeg
        str(tmp_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.error("  ffmpeg timed out after 10 minutes")
        tmp_path.unlink(missing_ok=True)
        return False

    if result.returncode != 0:
        log.error("  ffmpeg failed (code %d): %s", result.returncode, result.stderr[:200])
        tmp_path.unlink(missing_ok=True)
        return False

    # Atomic rename — only on success.
    tmp_path.rename(wav_path)
    return True


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_wav(wav_path: Path) -> tuple[bool, str]:
    """Validate a WAV file's header matches its actual file size."""
    try:
        file_size = wav_path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat: {exc}"
    if file_size == 0:
        return False, "empty file"
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
    except (wave.Error, EOFError, struct.error) as exc:
        return False, f"invalid WAV: {exc}"

    expected = n_frames * n_channels * sample_width
    if file_size - expected < 0:
        return False, f"truncated: {file_size} bytes but header says {expected} data bytes"
    return True, f"ok ({n_frames / _SAMPLE_RATE:.0f}s)"


def verify_cache(wav_root: Path) -> int:
    """Scan cache, delete corrupt WAVs. Returns number deleted."""
    deleted = 0
    for wav in sorted(wav_root.rglob("lfe-1000hz.wav")):
        ok, reason = verify_wav(wav)
        if not ok:
            log.warning("corrupt — deleting: %s (%s)", wav.relative_to(wav_root), reason)
            wav.unlink()
            # Remove empty parent dirs.
            try:
                wav.parent.rmdir()
            except OSError:
                pass
            deleted += 1
        else:
            log.debug("valid: %s (%s)", wav.relative_to(wav_root), reason)
    return deleted


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


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
    """Load saved config, merge with CLI args, prompt if missing, save.

    Config is stored at ``{beq_dir}/.extract_config.json``. WAVs go in
    ``{beq_dir}/wav-cache/``. First run prompts interactively; subsequent
    runs reuse saved paths. CLI args override saved config.

    Returns dict with keys: beq_dir (Path), wav_root (Path), media_roots (list[Path]).
    """
    # Step 1: determine beq_dir.
    beq_dir = beq_dir_arg
    if beq_dir is None:
        # Try to find an existing config in common locations.
        for candidate in [Path.cwd(), Path.home() / "beqdesigner"]:
            cfg = candidate / _CONFIG_NAME
            if cfg.exists():
                beq_dir = candidate
                break
    if beq_dir is None:
        beq_dir = Path(input("BEQ working directory (will be created if needed): ").strip())

    beq_dir = beq_dir.expanduser().resolve()
    beq_dir.mkdir(parents=True, exist_ok=True)

    # WAV cache is always a subdirectory of beq_dir.
    wav_root = beq_dir / "wav-cache"
    wav_root.mkdir(parents=True, exist_ok=True)

    # Step 2: load existing config.
    config_path = beq_dir / _CONFIG_NAME
    saved: dict = {}
    if config_path.exists():
        try:
            saved = json.loads(config_path.read_text())
            log.info("loaded config from %s", config_path)
        except (json.JSONDecodeError, OSError):
            pass

    # Step 3: determine media_roots (CLI args override saved).
    media_roots: list[Path] = []
    if media_roots_arg:
        media_roots = [p.expanduser().resolve() for p in media_roots_arg]
    elif saved.get("media_roots"):
        media_roots = [Path(p) for p in saved["media_roots"]]
        log.info("using %d saved media root(s)", len(media_roots))
    else:
        # Interactive prompt.
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

    # Step 4: save config for next run.
    config_data = {
        "media_roots": [str(p) for p in media_roots],
    }
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
        "--migrate-from", type=Path, dest="migrate_from",
        help="Migrate WAVs from an old path-mirrored cache to the portable structure.",
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

    # --- Config: load saved paths, prompt if missing, save for next run ---
    config = _load_or_prompt_config(args.beq_dir, args.media_roots)
    beq_dir = config["beq_dir"]
    wav_root = config["wav_root"]
    media_roots = config["media_roots"]

    # Clean up any .tmp files from interrupted runs.
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

    # Discover media.
    media, missing_ids = discover_media(media_roots)
    log.info("found %d extractable titles (sorted by size, smallest first)", len(media))

    # Save missing IDs list.
    if missing_ids:
        missing_file = beq_dir / "missing_ids.txt"
        missing_file.write_text("\n".join(sorted(set(missing_ids))) + "\n")
        log.info("missing TMDb IDs written to %s", missing_file)

    # Apply limit.
    if args.limit > 0:
        media = media[:args.limit]
        log.info("limited to %d titles", len(media))

    # Extract.
    total = len(media)
    extracted = 0
    skipped = 0
    errors = 0
    start_time = time.time()

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
        prefix = f"[{i + 1}/{total} {pct}%]"

        if wav.exists():
            log.info("%s CACHED: %s (%s)%s [%s]", prefix, title, year, ep_label, media_id)
            skipped += 1
            continue

        log.info("%s Extracting: %s (%s)%s [%s] — %.0f MB",
                 prefix, title, year, ep_label, media_id, size_mb)
        log.info("  source: %s", m["path"])

        t0 = time.time()
        ok = extract_one(m["path"], wav)
        elapsed = time.time() - t0

        if ok:
            wav_size = wav.stat().st_size
            wav_duration = wav_size / (_SAMPLE_RATE * 2)  # 16-bit mono
            log.info("  done in %.1fs — %d bytes (%.0fs audio)",
                     elapsed, wav_size, wav_duration)
            extracted += 1
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
