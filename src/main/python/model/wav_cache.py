"""WAV cache layout: paths, buckets, and naming conventions.

The WAV cache is a portable, filesystem-friendly directory layout for
extracted LFE audio. All code that reads or writes the cache (extraction,
profile generation, verification, status reports) MUST use these helpers
so paths agree across machines.

Two layouts coexist:

**Canonical (writes use this) -- ID-based, antifragile:**
::

    wav-cache/
      tmdb/19/19995/Avatar (2009) [tmdb-19995].lfe-1000hz.wav
      tvdb/37/377543/Jujutsu Kaisen [tvdb-377543]/Season 01/S01E01.lfe-1000hz.wav

ID values are guaranteed safe across all filesystems (digits + ASCII).
Sharded by the first 2 ID chars to avoid 10000+ entries at one level.

**Legacy (reads only) -- title-bucket, fragile but populated:**
::

    wav-cache/
      AV/Avatar (2009) [tmdb-19995].lfe-1000hz.wav

Used the first 2 chars of the title, which is brittle across filesystems
(Mac strips trailing whitespace, Linux preserves it; NFD vs NFC unicode).
Existing extractions live here; reads fall back to it after the canonical
path misses, so no migration is needed.

Stdlib-only -- safe to import from CLI scripts and headless Docker.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

# Suffix used by all extracted LFE WAV files.
WAV_SUFFIX = ".lfe-1000hz.wav"


# ---------------------------------------------------------------------------
# Canonical (ID-based) layout
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
    """Build the canonical cache path for a title using its media DB ID.

    Layout: ``{id_type}/{shard}/{id_value}/{descriptive_filename}``

    The ID-based directory structure is portable across all filesystems
    because IDs are pure ASCII (digits + dashes + lowercase ``tt`` prefix
    for IMDB). The descriptive filename is included for human-readability
    when browsing the cache with ``ls``/``find``.

    Films:
        ``wav-cache/tmdb/19/19995/Avatar (2009) [tmdb-19995].lfe-1000hz.wav``
    TV:
        ``wav-cache/tvdb/37/377543/Show [tvdb-377543]/Season 01/S01E01.lfe-1000hz.wav``

    ``media_id`` must be in the form ``tmdb-NNN`` / ``tvdb-NNN`` / ``imdb-ttNNN``.
    """
    id_type, id_value = _split_media_id(media_id)
    # Strip non-numeric prefix for sharding (IMDB IDs start with 'tt',
    # which would put all IMDB titles in a single 'tt' shard).
    # Pad to 2 chars for single-digit IDs (tmdb-1 -> shard '01').
    numeric_part = id_value.lstrip("abcdefghijklmnopqrstuvwxyz") or id_value
    shard = numeric_part[:2].zfill(2)
    id_dir = wav_root / id_type / shard / id_value

    if content_type.upper() == "TV" and season is not None and episode is not None:
        title_dir = f"{title} [{media_id}]"
        season_dir = f"Season {season:02d}"
        wav_name = f"S{season:02d}E{episode:02d}{WAV_SUFFIX}"
        return id_dir / title_dir / season_dir / wav_name

    wav_name = f"{title} ({year}) [{media_id}]{WAV_SUFFIX}"
    return id_dir / wav_name


def find_cached_wav(
    wav_root: Path,
    title: str,
    year: str,
    media_id: str,
    content_type: str = "film",
    season: int | None = None,
    episode: int | None = None,
) -> Path | None:
    """Find a cached WAV, checking canonical layout then legacy fallback.

    Returns the path to an existing WAV file, or None if not found.
    Use this for cache hit/miss checks. Use ``cache_path()`` for the
    write target when extracting (always canonical).
    """
    canonical = cache_path(
        wav_root, title, year, media_id, content_type, season, episode,
    )
    if canonical.exists():
        return canonical
    legacy = legacy_title_cache_path(
        wav_root, title, year, media_id, content_type, season, episode,
    )
    if legacy.exists():
        return legacy
    return None


def _split_media_id(media_id: str) -> tuple[str, str]:
    """Split ``tmdb-19995`` -> (``tmdb``, ``19995``).

    Raises ValueError if the ID isn't in the expected format.
    """
    if "-" not in media_id:
        raise ValueError(
            f"media_id must be 'type-value' (e.g. 'tmdb-19995'), got: {media_id!r}",
        )
    id_type, id_value = media_id.split("-", 1)
    if not id_type or not id_value:
        raise ValueError(f"media_id has empty type or value: {media_id!r}")
    return id_type, id_value


# ---------------------------------------------------------------------------
# Legacy (title-bucket) layout -- READS ONLY
# ---------------------------------------------------------------------------


def bucket_name(title: str) -> str:
    """Build a cross-filesystem-safe two-character bucket name from a title.

    Used by the legacy title-bucket layout. New code should use the
    ID-based ``cache_path()`` instead.

    - NFC-normalises unicode BEFORE slicing (Mac filesystems use NFD, Linux
      uses NFC; without this, "Bā" produces a different byte sequence on
      each side and we'd write to two different bucket dirs).
    - Strips whitespace then pads with ``_`` so trailing spaces never appear
      (Mac silently drops them, Linux preserves them).
    - Always returns exactly 2 characters, uppercased.

    Examples:
        "Avatar"   -> "AV"
        "3 Days"   -> "3_"   (was "3 " on Linux, "3" on Mac)
        "A"        -> "A_"
        "Bā'al"    -> "BĀ"   (NFC-normalised first, then sliced)
    """
    normalised = unicodedata.normalize("NFC", title)
    raw = normalised[:2].upper().strip()
    return raw.ljust(2, "_")


def legacy_title_cache_path(
    wav_root: Path,
    title: str,
    year: str,
    media_id: str,
    content_type: str = "film",
    season: int | None = None,
    episode: int | None = None,
) -> Path:
    """Build the legacy title-bucket cache path.

    Used for fallback reads only -- new writes always go to ``cache_path()``.

    Films:  ``wav-cache/AV/Avatar (2009) [tmdb-19995].lfe-1000hz.wav``
    TV:     ``wav-cache/JU/Jujutsu Kaisen [tvdb-377543]/Season 01/S01E01.lfe-1000hz.wav``
    """
    bucket = bucket_name(title)

    if content_type.upper() == "TV" and season is not None and episode is not None:
        title_dir = f"{title} [{media_id}]"
        season_dir = f"Season {season:02d}"
        wav_name = f"S{season:02d}E{episode:02d}{WAV_SUFFIX}"
        return wav_root / bucket / title_dir / season_dir / wav_name

    wav_name = f"{title} ({year}) [{media_id}]{WAV_SUFFIX}"
    return wav_root / bucket / wav_name
