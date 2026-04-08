"""TMDb metadata fetcher for the ML training pipeline (Experiment 18).

Fetches studio/distributor, sound re-recording mixer, director, and
production country for BEQ catalogue entries using the TMDb API.

Three-tier cache strategy (no user fetches metadata that's already known):

1. **Repo-committed mirror** — ``src/test/resources/auto_beq/tmdb_metadata.json``
   Ships with the repo so new users / contributors have all known catalogue
   metadata without hitting TMDb. Updated periodically and committed.
2. **Local user cache** — ``~/.config/beqdesigner/tmdb_metadata_cache.json``
   Warm cache for entries fetched during this user's sessions (includes
   entries not yet committed to the repo mirror).
3. **Live TMDb API** — only for TMDb IDs not found in either cache.

Usage::

    from model.auto_beq_metadata import load_metadata, fetch_metadata_batch, enrich_media_metadata

    # Load from repo mirror + local cache (no network):
    cache = load_metadata()

    # Fetch any missing entries from TMDb and merge into cache:
    cache = fetch_metadata_batch(catalogue_entries, cache=cache)

    # Enrich a single MediaMetadata for the ML pipeline:
    metadata = enrich_media_metadata(entry, cache)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("auto_beq_metadata")

# TMDb API — same key used in model/postbuilder.py for the existing UI
# integration. Public API key embedded in the application.
_TMDB_API_KEY = "5e23b4412adb55e7cca19cfb9d0196b6"
_TMDB_BASE = "https://api.themoviedb.org/3"

# TMDb rate limit: ~40 requests per 10 seconds. We don't artificially
# throttle — just respect 429 Retry-After headers when they come.
_REQUEST_TIMEOUT_S = 15
_MAX_RETRIES = 3

# Tier 1: repo-committed mirror (ships with the repo).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_REPO_METADATA_FILE = _REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "tmdb_metadata.json"

# Tier 2: local user cache (warm cache for user-fetched entries).
_CACHE_DIR = Path.home() / ".config" / "beqdesigner"
_CACHE_FILE = _CACHE_DIR / "tmdb_metadata_cache.json"


# ---------------------------------------------------------------------------
# Cache I/O — three-tier loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, dict]:
    """Load a JSON dict from path, returning empty dict on any failure."""
    if path.exists():
        try:
            with path.open() as f:
                data = json.load(f)
            log.info("loaded %d entries from %s", len(data), path)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to load %s: %s", path, exc)
    return {}


def load_metadata() -> dict[str, dict]:
    """Load TMDb metadata from repo mirror + local user cache (no network).

    Merges both sources. Local user cache entries take precedence over
    repo entries (in case the user has fresher data).
    """
    repo = _load_json(_REPO_METADATA_FILE)
    local = _load_json(_CACHE_FILE)
    merged = {**repo, **local}
    if repo and local:
        log.info("merged metadata: %d repo + %d local = %d total (after dedup)",
                 len(repo), len(local), len(merged))
    return merged


# Back-compat alias.
load_cache = load_metadata


def save_cache(cache: dict[str, dict]) -> None:
    """Persist the TMDb metadata to the local user cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    tmp.replace(_CACHE_FILE)
    log.info("saved TMDb cache: %d entries to %s", len(cache), _CACHE_FILE)


def save_repo_metadata(cache: dict[str, dict]) -> None:
    """Write the full metadata cache to the repo-committed mirror.

    Call this after a batch fetch to update the repo copy. The updated
    file should then be committed to git so other users get it for free.
    """
    tmp = _REPO_METADATA_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    tmp.replace(_REPO_METADATA_FILE)
    log.info("saved repo metadata mirror: %d entries to %s", len(cache), _REPO_METADATA_FILE)


# ---------------------------------------------------------------------------
# Single-title TMDb fetch
# ---------------------------------------------------------------------------


def _fetch_tmdb_details(tmdb_id: str) -> dict | None:
    """Fetch movie details + credits from TMDb for a single ID.

    No artificial throttling — fires requests as fast as possible and
    respects 429 Retry-After headers when TMDb tells us to slow down.

    Returns a dict with extracted fields, or None on failure.
    """
    url = f"{_TMDB_BASE}/movie/{tmdb_id}"
    params = {
        "api_key": _TMDB_API_KEY,
        "append_to_response": "credits",
    }
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            log.warning("TMDb request failed for ID %s: %s", tmdb_id, exc)
            return None

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "2"))
            log.info("TMDb rate-limited (attempt %d/%d), sleeping %ds",
                     attempt + 1, _MAX_RETRIES, retry_after)
            time.sleep(retry_after)
            continue

        if r.status_code != 200:
            log.warning("TMDb returned %d for ID %s", r.status_code, tmdb_id)
            return None

        return _extract_fields(r.json())

    log.warning("TMDb rate-limited %d times for ID %s, giving up", _MAX_RETRIES, tmdb_id)
    return None


def _extract_fields(data: dict) -> dict:
    """Extract the ML-relevant fields from a TMDb movie+credits response."""
    # Studio: first production company (typically the primary studio).
    studios = data.get("production_companies", [])
    studio = studios[0]["name"] if studios else None

    # All production company names (for richer embedding later).
    all_studios = [c["name"] for c in studios]

    # Production country ISO codes.
    countries = [c["iso_3166_1"] for c in data.get("production_countries", [])]
    country = countries[0] if countries else None

    # Crew extraction.
    crew = data.get("credits", {}).get("crew", [])

    # Sound re-recording mixer(s) — the people who set the bass rolloff.
    mixers = [c["name"] for c in crew if c.get("job") == "Sound Re-Recording Mixer"]
    # Supervising sound editor as a fallback signal.
    sound_editors = [c["name"] for c in crew if c.get("job") == "Supervising Sound Editor"]
    # Sound designers.
    sound_designers = [c["name"] for c in crew if c.get("job") == "Sound Designer"]

    # Director.
    directors = [c["name"] for c in crew if c.get("job") == "Director"]

    return {
        "studio": studio,
        "all_studios": all_studios,
        "country": country,
        "countries": countries,
        "mixers": mixers,
        "sound_editors": sound_editors,
        "sound_designers": sound_designers,
        "directors": directors,
        "tmdb_id": str(data.get("id", "")),
    }


# ---------------------------------------------------------------------------
# Batch fetch
# ---------------------------------------------------------------------------


def fetch_metadata_batch(
    entries: list[dict],
    cache: dict[str, dict] | None = None,
    progress_every: int = 50,
) -> dict[str, dict]:
    """Fetch TMDb metadata for all catalogue entries that have a theMovieDB ID.

    Skips entries already in the cache. Rate-limits requests. Saves cache to
    disk periodically and at the end.

    Args:
        entries: BEQ catalogue entries (dicts with 'theMovieDB' field).
        cache: Existing cache to extend. If None, loads from disk.
        progress_every: Log progress every N fetches.

    Returns:
        The updated cache dict (also persisted to disk).
    """
    if cache is None:
        cache = load_cache()

    # Collect unique TMDb IDs not yet in cache.
    to_fetch: list[str] = []
    seen: set[str] = set()
    for e in entries:
        tmdb_id = str(e.get("theMovieDB", "") or "").strip()
        if tmdb_id and tmdb_id not in cache and tmdb_id not in seen:
            to_fetch.append(tmdb_id)
            seen.add(tmdb_id)

    if not to_fetch:
        log.info("all %d entries already cached, nothing to fetch", len(cache))
        return cache

    log.info("fetching TMDb metadata for %d new entries (%d already cached)",
             len(to_fetch), len(cache))

    fetched = 0
    errors = 0
    for i, tmdb_id in enumerate(to_fetch):
        result = _fetch_tmdb_details(tmdb_id)
        if result is not None:
            cache[tmdb_id] = result
            fetched += 1
        else:
            errors += 1

        if (i + 1) % progress_every == 0:
            log.info("progress: %d/%d fetched, %d errors", i + 1, len(to_fetch), errors)
            save_cache(cache)

    save_cache(cache)
    log.info("batch complete: %d fetched, %d errors, %d total cached",
             fetched, errors, len(cache))
    return cache


# ---------------------------------------------------------------------------
# Enrichment helper for the ML pipeline
# ---------------------------------------------------------------------------


def enrich_media_metadata(entry: dict, cache: dict[str, dict]):
    """Create a fully-enriched MediaMetadata from a catalogue entry + TMDb cache.

    Returns a MediaMetadata with all Tier 1-3 fields populated where data
    is available.
    """
    from model.auto_beq_advisor import MediaMetadata

    tmdb_id = str(entry.get("theMovieDB", "") or "").strip()
    tmdb = cache.get(tmdb_id, {})

    # Primary mixer: first "Sound Re-Recording Mixer" from TMDb credits.
    mixers = tmdb.get("mixers", [])
    primary_mixer = mixers[0] if mixers else None

    return MediaMetadata(
        title=str(entry.get("title", "")),
        year=int(entry.get("year", 0) or 0) or None,
        audio_codec=entry.get("audioTypes", [None])[0] if entry.get("audioTypes") else None,
        audio_types=tuple(entry.get("audioTypes", [])),
        source=entry.get("source") or None,
        genres=tuple(entry.get("genres", [])),
        language=entry.get("language") or None,
        studio=tmdb.get("studio"),
        supervising_mixer=primary_mixer,
        rating=entry.get("rating") or None,
        runtime_min=int(entry.get("runtime", 0) or 0) or None,
    )


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------


def fetch_all(
    catalogue_entries: list[dict] | None = None,
    update_repo_mirror: bool = True,
) -> dict[str, dict]:
    """One-shot: fetch metadata for the full BEQ catalogue.

    If ``catalogue_entries`` is None, fetches the full catalogue via
    ``auto_beq_catalogue._fetch_or_cache()``.

    When ``update_repo_mirror`` is True (default), writes the full merged
    cache to the repo-committed mirror so it can be committed to git.
    """
    if catalogue_entries is None:
        from model.auto_beq_catalogue import _fetch_or_cache
        catalogue_entries = _fetch_or_cache()

    cache = fetch_metadata_batch(catalogue_entries)
    if update_repo_mirror:
        save_repo_metadata(cache)
    return cache
