"""TMDb metadata fetcher for the ML training pipeline (Experiment 18).

Fetches studio/distributor, sound re-recording mixer, director, and
production country for BEQ catalogue entries using the TMDb API. Results
are cached locally — once fetched, a title's metadata is never re-fetched.

The BEQ catalogue already carries ``theMovieDB`` IDs on every entry, so
we skip the search step and go straight to the details endpoint.

Future: the enriched metadata will be persisted into the git DB catalogue
output alongside filter parameters — see the TODO in
docs/design/auto_beq_ml_experiments.md.

Usage::

    from model.auto_beq_metadata import load_cache, fetch_metadata_batch, enrich_media_metadata

    # Load from local cache (no network):
    cache = load_cache()

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

# Local user cache — never expires (TMDb metadata doesn't change).
_CACHE_DIR = Path.home() / ".config" / "beqdesigner"
_CACHE_FILE = _CACHE_DIR / "tmdb_metadata_cache.json"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def load_cache() -> dict[str, dict]:
    """Load the TMDb metadata cache from disk. Returns empty dict if missing."""
    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open() as f:
                data = json.load(f)
            log.info("loaded TMDb cache: %d entries from %s", len(data), _CACHE_FILE)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("failed to load TMDb cache: %s", exc)
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    """Persist the TMDb metadata cache to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    tmp.replace(_CACHE_FILE)
    log.info("saved TMDb cache: %d entries to %s", len(cache), _CACHE_FILE)


# ---------------------------------------------------------------------------
# Single-title TMDb fetch
# ---------------------------------------------------------------------------


def _fetch_tmdb_details(tmdb_id: str, is_tv: bool = False) -> dict | None:
    """Fetch movie or TV details + credits from TMDb for a single ID.

    Uses ``/movie/`` or ``/tv/`` endpoint based on ``is_tv``. The BEQ
    catalogue's ``content_type`` field ("film" vs "TV") determines which
    to use — no guessing or fallback needed.

    Returns a dict with extracted fields, or None on failure.
    """
    media_type = "tv" if is_tv else "movie"
    url = f"{_TMDB_BASE}/{media_type}/{tmdb_id}"
    params = {
        "api_key": _TMDB_API_KEY,
        "append_to_response": "credits" if not is_tv else "aggregate_credits",
    }
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            log.warning("TMDb request failed for %s ID %s: %s", media_type, tmdb_id, exc)
            return None

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "2"))
            log.info("TMDb rate-limited (attempt %d/%d), sleeping %ds",
                     attempt + 1, _MAX_RETRIES, retry_after)
            time.sleep(retry_after)
            continue

        if r.status_code != 200:
            log.warning("TMDb returned %d for %s ID %s", r.status_code, media_type, tmdb_id)
            return None

        return _extract_fields(r.json(), is_tv=is_tv)

    log.warning("TMDb rate-limited %d times for %s ID %s, giving up",
                _MAX_RETRIES, media_type, tmdb_id)
    return None


def _extract_fields(data: dict, is_tv: bool = False) -> dict:
    """Extract the ML-relevant fields from a TMDb movie or TV response."""
    # Studio / network: movies have production_companies, TV has networks + production_companies.
    studios = data.get("production_companies", [])
    if is_tv and not studios:
        studios = data.get("networks", [])
    studio = studios[0]["name"] if studios else None
    all_studios = [c["name"] for c in studios]

    # Production country / origin country.
    countries = [c["iso_3166_1"] for c in data.get("production_countries", [])]
    if not countries and is_tv:
        countries = data.get("origin_country", [])
    country = countries[0] if countries else None

    # Crew extraction — TV uses aggregate_credits with different structure.
    if is_tv:
        crew = data.get("aggregate_credits", {}).get("crew", [])
        # aggregate_credits nests jobs: {"jobs": [{"job": "..."}]}
        def _has_job(member: dict, job: str) -> bool:
            return any(j.get("job") == job for j in member.get("jobs", []))
    else:
        crew = data.get("credits", {}).get("crew", [])
        def _has_job(member: dict, job: str) -> bool:
            return member.get("job") == job

    mixers = [c["name"] for c in crew if _has_job(c, "Sound Re-Recording Mixer")]
    sound_editors = [c["name"] for c in crew if _has_job(c, "Supervising Sound Editor")]
    sound_designers = [c["name"] for c in crew if _has_job(c, "Sound Designer")]
    directors = [c["name"] for c in crew if _has_job(c, "Director")]

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

    # Collect unique TMDb IDs not yet in cache, along with content_type.
    to_fetch: list[tuple[str, bool]] = []  # (tmdb_id, is_tv)
    seen: set[str] = set()
    # Index entries by TMDb ID for content_type lookup.
    type_by_id: dict[str, str] = {}
    for e in entries:
        tmdb_id = str(e.get("theMovieDB", "") or "").strip()
        if tmdb_id:
            type_by_id[tmdb_id] = e.get("content_type", "film")
        if tmdb_id and tmdb_id not in cache and tmdb_id not in seen:
            is_tv = e.get("content_type", "film").upper() == "TV"
            to_fetch.append((tmdb_id, is_tv))
            seen.add(tmdb_id)

    if not to_fetch:
        log.info("all %d entries already cached, nothing to fetch", len(cache))
        return cache

    log.info("fetching TMDb metadata for %d new entries (%d already cached)",
             len(to_fetch), len(cache))

    fetched = 0
    errors = 0
    for i, (tmdb_id, is_tv) in enumerate(to_fetch):
        result = _fetch_tmdb_details(tmdb_id, is_tv=is_tv)
        if result is not None:
            cache[tmdb_id] = result
            fetched += 1
        else:
            # Cache the miss so we don't re-fetch on every run.
            cache[tmdb_id] = {"_not_found": True}
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
        all_studios=tuple(tmdb.get("all_studios", [])),
        supervising_mixer=primary_mixer,
        rating=entry.get("rating") or None,
        runtime_min=int(entry.get("runtime", 0) or 0) or None,
    )


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------


def fetch_all(catalogue_entries: list[dict] | None = None) -> dict[str, dict]:
    """One-shot: fetch metadata for the full BEQ catalogue.

    If ``catalogue_entries`` is None, fetches the full catalogue via
    ``auto_beq_catalogue._fetch_or_cache()``.
    """
    if catalogue_entries is None:
        from model.auto_beq_catalogue import _fetch_or_cache
        catalogue_entries = _fetch_or_cache()

    return fetch_metadata_batch(catalogue_entries)
