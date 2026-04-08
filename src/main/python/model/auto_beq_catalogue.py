"""Catalogue lookup for auto-BEQ.

The BEQ catalogue (~14,700 entries) is the primary source for filter
chains. When a title matches a catalogue entry, we use the expert's
hand-crafted filters directly — no auto-generation needed.

Auto-generation (via MeasurementAdvisor) is the FALLBACK for
uncatalogued content only.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path

log = logging.getLogger("auto_beq_catalogue")

_CATALOGUE_URL = (
    "https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/"
    "master/docs/database.json"
)
_CACHE_DIR = Path.home() / ".config" / "beqdesigner"
_CACHE_FILE = _CACHE_DIR / "catalogue_cache.json"
_CACHE_MAX_AGE_HOURS = 24


def _fetch_or_cache() -> list[dict]:
    """Return the full catalogue, fetching from GitHub if stale/missing."""
    import time

    if _CACHE_FILE.exists():
        age_hours = (time.time() - _CACHE_FILE.stat().st_mtime) / 3600
        if age_hours < _CACHE_MAX_AGE_HOURS:
            log.debug("catalogue cache hit (%d entries, %.1fh old)",
                      0, age_hours)
            with _CACHE_FILE.open() as f:
                return json.load(f)

    log.info("fetching catalogue from %s", _CATALOGUE_URL)
    try:
        with urllib.request.urlopen(_CATALOGUE_URL, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _CACHE_FILE.open("w") as f:
            json.dump(data, f)
        log.info("cached %d catalogue entries", len(data))
        return data
    except Exception as exc:
        log.warning("catalogue fetch failed: %s", exc)
        if _CACHE_FILE.exists():
            with _CACHE_FILE.open() as f:
                return json.load(f)
        return []


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _codec_contradicted(entry: dict, audio_codec: str | None) -> bool:
    """Check if a catalogue entry's warning field explicitly
    contradicts the given audio codec.

    E.g. "This is ONLY for the Atmos track!!!" should be skipped
    when audio_codec is 'dts' (DTS-HD).
    """
    if not audio_codec:
        return False
    warning = (entry.get("warning") or "").lower()
    if not warning:
        return False
    codec_lower = audio_codec.lower()

    # "ONLY for the Atmos track" → skip if codec is NOT atmos-related
    if "only for the atmos" in warning or "only for atmos" in warning:
        if "atmos" not in codec_lower and "truehd" not in codec_lower:
            return True

    # "DO NOT USE WITH BLU-RAY" → skip if codec looks like disc
    if "do not use with blu-ray" in warning:
        if codec_lower in ("dts", "truehd", "pcm_bluray"):
            return True

    return False


def lookup_catalogue(
    title: str,
    year: int | None = None,
    audio_codec: str | None = None,
    catalogue: list[dict] | None = None,
) -> dict | None:
    """Find the best-matching catalogue entry for a title.

    Matching strategy:
    1. Normalised title match (case-insensitive, punctuation-stripped).
    2. If year given, prefer entries with matching year.
    3. Skip entries whose warning field contradicts the audio codec.
    4. Among remaining matches, prefer the one with the most filters
       (richest correction chain).

    Returns the full catalogue entry dict, or None if no match.
    """
    if catalogue is None:
        catalogue = _fetch_or_cache()
    if not catalogue:
        return None

    norm_title = _normalise_title(title)
    candidates: list[dict] = []

    for entry in catalogue:
        entry_title = _normalise_title(entry.get("title", ""))
        if entry_title != norm_title:
            continue
        entry_year = entry.get("year")
        if year is not None and entry_year:
            # Catalogue stores year as string or int inconsistently.
            if str(entry_year) != str(year):
                continue
        if _codec_contradicted(entry, audio_codec):
            log.info(
                "skipping catalogue entry for %r: warning contradicts codec %r (%s)",
                title, audio_codec, (entry.get("warning") or "")[:60],
            )
            continue
        candidates.append(entry)

    if not candidates:
        return None

    # Prefer entry with most filters (richest chain).
    candidates.sort(key=lambda e: len(e.get("filters", [])), reverse=True)

    # Skip entries with 0 filters (placeholders).
    for c in candidates:
        if c.get("filters"):
            log.info(
                "catalogue match for %r: %d filters, author=%s, source=%s",
                title, len(c["filters"]),
                c.get("author", "?"), c.get("source", "?"),
            )
            return c

    return None
