"""Spike tests for the TMDb metadata fetcher (Experiment 18 infrastructure).

Tests the full pipeline: TMDb API fetch → cache → enrich_media_metadata →
build_feature_vector with real studio/mixer data.

These tests hit the real TMDb API and are skipped if the network is
unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import requests
from model.auto_beq import DEFAULT_GRID, evaluate_filter_chain
from model.auto_beq_advisor import MediaMetadata, extract_curve_features
from model.auto_beq_metadata import (
    _extract_fields,
    _fetch_tmdb_details,
    enrich_media_metadata,
    fetch_metadata_batch,
    load_cache,
    save_cache,
)
from model.auto_beq_nn import (
    N_FEATURES,
    build_feature_vector,
    build_metadata_features,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CATALOGUE_SNAPSHOT = _REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "database.json"
_DEFAULT_FS = 1000


def _load_snapshot() -> list[dict]:
    with _CATALOGUE_SNAPSHOT.open() as f:
        return json.load(f)


def _have_network() -> bool:
    try:
        r = requests.get("https://api.themoviedb.org/3/configuration",
                         params={"api_key": "5e23b4412adb55e7cca19cfb9d0196b6"},
                         timeout=5)
        return r.status_code == 200
    except Exception:
        return False


_NETWORK = _have_network()


# ---------------------------------------------------------------------------
# 1. Single-title TMDb fetch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NETWORK, reason="TMDb API unreachable")
def test_fetch_battle_los_angeles():
    """Fetch Battle: LA (TMDb 44943) and verify studio + mixer fields."""
    result = _fetch_tmdb_details("44943")
    assert result is not None

    # Columbia Pictures is the primary studio.
    assert result["studio"] is not None
    assert "Columbia" in result["studio"], f"unexpected studio: {result['studio']}"

    # Should have sound re-recording mixers.
    assert len(result["mixers"]) > 0, "no mixers found"
    print(f"Studio: {result['studio']}")
    print(f"All studios: {result['all_studios']}")
    print(f"Mixers: {result['mixers']}")
    print(f"Directors: {result['directors']}")
    print(f"Country: {result['country']}")


@pytest.mark.skipif(not _NETWORK, reason="TMDb API unreachable")
def test_fetch_mad_max():
    """Mad Max: Fury Road (TMDb 76341) should have Warner Bros."""
    result = _fetch_tmdb_details("76341")
    assert result is not None
    assert "Warner" in result["studio"], f"unexpected studio: {result['studio']}"
    assert len(result["mixers"]) > 0


# ---------------------------------------------------------------------------
# 2. Extract fields from mock response
# ---------------------------------------------------------------------------


def test_extract_fields_minimal():
    """_extract_fields handles missing/empty data gracefully."""
    result = _extract_fields({})
    assert result["studio"] is None
    assert result["country"] is None
    assert result["mixers"] == []
    assert result["directors"] == []


def test_extract_fields_full():
    """_extract_fields extracts all fields from a realistic response."""
    data = {
        "id": 12345,
        "production_companies": [
            {"name": "Universal Pictures", "id": 33},
            {"name": "Legendary Entertainment", "id": 923},
        ],
        "production_countries": [
            {"iso_3166_1": "US", "name": "United States"},
        ],
        "credits": {
            "crew": [
                {"name": "Chris Jenkins", "job": "Sound Re-Recording Mixer", "department": "Sound"},
                {"name": "George Miller", "job": "Director", "department": "Directing"},
                {"name": "Mark Mangini", "job": "Supervising Sound Editor", "department": "Sound"},
                {"name": "Mark Mangini", "job": "Sound Designer", "department": "Sound"},
            ],
        },
    }
    result = _extract_fields(data)
    assert result["studio"] == "Universal Pictures"
    assert result["all_studios"] == ["Universal Pictures", "Legendary Entertainment"]
    assert result["country"] == "US"
    assert result["mixers"] == ["Chris Jenkins"]
    assert result["directors"] == ["George Miller"]
    assert result["sound_editors"] == ["Mark Mangini"]
    assert result["sound_designers"] == ["Mark Mangini"]


# ---------------------------------------------------------------------------
# 3. Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path, monkeypatch):
    """save_cache + load_cache preserves data."""
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_FILE", tmp_path / "test_cache.json")
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_DIR", tmp_path)

    original = {"44943": {"studio": "Columbia Pictures", "mixers": ["Paul Massey"]}}
    save_cache(original)

    loaded = load_cache()
    assert loaded == original


# ---------------------------------------------------------------------------
# 4. Batch fetch with cache
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NETWORK, reason="TMDb API unreachable")
def test_batch_fetch_snapshot(tmp_path, monkeypatch):
    """Batch-fetch metadata for the 18-entry test snapshot.

    Verifies caching: second call should fetch 0 new entries.
    """
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_FILE", tmp_path / "test_cache.json")
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_DIR", tmp_path)

    entries = _load_snapshot()
    cache = fetch_metadata_batch(entries, cache={})

    # All entries in the snapshot have TMDb IDs.
    unique_ids = {str(e.get("theMovieDB", "")).strip() for e in entries if e.get("theMovieDB")}
    assert len(cache) >= len(unique_ids) - 1, (
        f"expected ≥{len(unique_ids)-1} cached, got {len(cache)}"
    )

    # Spot-check: Battle: LA should be in cache.
    assert "44943" in cache
    assert cache["44943"]["studio"] is not None

    # Second call: nothing new to fetch.
    cache2 = fetch_metadata_batch(entries, cache=cache)
    assert len(cache2) == len(cache), "second fetch should add nothing"

    # Print studio distribution for inspection.
    studios = [v.get("studio", "?") for v in cache.values()]
    print(f"\nStudio distribution ({len(studios)} titles):")
    for s in sorted(set(studios)):
        print(f"  {s}: {studios.count(s)}")


# ---------------------------------------------------------------------------
# 5. enrich_media_metadata integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NETWORK, reason="TMDb API unreachable")
def test_enrich_media_metadata(tmp_path, monkeypatch):
    """enrich_media_metadata produces a MediaMetadata with studio + mixer."""
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_FILE", tmp_path / "test_cache.json")
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_DIR", tmp_path)

    entries = _load_snapshot()
    cache = fetch_metadata_batch(entries, cache={})

    # Enrich Battle: LA.
    entry = next(e for e in entries if "Battle" in str(e.get("title", "")))
    metadata = enrich_media_metadata(entry, cache)

    assert metadata.studio is not None, "studio should be populated"
    assert "Columbia" in metadata.studio
    assert metadata.supervising_mixer is not None, "mixer should be populated"
    assert metadata.genres  # should have genres from catalogue
    assert metadata.year == 2011

    print(f"\nEnriched metadata for {metadata.title}:")
    print(f"  studio={metadata.studio}")
    print(f"  mixer={metadata.supervising_mixer}")
    print(f"  genres={metadata.genres}")
    print(f"  source={metadata.source}")


# ---------------------------------------------------------------------------
# 6. Feature vector with real studio data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NETWORK, reason="TMDb API unreachable")
def test_feature_vector_with_studio(tmp_path, monkeypatch):
    """Feature vector has non-zero studio embedding when enriched with TMDb data."""
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_FILE", tmp_path / "test_cache.json")
    monkeypatch.setattr("model.auto_beq_metadata._CACHE_DIR", tmp_path)

    entries = _load_snapshot()
    cache = fetch_metadata_batch(entries, cache={})

    entry = next(e for e in entries if "Battle" in str(e.get("title", "")))
    metadata = enrich_media_metadata(entry, cache)

    # Build synthetic audio features.
    correction = evaluate_filter_chain(entry["filters"], DEFAULT_GRID, fs=_DEFAULT_FS)
    rolloff = -correction
    anchor_idx = int(np.argmin(np.abs(DEFAULT_GRID - 80.0)))
    rolloff_norm = rolloff - rolloff[anchor_idx]
    features = extract_curve_features(rolloff_norm, DEFAULT_GRID)

    vec = build_feature_vector(features, metadata)
    assert vec.shape == (N_FEATURES,)

    # Studio slot should have exactly one 1.0 (one-hot against hybrid vocab).
    # In the full vector: audio(9) + year(1) + format(6) + source(3) = 19
    from model.auto_beq_nn import N_STUDIO, _STUDIO_VOCAB
    studio_slice = vec[19:19 + N_STUDIO]
    assert studio_slice.sum() == 1.0, (
        f"studio should be one-hot when studio='{metadata.studio}', "
        f"got sum={studio_slice.sum()}"
    )
    # "Columbia Pictures" should resolve to "sony/columbia" parent group.
    sony_idx = _STUDIO_VOCAB.index("sony/columbia")
    assert studio_slice[sony_idx] == 1.0, (
        f"Columbia Pictures should resolve to 'sony/columbia' (idx {sony_idx}), "
        f"got argmax={int(studio_slice.argmax())} ({_STUDIO_VOCAB[int(studio_slice.argmax())]})"
    )

    # Without studio, the "other" bucket (last position) should be 1.0.
    meta_no_studio = MediaMetadata(
        title=metadata.title, year=metadata.year,
        audio_types=metadata.audio_types, source=metadata.source,
        genres=metadata.genres, language=metadata.language,
        rating=metadata.rating, runtime_min=metadata.runtime_min,
        studio=None, supervising_mixer=None,
    )
    vec_no_studio = build_feature_vector(features, meta_no_studio)
    studio_slice_none = vec_no_studio[19:19 + N_STUDIO]
    assert studio_slice_none[-1] == 1.0, "studio=None should activate 'other' bucket"

    print(f"\nWith studio '{metadata.studio}': resolved to '{_STUDIO_VOCAB[int(studio_slice.argmax())]}' (idx {int(studio_slice.argmax())})")
    print(f"Without studio: 'other' bucket at idx={int(studio_slice_none.argmax())}")
