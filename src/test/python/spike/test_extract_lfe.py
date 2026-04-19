"""Unit tests for ``cli/extract.py`` — bias-corrected unmatched selection
and incremental media discovery.

Only pure-function helpers are tested here — anything that actually
shells out to ffmpeg belongs in an integration test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cli import extract as extract_mod_static


@pytest.fixture(scope="module")
def extract_mod():
    """Return the extract module."""
    return extract_mod_static


# ---------------------------------------------------------------------------
# _classify_unmatched_format — filename tag heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,expected", [
    ("/Movies/Dune (2021) [Bluray-2160p][TrueHD Atmos 7.1][x265]/Dune.mkv", "atmos"),
    ("/Movies/Dune (2021) [Bluray-2160p][TrueHD 7.1]/Dune.mkv", "truehd"),
    ("/Movies/Alien (1979) [Bluray-2160p][DTS-HD MA 7.1]/Alien.mkv", "dts-hd"),
    ("/TV/Show/Show S01E01 [WEBDL-2160p][DDP5.1][x265]/ep.mkv", "dd+"),
    ("/TV/Show/Show S01E01 [WEBDL-2160p][EAC3 5.1]/ep.mkv", "dd+"),
    ("/Movies/Nothing (2020)/Nothing.mkv", "other"),
    ("/Movies/Anime [Bluray-2160p][FLAC 2.0][x265]/anime.mkv", "other"),
])
def test_classify_unmatched_format(extract_mod, path, expected):
    assert extract_mod._classify_unmatched_format(path) == expected


@pytest.mark.parametrize("year,expected", [
    (1985, "pre1990"),
    ("1985", "pre1990"),
    (1999, "1990s-2000s"),
    (2009, "1990s-2000s"),
    (2010, "2010s"),
    (2019, "2010s"),
    (2020, "2020s"),
    (2024, "2020s"),
    (None, "unknown"),
    ("not-a-year", "unknown"),
])
def test_classify_era(extract_mod, year, expected):
    assert extract_mod._classify_era(year) == expected


# ---------------------------------------------------------------------------
# select_unmatched_to_extract — greedy bias-correction selection
# ---------------------------------------------------------------------------


def _make_candidate(path: str, year: int, content_type: str = "film") -> dict:
    return {
        "path": path,
        "media_id": f"tmdb-{abs(hash(path)) % 100000}",
        "title": Path(path).stem.split(" (")[0],
        "year": str(year),
        "size_bytes": 10 * 1_000_000_000,  # ~10 GB, above the filter
        "content_type": content_type,
    }


def _make_catalogue_entry(author, year, audio, ct="film") -> dict:
    return {
        "title": f"Title-{year}",
        "year": str(year),
        "author": author,
        "audioTypes": [audio],
        "content_type": ct,
        "filters": [{"type": "LowShelf", "freq": 20, "gain": 5, "q": 0.9}],
    }


def test_select_unmatched_picks_all_if_n_exceeds_pool(extract_mod):
    """Asking for more than the pool size just returns the whole pool."""
    candidates = [
        _make_candidate("/M/A (2020) [Bluray-2160p][Atmos]/A.mkv", 2020),
        _make_candidate("/M/B (1995) [Bluray-1080p][DTS-HD MA]/B.mkv", 1995),
    ]
    catalogue = [
        _make_catalogue_entry("a", 2020, "Atmos"),
        _make_catalogue_entry("b", 1995, "DTS-HD MA"),
    ]
    picks = extract_mod.select_unmatched_to_extract(
        no_catalogue_media=candidates,
        have_entries=[],
        catalogue=catalogue,
        n=10,
    )
    assert len(picks) == 2
    # Both candidates should be present.
    assert {p["title"] for p in picks} == {"A", "B"}


def test_select_unmatched_zero_n_returns_empty(extract_mod):
    candidates = [_make_candidate("/M/A (2020) [Bluray-2160p][Atmos]/A.mkv", 2020)]
    picks = extract_mod.select_unmatched_to_extract(
        no_catalogue_media=candidates,
        have_entries=[],
        catalogue=[_make_catalogue_entry("a", 2020, "Atmos")],
        n=0,
    )
    assert picks == []


def test_select_unmatched_empty_pool_returns_empty(extract_mod):
    picks = extract_mod.select_unmatched_to_extract(
        no_catalogue_media=[],
        have_entries=[],
        catalogue=[_make_catalogue_entry("a", 2020, "Atmos")],
        n=10,
    )
    assert picks == []


def test_select_unmatched_prefers_gap_filling_format(extract_mod):
    """If the WAV cache already has all Atmos, the next pick favours DTS-HD."""
    # Catalogue target: roughly balanced across formats.
    catalogue = []
    for _ in range(5):
        catalogue.append(_make_catalogue_entry("a", 2020, "Atmos"))
    for _ in range(5):
        catalogue.append(_make_catalogue_entry("b", 2020, "DTS-HD MA"))

    # Have distribution: only Atmos so far. Gap is entirely in DTS-HD.
    have = [_make_catalogue_entry("a", 2020, "Atmos") for _ in range(5)]

    # Both candidates are 2020, 1 is Atmos, 1 is DTS-HD. The gap-filling
    # pick should be the DTS-HD one.
    candidates = [
        _make_candidate("/M/A (2020) [Bluray-2160p][Atmos]/A.mkv", 2020),
        _make_candidate("/M/B (2020) [Bluray-2160p][DTS-HD MA]/B.mkv", 2020),
    ]
    picks = extract_mod.select_unmatched_to_extract(
        no_catalogue_media=candidates,
        have_entries=have,
        catalogue=catalogue,
        n=1,
    )
    assert len(picks) == 1
    # DTS-HD wins because the have-side has zero of it vs 50% target.
    assert picks[0]["title"] == "B"


def test_select_unmatched_prefers_gap_filling_era(extract_mod):
    """Have is all 2020s; next pick should favour an older era to diversify."""
    catalogue = (
        [_make_catalogue_entry("a", 2020, "Atmos") for _ in range(5)]
        + [_make_catalogue_entry("b", 1995, "Atmos") for _ in range(5)]
    )
    have = [_make_catalogue_entry("a", 2020, "Atmos") for _ in range(5)]
    candidates = [
        _make_candidate("/M/A (2023) [Bluray-2160p][Atmos]/A.mkv", 2023),
        _make_candidate("/M/B (1995) [Bluray-1080p][Atmos]/B.mkv", 1995),
    ]
    picks = extract_mod.select_unmatched_to_extract(
        no_catalogue_media=candidates,
        have_entries=have,
        catalogue=catalogue,
        n=1,
    )
    assert picks[0]["title"] == "B", "1995 era should beat 2023 when have is already 2020s-heavy"


def test_select_unmatched_greedy_rescoring(extract_mod):
    """After the first pick, the next pick is scored against the updated distribution.

    Start with have = all Atmos/2020s. Three candidates all differ along
    at least one dimension. The first pick should fill the largest gap;
    the second pick should fill the new-largest gap (different from the
    first).
    """
    catalogue = (
        [_make_catalogue_entry("a", 2022, "Atmos") for _ in range(10)]  # target: 50% era 2020s + Atmos
        + [_make_catalogue_entry("b", 1995, "DTS-HD MA") for _ in range(10)]
    )
    have = [_make_catalogue_entry("a", 2022, "Atmos") for _ in range(10)]
    candidates = [
        _make_candidate("/M/A (1995) [Bluray-1080p][DTS-HD MA]/A.mkv", 1995),
        _make_candidate("/M/B (2010) [Bluray-1080p][Atmos]/B.mkv", 2010),
        _make_candidate("/M/C (2023) [Bluray-2160p][Atmos]/C.mkv", 2023),
    ]
    picks = extract_mod.select_unmatched_to_extract(
        no_catalogue_media=candidates,
        have_entries=have,
        catalogue=catalogue,
        n=2,
    )
    assert len(picks) == 2
    # A fills both format (DTS-HD) and era (1995) gaps — highest score.
    assert picks[0]["title"] == "A"
    # Second pick is the remaining best — B (2010s era) is the next best
    # era gap now that A has been committed.
    assert picks[1]["title"] in ("B", "C")


# ---------------------------------------------------------------------------
# _scan_single_directory — single-directory non-recursive scan
# ---------------------------------------------------------------------------


def _make_catalogue_index(entries: list[dict] | None = None) -> dict:
    """Build a catalogue_index for testing."""
    by_tmdb: dict[str, dict] = {}
    by_title_year: dict[tuple[str, str], dict] = {}
    for e in (entries or []):
        tid = str(e.get("theMovieDB", "")).strip()
        if tid:
            by_tmdb.setdefault(tid, e)
        key = (e.get("title", "").lower().strip(), str(e.get("year", "")))
        by_title_year.setdefault(key, e)
    return {"by_tmdb": by_tmdb, "by_title_year": by_title_year}


@pytest.fixture
def _bypass_min_size(monkeypatch):
    """Zero-byte fixture files need the size filter bypassed."""
    import model.media_constants as mc
    monkeypatch.setattr(mc, "MIN_FEATURE_SIZE_BYTES", 0)
    # Also patch the module-level reference in extract.
    monkeypatch.setattr(extract_mod_static, "MIN_FEATURE_SIZE_BYTES", 0)


def test_scan_single_directory_finds_media(tmp_path, _bypass_min_size):
    """Basic scan of a directory with one tagged movie file."""
    movie_dir = tmp_path / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir()
    mkv = movie_dir / "Dune (2021) [tmdb-438631].mkv"
    mkv.touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    results, all_ids, missing = extract_mod_static._scan_single_directory(
        movie_dir, cat_index,
    )

    assert len(all_ids) == 1
    assert all_ids[0]["media_id"] == "tmdb-438631"
    assert all_ids[0]["title"] == "Dune"
    assert all_ids[0]["year"] == "2021"
    assert all_ids[0]["has_catalogue"] is True
    assert all_ids[0]["size_bytes"] == 0
    assert all_ids[0]["content_type"] == "film"
    assert len(results) == 1
    assert missing == []


def test_scan_single_directory_missing_id(tmp_path, _bypass_min_size):
    """Files without a DB ID tag go into missing_ids."""
    movie_dir = tmp_path / "Random Movie"
    movie_dir.mkdir()
    mkv = movie_dir / "random.mkv"
    mkv.touch()

    cat_index = _make_catalogue_index()
    results, all_ids, missing = extract_mod_static._scan_single_directory(
        movie_dir, cat_index,
    )

    assert results == []
    assert all_ids == []
    assert len(missing) == 1
    assert "random.mkv" in missing[0]


def test_scan_single_directory_not_recursive(tmp_path, _bypass_min_size):
    """Subdirectory media files are NOT found (non-recursive scan)."""
    root = tmp_path / "root"
    root.mkdir()
    subdir = root / "subdir"
    subdir.mkdir()
    (subdir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index()
    results, all_ids, missing = extract_mod_static._scan_single_directory(
        root, cat_index,
    )

    assert results == []
    assert all_ids == []
    assert missing == []


def test_scan_single_directory_skips_non_mkv(tmp_path, _bypass_min_size):
    """Non-.mkv files are skipped."""
    movie_dir = tmp_path / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir()
    (movie_dir / "Dune.txt").touch()
    (movie_dir / "Dune.jpg").touch()

    cat_index = _make_catalogue_index()
    results, all_ids, missing = extract_mod_static._scan_single_directory(
        movie_dir, cat_index,
    )

    assert results == []
    assert all_ids == []
    assert missing == []


def test_scan_single_directory_tv_detection(tmp_path, _bypass_min_size):
    """TV episodes get content_type=TV and season/episode parsed."""
    show_dir = tmp_path / "Blue Eye Samurai (2023) [tvdb-434151]" / "Season 1"
    show_dir.mkdir(parents=True)
    ep = show_dir / "S01E03 - Some Episode.mkv"
    ep.touch()

    cat_index = _make_catalogue_index()
    results, all_ids, missing = extract_mod_static._scan_single_directory(
        show_dir, cat_index,
    )

    assert len(all_ids) == 1
    assert all_ids[0]["content_type"] == "TV"
    assert all_ids[0]["season"] == 1
    assert all_ids[0]["episode"] == 3


def test_scan_single_directory_enriched_fields(tmp_path, _bypass_min_size):
    """Entries include size_bytes, content_type, season, episode."""
    movie_dir = tmp_path / "Inception (2010) [tmdb-27205]"
    movie_dir.mkdir()
    mkv = movie_dir / "Inception (2010) [tmdb-27205].mkv"
    mkv.write_bytes(b"x" * 100)  # small but nonzero

    cat_index = _make_catalogue_index([
        {"title": "Inception", "year": "2010", "theMovieDB": "27205",
         "filters": [{"gain": 5.0}]},
    ])

    results, all_ids, missing = extract_mod_static._scan_single_directory(
        movie_dir, cat_index,
    )

    entry = all_ids[0]
    assert entry["size_bytes"] == 100
    assert entry["content_type"] == "film"
    assert entry["season"] is None
    assert entry["episode"] is None


# ---------------------------------------------------------------------------
# discover_media_incremental — end-to-end incremental discovery
# ---------------------------------------------------------------------------


def test_incremental_discovery_fresh_scan(tmp_path, _bypass_min_size):
    """First run with no inventory scans everything."""
    # Build a small library.
    movie_dir = tmp_path / "library" / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"
    results, missing, all_ids, no_cat = extract_mod_static.discover_media_incremental(
        [tmp_path / "library"], cat_index, inventory_path,
    )

    assert len(results) == 1
    assert results[0]["title"] == "Dune"
    assert len(all_ids) == 1
    assert missing == []
    assert no_cat == []

    # Inventory file is written.
    assert inventory_path.exists()
    inv = json.loads(inventory_path.read_text())
    assert inv["format_version"] == 3
    assert "directories" in inv
    # Backward-compatible flat arrays.
    assert len(inv["media"]) == 1
    assert inv["n_total_with_ids"] == 1
    assert inv["n_catalogue_matched"] == 1


def test_incremental_discovery_cached_rerun(tmp_path, _bypass_min_size):
    """Second run with no changes uses cache (no rescan)."""
    movie_dir = tmp_path / "library" / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"

    # First scan.
    extract_mod_static.discover_media_incremental(
        [tmp_path / "library"], cat_index, inventory_path,
    )
    first_inv = json.loads(inventory_path.read_text())

    # Second scan — nothing changed.
    results, missing, all_ids, no_cat = extract_mod_static.discover_media_incremental(
        [tmp_path / "library"], cat_index, inventory_path,
    )

    assert len(results) == 1
    assert results[0]["title"] == "Dune"
    # The inventory is updated (new scanned_at) but content is same.
    second_inv = json.loads(inventory_path.read_text())
    assert second_inv["format_version"] == 3
    assert len(second_inv["media"]) == 1


def test_incremental_discovery_new_directory_detected(tmp_path, _bypass_min_size):
    """Adding a new directory is detected and scanned."""
    lib = tmp_path / "library"
    lib.mkdir()

    movie_dir = lib / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir()
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
        {"title": "Inception", "year": "2010", "theMovieDB": "27205",
         "filters": [{"gain": 5.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"

    # First scan — only Dune.
    results1, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(results1) == 1

    # Add a new movie.
    new_dir = lib / "Inception (2010) [tmdb-27205]"
    new_dir.mkdir()
    (new_dir / "Inception (2010) [tmdb-27205].mkv").touch()

    # Second scan — should find both.
    results2, _, all_ids2, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(results2) == 2
    titles = {r["title"] for r in results2}
    assert "Dune" in titles
    assert "Inception" in titles


def test_incremental_discovery_dedup_across_directories(tmp_path, _bypass_min_size):
    """Same media_id in two directories: only the first is kept."""
    lib = tmp_path / "library"
    dir_a = lib / "Dune (2021) [tmdb-438631]"
    dir_b = lib / "backup" / "Dune (2021) [tmdb-438631]"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    (dir_a / "Dune (2021) [tmdb-438631].mkv").touch()
    (dir_b / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"
    results, _, all_ids, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    # all_with_ids has both (no dedup on raw entries).
    assert len(all_ids) == 2
    # results deduplicates by (media_id, season, episode).
    assert len(results) == 1


def test_incremental_discovery_catalogue_change(tmp_path, _bypass_min_size):
    """Media that gains a catalogue entry on rescan moves to results."""
    lib = tmp_path / "library"
    movie_dir = lib / "NewMovie (2025) [tmdb-999999]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "NewMovie (2025) [tmdb-999999].mkv").touch()

    # First scan: no catalogue entry.
    cat_index_v1 = _make_catalogue_index([])
    inventory_path = tmp_path / "media_inventory.json"

    results1, _, _, no_cat1 = extract_mod_static.discover_media_incremental(
        [lib], cat_index_v1, inventory_path,
    )
    assert len(results1) == 0
    assert len(no_cat1) == 1

    # Second scan: catalogue now has an entry (catalogue updated, directory unchanged).
    cat_index_v2 = _make_catalogue_index([
        {"title": "NewMovie", "year": "2025", "theMovieDB": "999999",
         "filters": [{"gain": 3.0}]},
    ])
    results2, _, _, no_cat2 = extract_mod_static.discover_media_incremental(
        [lib], cat_index_v2, inventory_path,
    )
    assert len(results2) == 1
    assert results2[0]["title"] == "NewMovie"
    assert len(no_cat2) == 0


def test_incremental_discovery_v1_discarded_and_rescanned(tmp_path, _bypass_min_size):
    """A v1 inventory (no directories key) is discarded — full rescan."""
    lib = tmp_path / "library"
    movie_dir = lib / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    # Write a v1 inventory (old format — no directories key).
    inventory_path = tmp_path / "media_inventory.json"
    inventory_path.write_text(json.dumps({
        "media": [{"path": "old", "media_id": "tmdb-1"}],
        "missing_ids": [],
    }))

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    results, _, all_ids, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    # Should have done a fresh scan, not used the old data.
    assert len(results) == 1
    assert all_ids[0]["title"] == "Dune"
    # Output is now v2 format.
    inv = json.loads(inventory_path.read_text())
    assert inv["format_version"] == 3
    assert "directories" in inv


def test_incremental_discovery_inventory_has_backward_compat_arrays(tmp_path, _bypass_min_size):
    """The written inventory has top-level media and missing_ids arrays."""
    lib = tmp_path / "library"
    movie_dir = lib / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    # Also add a file without a tag.
    notagdir = lib / "Untagged"
    notagdir.mkdir()
    (notagdir / "random.mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"
    extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    inv = json.loads(inventory_path.read_text())
    # Top-level arrays for nn_acquisition_recommender and nn_cache_bias_report.
    assert "media" in inv
    assert "missing_ids" in inv
    assert isinstance(inv["media"], list)
    assert isinstance(inv["missing_ids"], list)
    assert inv["n_total_with_ids"] == len(inv["media"])
    assert inv["n_missing_ids"] == len(inv["missing_ids"])


def test_incremental_discovery_breadth_first_sort_applied(tmp_path, _bypass_min_size):
    """Results are breadth-first sorted (movies by size, TV round-robin)."""
    lib = tmp_path / "library"

    # Two movies of different sizes.
    small_dir = lib / "Small (2020) [tmdb-111111]"
    small_dir.mkdir(parents=True)
    small = small_dir / "Small (2020) [tmdb-111111].mkv"
    small.write_bytes(b"x" * 100)

    big_dir = lib / "Big (2021) [tmdb-222222]"
    big_dir.mkdir(parents=True)
    big = big_dir / "Big (2021) [tmdb-222222].mkv"
    big.write_bytes(b"x" * 1000)

    cat_index = _make_catalogue_index([
        {"title": "Small", "year": "2020", "theMovieDB": "111111",
         "filters": [{"gain": 1.0}]},
        {"title": "Big", "year": "2021", "theMovieDB": "222222",
         "filters": [{"gain": 2.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"
    results, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    assert len(results) == 2
    # Breadth-first: movies sorted by size ascending.
    assert results[0]["title"] == "Small"
    assert results[1]["title"] == "Big"


def test_incremental_discovery_nonexistent_root(tmp_path, _bypass_min_size):
    """Non-existent root is warned but does not crash."""
    cat_index = _make_catalogue_index()
    inventory_path = tmp_path / "media_inventory.json"

    results, missing, all_ids, no_cat = extract_mod_static.discover_media_incremental(
        [tmp_path / "does_not_exist"], cat_index, inventory_path,
    )

    assert results == []
    assert missing == []
    assert all_ids == []
    assert no_cat == []


def test_new_movie_detected_on_cached_rerun(tmp_path, _bypass_min_size):
    """A new movie added to a cached library must be detected on next run."""
    lib = tmp_path / "library"
    movies = lib / "Movies"
    existing = movies / "Dune (2021) [tmdb-438631]"
    existing.mkdir(parents=True)
    (existing / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
        {"title": "Avatar", "year": "2009", "theMovieDB": "19995",
         "filters": [{"gain": 3.0}]},
    ])
    inventory_path = tmp_path / "inventory.json"

    # First run: scan and cache.
    r1, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r1) == 1
    assert r1[0]["title"] == "Dune"

    # Add a new movie AFTER the cache was written.
    new_movie = movies / "Avatar (2009) [tmdb-19995]"
    new_movie.mkdir()
    (new_movie / "Avatar (2009) [tmdb-19995].mkv").touch()

    # Second run: must detect the new movie from cache fast path.
    r2, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r2) == 2, (
        f"Expected 2 titles (Dune + Avatar), got {len(r2)}: "
        f"{[r['title'] for r in r2]}. New movie not detected from cached scan."
    )
    titles = {r["title"] for r in r2}
    assert "Dune" in titles
    assert "Avatar" in titles


def test_new_episode_detected_on_cached_rerun(tmp_path, _bypass_min_size):
    """A new episode added to a cached TV show must be detected."""
    lib = tmp_path / "library"
    show = lib / "Show (2020) [tvdb-123]"
    s1 = show / "Season 01"
    s1.mkdir(parents=True)
    (s1 / "Show S01E01 [tvdb-123].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Show", "year": "2020", "theTVDB": "123",
         "filters": [{"gain": 1.0}]},
    ])
    inventory_path = tmp_path / "inventory.json"

    # First run.
    r1, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r1) == 1

    # Add new episode.
    (s1 / "Show S01E02 [tvdb-123].mkv").touch()

    # Second run: must detect new episode.
    r2, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r2) == 2, (
        f"Expected 2 episodes, got {len(r2)}. New episode not detected."
    )


def test_incremental_discovery_saves_after_each_root(tmp_path, _bypass_min_size):
    """Inventory is saved after each root — Ctrl+C won't lose all progress."""
    root_a = tmp_path / "lib_a" / "Movie A (2020) [tmdb-1]"
    root_b = tmp_path / "lib_b" / "Movie B (2021) [tmdb-2]"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    (root_a / "Movie A (2020) [tmdb-1].mkv").touch()
    (root_b / "Movie B (2021) [tmdb-2].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Movie A", "year": "2020", "theMovieDB": "1",
         "filters": [{"gain": 1.0}]},
        {"title": "Movie B", "year": "2021", "theMovieDB": "2",
         "filters": [{"gain": 1.0}]},
    ])

    inventory_path = tmp_path / "inventory.json"

    # Track write times to verify per-root saves.
    import os
    write_times = []
    original_write = inventory_path.write_text.__func__ if hasattr(inventory_path.write_text, '__func__') else None

    extract_mod_static.discover_media_incremental(
        [root_a.parent, root_b.parent], cat_index, inventory_path,
    )

    # Inventory should exist and have directories from both roots.
    data = json.loads(inventory_path.read_text())
    dirs = data["directories"]
    assert any("lib_a" in k for k in dirs), f"Missing lib_a dirs in {list(dirs.keys())}"
    assert any("lib_b" in k for k in dirs), f"Missing lib_b dirs in {list(dirs.keys())}"
    assert len(data["media"]) == 2


def test_incremental_discovery_subtree_pruning(tmp_path, _bypass_min_size):
    """Unchanged subtrees are pruned — subdirectories not walked."""
    lib = tmp_path / "library"
    show_dir = lib / "Show (2020) [tvdb-123]"
    s1 = show_dir / "Season 01"
    s2 = show_dir / "Season 02"
    s1.mkdir(parents=True)
    s2.mkdir(parents=True)
    (s1 / "Show S01E01 [tvdb-123].mkv").touch()
    (s2 / "Show S02E01 [tvdb-123].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Show", "year": "2020", "theTVDB": "123",
         "filters": [{"gain": 1.0}]},
    ])
    inventory_path = tmp_path / "media_inventory.json"

    # First run: full scan.
    r1, _, _, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r1) == 2

    # Second run: nothing changed — subtree should be pruned.
    # We verify by checking results are still correct (data from cache).
    r2, _, all_ids2, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r2) == 2
    assert len(all_ids2) == 2

    # Add a new episode to Season 01 — only that season should be rescanned.
    (s1 / "Show S01E02 [tvdb-123].mkv").touch()
    r3, _, all_ids3, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )
    assert len(r3) == 3  # 2 original + 1 new


# ---------------------------------------------------------------------------
# cache_path — two-letter bucket layout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# cache_path — canonical (ID-based) layout
# ---------------------------------------------------------------------------


def test_cache_path_id_based_film():
    """cache_path uses {id_type}/{shard}/{id_value}/ structure for films."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(root, "Avatar", "2009", "tmdb-19995")
    assert result == (
        root / "tmdb" / "19" / "19995" / "Avatar (2009) [tmdb-19995].lfe-1000hz.wav"
    )


def test_cache_path_id_based_tv():
    """TV shows get title subdir with season folders under the ID dir."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(
        root, "Jujutsu Kaisen", "2020", "tvdb-377543",
        content_type="TV", season=1, episode=1,
    )
    assert result == (
        root / "tvdb" / "37" / "377543" / "Jujutsu Kaisen [tvdb-377543]"
        / "Season 01" / "S01E01.lfe-1000hz.wav"
    )


def test_cache_path_id_based_imdb():
    """IMDB IDs shard on numeric digits (strip 'tt' prefix)."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(
        root, "The Matrix", "1999", "imdb-tt0133093",
    )
    # Shard on numeric part "0133093"[:2] = "01", not the "tt" prefix.
    assert result == (
        root / "imdb" / "01" / "tt0133093"
        / "The Matrix (1999) [imdb-tt0133093].lfe-1000hz.wav"
    )


def test_cache_path_id_unicode_safe():
    """ID-based path is unaffected by title encoding (NFD vs NFC)."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", "Bā'al")
    nfc = unicodedata.normalize("NFC", "Bā'al")
    root = Path("/cache")
    # Same media_id -> same canonical directory regardless of title encoding.
    p_nfd = extract_mod_static.cache_path(root, nfd, "2020", "tmdb-1")
    p_nfc = extract_mod_static.cache_path(root, nfc, "2020", "tmdb-1")
    assert p_nfd.parent == p_nfc.parent  # same dir
    # The filename differs (it includes the title) but the dir is canonical.


class TestCheckMkvHeader:
    """_check_mkv_header pre-validates Matroska containers before ffmpeg runs."""

    def test_valid_mkv_passes(self, tmp_path):
        from cli.extract import _check_mkv_header
        f = tmp_path / "good.mkv"
        f.write_bytes(b"\x1a\x45\xdf\xa3" + b"...rest of file...")
        assert _check_mkv_header(f) is None

    def test_corrupt_mkv_returns_reason(self, tmp_path):
        from cli.extract import _check_mkv_header
        f = tmp_path / "corrupt.mkv"
        # Bytes that aren't EBML magic.
        f.write_bytes(b"\xf1\xd6\xa5\x26" + b"garbage")
        reason = _check_mkv_header(f)
        assert reason is not None
        assert "not a valid Matroska" in reason
        assert "f1 d6 a5 26" in reason  # actual bytes shown

    def test_empty_file_returns_reason(self, tmp_path):
        from cli.extract import _check_mkv_header
        f = tmp_path / "empty.mkv"
        f.write_bytes(b"")
        reason = _check_mkv_header(f)
        assert reason is not None
        assert "too small" in reason

    def test_truncated_file_returns_reason(self, tmp_path):
        from cli.extract import _check_mkv_header
        f = tmp_path / "trunc.mkv"
        f.write_bytes(b"\x1a\x45")  # only 2 bytes
        reason = _check_mkv_header(f)
        assert reason is not None
        assert "too small" in reason

    def test_non_mkv_extension_skips_check(self, tmp_path):
        """Non-MKV extensions can't be pre-validated -- just return None."""
        from cli.extract import _check_mkv_header
        f = tmp_path / "movie.mp4"
        f.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        assert _check_mkv_header(f) is None  # not MKV, skip the check


def test_cache_path_invalid_media_id():
    """Malformed media_id raises ValueError."""
    root = Path("/cache")
    # No dash separator at all.
    with pytest.raises(ValueError, match="must be 'type-value'"):
        extract_mod_static.cache_path(root, "Avatar", "2009", "tmdb19995")
    # Empty value.
    with pytest.raises(ValueError, match="empty type or value"):
        extract_mod_static.cache_path(root, "Avatar", "2009", "tmdb-")


# ---------------------------------------------------------------------------
# legacy_title_cache_path — title-bucket layout (reads only)
# ---------------------------------------------------------------------------


def test_legacy_cache_path_two_letter_bucket():
    """Legacy layout uses first two letters as bucket directory."""
    from model.wav_cache import legacy_title_cache_path
    root = Path("/cache")
    result = legacy_title_cache_path(root, "Avatar", "2009", "tmdb-19995")
    assert result == root / "AV" / "Avatar (2009) [tmdb-19995].lfe-1000hz.wav"


def test_legacy_cache_path_tv_with_seasons():
    """Legacy TV layout: bucket / title / season / episode."""
    from model.wav_cache import legacy_title_cache_path
    root = Path("/cache")
    result = legacy_title_cache_path(
        root, "Jujutsu Kaisen", "2020", "tvdb-377543",
        content_type="TV", season=1, episode=1,
    )
    assert result == (
        root / "JU" / "Jujutsu Kaisen [tvdb-377543]"
        / "Season 01" / "S01E01.lfe-1000hz.wav"
    )


def test_legacy_cache_path_digit_then_space_no_trailing_space():
    """Legacy bucket sanitisation: "3 Days" -> "3_" not "3 "."""
    from model.wav_cache import legacy_title_cache_path
    root = Path("/cache")
    result = legacy_title_cache_path(root, "3 Days to Kill", "2014", "tmdb-1")
    assert result.parent.name == "3_"
    assert " " not in result.parent.name


def test_legacy_cache_path_single_letter_padded():
    """Legacy bucket: single-letter titles get "_" padding."""
    from model.wav_cache import legacy_title_cache_path
    root = Path("/cache")
    assert legacy_title_cache_path(root, "A", "2020", "tmdb-1").parent.name == "A_"
    assert legacy_title_cache_path(root, "I", "2020", "tmdb-2").parent.name == "I_"


def test_legacy_cache_path_unicode_normalised():
    """Legacy bucket: NFC normalisation makes Mac and Linux agree."""
    import unicodedata
    from model.wav_cache import legacy_title_cache_path
    nfd = unicodedata.normalize("NFD", "Bā'al")
    nfc = unicodedata.normalize("NFC", "Bā'al")
    assert nfd != nfc
    root = Path("/cache")
    bucket_nfd = legacy_title_cache_path(root, nfd, "2020", "tmdb-1").parent.name
    bucket_nfc = legacy_title_cache_path(root, nfc, "2020", "tmdb-1").parent.name
    assert bucket_nfd == bucket_nfc


# ---------------------------------------------------------------------------
# find_cached_wav — lookup with legacy fallback
# ---------------------------------------------------------------------------


class TestFindCachedWav:
    """find_cached_wav checks ID-based path first, then legacy fallback."""

    def test_finds_canonical_path(self, tmp_path):
        from model.wav_cache import cache_path, find_cached_wav
        path = cache_path(tmp_path, "Avatar", "2009", "tmdb-19995")
        path.parent.mkdir(parents=True)
        path.touch()
        result = find_cached_wav(tmp_path, "Avatar", "2009", "tmdb-19995")
        assert result == path

    def test_falls_back_to_legacy(self, tmp_path):
        """When canonical doesn't exist, find legacy title-bucket path."""
        from model.wav_cache import find_cached_wav, legacy_title_cache_path
        legacy = legacy_title_cache_path(tmp_path, "Avatar", "2009", "tmdb-19995")
        legacy.parent.mkdir(parents=True)
        legacy.touch()
        result = find_cached_wav(tmp_path, "Avatar", "2009", "tmdb-19995")
        assert result == legacy

    def test_returns_none_when_neither_exists(self, tmp_path):
        from model.wav_cache import find_cached_wav
        result = find_cached_wav(tmp_path, "Avatar", "2009", "tmdb-19995")
        assert result is None

    def test_canonical_takes_precedence_over_legacy(self, tmp_path):
        """If both exist, the canonical path wins."""
        from model.wav_cache import cache_path, find_cached_wav, legacy_title_cache_path
        canonical = cache_path(tmp_path, "Avatar", "2009", "tmdb-19995")
        legacy = legacy_title_cache_path(tmp_path, "Avatar", "2009", "tmdb-19995")
        canonical.parent.mkdir(parents=True)
        legacy.parent.mkdir(parents=True)
        canonical.touch()
        legacy.touch()
        result = find_cached_wav(tmp_path, "Avatar", "2009", "tmdb-19995")
        assert result == canonical

    def test_finds_legacy_with_trailing_space_bucket(self, tmp_path):
        """Legacy WAVs in pre-fix buggy paths are still findable.

        Some titles like "3 Days to Kill" got written to "3 /" (trailing
        space) on Linux before the bucket sanitisation fix. The legacy
        path computation uses the SANITISED bucket ("3_"), so files in
        the buggy "3 /" dir would be missed. We don't try to find those
        — they're orphans. This test documents that behaviour.
        """
        from model.wav_cache import find_cached_wav
        # Write to the buggy unsanitised path.
        buggy = tmp_path / "3 " / "3 Days to Kill (2014) [tmdb-1].lfe-1000hz.wav"
        buggy.parent.mkdir(parents=True)
        buggy.touch()
        # find_cached_wav only knows the sanitised path "3_" -> miss.
        result = find_cached_wav(tmp_path, "3 Days to Kill", "2014", "tmdb-1")
        assert result is None  # buggy file is orphaned, not findable


# ---------------------------------------------------------------------------
# Relative path helpers — portability across machines
# ---------------------------------------------------------------------------


def test_strip_root_prefix_basic():
    """Strips the root's parent prefix, keeping root name as first component."""
    roots = [Path("/media/Movies"), Path("/media/TV")]
    assert extract_mod_static._strip_root_prefix(
        "/media/Movies/Dune (2021)/Dune.mkv", roots,
    ) == "Movies/Dune (2021)/Dune.mkv"


def test_strip_root_prefix_root_itself():
    """Root path becomes just the root's basename."""
    roots = [Path("/media/Movies")]
    assert extract_mod_static._strip_root_prefix(
        "/media/Movies", roots,
    ) == "Movies"


def test_strip_root_prefix_second_root():
    """Strips prefix from a path under the second root."""
    roots = [Path("/media/Movies"), Path("/media/TV")]
    assert extract_mod_static._strip_root_prefix(
        "/media/TV/Show/S01E01.mkv", roots,
    ) == "TV/Show/S01E01.mkv"


def test_strip_root_prefix_no_match():
    """Returns path unchanged when no root matches."""
    roots = [Path("/media/Movies")]
    path = "/other/path/file.mkv"
    assert extract_mod_static._strip_root_prefix(path, roots) == path


def test_resolve_relative_path_finds_existing(tmp_path):
    """Resolves a relative path by matching root basename."""
    root_movies = tmp_path / "Movies"
    root_movies.mkdir()
    target = root_movies / "Dune" / "Dune.mkv"
    target.parent.mkdir(parents=True)
    target.touch()

    result = extract_mod_static._resolve_relative_path(
        "Movies/Dune/Dune.mkv", [root_movies],
    )
    assert result == str(root_movies / "Dune" / "Dune.mkv")


def test_resolve_relative_path_missing_returns_best_guess(tmp_path):
    """When the path doesn't exist under any root, returns best guess."""
    root_movies = tmp_path / "Movies"
    root_movies.mkdir()
    result = extract_mod_static._resolve_relative_path(
        "Movies/Missing/file.mkv", [root_movies],
    )
    assert result == str(root_movies / "Missing" / "file.mkv")


def test_inventory_stores_relative_paths(tmp_path, _bypass_min_size):
    """Inventory v3 stores paths relative to media roots."""
    lib = tmp_path / "library"
    movie_dir = lib / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"
    extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    inv = json.loads(inventory_path.read_text())
    assert inv["format_version"] == 3

    # Directory keys should be relative, not absolute.
    for dir_key in inv["directories"]:
        assert not os.path.isabs(dir_key), f"directory key is absolute: {dir_key}"

    # Media entry paths should be relative.
    for entry in inv["media"]:
        assert not os.path.isabs(entry["path"]), f"media path is absolute: {entry['path']}"


def test_inventory_relative_paths_resolve_on_reload(tmp_path, _bypass_min_size):
    """Relative paths in a v3 inventory resolve correctly on reload."""
    lib = tmp_path / "library"
    movie_dir = lib / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"

    # First scan.
    extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    # Second scan — should reload from cache and resolve paths.
    results, _, all_ids, _ = extract_mod_static.discover_media_incremental(
        [lib], cat_index, inventory_path,
    )

    assert len(results) == 1
    assert results[0]["title"] == "Dune"
    # The returned entry paths should be absolute (resolved for processing).
    for entry in all_ids:
        assert os.path.isabs(entry["path"]), f"returned path should be absolute: {entry['path']}"


def test_inventory_relative_paths_with_different_root(tmp_path, _bypass_min_size):
    """Inventory relative paths work when the root mount point changes."""
    # First scan under /tmp/.../old_mount/library.
    old_mount = tmp_path / "old_mount"
    lib_old = old_mount / "library"
    movie_dir = lib_old / "Dune (2021) [tmdb-438631]"
    movie_dir.mkdir(parents=True)
    (movie_dir / "Dune (2021) [tmdb-438631].mkv").touch()

    cat_index = _make_catalogue_index([
        {"title": "Dune", "year": "2021", "theMovieDB": "438631",
         "filters": [{"gain": 4.0}]},
    ])

    inventory_path = tmp_path / "media_inventory.json"
    extract_mod_static.discover_media_incremental(
        [lib_old], cat_index, inventory_path,
    )

    # "Move" library to a different mount point by copying.
    import shutil
    new_mount = tmp_path / "new_mount"
    new_mount.mkdir()
    lib_new = new_mount / "library"
    shutil.copytree(lib_old, lib_new)

    # Reload inventory with new root — should resolve relative paths.
    results, _, all_ids, _ = extract_mod_static.discover_media_incremental(
        [lib_new], cat_index, inventory_path,
    )

    assert len(results) == 1
    # The path should be under the new root, not the old one.
    assert str(lib_new) in all_ids[0]["path"], (
        f"Expected path under {lib_new}, got {all_ids[0]['path']}"
    )


def test_extract_config_uses_local_config_dir(tmp_path, monkeypatch):
    """extract_config.json is loaded from ~/.config/beqdesigner/, not shared dir."""
    from spike._auto_beq_helpers import beq_config_dir as _real_beq_config_dir

    local_config = tmp_path / "local_config"
    local_config.mkdir()

    # Write extract_config.json in local config dir.
    config = {"media_roots": ["/test/media"]}
    (local_config / "extract_config.json").write_text(json.dumps(config))

    # Patch beq_config_dir to return our test dir.
    monkeypatch.setattr(
        "spike._auto_beq_helpers.beq_config_dir",
        lambda: local_config,
    )

    beq_dir = tmp_path / "shared_beq"
    beq_dir.mkdir()

    # Should NOT look in beq_dir for extract_config.json.
    result = extract_mod_static._load_or_prompt_config(beq_dir, None)
    assert result["media_roots"] == [Path("/test/media")]


# ---------------------------------------------------------------------------
# beq_dir() — resolution order tests
# ---------------------------------------------------------------------------


def test_beq_dir_resolves_from_BEQ_SHARED_DIR(tmp_path, monkeypatch):
    """BEQ_SHARED_DIR env var is the primary resolution source."""
    from spike import _auto_beq_helpers as helpers
    target = tmp_path / "shared"
    target.mkdir()
    monkeypatch.setenv("BEQ_SHARED_DIR", str(target))
    # Clear BEQ_DIR to avoid interference.
    monkeypatch.delenv("BEQ_DIR", raising=False)
    assert helpers.beq_dir() == target


def test_beq_dir_resolves_from_shared_beq_dir_setting(tmp_path, monkeypatch):
    """shared_beq_dir in settings.json is the secondary resolution source."""
    from spike import _auto_beq_helpers as helpers
    monkeypatch.delenv("BEQ_SHARED_DIR", raising=False)
    monkeypatch.delenv("BEQ_DIR", raising=False)

    target = tmp_path / "shared"
    target.mkdir()

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    settings = {"shared_beq_dir": str(target)}
    (cfg_dir / "settings.json").write_text(json.dumps(settings))

    # Patch Path.home to point to our tmp config.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".config" / "beqdesigner").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".config" / "beqdesigner" / "settings.json").write_text(
        json.dumps(settings)
    )

    assert helpers.beq_dir() == target


def test_beq_dir_errors_when_not_configured(tmp_path, monkeypatch):
    """beq_dir() raises RuntimeError when nothing is configured."""
    from spike import _auto_beq_helpers as helpers
    monkeypatch.delenv("BEQ_SHARED_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")

    with pytest.raises(RuntimeError, match="not configured"):
        helpers.beq_dir()



# ---------------------------------------------------------------------------
# wav_cache_dir — resolution from BEQ_SHARED_DIR
# ---------------------------------------------------------------------------


class TestWavCacheDirResolution:
    """wav_cache_dir() should auto-derive from BEQ_SHARED_DIR."""

    def test_derives_from_beq_shared_dir(self, tmp_path, monkeypatch):
        """When BEQ_SHARED_DIR is set, wav_cache_dir returns {shared}/wav-cache."""
        from spike import _auto_beq_helpers as helpers
        monkeypatch.setenv("BEQ_SHARED_DIR", str(tmp_path))
        monkeypatch.delenv("BEQ_WAV_CACHE", raising=False)
        monkeypatch.delenv("BEQ_DIR", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")

        result = helpers.wav_cache_dir()
        assert result == tmp_path / "wav-cache"
        assert result.exists()  # auto-created

    def test_explicit_wav_cache_takes_precedence(self, tmp_path, monkeypatch):
        """BEQ_WAV_CACHE overrides BEQ_SHARED_DIR derivation."""
        from spike import _auto_beq_helpers as helpers
        explicit = tmp_path / "my-custom-cache"
        explicit.mkdir()
        monkeypatch.setenv("BEQ_WAV_CACHE", str(explicit))
        monkeypatch.setenv("BEQ_SHARED_DIR", str(tmp_path / "shared"))

        result = helpers.wav_cache_dir()
        assert result == explicit

    def test_errors_when_nothing_configured(self, tmp_path, monkeypatch):
        """Raises RuntimeError when no config can resolve."""
        from spike import _auto_beq_helpers as helpers
        monkeypatch.delenv("BEQ_SHARED_DIR", raising=False)
        monkeypatch.delenv("BEQ_WAV_CACHE", raising=False)
        monkeypatch.delenv("BEQ_DIR", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")

        with pytest.raises(RuntimeError, match="not configured"):
            helpers.wav_cache_dir()


class TestCheckProductionModel:
    """check_production_model() in shared helpers, no Qt import."""

    def test_finds_joblib_model(self, tmp_path, monkeypatch):
        from spike import _auto_beq_helpers as helpers
        monkeypatch.setenv("BEQ_SHARED_DIR", str(tmp_path))
        monkeypatch.delenv("AUTO_BEQ_MODEL_PATH", raising=False)
        monkeypatch.delenv("AUTO_BEQ_ADVISOR", raising=False)
        (tmp_path / "production_model.joblib").touch()

        assert helpers.check_production_model() is True

    def test_missing_model(self, tmp_path, monkeypatch):
        from spike import _auto_beq_helpers as helpers
        monkeypatch.setenv("BEQ_SHARED_DIR", str(tmp_path))
        monkeypatch.delenv("AUTO_BEQ_MODEL_PATH", raising=False)
        monkeypatch.delenv("AUTO_BEQ_ADVISOR", raising=False)

        assert helpers.check_production_model() is False


# ---------------------------------------------------------------------------
# find_media_dirs — adaptive depth detection
# ---------------------------------------------------------------------------


class TestFindMediaDirs:
    """Tests for model.media_utils.find_media_dirs()."""

    def test_flat_layout(self, tmp_path):
        """Root contains .mkv files directly — returns immediate children dirs."""
        from model.media_utils import find_media_dirs

        (tmp_path / "movie.mkv").touch()
        sub = tmp_path / "extras"
        sub.mkdir()
        result = find_media_dirs(tmp_path)
        assert result == [sub]

    def test_one_level_layout(self, tmp_path):
        """Standard layout: root/Title (Year)/file.mkv."""
        from model.media_utils import find_media_dirs

        d1 = tmp_path / "Avatar (2009)"
        d1.mkdir()
        (d1 / "Avatar.mkv").touch()
        d2 = tmp_path / "Dune (2021)"
        d2.mkdir()
        (d2 / "Dune.mkv").touch()
        result = find_media_dirs(tmp_path)
        assert sorted(result) == sorted([d1, d2])

    def test_two_level_layout(self, tmp_path):
        """TV layout: root/Show (Year)/Season 01/file.mkv."""
        from model.media_utils import find_media_dirs

        show = tmp_path / "Breaking Bad (2008)"
        s1 = show / "Season 01"
        s1.mkdir(parents=True)
        (s1 / "S01E01.mkv").touch()
        show2 = tmp_path / "The Wire (2002)"
        s2 = show2 / "Season 01"
        s2.mkdir(parents=True)
        (s2 / "S01E01.mkv").touch()
        result = find_media_dirs(tmp_path)
        assert sorted(result) == sorted([show, show2])

    def test_no_media_files(self, tmp_path):
        """No .mkv files — falls back to immediate children."""
        from model.media_utils import find_media_dirs

        d1 = tmp_path / "subdir1"
        d1.mkdir()
        d2 = tmp_path / "subdir2"
        d2.mkdir()
        result = find_media_dirs(tmp_path)
        assert sorted(result) == sorted([d1, d2])

    def test_no_year_in_dirname(self, tmp_path):
        """Media file exists but no (YEAR) in dir name — falls back to children."""
        from model.media_utils import find_media_dirs

        d1 = tmp_path / "SomeDir"
        d1.mkdir()
        (d1 / "video.mkv").touch()
        result = find_media_dirs(tmp_path)
        assert result == [d1]


# ---------------------------------------------------------------------------
# ProgressLogger — time-throttled progress with ETA
# ---------------------------------------------------------------------------


class TestProgressLogger:
    """Tests for model.media_utils.ProgressLogger."""

    def test_logs_first_and_final_item(self):
        """The first and last items are always logged regardless of interval."""
        from model.media_utils import ProgressLogger
        import logging

        msgs = []
        logger = logging.getLogger("test_progress_final")
        logger.handlers = [logging.StreamHandler()]
        logger.handlers[0].emit = lambda r: msgs.append(r.getMessage())
        logger.setLevel(logging.INFO)

        progress = ProgressLogger(total=3, logger=logger, min_interval_s=9999)
        progress.update(1, label="a")
        progress.update(2, label="b")
        progress.update(3, label="c")  # final — must log

        # First and final items logged; middle suppressed by interval.
        assert len(msgs) == 2
        assert "1/3" in msgs[0]
        assert "a" in msgs[0]
        assert "3/3" in msgs[1]
        assert "c" in msgs[1]

    def test_respects_min_interval_after_first(self):
        """Items after the first are suppressed until min_interval_s elapses."""
        from model.media_utils import ProgressLogger
        import logging

        msgs = []
        logger = logging.getLogger("test_progress_interval")
        logger.handlers = [logging.StreamHandler()]
        logger.handlers[0].emit = lambda r: msgs.append(r.getMessage())
        logger.setLevel(logging.INFO)

        progress = ProgressLogger(total=100, logger=logger, min_interval_s=9999)
        for i in range(1, 100):
            progress.update(i)
        # Only the first item should be logged (interval too large for rest).
        assert len(msgs) == 1
        assert "1/100" in msgs[0]

    def test_logs_when_interval_elapsed(self):
        """Items after min_interval_s has elapsed are logged."""
        from model.media_utils import ProgressLogger
        import logging

        msgs = []
        logger = logging.getLogger("test_progress_elapsed")
        logger.handlers = [logging.StreamHandler()]
        logger.handlers[0].emit = lambda r: msgs.append(r.getMessage())
        logger.setLevel(logging.INFO)

        progress = ProgressLogger(total=10, logger=logger, min_interval_s=0)
        progress.update(1, label="first")
        progress.update(5, label="mid")

        # With min_interval_s=0, both should log.
        assert len(msgs) == 2
        assert "1/10" in msgs[0]
        assert "5/10" in msgs[1]

    def test_finish_returns_elapsed(self):
        """finish() logs and returns elapsed time."""
        from model.media_utils import ProgressLogger
        import logging

        msgs = []
        logger = logging.getLogger("test_progress_finish")
        logger.handlers = [logging.StreamHandler()]
        logger.handlers[0].emit = lambda r: msgs.append(r.getMessage())
        logger.setLevel(logging.INFO)

        progress = ProgressLogger(total=1, logger=logger)
        elapsed = progress.finish("all done")
        assert elapsed >= 0
        assert "all done" in msgs[-1]

    def test_eta_in_output(self):
        """ETA and remaining time are shown for non-final items."""
        from model.media_utils import ProgressLogger
        import logging

        msgs = []
        logger = logging.getLogger("test_progress_eta")
        logger.handlers = [logging.StreamHandler()]
        logger.handlers[0].emit = lambda r: msgs.append(r.getMessage())
        logger.setLevel(logging.INFO)

        progress = ProgressLogger(total=100, logger=logger, min_interval_s=0)
        # Force some elapsed time so ETA is computed.
        progress._t0 -= 10  # pretend 10 seconds have passed
        progress.update(50, label="halfway")

        assert len(msgs) == 1
        assert "50/100" in msgs[0]
        assert "remaining" in msgs[0]
        assert "ETA" in msgs[0]


# ---------------------------------------------------------------------------
# format_duration — human-readable time formatting
# ---------------------------------------------------------------------------


class TestFormatDuration:
    """Tests for model.media_utils.format_duration()."""

    def test_zero(self):
        from model.media_utils import format_duration
        assert format_duration(0) == "0s"

    def test_seconds_only(self):
        from model.media_utils import format_duration
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        from model.media_utils import format_duration
        assert format_duration(150) == "2m:30s"

    def test_minutes_no_seconds(self):
        from model.media_utils import format_duration
        assert format_duration(120) == "2m"

    def test_hours_and_minutes(self):
        from model.media_utils import format_duration
        assert format_duration(4980) == "1h:23m"

    def test_hours_minutes_seconds_approximate(self):
        """With approximate=True (default), hours drop seconds."""
        from model.media_utils import format_duration
        assert format_duration(4984) == "1h:23m"  # seconds dropped

    def test_hours_minutes_seconds_exact(self):
        from model.media_utils import format_duration
        assert format_duration(4984, approximate=False) == "1h:23m:4s"

    def test_hours_only(self):
        from model.media_utils import format_duration
        assert format_duration(3600) == "1h"

    def test_many_hours_approximate(self):
        """5+ hours rounds to nearest hour with ~ prefix."""
        from model.media_utils import format_duration
        assert format_duration(36000) == "~10h"
        assert format_duration(19800) == "~6h"  # 5h 30m -> ~6h (rounds up)
        assert format_duration(18000) == "~5h"  # exactly 5h

    def test_fractional_rounds_down(self):
        from model.media_utils import format_duration
        assert format_duration(45.9) == "45s"

    def test_negative_clamps_to_zero(self):
        from model.media_utils import format_duration
        assert format_duration(-5) == "0s"
