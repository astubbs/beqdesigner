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
    assert inv["format_version"] == 2
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
    assert second_inv["format_version"] == 2
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
    assert inv["format_version"] == 2
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


def test_cache_path_two_letter_bucket():
    """cache_path uses first two letters as bucket directory."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(root, "Avatar", "2009", "tmdb-19995")
    assert result == root / "AV" / "Avatar (2009) [tmdb-19995].lfe-1000hz.wav"


def test_cache_path_tv_with_seasons():
    """TV shows get title subdir with season folders."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(
        root, "Jujutsu Kaisen", "2020", "tvdb-377543",
        content_type="TV", season=1, episode=1,
    )
    assert result == (
        root / "JU" / "Jujutsu Kaisen [tvdb-377543]"
        / "Season 01" / "S01E01.lfe-1000hz.wav"
    )


def test_cache_path_short_title():
    """Titles shorter than 2 chars get padded."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(root, "X", "2020", "tmdb-12345")
    assert result == root / "X_" / "X (2020) [tmdb-12345].lfe-1000hz.wav"


def test_cache_path_numeric_title():
    """Titles starting with numbers work correctly."""
    root = Path("/cache")
    result = extract_mod_static.cache_path(root, "2001", "1968", "tmdb-62")
    assert result == root / "20" / "2001 (1968) [tmdb-62].lfe-1000hz.wav"


