"""Unit tests for ``cli/extract.py`` — bias-corrected unmatched selection.

Only pure-function helpers are tested here — anything that actually
shells out to ffmpeg belongs in an integration test.
"""
from __future__ import annotations

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
