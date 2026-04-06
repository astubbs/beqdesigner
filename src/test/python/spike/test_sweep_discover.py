"""Unit tests for the discovery CLI.

Uses the committed fake library fixture at
``src/test/resources/auto_beq/fake_library/`` — zero-byte ``.mkv``
files in standard media naming subdirectories. Discovery only reads filenames,
not content, so empty placeholders are sufficient.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spike import sweep_discover as sd

_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "resources" / "auto_beq" / "fake_library"
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("candidate,expected_title,expected_year", [
    # Standard media directory names
    ("Dune (2021) [tmdb-438631]", "Dune", 2021),
    ("The Matrix (1999) [imdb-tt0133093]", "The Matrix", 1999),
    ("Inception (2010)", "Inception", 2010),
    ("Blade Runner 2049 (2017)", "Blade Runner 2049", 2017),
    # tvdb tag (TV shows)
    ("Blue Eye Samurai (2023) [tvdb-434151]", "Blue Eye Samurai", 2023),
    ("X-Men '97 (2024) [tvdb-412432]", "X-Men '97", 2024),
])
def test_parse_dir_variants(candidate, expected_title, expected_year, tmp_path):
    p = tmp_path / candidate / "whatever.mkv"
    p.parent.mkdir(parents=True)
    p.touch()
    result = sd.parse_media_filename(p)
    assert result is not None
    assert result.title == expected_title
    assert result.year == expected_year


@pytest.mark.parametrize("candidate,expected_title,expected_year", [
    # Scene-style (dots as separators — parsed from directory or stem)
    ("Edge.of.Tomorrow.2014.UHD.2160p.x265.DTS-HD.MA.7.1", "Edge of Tomorrow", 2014),
    ("Mad.Max.Fury.Road.2015.BluRay.1080p.DTS.x264", "Mad Max Fury Road", 2015),
    ("The.Dark.Knight.2008.REMUX", "The Dark Knight", 2008),
])
def test_parse_scene_style_variants(candidate, expected_title, expected_year, tmp_path):
    p = tmp_path / candidate / "whatever.mkv"
    p.parent.mkdir(parents=True)
    p.touch()
    result = sd.parse_media_filename(p)
    assert result is not None
    assert result.title == expected_title
    assert result.year == expected_year


@pytest.mark.parametrize("candidate", [
    "Some TV Show S01E01",
    "No year here",
    "Movie (99)",        # 2-digit year rejected
    "Movie (20100)",     # 5-digit year rejected
])
def test_parse_rejected_filenames(candidate, tmp_path):
    p = tmp_path / candidate / "whatever.mkv"
    p.parent.mkdir(parents=True)
    p.touch()
    assert sd.parse_media_filename(p) is None


def test_parse_media_filename_falls_back_to_stem(tmp_path):
    p = tmp_path / "Random Folder" / "Inception (2010).mkv"
    p.parent.mkdir(parents=True)
    p.touch()
    result = sd.parse_media_filename(p)
    assert result is not None
    assert result.title == "Inception"
    assert result.year == 2010


def test_parse_media_filename_tv_show_season_structure(tmp_path):
    # TV shows: Show (YEAR) [tvdb-NNN] / Season N / S01E01.mkv
    p = (tmp_path / "Blue Eye Samurai (2023) [tvdb-434151]"
         / "Season 1" / "S01E01 - The Great Fire of 1657.mkv")
    p.parent.mkdir(parents=True)
    p.touch()
    result = sd.parse_media_filename(p)
    assert result is not None
    assert result.title == "Blue Eye Samurai"
    assert result.year == 2023
    assert result.season == 1
    assert result.episode == 1


def test_parse_media_filename_tv_show_no_season_subdir(tmp_path):
    p = (tmp_path / "MINDHUNTER (2017) [tvdb-328708]"
         / "S01E01 - Episode 1.mkv")
    p.parent.mkdir(parents=True)
    p.touch()
    result = sd.parse_media_filename(p)
    assert result is not None
    assert result.title == "MINDHUNTER"
    assert result.year == 2017
    assert result.season == 1
    assert result.episode == 1


def test_parse_media_filename_stem_with_episode_info(tmp_path):
    """Splinter Cell case: no year in dir, year+episode in file stem."""
    p = (tmp_path / "Splinter Cell - Deathwatch [tvdb-386210]"
         / "Season 01"
         / "Splinter Cell - Deathwatch (2025) - S01E01 - Up From The Grave [WEBDL-2160p].mkv")
    p.parent.mkdir(parents=True)
    p.touch()
    result = sd.parse_media_filename(p)
    assert result is not None
    assert result.title == "Splinter Cell - Deathwatch"
    assert result.year == 2025
    assert result.season == 1
    assert result.episode == 1


# ---------------------------------------------------------------------------
# Title normalisation + catalogue matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Dune", "dune"),
    ("The Matrix", "thematrix"),
    ("Spider-Man: No Way Home", "spidermannowayhome"),
    ("Wall·E", "walle"),
    ("  Spaces  ", "spaces"),
])
def test_normalise_title(raw, expected):
    assert sd.normalise_title(raw) == expected


def test_match_catalogue_exact():
    catalogue = [
        {"title": "Dune", "year": "2021", "filters": [{"gain": 4.0}]},
        {"title": "Dune", "year": "1984", "filters": [{"gain": 2.0}]},
        {"title": "Other", "year": "2021", "filters": [{"gain": 1.0}]},
    ]
    m = sd.match_catalogue(catalogue, "Dune", 2021)
    assert m is not None
    assert m["year"] == "2021"
    assert m["filters"][0]["gain"] == 4.0


def test_match_catalogue_prefers_most_filters_on_tie():
    catalogue = [
        {"title": "Dune", "year": "2021", "filters": [{"gain": 1.0}]},
        {"title": "Dune", "year": "2021", "filters": [{"g": 1}, {"g": 2}, {"g": 3}]},
        {"title": "Dune", "year": "2021", "filters": [{"g": 1}, {"g": 2}]},
    ]
    m = sd.match_catalogue(catalogue, "Dune", 2021)
    assert len(m["filters"]) == 3


def test_match_catalogue_skips_empty_filters():
    catalogue = [
        {"title": "Dune", "year": "2021", "filters": []},
        {"title": "Dune", "year": "2021", "filters": [{"gain": 1.0}]},
    ]
    m = sd.match_catalogue(catalogue, "Dune", 2021)
    assert m is not None
    assert m["filters"] == [{"gain": 1.0}]


def test_match_catalogue_case_and_punctuation_insensitive():
    catalogue = [
        {"title": "The Matrix", "year": "1999", "filters": [{"g": 1}]},
    ]
    assert sd.match_catalogue(catalogue, "the matrix", 1999) is not None
    assert sd.match_catalogue(catalogue, "The-Matrix!", 1999) is not None


def test_match_catalogue_case_insensitive_all_caps():
    """Catalogue has 'BLUE EYE SAMURAI', library has 'Blue Eye Samurai'."""
    catalogue = [
        {"title": "BLUE EYE SAMURAI", "year": "2023", "filters": [{"g": 1}]},
    ]
    assert sd.match_catalogue(catalogue, "Blue Eye Samurai", 2023) is not None


def test_match_catalogue_episode_specific():
    """Scavengers Reign style: different profiles for different episode sets."""
    catalogue = [
        {"title": "Scavengers Reign", "year": "2023",
         "season": "1", "episode": "1,7,9,11",
         "filters": [{"g": 1}, {"g": 2}, {"g": 3}, {"g": 4}]},
        {"title": "Scavengers Reign", "year": "2023",
         "season": "1", "episode": "2,3,4,5,6,8,10,12",
         "filters": [{"g": 1}, {"g": 2}, {"g": 3}, {"g": 4}, {"g": 5}, {"g": 6}]},
    ]
    # Episode 7 should match the first profile (episodes 1,7,9,11).
    m = sd.match_catalogue(catalogue, "Scavengers Reign", 2023, season=1, episode=7)
    assert m is not None
    assert len(m["filters"]) == 4
    # Episode 3 should match the second profile (episodes 2,3,...).
    m = sd.match_catalogue(catalogue, "Scavengers Reign", 2023, season=1, episode=3)
    assert m is not None
    assert len(m["filters"]) == 6


def test_match_catalogue_south_park_season_episode_format():
    """South Park style: season field contains 'S27E01' format."""
    catalogue = [
        {"title": "South Park", "year": "1997",
         "season": "27E01", "episode": "",
         "filters": [{"g": 1}, {"g": 2}, {"g": 3}, {"g": 4}]},
        {"title": "South Park", "year": "1997",
         "season": "27E02", "episode": "",
         "filters": [{"g": 1}, {"g": 2}, {"g": 3}]},
    ]
    m = sd.match_catalogue(catalogue, "South Park", 1997, season=27, episode=1)
    assert m is not None
    assert len(m["filters"]) == 4
    m = sd.match_catalogue(catalogue, "South Park", 1997, season=27, episode=2)
    assert m is not None
    assert len(m["filters"]) == 3


def test_match_catalogue_no_match_returns_none():
    catalogue = [{"title": "Dune", "year": "2021", "filters": [{"g": 1}]}]
    assert sd.match_catalogue(catalogue, "Dune", 1984) is None
    assert sd.match_catalogue(catalogue, "Nonexistent", 2021) is None


# ---------------------------------------------------------------------------
# Rating extraction / bucketing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rating,expected", [
    (7.5, 7.5),
    ("8.1", 8.1),
    ("PG-13", None),
    (None, None),
    ("", None),
])
def test_extract_rating(rating, expected):
    assert sd.extract_rating({"rating": rating}) == expected


@pytest.mark.parametrize("rating,expected", [
    (7.9, 7.5),
    (7.0, 7.0),
    (0.3, 0.0),
    (None, -1.0),
])
def test_bucket_rating(rating, expected):
    assert sd.bucket_rating(rating) == expected


# ---------------------------------------------------------------------------
# Discovery against the fake library fixture
# ---------------------------------------------------------------------------

def _mock_catalogue() -> list[dict]:
    """Small catalogue matching the fake library fixture."""
    return [
        {"title": "Dune", "year": "2021", "rating": 7.8,
         "filters": [{"type": "LowShelf", "gain": 4.0}]},
        {"title": "The Matrix", "year": "1999", "rating": 8.2,
         "filters": [{"type": "LowShelf", "gain": 3.0}]},
        {"title": "Inception", "year": "2010", "rating": 8.8,
         "filters": [{"type": "LowShelf", "gain": 5.0}]},
        # TV show — catalogue has ALL-CAPS title, library has mixed case.
        {"title": "BLUE EYE SAMURAI", "year": "2023", "rating": "TV-MA",
         "filters": [{"type": "LowShelf", "gain": 3.0}],
         "content_type": "TV"},
        # Deliberately no entry for "Unmatched Movie (2030)" or "Some TV Show S01E01".
    ]


def test_discover_against_fake_library():
    """Integration test covering multiple directory structures:
    - tmdb-tagged movie: ``Dune (2021) [tmdb-438631]/Dune (2021) [tmdb-438631].mkv``
    - imdb-tagged movie: ``The Matrix (1999) [imdb-tt0133093]/The Matrix (1999).mkv``
    - Bare movie (no tag): ``Inception (2010)/Inception (2010).mkv``
    - TV show with Season subdir: ``Blue Eye Samurai (2023) [tvdb-434151]/Season 1/S01E01.mkv``
    - Unmatched movie (no catalogue entry): ``Unmatched Movie (2030)/...``
    - Unparseable TV episode (no year): ``Some TV Show S01E01/...``
    """
    assert _FIXTURE_ROOT.is_dir(), f"fake library fixture missing: {_FIXTURE_ROOT}"
    catalogue = _mock_catalogue()
    matches, total_scanned = sd.discover_matches(_FIXTURE_ROOT, catalogue)
    assert total_scanned > 0

    titles = {m.title for m in matches}
    assert "Dune" in titles, "[tmdb-] tagged movie should match"
    assert "The Matrix" in titles, "imdb-tagged movie should match"
    assert "Inception" in titles, "bare Title (YEAR) movie should match"
    assert "Blue Eye Samurai" in titles, (
        "TV show with Season subdir and [tvdb-] tag should match"
    )

    # Unmatched content is absent.
    assert "Unmatched Movie" not in titles
    assert "Some TV Show S01E01" not in titles

    # TV show with 2 episodes should produce 2 matches (one per .mkv).
    bes_matches = [m for m in matches if m.title == "Blue Eye Samurai"]
    assert len(bes_matches) == 2, (
        f"expected 2 episode matches for Blue Eye Samurai, got {len(bes_matches)}"
    )

    # Library root recorded on each match.
    for m in matches:
        assert m.library_root == str(_FIXTURE_ROOT)
        assert Path(m.path).exists()


def test_sort_matches_by_rating_then_year():
    films = [
        sd.SweepFilm(path="a", library_root="/", title="A", year=2000,
                     rating=7.0, catalogue_entry={}),
        sd.SweepFilm(path="b", library_root="/", title="B", year=2020,
                     rating=8.5, catalogue_entry={}),
        sd.SweepFilm(path="c", library_root="/", title="C", year=2010,
                     rating=8.5, catalogue_entry={}),
        sd.SweepFilm(path="d", library_root="/", title="D", year=2024,
                     rating=None, catalogue_entry={}),
    ]
    sorted_films = sd.sort_matches(films)
    assert [f.title for f in sorted_films] == ["B", "C", "A", "D"]


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------

def test_write_and_load_config(tmp_path):
    films = [
        sd.SweepFilm(
            path="/films/dune.mkv", library_root="/films",
            title="Dune", year=2021, rating=7.8,
            catalogue_entry={"title": "Dune", "filters": [{"g": 1}]},
        ),
    ]
    out = tmp_path / "sweep.json"
    written = sd.write_config(
        films=films,
        library_roots=[Path("/films")],
        test_limit=5,
        output=out,
    )
    assert written == out
    assert written.exists()

    loaded = sd.load_config(out)
    assert loaded is not None
    assert loaded["schema_version"] == 1
    assert loaded["test_limit"] == 5
    assert loaded["library_roots"] == ["/films"]
    assert len(loaded["films"]) == 1
    assert loaded["films"][0]["title"] == "Dune"
    assert loaded["films"][0]["catalogue_entry"]["filters"] == [{"g": 1}]
    # Timestamp is a valid ISO string.
    assert "T" in loaded["generated_at"]


def test_load_config_missing_returns_none(tmp_path):
    assert sd.load_config(tmp_path / "does_not_exist.json") is None


# ---------------------------------------------------------------------------
# Library-root resolution
# ---------------------------------------------------------------------------

def test_resolve_library_roots_from_cli_flags(monkeypatch):
    monkeypatch.delenv("AUTO_BEQ_LIBRARY_ROOTS", raising=False)
    args = sd._parse_args(["--library", "/a", "--library", "/b"])
    roots = sd._resolve_library_roots(args)
    assert roots == [Path("/a"), Path("/b")]


def test_resolve_library_roots_from_env_var(monkeypatch):
    monkeypatch.setenv("AUTO_BEQ_LIBRARY_ROOTS", "/foo:/bar:/baz")
    args = sd._parse_args([])
    roots = sd._resolve_library_roots(args)
    assert roots == [Path("/foo"), Path("/bar"), Path("/baz")]


def test_resolve_library_roots_expands_user(monkeypatch):
    monkeypatch.setenv("AUTO_BEQ_LIBRARY_ROOTS", "~/media")
    args = sd._parse_args([])
    roots = sd._resolve_library_roots(args)
    assert roots == [Path.home() / "media"]


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def test_dotenv_parser(tmp_path, monkeypatch):
    # Do not clobber existing vars.
    monkeypatch.setenv("ALREADY_SET", "keep")
    monkeypatch.delenv("DOTENV_FRESH", raising=False)
    monkeypatch.delenv("DOTENV_QUOTED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "ALREADY_SET=from_dotenv\n"
        "DOTENV_FRESH=plain_value\n"
        'DOTENV_QUOTED="value with spaces"\n'
        "INVALID_LINE_NO_EQUALS\n"
    )
    parsed = sd.load_dotenv(env_file)
    assert parsed["ALREADY_SET"] == "from_dotenv"
    assert parsed["DOTENV_FRESH"] == "plain_value"
    assert parsed["DOTENV_QUOTED"] == "value with spaces"
    assert "INVALID_LINE_NO_EQUALS" not in parsed
    # setdefault semantics: existing env var is preserved.
    assert os.environ["ALREADY_SET"] == "keep"
    # Fresh var IS set.
    assert os.environ["DOTENV_FRESH"] == "plain_value"
    # Cleanup.
    monkeypatch.delenv("DOTENV_FRESH", raising=False)
    monkeypatch.delenv("DOTENV_QUOTED", raising=False)


def test_dotenv_missing_file_returns_empty(tmp_path):
    assert sd.load_dotenv(tmp_path / "nonexistent.env") == {}


# ---------------------------------------------------------------------------
# End-to-end main() against fake library
# ---------------------------------------------------------------------------

def test_main_against_fake_library(tmp_path, monkeypatch, capsys):
    """Full CLI run — fake library + mocked catalogue fetch → config on disk."""
    # Stub out the network fetch by seeding the cache.
    cache_path = tmp_path / "catalogue_cache.json"
    cache_path.write_text(json.dumps(_mock_catalogue()))
    monkeypatch.setattr(sd, "_catalogue_cache_path", lambda: cache_path)

    config_path = tmp_path / "sweep.json"
    exit_code = sd.main([
        "--library", str(_FIXTURE_ROOT),
        "--yes",
        "--output", str(config_path),
        "--test-limit", "3",
    ])
    assert exit_code == 0
    assert config_path.exists()

    loaded = json.loads(config_path.read_text())
    assert loaded["test_limit"] == 3
    titles = {f["title"] for f in loaded["films"]}
    assert "Dune" in titles
    assert "The Matrix" in titles
    assert "Inception" in titles
    assert "Blue Eye Samurai" in titles

    # Blue Eye Samurai has 2 episodes → 2 entries.
    bes_count = sum(1 for f in loaded["films"] if f["title"] == "Blue Eye Samurai")
    assert bes_count == 2

    captured = capsys.readouterr()
    assert "Total:" in captured.out
    assert "5 titles" in captured.out or "titles" in captured.out
    assert str(config_path) in captured.out


def test_main_nonexistent_library_errors(tmp_path, capsys):
    exit_code = sd.main(["--library", str(tmp_path / "nope"), "--yes"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
