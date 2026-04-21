"""Tests for model/wav_discovery.py - WAV-to-catalogue matching logic."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _fake_catalogue():
    """Minimal BEQ catalogue entries for testing matching."""
    return [
        {"title": "Avatar", "year": 2009, "theMovieDB": "19995",
         "filters": [{"freq": 30, "gain": -5, "q": 0.7}]},
        {"title": "Inception", "year": 2010, "theMovieDB": "27205",
         "filters": [{"freq": 25, "gain": -3, "q": 0.9}]},
        {"title": "Dune", "year": 2021, "theMovieDB": "438631",
         "filters": []},
        {"title": "Blue Eye Samurai", "year": 2023, "theMovieDB": "",
         "tvdbid": "434151",
         "filters": [{"freq": 40, "gain": -4, "q": 0.8}]},
    ]


class TestMatchWavsToCatalogue:
    """Unit tests for _match_wavs_to_catalogue."""

    def _run_match(self, wav_paths, catalogue=None):
        """Run matching with a fake catalogue."""
        import model.auto_beq_catalogue as cat_mod
        with patch.object(cat_mod, "_fetch_or_cache",
                          return_value=catalogue or _fake_catalogue()):
            from model.wav_discovery import _match_wavs_to_catalogue
            return _match_wavs_to_catalogue(wav_paths)

    def test_tmdb_direct_match(self):
        wav = Path("/cache/Avatar (2009) [tmdb-19995]/Avatar.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 1
        assert len(unmatched) == 0
        assert matched[0]["catalogue_entry"]["title"] == "Avatar"
        assert matched[0]["tmdb_id"] == "19995"
        assert matched[0]["media_id"] == "tmdb-19995"

    def test_tmdb_match_different_id(self):
        wav = Path("/cache/Inception (2010) [tmdb-27205]/Inception.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 1
        assert matched[0]["catalogue_entry"]["title"] == "Inception"

    def test_tvdb_fallback_via_title_year(self):
        """tvdb IDs aren't in the catalogue's theMovieDB field, so matching
        falls back to title+year extracted from the directory name."""
        wav = Path("/cache/Blue Eye Samurai (2023) [tvdb-434151]/Season 1/S01E01.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 1
        assert matched[0]["catalogue_entry"]["title"] == "Blue Eye Samurai"

    def test_unmatched_wav_no_catalogue_entry(self):
        """WAV with an ID that doesn't match any catalogue entry."""
        wav = Path("/cache/Unknown Movie (2030) [tmdb-99999]/file.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 0
        assert len(unmatched) == 1
        assert unmatched[0] == wav

    def test_wav_without_id_tag_skipped(self):
        """WAVs without [tmdb-NNN] or similar tags are silently skipped."""
        wav = Path("/cache/legacy/Some Movie/audio.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 0
        assert len(unmatched) == 0  # not matched, but also not in unmatched

    def test_empty_wav_list(self):
        matched, unmatched = self._run_match([])
        assert matched == []
        assert unmatched == []

    def test_multiple_wavs_mixed_results(self):
        wavs = [
            Path("/cache/Avatar (2009) [tmdb-19995]/Avatar.lfe-1000hz.wav"),
            Path("/cache/Unknown (2030) [tmdb-99999]/file.wav"),
            Path("/cache/Inception (2010) [tmdb-27205]/Inception.lfe-1000hz.wav"),
        ]
        matched, unmatched = self._run_match(wavs)
        assert len(matched) == 2
        assert len(unmatched) == 1
        titles = {m["catalogue_entry"]["title"] for m in matched}
        assert titles == {"Avatar", "Inception"}

    def test_imdb_fallback_via_title_year(self):
        """imdb IDs aren't in theMovieDB, so matching uses title+year."""
        wav = Path("/cache/Inception (2010) [imdb-tt1375666]/Inception.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 1
        assert matched[0]["catalogue_entry"]["title"] == "Inception"

    def test_title_year_case_insensitive(self):
        """Title matching is case-insensitive."""
        wav = Path("/cache/INCEPTION (2010) [imdb-tt1375666]/file.lfe-1000hz.wav")
        matched, unmatched = self._run_match([wav])
        assert len(matched) == 1
