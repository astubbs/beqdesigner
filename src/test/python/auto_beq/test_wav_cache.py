"""Tests for model.wav_cache - cache layout and walking."""

from pathlib import Path

from model.wav_cache import WAV_SUFFIX, iter_cached_wavs


def _make_wav(parent: Path, name: str) -> Path:
    """Create a fake WAV file in the given directory."""
    p = parent / f"{name}{WAV_SUFFIX}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"RIFF" + b"\x00" * 40)
    return p


class TestIterCachedWavs:

    def test_finds_wavs_in_flat_buckets(self, tmp_path):
        """Flat bucket layout (legacy title-bucket)."""
        _make_wav(tmp_path / "AV", "Avatar (2009) [tmdb-19995]")
        _make_wav(tmp_path / "DU", "Dune (2021) [tmdb-438631]")
        result = iter_cached_wavs(tmp_path)
        assert len(result) == 2
        assert all(str(p).endswith(WAV_SUFFIX) for p in result)

    def test_finds_wavs_in_nested_id_layout(self, tmp_path):
        """ID-based layout with shard directories."""
        _make_wav(tmp_path / "tmdb" / "19" / "19995", "Avatar (2009) [tmdb-19995]")
        _make_wav(tmp_path / "tvdb" / "37" / "377543" / "Show [tvdb-377543]" / "Season 01", "S01E01")
        result = iter_cached_wavs(tmp_path)
        assert len(result) == 2

    def test_empty_cache_returns_empty(self, tmp_path):
        result = iter_cached_wavs(tmp_path)
        assert result == []

    def test_nonexistent_root_returns_empty(self, tmp_path):
        result = iter_cached_wavs(tmp_path / "does_not_exist")
        assert result == []

    def test_non_wav_files_excluded(self, tmp_path):
        """Only files ending in WAV_SUFFIX are returned."""
        _make_wav(tmp_path / "AV", "Avatar (2009) [tmdb-19995]")
        (tmp_path / "AV" / "readme.txt").write_text("not a wav")
        (tmp_path / "AV" / "data.npy").write_bytes(b"\x00")
        result = iter_cached_wavs(tmp_path)
        assert len(result) == 1

    def test_results_are_sorted(self, tmp_path):
        _make_wav(tmp_path / "ZZ", "Zorro")
        _make_wav(tmp_path / "AA", "Aardvark")
        _make_wav(tmp_path / "MM", "Movie")
        result = iter_cached_wavs(tmp_path)
        assert result == sorted(result)
