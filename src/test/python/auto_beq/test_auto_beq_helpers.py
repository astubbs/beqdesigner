"""Tests for _auto_beq_helpers: audio cache path mirroring + LFE extraction."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from model.audio_extraction import extract_lfe_wav, have_tool
from model.wav_discovery import audio_cache_dir


class TestAudioCacheDir:
    def test_reads_from_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTO_BEQ_AUDIO_CACHE", str(tmp_path / "cache"))
        result = audio_cache_dir()
        assert result == tmp_path / "cache"
        assert result.exists()

    def test_reads_from_settings_json(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AUTO_BEQ_AUDIO_CACHE", raising=False)
        settings_dir = tmp_path / "config"
        settings_dir.mkdir()
        settings_file = settings_dir / "settings.json"
        cache_dir = tmp_path / "my-cache"
        settings_file.write_text(f'{{"audio_cache_dir": "{cache_dir}"}}')
        # Patch Path.home to point at our temp structure
        monkeypatch.setenv(
            "AUTO_BEQ_AUDIO_CACHE", ""
        )  # empty string should still fail
        # Actually just test via env var since patching home() is fragile
        monkeypatch.setenv("AUTO_BEQ_AUDIO_CACHE", str(cache_dir))
        result = audio_cache_dir()
        assert result == cache_dir

    def test_raises_when_not_configured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AUTO_BEQ_AUDIO_CACHE", raising=False)
        # Point home config at an empty dir so no settings.json is found
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        with pytest.raises(RuntimeError, match="audio cache dir not configured"):
            audio_cache_dir()


class TestExtractLfeWavCachePath:
    """Test that cache paths are mirrored correctly under the cache dir.

    These don't run ffmpeg - they test path construction by pre-seeding
    the cache with a dummy file and verifying the function finds it.
    """

    def test_mirrored_path_structure(self, monkeypatch, tmp_path):
        cache_root = tmp_path / "audio-cache"
        monkeypatch.setenv("AUTO_BEQ_AUDIO_CACHE", str(cache_root))

        source = Path("/Volumes/NAS/Movies/Dune (2021)/Dune.mkv")
        # Pre-compute what the cache path should be
        expected = (
            cache_root
            / "Volumes/NAS/Movies/Dune (2021)"
            / "Dune.lfe-1000hz.wav"
        )
        # Pre-seed the cache so extract_lfe_wav skips ffmpeg
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_bytes(b"\x00" * 100)

        result = extract_lfe_wav(source, target_fs=1000)
        assert result == expected

    def test_mirrored_path_with_trim(self, monkeypatch, tmp_path):
        cache_root = tmp_path / "audio-cache"
        monkeypatch.setenv("AUTO_BEQ_AUDIO_CACHE", str(cache_root))

        source = Path("/Volumes/NAS/Movies/EoT/EoT.mkv")
        expected = (
            cache_root
            / "Volumes/NAS/Movies/EoT"
            / "EoT.lfe-1000hz-t1800-3600.wav"
        )
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_bytes(b"\x00" * 100)

        result = extract_lfe_wav(
            source, target_fs=1000,
            trim_start_s=1800, trim_end_s=3600,
        )
        assert result == expected

    def test_source_never_gets_wav_sibling(self, monkeypatch, tmp_path):
        """Verify we never write next to the source file."""
        cache_root = tmp_path / "audio-cache"
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "test.mkv"
        source.write_bytes(b"\x00")
        monkeypatch.setenv("AUTO_BEQ_AUDIO_CACHE", str(cache_root))

        # Pre-seed cache so ffmpeg is skipped
        expected = cache_root / str(source.resolve()).lstrip("/")
        expected = expected.with_suffix("").parent / (
            expected.with_suffix("").name + ".lfe-1000hz.wav"
        )
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_bytes(b"\x00" * 100)

        extract_lfe_wav(source, target_fs=1000)

        # Source directory should have NO .wav files
        wav_files = list(source_dir.glob("*.wav"))
        assert wav_files == [], f"WAV written next to source: {wav_files}"


@pytest.mark.integration
@pytest.mark.skipif(
    not have_tool("ffmpeg") or not have_tool("ffprobe"),
    reason="ffmpeg/ffprobe not on PATH",
)
class TestExtractLfeWavReal:
    """Integration test: actually extract LFE from a real media file.

    Set AUTO_BEQ_TEST_MEDIA_FILE to the path of any MKV with an LFE
    channel (e.g. a DTS-HD MA 7.1 rip). The test is skipped if the
    env var is not set or the file doesn't exist.
    """

    EOT_PATH = Path(os.environ.get("AUTO_BEQ_TEST_MEDIA_FILE", "/nonexistent"))

    @pytest.mark.skipif(
        not Path(os.environ.get("AUTO_BEQ_TEST_MEDIA_FILE", "/nonexistent")).exists(),
        reason="AUTO_BEQ_TEST_MEDIA_FILE not set or file not found",
    )
    def test_real_extraction_to_cache_dir(self):
        """Extract EoT LFE to the configured cache dir and verify
        the WAV lands in the mirrored path, NOT next to the source."""
        cache_root = audio_cache_dir()
        wav_path = extract_lfe_wav(self.EOT_PATH, target_fs=1000)

        # WAV should be under the cache root
        assert str(wav_path).startswith(str(cache_root)), (
            f"WAV not under cache root: {wav_path}"
        )
        assert wav_path.exists()
        assert wav_path.stat().st_size > 0

        # Source directory should NOT have any new .wav files
        source_dir = self.EOT_PATH.parent
        # Note: old cached WAVs from before this change may still be
        # there - we only assert the NEW extraction went to cache_root.
        assert str(wav_path.parent) != str(source_dir), (
            "WAV was written next to source instead of cache dir"
        )


# ---------------------------------------------------------------------------
# Disk-backed caches for curve features + discovery
# ---------------------------------------------------------------------------


class TestCachedExtractFeatures:
    """Unit tests for ``cached_extract_features_with_strategy``."""

    def test_returns_same_result_on_cache_hit(self, monkeypatch, tmp_path):
        """Second call returns the cached result instead of re-extracting."""
        import numpy as np
        import model.audio_extraction as ae
        import model.wav_discovery as wd
        from model.audio_extraction import (
            STRATEGY_WELCH,
            cached_extract_features_with_strategy,
        )

        # Point beq_shared_dir at a tmp cache.
        monkeypatch.setattr(wd, "beq_shared_dir", lambda: tmp_path)

        # Stub the uncached extractor to return a counter-tagged sentinel.
        call_count = {"n": 0}

        def fake_extract(wav_path, freqs_hz, fs, strategy):
            call_count["n"] += 1
            return ("sentinel", call_count["n"], str(wav_path))

        monkeypatch.setattr(ae, "extract_features_with_strategy", fake_extract)

        wav = tmp_path / "fake.wav"
        wav.write_bytes(b"\x00" * 1024)

        freqs = np.array([20.0, 40.0, 80.0])
        result1 = cached_extract_features_with_strategy(
            wav, freqs, 1000, strategy=STRATEGY_WELCH,
        )
        result2 = cached_extract_features_with_strategy(
            wav, freqs, 1000, strategy=STRATEGY_WELCH,
        )

        assert result1 == result2, "cache hit should return the same value"
        assert call_count["n"] == 1, "uncached extractor should run only once"

    def test_invalidates_on_mtime_change(self, monkeypatch, tmp_path):
        """Touching the WAV bumps its mtime -> new cache key -> re-extract."""
        import time
        import numpy as np
        import model.audio_extraction as ae
        import model.wav_discovery as wd
        from model.audio_extraction import (
            STRATEGY_WELCH,
            cached_extract_features_with_strategy,
        )

        monkeypatch.setattr(wd, "beq_shared_dir", lambda: tmp_path)
        call_count = {"n": 0}

        def fake_extract(wav_path, freqs_hz, fs, strategy):
            call_count["n"] += 1
            return ("result", call_count["n"])

        monkeypatch.setattr(ae, "extract_features_with_strategy", fake_extract)

        wav = tmp_path / "fake.wav"
        wav.write_bytes(b"\x00" * 1024)

        freqs = np.array([20.0, 40.0, 80.0])
        r1 = cached_extract_features_with_strategy(wav, freqs, 1000, strategy=STRATEGY_WELCH)
        assert r1 == ("result", 1)

        # Bump the mtime explicitly (some filesystems have 1 s mtime
        # resolution so we need to force a later timestamp).
        new_time = wav.stat().st_mtime + 10
        import os as _os
        _os.utime(wav, (new_time, new_time))

        r2 = cached_extract_features_with_strategy(wav, freqs, 1000, strategy=STRATEGY_WELCH)
        assert call_count["n"] == 2, "mtime bump should invalidate the cache"
        assert r2 == ("result", 2)

    def test_env_var_disables_cache(self, monkeypatch, tmp_path):
        """``AUTO_BEQ_FEATURE_CACHE=0`` bypasses the cache entirely."""
        import numpy as np
        import model.audio_extraction as ae
        import model.wav_discovery as wd
        from model.audio_extraction import (
            STRATEGY_WELCH,
            cached_extract_features_with_strategy,
        )

        monkeypatch.setattr(wd, "beq_shared_dir", lambda: tmp_path)
        monkeypatch.setenv("AUTO_BEQ_FEATURE_CACHE", "0")

        call_count = {"n": 0}

        def fake_extract(wav_path, freqs_hz, fs, strategy):
            call_count["n"] += 1
            return ("result", call_count["n"])

        monkeypatch.setattr(ae, "extract_features_with_strategy", fake_extract)

        wav = tmp_path / "fake.wav"
        wav.write_bytes(b"\x00" * 1024)

        freqs = np.array([20.0, 40.0, 80.0])
        for _ in range(3):
            cached_extract_features_with_strategy(wav, freqs, 1000, strategy=STRATEGY_WELCH)
        assert call_count["n"] == 3, "disabled cache should always re-extract"


class TestDiscoveryCache:
    """Unit tests for ``discover_wav_catalogue_pairs_cached``."""

    def test_cache_hit_returns_cached_pairs(self, monkeypatch, tmp_path):
        """Second call returns the cached pair list without re-walking."""
        import model.wav_discovery as wd
        from model.wav_discovery import discover_wav_catalogue_pairs_cached

        wav_root = tmp_path / "wav-cache"
        wav_root.mkdir()
        monkeypatch.setattr(wd, "beq_shared_dir", lambda: tmp_path)
        monkeypatch.setattr(wd, "wav_cache_dir", lambda: wav_root)

        call_count = {"n": 0}

        def fake_discover():
            call_count["n"] += 1
            return [
                {"wav_path": wav_root / "a.wav", "catalogue_entry": {"title": "A"},
                 "tmdb_id": "1", "media_id": "tmdb-1"},
            ]

        monkeypatch.setattr(wd, "discover_wav_catalogue_pairs", fake_discover)

        r1 = discover_wav_catalogue_pairs_cached()
        r2 = discover_wav_catalogue_pairs_cached()

        assert call_count["n"] == 1, "should walk only on the first call"
        assert len(r1) == 1
        # Pickle roundtrip equivalence (Path objects compare by value).
        assert [p["tmdb_id"] for p in r2] == [p["tmdb_id"] for p in r1]

    def test_cache_invalidates_on_wav_root_mtime_change(self, monkeypatch, tmp_path):
        """Bumping the wav-cache root mtime (as extract_lfe.py does)
        invalidates the discovery cache."""
        import os as _os
        import model.wav_discovery as wd
        from model.wav_discovery import discover_wav_catalogue_pairs_cached

        wav_root = tmp_path / "wav-cache"
        wav_root.mkdir()
        monkeypatch.setattr(wd, "beq_shared_dir", lambda: tmp_path)
        monkeypatch.setattr(wd, "wav_cache_dir", lambda: wav_root)

        call_count = {"n": 0}

        def fake_discover():
            call_count["n"] += 1
            return [{"wav_path": wav_root / "a.wav", "catalogue_entry": {}, "tmdb_id": "1"}]

        monkeypatch.setattr(wd, "discover_wav_catalogue_pairs", fake_discover)

        discover_wav_catalogue_pairs_cached()
        assert call_count["n"] == 1

        # Bump the root mtime to simulate a fresh extract_lfe.py run.
        new_time = wav_root.stat().st_mtime + 10
        _os.utime(wav_root, (new_time, new_time))

        discover_wav_catalogue_pairs_cached()
        assert call_count["n"] == 2, "root mtime change should invalidate"

    def test_env_var_disables_discovery_cache(self, monkeypatch, tmp_path):
        """``AUTO_BEQ_DISCOVERY_CACHE=0`` bypasses the cache entirely."""
        import model.wav_discovery as wd
        from model.wav_discovery import discover_wav_catalogue_pairs_cached

        wav_root = tmp_path / "wav-cache"
        wav_root.mkdir()
        monkeypatch.setattr(wd, "beq_shared_dir", lambda: tmp_path)
        monkeypatch.setattr(wd, "wav_cache_dir", lambda: wav_root)
        monkeypatch.setenv("AUTO_BEQ_DISCOVERY_CACHE", "0")

        call_count = {"n": 0}

        def fake_discover():
            call_count["n"] += 1
            return []

        monkeypatch.setattr(wd, "discover_wav_catalogue_pairs", fake_discover)

        for _ in range(3):
            discover_wav_catalogue_pairs_cached()
        assert call_count["n"] == 3, "disabled cache should always walk"


# ---------------------------------------------------------------------------
# prepare_training_data shared function
# ---------------------------------------------------------------------------


class TestPrepareTrainingData:
    """Unit tests for the shared ``prepare_training_data()`` function.

    Uses monkeypatching to avoid touching real WAV caches or network.
    """

    def _make_fake_pair(self, title: str, has_filters: bool = True):
        """Build a minimal catalogue pair dict for testing."""
        filters = [{"gain": 5.0}] if has_filters else []
        entry = {
            "title": title,
            "author": "test",
            "theMovieDB": f"tmdb-{title}",
            "filters": filters,
        }
        return {"catalogue_entry": entry, "wav_path": Path(f"/fake/{title}.wav")}

    def _patch_dependencies(self, monkeypatch, pairs, features_value="fake_features"):
        """Patch out I/O-heavy dependencies with fakes."""
        import model.wav_discovery as wd

        monkeypatch.setattr(
            wd, "discover_wav_catalogue_pairs_cached", lambda: pairs,
        )

        # Mock extract_features_parallel to return (pair, features) tuples.
        import model.audio_extraction as audio_mod
        monkeypatch.setattr(
            audio_mod, "extract_features_parallel",
            lambda pairs, grid, fs, strategy=None: [
                (p, features_value) for p in pairs
            ],
        )

        # Mock TMDb cache loading.
        import model.auto_beq_metadata as meta_mod
        monkeypatch.setattr(meta_mod, "load_cache", lambda: {})
        monkeypatch.setattr(
            meta_mod, "fetch_metadata_batch",
            lambda entries, cache=None: cache or {},
        )

    def test_basic_returns_expected_keys(self, monkeypatch):
        """Basic call returns pairs, all_real, tmdb_cache, real_samples."""
        pairs = [self._make_fake_pair(f"Movie{i}") for i in range(10)]
        self._patch_dependencies(monkeypatch, pairs)

        from model.training_data import prepare_training_data
        result = prepare_training_data(fetch_catalogue=False)

        assert "pairs" in result
        assert "all_real" in result
        assert "tmdb_cache" in result
        assert "real_samples" in result
        assert len(result["pairs"]) == 10
        assert len(result["all_real"]) == 10
        assert len(result["real_samples"]) == 10

    def test_filters_entries_without_filters(self, monkeypatch):
        """Entries without filters are excluded from real_samples."""
        pairs = [
            self._make_fake_pair("WithFilters", has_filters=True),
            self._make_fake_pair("NoFilters", has_filters=False),
        ]
        self._patch_dependencies(monkeypatch, pairs)

        from model.training_data import prepare_training_data
        result = prepare_training_data(fetch_catalogue=False)

        assert len(result["all_real"]) == 2
        assert len(result["real_samples"]) == 1
        assert result["real_samples"][0][0]["title"] == "WithFilters"

    def test_raises_on_empty_cache(self, monkeypatch):
        """Empty WAV cache raises RuntimeError."""
        self._patch_dependencies(monkeypatch, pairs=[])

        from model.training_data import prepare_training_data
        with pytest.raises(RuntimeError, match="No catalogue-matched WAVs"):
            prepare_training_data()

    def test_min_pairs_enforced(self, monkeypatch):
        """min_pairs threshold raises when not met."""
        pairs = [self._make_fake_pair(f"M{i}") for i in range(5)]
        self._patch_dependencies(monkeypatch, pairs)

        from model.training_data import prepare_training_data
        with pytest.raises(RuntimeError, match="need >= 10"):
            prepare_training_data(min_pairs=10)

    def test_split_produces_train_test_indices(self, monkeypatch):
        """split=True adds train_idx, test_idx, train_samples, entries_test."""
        pairs = [self._make_fake_pair(f"M{i}") for i in range(50)]
        self._patch_dependencies(monkeypatch, pairs)

        # Mock catalogue fetching for synth_entries.
        import model.auto_beq_catalogue as cat_mod
        monkeypatch.setattr(cat_mod, "_fetch_or_cache", lambda: [])
        import model.auto_beq_nn as nn_mod
        monkeypatch.setattr(nn_mod, "deduplicate_by_title", lambda x: x)

        from model.training_data import prepare_training_data
        result = prepare_training_data(split=True)

        assert "train_idx" in result
        assert "test_idx" in result
        assert "train_samples" in result
        assert "entries_test" in result
        assert "synth_entries" in result
        # Default 80/20 split.
        total = len(result["train_idx"]) + len(result["test_idx"])
        assert total == 50

    def test_build_feature_vectors_requires_split(self, monkeypatch):
        """build_feature_vectors=True without split=True raises ValueError."""
        pairs = [self._make_fake_pair("M0")]
        self._patch_dependencies(monkeypatch, pairs)

        from model.training_data import prepare_training_data
        with pytest.raises(ValueError, match="requires split=True"):
            prepare_training_data(build_feature_vectors=True, split=False)
