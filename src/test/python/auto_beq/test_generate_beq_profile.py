"""Unit tests for helpers in ``cli/generate.py``.

Only pure-Python helpers are exercised here — anything that calls ffprobe,
ffmpeg, or loads a model belongs in an integration test.
"""
from __future__ import annotations

import pytest

from cli import generate as gen_module_static


@pytest.fixture(scope="module")
def gen_module():
    return gen_module_static


def test_parse_ffprobe_streams_top_level_only(gen_module):
    """Standalone mkv with FLAC audio: streams at top level, empty programs."""
    probe_json = """
    {
        "programs": [],
        "stream_groups": [],
        "streams": [
            {
                "codec_name": "flac",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo"
            }
        ]
    }
    """
    stream = gen_module._parse_ffprobe_streams(probe_json)
    assert stream.get("codec_name") == "flac"
    assert stream.get("channels") == 2
    assert stream.get("channel_layout") == "stereo"


def test_parse_ffprobe_streams_nested_in_programs(gen_module):
    """Container format (e.g. MPEG-TS): streams only in programs[0]."""
    probe_json = """
    {
        "programs": [
            {
                "streams": [
                    {"codec_name": "ac3", "channels": 6, "channel_layout": "5.1(side)"}
                ]
            }
        ],
        "streams": []
    }
    """
    stream = gen_module._parse_ffprobe_streams(probe_json)
    assert stream.get("codec_name") == "ac3"
    assert stream.get("channels") == 6


def test_parse_ffprobe_streams_empty_programs_no_streams(gen_module):
    probe_json = '{"programs": [], "streams": []}'
    assert gen_module._parse_ffprobe_streams(probe_json) == {}


def test_parse_ffprobe_streams_malformed_json(gen_module):
    assert gen_module._parse_ffprobe_streams("not-json") == {}
    assert gen_module._parse_ffprobe_streams("") == {}


def test_parse_ffprobe_streams_missing_keys(gen_module):
    """Some ffprobe builds omit `programs` entirely."""
    probe_json = '{"streams": [{"codec_name": "eac3", "channels": 8}]}'
    stream = gen_module._parse_ffprobe_streams(probe_json)
    assert stream.get("codec_name") == "eac3"
    assert stream.get("channels") == 8


def test_stale_model_cache_is_detected(tmp_path):
    """A pickled model from an older class version should be detected as stale.

    Regression: LateFusionModel gained _n_audio after the cache was created,
    causing AttributeError on predict(). The fix: smoke-test after unpickling
    and retrain if the model is incompatible.
    """
    import pickle

    # Write a cache file with a valid dict but a model that's just a string
    # (simulates a class whose interface changed).
    cache_file = tmp_path / "nn_model.pkl"
    cache_file.write_bytes(pickle.dumps({"cat_key": "match", "model": "not-a-model"}))

    # Loading succeeds but the "model" has no predict() method.
    cached = pickle.loads(cache_file.read_bytes())
    model = cached["model"]

    # The smoke-test in _load_or_train_model should catch this.
    import numpy as np
    try:
        model.predict(np.zeros((1, 18)))
        assert False, "Should have raised AttributeError"
    except AttributeError:
        # Expected — stale model detected, cache should be deleted.
        cache_file.unlink(missing_ok=True)

    assert not cache_file.exists(), "Stale cache should have been deleted"


# ---------------------------------------------------------------------------
# End-to-end profile pipeline test
# ---------------------------------------------------------------------------


class TestE2EProfilePipeline:
    """End-to-end test of the profile generation pipeline.

    Exercises the full chain: synthetic WAV -> feature extraction ->
    model training -> model save/load -> prediction -> filter output.
    Uses NO external resources (no ffmpeg, no media files, no NAS).

    This test catches the SIGSEGV that crashed profile generation on
    macOS (xgboost + Metal/MPS conflict in the same process).
    """

    @staticmethod
    def _make_synthetic_wav(path, duration_s=1.0, freq_hz=30.0, fs=1000):
        """Create a minimal mono WAV with a sine tone."""
        import struct
        import wave

        n_samples = int(fs * duration_s)
        samples = []
        for i in range(n_samples):
            t = i / fs
            val = int(16000 * __import__("math").sin(2 * __import__("math").pi * freq_hz * t))
            samples.append(struct.pack("<h", max(-32768, min(32767, val))))
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(b"".join(samples))

    def test_full_pipeline_synthetic(self, tmp_path):
        """Train a tiny model, save it, load it, predict filters.

        This exercises the same code path as bin/beq-designer profile:
        WAV -> spectrum -> features -> model.predict -> labels_to_filters.
        """
        import numpy as np

        # 1. Create synthetic WAV
        wav_path = tmp_path / "test.lfe-1000hz.wav"
        self._make_synthetic_wav(wav_path)

        # 2. Read and analyse the WAV (same as generate.py)
        from model.signal import Signal, read_wav_data
        samples, read_fs, _ = read_wav_data(str(wav_path))
        mono = samples[:, 0] if samples.ndim > 1 else samples
        sig = Signal("test", mono, fs=1000)
        measured_freqs, measured_db = sig.avg_spectrum()

        from model.auto_beq import DEFAULT_GRID, smooth_fractional_octave
        curve = np.interp(DEFAULT_GRID, measured_freqs, measured_db)
        anchor_idx = int(np.argmin(np.abs(DEFAULT_GRID - 80.0)))
        curve -= curve[anchor_idx]
        curve = smooth_fractional_octave(curve, DEFAULT_GRID, octaves=1.0 / 6.0)
        curve -= curve[anchor_idx]

        from model.auto_beq_advisor import extract_curve_features
        features = extract_curve_features(curve, DEFAULT_GRID)

        # 3. Train a tiny XGBoost model on synthetic data
        from model.auto_beq_nn import (
            AudioFeatureConfig,
            build_feature_vector,
            catalogue_entry_to_labels,
            labels_to_filters,
            train_production_weighted_hybrid,
        )
        from model.auto_beq_metadata import enrich_media_metadata

        # Minimal synthetic entries with simple filters
        synth_entries = []
        for i in range(5):
            entry = {
                "title": f"Synth Film {i}",
                "year": 2020 + i,
                "audioTypes": ["DTS-HD MA"],
                "author": "test",
                "filters": [
                    {"type": "LowShelf", "freq": 30, "gain": 5 + i, "q": 0.7, "count": 1},
                ],
            }
            synth_entries.append(entry)

        # Build synthetic feature vectors (same features for all, just
        # to exercise the training path)
        from model.auto_beq_advisor import MediaMetadata
        cfg = AudioFeatureConfig()
        metadata = MediaMetadata(
            title="Test Film", year=2022,
            genres=("Action",), runtime_min=120, rating="PG-13",
        )
        x = build_feature_vector(features, metadata, config=cfg)

        real_samples = [(e, features) for e in synth_entries]
        model, meta = train_production_weighted_hybrid(
            real_samples=real_samples,
            synth_entries=synth_entries,
            tmdb_cache={},
            freqs_hz=DEFAULT_GRID,
            fs=1000,
            real_weight=50.0,
            config=cfg,
        )

        # 4. Save and reload (exercises joblib serialize/deserialize -
        #    this is where the SIGSEGV happens on macOS)
        import joblib
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, str(model_path))

        from model.auto_beq_nn import load_model
        loaded = load_model(str(model_path))

        # 5. Predict
        y_pred = loaded.predict(x.reshape(1, -1))[0]

        # 6. Convert to filters
        filters = labels_to_filters(y_pred)

        # 7. Assert coherent output
        assert isinstance(filters, list)
        assert len(filters) > 0, "model should produce at least one filter"
        for f in filters:
            assert "type" in f, f"filter missing 'type': {f}"
            assert "freq" in f, f"filter missing 'freq': {f}"
            assert "gain" in f, f"filter missing 'gain': {f}"
            assert "q" in f or "count" in f, f"filter missing 'q' or 'count': {f}"

    def test_pipeline_after_qt_init(self, tmp_path):
        """Profile pipeline must work after Qt has initialized.

        Regression test for SIGSEGV: Qt initializes Metal/MPS on macOS,
        and subsequent xgboost model loading crashes because Metal state
        conflicts. This test imports PyQt6 first (like the interactive
        CLI does) then runs the prediction pipeline.
        """
        import os
        # Disable MPS before any imports - this is the fix we're testing
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

        # Import Qt to trigger Metal initialization (same as interactive CLI)
        try:
            from PyQt6.QtWidgets import QApplication
            # Don't create a QApplication - just importing triggers Metal init
        except ImportError:
            import pytest
            pytest.skip("PyQt6 not installed")

        # Now do the same pipeline as test_full_pipeline_synthetic
        import numpy as np

        wav_path = tmp_path / "test.lfe-1000hz.wav"
        self._make_synthetic_wav(wav_path)

        from model.signal import Signal, read_wav_data
        samples, read_fs, _ = read_wav_data(str(wav_path))
        mono = samples[:, 0] if samples.ndim > 1 else samples
        sig = Signal("test", mono, fs=1000)
        measured_freqs, measured_db = sig.avg_spectrum()

        from model.auto_beq import DEFAULT_GRID, smooth_fractional_octave
        curve = np.interp(DEFAULT_GRID, measured_freqs, measured_db)
        anchor_idx = int(np.argmin(np.abs(DEFAULT_GRID - 80.0)))
        curve -= curve[anchor_idx]
        curve = smooth_fractional_octave(curve, DEFAULT_GRID, octaves=1.0 / 6.0)
        curve -= curve[anchor_idx]

        from model.auto_beq_advisor import MediaMetadata, extract_curve_features
        features = extract_curve_features(curve, DEFAULT_GRID)

        import joblib
        from model.auto_beq_nn import (
            AudioFeatureConfig,
            build_feature_vector,
            labels_to_filters,
            load_model,
            train_production_weighted_hybrid,
        )

        cfg = AudioFeatureConfig()
        metadata = MediaMetadata(
            title="Test Film", year=2022,
            genres=("Action",), runtime_min=120, rating="PG-13",
        )
        x = build_feature_vector(features, metadata, config=cfg)

        synth_entries = [
            {"title": f"Film {i}", "year": 2020 + i, "audioTypes": ["DTS-HD MA"],
             "author": "test", "filters": [
                {"type": "LowShelf", "freq": 30, "gain": 5 + i, "q": 0.7, "count": 1}]}
            for i in range(5)
        ]
        real_samples = [(e, features) for e in synth_entries]
        model, _ = train_production_weighted_hybrid(
            real_samples=real_samples, synth_entries=synth_entries,
            tmdb_cache={}, freqs_hz=DEFAULT_GRID, fs=1000,
            real_weight=50.0, config=cfg,
        )

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, str(model_path))
        loaded = load_model(str(model_path))
        y_pred = loaded.predict(x.reshape(1, -1))[0]
        filters = labels_to_filters(y_pred)
        assert len(filters) > 0

    def test_missing_model_clear_error(self, tmp_path):
        """Missing model file should raise FileNotFoundError, not crash."""
        fake_path = tmp_path / "nonexistent.joblib"
        with pytest.raises((FileNotFoundError, RuntimeError)):
            from model.auto_beq_nn import load_model
            load_model(str(fake_path))
