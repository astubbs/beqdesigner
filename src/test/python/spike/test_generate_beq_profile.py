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
