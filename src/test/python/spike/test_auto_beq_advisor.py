"""Unit tests for the Advisor abstraction.

Fast, deterministic, offline. No real LLM calls - LLM paths are
covered by the real-media integration test when Ollama is available.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from model.auto_beq_advisor import (
    Advice,
    CurveFeatures,
    HeuristicAdvisor,
    MediaMetadata,
    MockAdvisor,
    _clamp_advice,
    _render_ollama_user_prompt,
    _slugify,
    extract_curve_features,
    get_advisor,
)


def test_slugify_basic():
    assert _slugify("Edge of Tomorrow") == "edge-of-tomorrow"
    assert _slugify("Mad Max: Fury Road") == "mad-max-fury-road"
    assert _slugify("1917") == "1917"
    assert _slugify("  John  Wick  ") == "john-wick"


def test_extract_curve_features_on_synthetic_curve():
    # Synthetic curve: peak at 20 Hz, 0 dB anchor at 80 Hz, -20 dB at 5 Hz.
    freqs = np.logspace(np.log10(5.0), np.log10(200.0), 240)
    curve = np.zeros_like(freqs)
    # Simple V-shape: -20 at 5 Hz, +8 at 20 Hz, 0 at 80 Hz.
    for i, f in enumerate(freqs):
        if f < 20:
            curve[i] = -20 + (f - 5) * (28 / 15)
        elif f < 80:
            curve[i] = 8 * (80 - f) / 60
        else:
            curve[i] = 0

    features = extract_curve_features(curve, freqs)
    assert features.shoulder_peak_db == pytest.approx(8.0, abs=0.5)
    assert features.shoulder_peak_hz == pytest.approx(20.0, abs=2.0)
    assert features.level_at_5hz_db == pytest.approx(-20.0, abs=1.0)
    assert features.level_at_20hz_db == pytest.approx(8.0, abs=1.0)
    assert features.rolloff_depth_db > 20.0
    assert len(features.curve_sample_points) == 12


def test_heuristic_advisor_returns_clamped_advice():
    meta = MediaMetadata(title="Test Film", year=2020)
    # Large rolloff => cliff profile => 12 dB cap
    features = CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
        level_at_5hz_db=-30.0, level_at_10hz_db=-25.0, level_at_20hz_db=3.0,
        rolloff_depth_db=30.0, rolloff_slope_db_per_oct=10.0,
        dynamic_range_db=35.0, curve_sample_points=(),
    )
    advice = HeuristicAdvisor().advise(meta, features)
    assert advice.source == "heuristic"
    assert advice.max_gain_db == 12.0
    assert advice.knee_hz is None
    assert "cliff" in advice.reasoning


def test_heuristic_advisor_mild_profile():
    features = CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
        level_at_5hz_db=-5.0, level_at_10hz_db=-3.0, level_at_20hz_db=2.0,
        rolloff_depth_db=10.0, rolloff_slope_db_per_oct=2.0,
        dynamic_range_db=10.0, curve_sample_points=(),
    )
    advice = HeuristicAdvisor().advise(MediaMetadata(title="x"), features)
    assert advice.max_gain_db == 15.0
    assert "mild" in advice.reasoning


def test_mock_advisor_loads_canned_response(tmp_path):
    path = tmp_path / "test-title.json"
    path.write_text(json.dumps({
        "max_gain_db": 22.5, "knee_hz": 20.0,
        "reasoning": "canned", "confidence": 0.9,
    }))
    advisor = MockAdvisor(responses_dir=tmp_path)
    meta = MediaMetadata(title="Test Title", year=2024)
    advice = advisor.advise(meta, _dummy_features())
    assert advice.source == "mock"
    assert advice.max_gain_db == 22.5
    assert advice.knee_hz == 20.0
    assert advice.confidence == 0.9
    assert advice.reasoning == "canned"


def test_mock_advisor_missing_file_raises(tmp_path):
    advisor = MockAdvisor(responses_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no canned MockAdvisor response"):
        advisor.advise(MediaMetadata(title="Unknown"), _dummy_features())


def test_mock_advisor_uses_committed_fixtures():
    # Real fixtures ship with the repo.
    advisor = MockAdvisor()
    for title in ("Edge of Tomorrow", "Mad Max: Fury Road", "John Wick"):
        advice = advisor.advise(MediaMetadata(title=title), _dummy_features())
        assert advice.max_gain_db > 0
        assert advice.source == "mock"


def test_clamp_advice_bounds():
    clamped = _clamp_advice(
        Advice(max_gain_db=99.0, knee_hz=200.0, confidence=1.5),
        source="test",
    )
    assert clamped.max_gain_db == 35.0
    assert clamped.knee_hz == 80.0
    assert clamped.confidence == 1.0
    assert clamped.source == "test"

    clamped2 = _clamp_advice(
        Advice(max_gain_db=-10.0, knee_hz=1.0, confidence=-0.5),
        source="test",
    )
    assert clamped2.max_gain_db == 0.0
    assert clamped2.knee_hz == 5.0
    assert clamped2.confidence == 0.0


def test_clamp_advice_preserves_none_knee():
    advice = _clamp_advice(
        Advice(max_gain_db=10.0, knee_hz=None),
        source="test",
    )
    assert advice.knee_hz is None


def test_get_advisor_returns_requested_impl():
    assert isinstance(get_advisor("heuristic"), HeuristicAdvisor)
    assert isinstance(get_advisor("mock"), MockAdvisor)
    # ollama construction shouldn't require the server to be up
    ollama = get_advisor("ollama")
    assert ollama.name == "ollama"


def test_get_advisor_default_is_heuristic(monkeypatch):
    monkeypatch.delenv("AUTO_BEQ_ADVISOR", raising=False)
    assert isinstance(get_advisor(None), HeuristicAdvisor)


def test_get_advisor_reads_env(monkeypatch):
    monkeypatch.setenv("AUTO_BEQ_ADVISOR", "mock")
    assert isinstance(get_advisor(None), MockAdvisor)


def test_get_advisor_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown advisor"):
        get_advisor("gpt5-dreamland")


def test_render_ollama_user_prompt_contains_placeholders():
    meta = MediaMetadata(
        title="Example", year=2024,
        audio_codec="dts", channel_layout="7.1",
    )
    features = _dummy_features()
    prompt = _render_ollama_user_prompt(meta, features)
    assert "Example (2024)" in prompt
    assert "dts 7.1" in prompt
    assert "Shoulder peak" in prompt
    assert "Level at 5 Hz" in prompt
    assert "Level at 10 Hz" in prompt
    assert "Level at 20 Hz" in prompt
    assert "max_gain_db" in prompt
    assert "knee_hz" in prompt


def test_render_ollama_prompt_handles_missing_metadata():
    features = _dummy_features()
    prompt = _render_ollama_user_prompt(MediaMetadata(title="Bare"), features)
    assert "unknown" in prompt  # year / codec / layout default to unknown


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dummy_features() -> CurveFeatures:
    return CurveFeatures(
        shoulder_peak_db=5.0,
        shoulder_peak_hz=25.0,
        level_at_5hz_db=-10.0,
        level_at_10hz_db=-5.0,
        level_at_20hz_db=4.0,
        rolloff_depth_db=15.0,
        rolloff_slope_db_per_oct=5.0,
        dynamic_range_db=18.0,
        curve_sample_points=tuple((float(hz), 0.0) for hz in (5, 10, 20, 40, 80)),
    )
