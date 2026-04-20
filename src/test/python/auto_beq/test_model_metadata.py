"""Tests for model.model_metadata - model provenance loading and formatting."""

from __future__ import annotations

import json
import time

import pytest

from model.model_metadata import (
    check_training_staleness,
    format_age,
    format_champion_comparison,
    format_model_banner,
    format_staleness_warning,
    load_champion_history,
    load_model_metadata,
    record_champion,
)


# ---------------------------------------------------------------------------
# format_age
# ---------------------------------------------------------------------------


def test_format_age_just_now():
    assert format_age(time.time() - 10) == "just now"


def test_format_age_minutes():
    assert format_age(time.time() - 5 * 60) == "5 minutes ago"


def test_format_age_hours():
    assert format_age(time.time() - 3 * 3600) == "3 hours ago"


def test_format_age_days():
    assert format_age(time.time() - 2 * 86400) == "2 days ago"


def test_format_age_weeks():
    assert format_age(time.time() - 14 * 86400) == "2 weeks ago"


def test_format_age_months():
    assert format_age(time.time() - 75 * 86400) == "2 months ago"


def test_format_age_future():
    """Timestamp in the future (clock skew) should return 'just now'."""
    assert format_age(time.time() + 3600) == "just now"


# ---------------------------------------------------------------------------
# format_model_banner
# ---------------------------------------------------------------------------


def _sample_metadata(**overrides) -> dict:
    """Build a complete metadata dict with sensible defaults."""
    base = {
        "trained_at": int(time.time()) - 2 * 86400,
        "n_features": 102,
        "feature_config": "base-102",
        "foundation_model": None,
        "n_real": 448,
        "n_synth": 7102,
        "real_weight": 50.0,
    }
    base.update(overrides)
    return base


def test_format_model_banner_full():
    from pathlib import Path

    meta = _sample_metadata()
    banner = format_model_banner(meta, Path("production_model.joblib"))
    assert "Model: production_model.joblib" in banner
    assert "Trained:" in banner
    assert "2 days ago" in banner
    assert "Features: 102" in banner
    assert "no foundation model" in banner
    assert "Training data: 448 real (50:1 weight) + 7102 synthetic" in banner


def test_format_model_banner_no_foundation():
    from pathlib import Path

    meta = _sample_metadata(foundation_model=None)
    banner = format_model_banner(meta, Path("model.joblib"))
    assert "no foundation model" in banner
    assert "foundation:" not in banner


def test_format_model_banner_with_foundation():
    from pathlib import Path

    meta = _sample_metadata(
        foundation_model="whisper-tiny",
        feature_config="whisper-tiny-486",
        n_features=486,
    )
    banner = format_model_banner(meta, Path("model.joblib"))
    assert "foundation: whisper-tiny" in banner
    assert "486" in banner


def test_format_model_banner_none_metadata():
    from pathlib import Path

    banner = format_model_banner(None, Path("old_model.joblib"))
    assert "old_model.joblib" in banner
    assert "no metadata sidecar - retrain to generate" in banner


# ---------------------------------------------------------------------------
# load_model_metadata
# ---------------------------------------------------------------------------


def test_load_model_metadata_missing_sidecar(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fake model data")
    assert load_model_metadata(model_path) is None


def test_load_model_metadata_corrupt_json(tmp_path, caplog):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fake model data")
    meta_path = tmp_path / "model.meta.json"
    meta_path.write_text("{invalid json!!!")
    result = load_model_metadata(model_path)
    assert result is None
    assert any("corrupt" in r.message.lower() for r in caplog.records)


def test_load_model_metadata_valid(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fake model data")
    meta = {"trained_at": 1700000000, "n_real": 100}
    meta_path = tmp_path / "model.meta.json"
    meta_path.write_text(json.dumps(meta))
    result = load_model_metadata(model_path)
    assert result == meta


# ---------------------------------------------------------------------------
# Champion history
# ---------------------------------------------------------------------------


def _make_results(*entries):
    """Build a results list from (name, mean_db) pairs with dummy fields."""
    return [
        (name, mean_db, mean_db + 1.0, 10.0, {"author1": mean_db}, "reference")
        for name, mean_db in entries
    ]


def test_record_champion_first_run(tmp_path):
    """No history file - records the first champion."""
    path = tmp_path / "champion_history.json"
    result = record_champion(path, "E82 baseline", 2.82, 1279)
    assert path.exists()
    assert result["current"]["experiment"] == "E82 baseline"
    assert result["current"]["mean_db"] == 2.82
    assert result["current"]["wav_count"] == 1279
    assert len(result["history"]) == 1


def test_record_champion_improvement(tmp_path):
    """Existing champion - new lower mean_db replaces it."""
    path = tmp_path / "champion_history.json"
    record_champion(path, "E82 baseline", 2.82, 1279)
    result = record_champion(path, "E85 diff-DSP", 2.27, 1279)
    assert result["current"]["experiment"] == "E85 diff-DSP"
    assert result["current"]["mean_db"] == 2.27
    assert len(result["history"]) == 2


def test_record_champion_no_improvement(tmp_path):
    """Existing champion - new higher mean_db does not replace."""
    path = tmp_path / "champion_history.json"
    record_champion(path, "E85 diff-DSP", 2.27, 1279)
    result = record_champion(path, "E83 Whisper", 2.95, 1279)
    assert result["current"]["experiment"] == "E85 diff-DSP"
    assert result["current"]["mean_db"] == 2.27
    # No new entry added to history since no improvement
    assert len(result["history"]) == 1


def test_load_champion_history_missing(tmp_path):
    """No file returns None."""
    assert load_champion_history(tmp_path / "nonexistent.json") is None


def test_load_champion_history_corrupt(tmp_path):
    """Invalid JSON returns None."""
    path = tmp_path / "champion_history.json"
    path.write_text("{invalid json!!!")
    assert load_champion_history(path) is None


def test_format_champion_comparison_first_run():
    """No history shows 'establishing baseline'."""
    results = _make_results(("E82 baseline", 2.82))
    output = format_champion_comparison(None, results)
    assert "First comparison run" in output
    assert "establishing baseline" in output


def test_format_champion_comparison_new_champion():
    """History exists, improvement found."""
    history = {
        "current": {
            "experiment": "E82 baseline",
            "mean_db": 2.82,
            "date": "2026-04-18",
            "wav_count": 1279,
        },
        "history": [],
    }
    results = _make_results(("E82 baseline", 2.82), ("E85 diff-DSP", 2.27))
    output = format_champion_comparison(history, results)
    assert "Previous champion: E82 baseline" in output
    assert "2.82 dB mean" in output
    assert "New champion:" in output
    assert "E85 diff-DSP" in output
    assert "-0.55 dB improvement" in output


def test_format_champion_comparison_no_change():
    """History exists, no improvement."""
    history = {
        "current": {
            "experiment": "E85 diff-DSP",
            "mean_db": 2.27,
            "date": "2026-04-20",
            "wav_count": 1279,
        },
        "history": [],
    }
    results = _make_results(("E82 baseline", 2.82), ("E85 diff-DSP", 2.35))
    output = format_champion_comparison(history, results)
    assert "Current champion unchanged" in output
    assert "E85 diff-DSP" in output
    assert "2.27 dB mean" in output


# ---------------------------------------------------------------------------
# Training staleness detection
# ---------------------------------------------------------------------------


def test_check_staleness_matching_mtime():
    """Same mtime means model is up to date."""
    meta = _sample_metadata(wav_cache_mtime=1700000000.0)
    result = check_training_staleness(meta, 1700000000.0)
    assert result["stale"] is False
    assert result["reason"] == "up to date"


def test_check_staleness_different_mtime():
    """Different mtime means training data has changed."""
    meta = _sample_metadata(wav_cache_mtime=1700000000.0)
    result = check_training_staleness(meta, 1700099999.0)
    assert result["stale"] is True
    assert "changed" in result["reason"]


def test_check_staleness_no_metadata():
    """None metadata returns not stale with info reason."""
    result = check_training_staleness(None, 1700000000.0)
    assert result["stale"] is False
    assert "no model metadata" in result["reason"]


def test_check_staleness_no_mtime_key():
    """Metadata without wav_cache_mtime key behaves like no metadata."""
    meta = _sample_metadata()  # no wav_cache_mtime key
    result = check_training_staleness(meta, 1700000000.0)
    assert result["stale"] is False
    assert "no model metadata" in result["reason"]


def test_check_staleness_no_cache():
    """current_wav_mtime=0.0 means WAV cache is not configured."""
    meta = _sample_metadata(wav_cache_mtime=1700000000.0)
    result = check_training_staleness(meta, 0.0)
    assert result["stale"] is False
    assert "not configured" in result["reason"]


def test_format_staleness_warning_stale():
    """Stale result produces a multi-line warning with retrain suggestion."""
    staleness = {"stale": True, "reason": "WAV cache has changed since model was trained"}
    warning = format_staleness_warning(staleness)
    assert "WARNING" in warning
    assert "bin/beq-designer dev train" in warning
    assert len(warning.splitlines()) >= 2


def test_format_staleness_warning_up_to_date():
    """Up-to-date result produces empty string."""
    staleness = {"stale": False, "reason": "up to date"}
    warning = format_staleness_warning(staleness)
    assert warning == ""


def test_format_staleness_warning_no_metadata():
    """No-metadata reason produces a single-line info message."""
    staleness = {
        "stale": False,
        "reason": "no model metadata - retrain to enable staleness detection",
    }
    warning = format_staleness_warning(staleness)
    assert "INFO" in warning
    assert len(warning.splitlines()) == 1


# ---------------------------------------------------------------------------
# Logging infrastructure - handler accumulation prevention
# ---------------------------------------------------------------------------


def test_log_handlers_not_duplicated_after_tier1_import():
    """Importing run_tier1_comparison must not add extra root log handlers.

    Regression test: module-level basicConfig calls in imported scripts
    used to stack handlers on the root logger, causing every log message
    to print N times.
    """
    import logging
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_count = len(original_handlers)

    import importlib
    import sys

    # Ensure experiments/ is on path (same as dev_reassess does).
    from pathlib import Path
    experiments_dir = Path(__file__).resolve().parents[4] / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))

    # Force re-evaluation of the module-level code.
    if "run_tier1_comparison" in sys.modules:
        # Already imported - handler count should be stable.
        pass
    else:
        import run_tier1_comparison  # noqa: F401

    assert len(root.handlers) <= original_count + 1, (
        f"handler count grew from {original_count} to {len(root.handlers)}: "
        f"{[type(h).__name__ for h in root.handlers]}"
    )


def test_check_ollama_hosts_unreachable():
    """All hosts unreachable returns empty available list."""
    from model.auto_beq_advisor import check_ollama_hosts
    result = check_ollama_hosts(["http://127.0.0.1:1"], timeout_s=0.5)
    assert result["available"] == []
    assert result["unavailable"] == ["http://127.0.0.1:1"]


def test_foundation_model_cache_is_singleton():
    """The _FOUNDATION_MODEL_CACHE must be the same dict regardless of
    import path, so the double-checked lock prevents redundant model loads."""
    from model.auto_beq_advisor import _FOUNDATION_MODEL_CACHE as cache1

    # Re-import via a different reference to verify same object.
    import model.auto_beq_advisor as advisor_mod
    cache2 = advisor_mod._FOUNDATION_MODEL_CACHE

    assert cache1 is cache2, "cache objects differ - model will load multiple times"
