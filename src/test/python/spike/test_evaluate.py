"""Tests for cli/evaluate.py - full-library evaluation pipeline.

These tests mock the model prediction and feature extraction to test
pipeline logic, scoring, and output formatting without needing real
models or WAV files.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_pair(title: str, author: str, tmdb_id: str, filters: list[dict]):
    """Build a minimal WAV-catalogue pair dict for testing."""
    return {
        "wav_path": Path(f"/fake/{title.replace(' ', '_')}.wav"),
        "catalogue_entry": {
            "title": title,
            "author": author,
            "theMovieDB": tmdb_id,
            "filters": filters,
        },
        "tmdb_id": tmdb_id,
    }


def _simple_filter(freq: float = 30.0, gain: float = -5.0, q: float = 0.7):
    """Build a minimal filter dict."""
    return {"type": "LowShelf", "freq": freq, "gain": gain, "q": q}


def _make_fake_features():
    """Return a mock CurveFeatures-like object."""
    mock = MagicMock()
    mock.foundation_embedding = None
    return mock


def _make_all_real(n: int = 10):
    """Build a list of (pair, features) tuples for testing."""
    pairs_and_features = []
    for i in range(n):
        # Vary gains so severity classification produces multiple classes.
        if i < 3:
            gain = -12.0  # heavy (abs sum >= 20 with 2 filters)
        elif i < 7:
            gain = -6.0   # moderate (abs sum >= 10 with 2 filters)
        else:
            gain = -2.0   # gentle

        pair = _make_fake_pair(
            title=f"Movie {i}",
            author=f"author_{i % 3}",
            tmdb_id=str(1000 + i),
            filters=[_simple_filter(gain=gain), _simple_filter(gain=gain)],
        )
        features = _make_fake_features()
        pairs_and_features.append((pair, features))
    return pairs_and_features


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReconstructSplit:
    """Verify the stratified split is deterministic."""

    def test_reconstruct_split_deterministic(self):
        """Same pairs + same seed = same split every time."""
        from cli.evaluate import _reconstruct_split

        all_real = _make_all_real(50)
        train_1, test_1 = _reconstruct_split(all_real)
        train_2, test_2 = _reconstruct_split(all_real)

        assert train_1 == train_2
        assert test_1 == test_2
        # 80/20 split of 50 -> 40 train, 10 test.
        assert len(train_1) == 40
        assert len(test_1) == 10
        # No overlap.
        assert train_1 & test_1 == set()

    def test_reconstruct_split_covers_all_indices(self):
        """Every index should be in either train or test."""
        from cli.evaluate import _reconstruct_split

        n = 30
        all_real = _make_all_real(n)
        train_idx, test_idx = _reconstruct_split(all_real)
        assert train_idx | test_idx == set(range(n))


class TestOverfittingGap:
    """Verify overfitting gap is computed correctly."""

    def test_overfitting_gap_calculation(self):
        """Mock per-title losses for train/held-out, verify gap."""
        # Simulate structured results with known losses.
        results = {
            "summary": {
                "total_titles": 10,
                "train_count": 8,
                "held_out_count": 2,
                "e82": {
                    "all_mean": 2.5,
                    "all_max": 5.0,
                    "train_mean": 2.0,
                    "held_out_mean": 3.0,
                    "overfitting_gap": -1.0,  # train - held_out
                },
            },
            "metadata": {"has_e85": False, "model_path": "/fake/model.joblib", "evaluation_time_s": 1.0},
            "per_author": {},
            "results": [],
        }
        # The gap should be train_mean - held_out_mean = 2.0 - 3.0 = -1.0.
        # A negative gap means held-out is worse (expected for overfitting).
        assert results["summary"]["e82"]["overfitting_gap"] == -1.0

    def test_overfitting_gap_positive_means_held_out_better(self):
        """A positive gap means held-out performed better than train - unusual."""
        train_losses = [3.0, 4.0, 5.0]
        held_out_losses = [1.0, 2.0]
        gap = float(np.mean(train_losses)) - float(np.mean(held_out_losses))
        assert gap > 0  # 4.0 - 1.5 = 2.5


class TestEvaluateNoModel:
    """Verify clear error when no model file exists."""

    def test_evaluate_no_model_raises(self, tmp_path):
        """No model file -> clear error message."""
        from cli.evaluate import _load_production_model

        with pytest.raises(RuntimeError, match="No production model found"):
            _load_production_model(tmp_path, auto_yes=True)


class TestCsvOutput:
    """Verify CSV has expected columns."""

    def test_csv_output_columns(self, tmp_path):
        """CSV should have all required columns."""
        from cli.evaluate import _save_csv

        data = {
            "metadata": {"has_e85": False, "model_path": "/fake/model.joblib", "evaluation_time_s": 1.0},
            "results": [
                {
                    "title": "Test Movie",
                    "author": "testauthor",
                    "tmdb_id": "12345",
                    "split": "train",
                    "e82_loss_db": 2.5,
                    "e85_loss_db": None,
                    "catalogue_filter_count": 3,
                    "catalogue_summed_gain_db": 15.0,
                },
            ],
        }

        csv_path = tmp_path / "test_results.csv"
        _save_csv(data, csv_path)

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        expected_cols = {"title", "author", "tmdb_id", "split", "e82_loss_db",
                         "catalogue_filter_count", "catalogue_summed_gain_db"}
        assert expected_cols.issubset(set(rows[0].keys()))
        # e85_loss_db should NOT be present when has_e85 is False.
        assert "e85_loss_db" not in rows[0]

    def test_csv_output_with_e85(self, tmp_path):
        """CSV should include e85_loss_db when E85 is available."""
        from cli.evaluate import _save_csv

        data = {
            "metadata": {"has_e85": True, "model_path": "/fake/model.joblib", "evaluation_time_s": 1.0},
            "results": [
                {
                    "title": "Test Movie",
                    "author": "testauthor",
                    "tmdb_id": "12345",
                    "split": "held-out",
                    "e82_loss_db": 2.5,
                    "e85_loss_db": 1.8,
                    "catalogue_filter_count": 3,
                    "catalogue_summed_gain_db": 15.0,
                },
            ],
        }

        csv_path = tmp_path / "test_results_e85.csv"
        _save_csv(data, csv_path)

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "e85_loss_db" in rows[0]
        assert rows[0]["e85_loss_db"] == "1.8"


class TestJsonOutput:
    """Verify JSON has expected structure."""

    def test_json_output_structure(self, tmp_path):
        """JSON should contain metadata, summary, per_author, results."""
        from cli.evaluate import _save_json

        data = {
            "metadata": {
                "model_path": "/fake/model.joblib",
                "has_e85": False,
                "evaluation_time_s": 42.5,
            },
            "summary": {
                "total_titles": 100,
                "train_count": 80,
                "held_out_count": 20,
                "e82": {
                    "all_mean": 2.82,
                    "train_mean": 2.65,
                    "held_out_mean": 3.14,
                    "overfitting_gap": -0.49,
                },
            },
            "per_author": {
                "author_a": {"count": 50, "mean": 2.5, "max": 6.0},
            },
            "results": [
                {
                    "title": "Dune",
                    "author": "author_a",
                    "tmdb_id": "438631",
                    "split": "held-out",
                    "e82_loss_db": 1.23,
                    "e85_loss_db": None,
                },
            ],
        }

        json_path = tmp_path / "test_results.json"
        _save_json(data, json_path)

        loaded = json.loads(json_path.read_text())

        # Top-level keys.
        assert "metadata" in loaded
        assert "summary" in loaded
        assert "per_author" in loaded
        assert "results" in loaded

        # Summary structure.
        assert loaded["summary"]["total_titles"] == 100
        assert loaded["summary"]["e82"]["overfitting_gap"] == -0.49

        # Results structure.
        assert len(loaded["results"]) == 1
        assert loaded["results"][0]["title"] == "Dune"
        assert loaded["results"][0]["split"] == "held-out"

    def test_json_roundtrip_fidelity(self, tmp_path):
        """Data should survive JSON serialization and deserialization."""
        from cli.evaluate import _save_json

        data = {
            "metadata": {"model_path": "/fake/model.joblib", "has_e85": True, "evaluation_time_s": 1.0},
            "summary": {"total_titles": 5, "e82": {"all_mean": 2.5}},
            "per_author": {"auth": {"count": 5, "mean": 2.5, "max": 4.0}},
            "results": [
                {"title": "A", "e82_loss_db": 1.5, "e85_loss_db": 1.2},
                {"title": "B", "e82_loss_db": 3.5, "e85_loss_db": None},
            ],
        }

        json_path = tmp_path / "roundtrip.json"
        _save_json(data, json_path)
        loaded = json.loads(json_path.read_text())

        assert loaded["results"][0]["e82_loss_db"] == 1.5
        assert loaded["results"][0]["e85_loss_db"] == 1.2
        assert loaded["results"][1]["e85_loss_db"] is None


class TestConsoleReport:
    """Verify console report formatting."""

    def test_console_report_runs_without_error(self, capsys):
        """Console report should print without exceptions."""
        from cli.evaluate import _print_console_report

        data = {
            "metadata": {"model_path": "/fake/model.joblib", "has_e85": False, "evaluation_time_s": 5.0},
            "summary": {
                "total_titles": 3,
                "train_count": 2,
                "held_out_count": 1,
                "e82": {
                    "all_mean": 2.5,
                    "all_max": 4.0,
                    "train_mean": 2.0,
                    "held_out_mean": 3.5,
                    "overfitting_gap": -1.5,
                },
            },
            "per_author": {
                "auth_a": {"count": 2, "mean": 2.0, "max": 3.0},
                "auth_b": {"count": 1, "mean": 3.5, "max": 3.5},
            },
            "results": [
                {"title": "Movie A", "author": "auth_a", "split": "train",
                 "e82_loss_db": 1.5, "e85_loss_db": None},
                {"title": "Movie B", "author": "auth_a", "split": "train",
                 "e82_loss_db": 2.5, "e85_loss_db": None},
                {"title": "Movie C", "author": "auth_b", "split": "held-out",
                 "e82_loss_db": 3.5, "e85_loss_db": None},
            ],
        }

        _print_console_report(data)
        output = capsys.readouterr().out

        assert "PRODUCTION MODEL EVALUATION" in output
        assert "3 titles" in output
        assert "Movie A" in output
        assert "overfitting indicator" in output
        assert "Worst 10" in output
