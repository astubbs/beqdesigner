"""Tests for cli/experiment_report.py - experiment comparison report logic."""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from cli.experiment_report import (
    ExperimentResult,
    GRADE_RANK,
    GRADE_SHORT,
    _load_baseline_sweep,
    _load_comparison_csv,
    load_all_results,
)


class TestExperimentResult:
    """ExperimentResult dataclass properties."""

    def test_avg_delta_with_data(self):
        r = ExperimentResult(experiment="E1", config="c1", total_delta=6.0, count=3)
        assert r.avg_delta == pytest.approx(2.0)

    def test_avg_delta_empty(self):
        r = ExperimentResult(experiment="E1", config="c1")
        assert r.avg_delta == 0.0

    def test_net_grade(self):
        r = ExperimentResult(experiment="E1", config="c1", improvements=5, degradations=2)
        assert r.net_grade == 3

    def test_net_grade_negative(self):
        r = ExperimentResult(experiment="E1", config="c1", improvements=1, degradations=4)
        assert r.net_grade == -3

    def test_label_with_different_config(self):
        r = ExperimentResult(experiment="E18", config="60s")
        assert r.label == "E18/60s"

    def test_label_same_config(self):
        r = ExperimentResult(experiment="baseline", config="baseline")
        assert r.label == "baseline"

    def test_label_empty_config(self):
        r = ExperimentResult(experiment="E1", config="")
        assert r.label == "E1"


class TestLoadBaselineSweep:
    """Loading baseline CSV data."""

    def test_loads_baseline_csv(self, tmp_path):
        csv_path = tmp_path / "auto_beq_sweep.csv"
        csv_path.write_text(textwrap.dedent("""\
            title,verdict,mean_err_db,max_err_db,extraction
            Avatar,PASS,1.2,3.4,welch
            Inception,MARGINAL,4.5,8.1,welch
            Dune,FAIL,9.2,15.3,welch
        """))
        result = _load_baseline_sweep(tmp_path)
        assert result is not None
        assert result.verdicts["PASS"] == 1
        assert result.verdicts["MARGINAL"] == 1
        assert result.verdicts["FAIL"] == 1
        assert result.count == 3
        assert "Avatar" in result.per_title
        assert result.per_title["Avatar"]["verdict"] == "PASS"
        assert result.per_title["Avatar"]["mean"] == pytest.approx(1.2)

    def test_returns_none_when_missing(self, tmp_path):
        assert _load_baseline_sweep(tmp_path) is None


class TestLoadComparisonCsv:
    """Loading comparison CSVs with baseline vs test columns."""

    def _write_csv(self, path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    def test_loads_comparison_with_improvements(self, tmp_path):
        csv_path = tmp_path / "comparison.csv"
        self._write_csv(csv_path, [
            {"title": "Avatar", "config": "60s", "baseline_verdict": "FAIL",
             "test_verdict": "PASS", "baseline_mean": "9.0", "test_mean": "1.5", "mean_delta": "-7.5"},
            {"title": "Dune", "config": "60s", "baseline_verdict": "PASS",
             "test_verdict": "PASS", "baseline_mean": "1.2", "test_mean": "1.0", "mean_delta": "-0.2"},
        ])
        results = _load_comparison_csv(csv_path, "E18", "config")
        assert len(results) == 1
        r = results[0]
        assert r.experiment == "E18"
        assert r.config == "60s"
        assert r.improvements == 1  # FAIL -> PASS
        assert r.degradations == 0
        assert r.same == 1  # PASS -> PASS
        assert r.count == 2

    def test_multiple_configs(self, tmp_path):
        csv_path = tmp_path / "comparison.csv"
        self._write_csv(csv_path, [
            {"title": "Avatar", "config": "30s", "baseline_verdict": "PASS",
             "test_verdict": "FAIL", "baseline_mean": "1.0", "test_mean": "9.0", "mean_delta": "8.0"},
            {"title": "Avatar", "config": "60s", "baseline_verdict": "PASS",
             "test_verdict": "PASS", "baseline_mean": "1.0", "test_mean": "0.8", "mean_delta": "-0.2"},
        ])
        results = _load_comparison_csv(csv_path, "E18", "config")
        assert len(results) == 2
        configs = {r.config for r in results}
        assert configs == {"30s", "60s"}

    def test_degradation_counted(self, tmp_path):
        csv_path = tmp_path / "comparison.csv"
        self._write_csv(csv_path, [
            {"title": "Avatar", "config": "c1", "baseline_verdict": "PASS",
             "test_verdict": "FAIL", "baseline_mean": "1.0", "test_mean": "12.0", "mean_delta": "11.0"},
        ])
        results = _load_comparison_csv(csv_path, "E1", "config")
        assert results[0].degradations == 1
        assert results[0].improvements == 0

    def test_returns_empty_when_missing(self, tmp_path):
        assert _load_comparison_csv(tmp_path / "nope.csv", "E1", "config") == []

    def test_all_grades_same(self, tmp_path):
        csv_path = tmp_path / "comparison.csv"
        self._write_csv(csv_path, [
            {"title": "A", "config": "c1", "baseline_verdict": "PASS",
             "test_verdict": "PASS", "baseline_mean": "1.0", "test_mean": "1.1", "mean_delta": "0.1"},
            {"title": "B", "config": "c1", "baseline_verdict": "FAIL",
             "test_verdict": "FAIL", "baseline_mean": "10.0", "test_mean": "9.5", "mean_delta": "-0.5"},
        ])
        results = _load_comparison_csv(csv_path, "E1", "config")
        r = results[0]
        assert r.improvements == 0
        assert r.degradations == 0
        assert r.same == 2


class TestLoadAllResults:
    """Integration: load_all_results with a populated CSV directory."""

    def test_loads_baseline_only(self, tmp_path):
        (tmp_path / "auto_beq_sweep.csv").write_text(textwrap.dedent("""\
            title,verdict,mean_err_db,max_err_db
            Avatar,PASS,1.2,3.4
        """))
        results = load_all_results(tmp_path)
        assert len(results) == 1
        assert results[0].experiment == "baseline"

    def test_empty_dir_returns_empty(self, tmp_path):
        assert load_all_results(tmp_path) == []


class TestGradeConstants:
    """Grade ranking and short-name maps."""

    def test_pass_ranks_lower_than_fail(self):
        assert GRADE_RANK["PASS"] < GRADE_RANK["FAIL"]

    def test_short_names(self):
        assert GRADE_SHORT["PASS"] == "P"
        assert GRADE_SHORT["MARGINAL"] == "M"
        assert GRADE_SHORT["FAIL"] == "F"
