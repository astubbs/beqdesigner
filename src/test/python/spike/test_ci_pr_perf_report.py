"""Unit tests for ``scripts/ci_pr_perf_report.py``.

The CI perf-report script is pure stdlib and runs on the grumpy
runner host outside Poetry, so we load it as a module directly from
its path instead of importing via a package. That keeps it testable
without adding a fake ``scripts/`` package layout.

Covers:

- pytest summary-line parsing (passed / failed / skipped / duration)
- CSV aggregation for both ``loss_db`` (F/G/H/I experiments) and
  ``mean_err_db`` (library sweep) column conventions
- Verdict counting (PASS / MARGINAL / FAIL)
- End-to-end CLI happy path via ``main()``
- Baseline-delta formatting
- Graceful handling of missing log files, empty CSVs, and malformed rows
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci_pr_perf_report.py"


def _load_script():
    """Load the bare script file as a module.

    Dataclass-defining modules loaded via ``spec_from_file_location``
    trip a Python 3.11 gotcha (`AttributeError: 'NoneType' object has
    no attribute '__dict__'` inside ``dataclasses._is_type``) unless
    the module is registered in ``sys.modules`` *before*
    ``exec_module`` runs — so do that explicitly here.
    """
    spec = importlib.util.spec_from_file_location("ci_pr_perf_report", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ci_pr_perf_report"] = mod
    spec.loader.exec_module(mod)
    return mod


perf = _load_script()


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".pytest_cache"
    d.mkdir()
    return d


def _write_log(path: Path, summary_line: str) -> None:
    path.write_text(
        "collecting ...\n"
        "test_something PASSED\n"
        "test_another PASSED\n"
        f"{summary_line}\n"
    )


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r))
    path.write_text("\n".join(lines) + "\n")


# --- parse_spike_log --------------------------------------------------------


def test_parse_spike_log_passed_only(cache_dir: Path) -> None:
    log = cache_dir / "spike_tests.log"
    _write_log(log, "===== 42 passed in 12.34s =====")
    status = perf.parse_spike_log(log, "spike-unit")
    assert status.name == "spike-unit"
    assert status.passed == 42
    assert status.failed == 0
    assert status.errors == 0
    assert status.skipped == 0
    assert status.duration_s == pytest.approx(12.34)
    assert status.present is True
    assert status.status_label == "OK"


def test_parse_spike_log_mixed_counts(cache_dir: Path) -> None:
    log = cache_dir / "spike_integration.log"
    _write_log(log, "===== 5 failed, 10 passed, 3 skipped in 90.10s =====")
    status = perf.parse_spike_log(log, "spike-integration")
    assert status.passed == 10
    assert status.failed == 5
    assert status.skipped == 3
    assert status.duration_s == pytest.approx(90.10)
    assert status.status_label == "FAIL"


def test_parse_spike_log_errors_counted(cache_dir: Path) -> None:
    log = cache_dir / "spike_experiments.log"
    _write_log(log, "===== 2 errors, 1 passed in 5.0s =====")
    status = perf.parse_spike_log(log, "spike-experiments")
    assert status.errors == 2
    assert status.passed == 1
    assert status.status_label == "FAIL"


def test_parse_spike_log_missing_file(cache_dir: Path) -> None:
    status = perf.parse_spike_log(cache_dir / "nope.log", "spike-unit")
    assert status.present is False
    assert status.passed == 0
    assert status.duration_s == 0.0
    assert status.status_label == "missing"


def test_parse_spike_log_no_summary_line(cache_dir: Path) -> None:
    log = cache_dir / "spike_tests.log"
    log.write_text("collecting ...\ntest_foo PASSED\n")  # no final summary
    status = perf.parse_spike_log(log, "spike-unit")
    assert status.present is True
    assert status.passed == 0  # no summary = nothing to parse
    assert status.status_label == "no run"


# --- parse_metric_csv -------------------------------------------------------


def test_parse_metric_csv_loss_db_column(cache_dir: Path) -> None:
    csv_path = cache_dir / "auto_beq_f_experiments.csv"
    _write_csv(
        csv_path,
        ["experiment", "title", "loss_db", "verdict"],
        [
            ["Baseline", "A", "1.5", "PASS"],
            ["Baseline", "B", "3.0", "MARGINAL"],
            ["Baseline", "C", "5.0", "FAIL"],
            ["Baseline", "D", "0.8", "PASS"],
        ],
    )
    m = perf.parse_metric_csv(csv_path)
    assert m is not None
    assert m.row_count == 4
    assert m.mean_loss_db == pytest.approx((1.5 + 3.0 + 5.0 + 0.8) / 4)
    assert m.max_loss_db == pytest.approx(5.0)
    assert m.pass_count == 2
    assert m.marginal_count == 1
    assert m.fail_count == 1


def test_parse_metric_csv_mean_err_db_column(cache_dir: Path) -> None:
    """Library sweep CSV uses ``mean_err_db``, not ``loss_db``."""
    csv_path = cache_dir / "auto_beq_sweep.csv"
    _write_csv(
        csv_path,
        ["title", "verdict", "mean_err_db", "max_err_db", "extraction"],
        [
            ["X", "PASS", "0.8", "1.1", "ok"],
            ["Y", "PASS", "1.2", "1.9", "ok"],
        ],
    )
    m = perf.parse_metric_csv(csv_path)
    assert m is not None
    assert m.row_count == 2
    assert m.mean_loss_db == pytest.approx(1.0)
    assert m.max_loss_db == pytest.approx(1.2)
    assert m.pass_count == 2
    assert m.marginal_count == 0
    assert m.fail_count == 0


def test_parse_metric_csv_empty_file(cache_dir: Path) -> None:
    csv_path = cache_dir / "auto_beq_empty.csv"
    csv_path.write_text("")
    m = perf.parse_metric_csv(csv_path)
    assert m is not None
    assert m.row_count == 0
    assert m.mean_loss_db is None
    assert m.max_loss_db is None


def test_parse_metric_csv_skips_unparseable_loss(cache_dir: Path) -> None:
    """Rows with a garbage loss value are just ignored by the mean."""
    csv_path = cache_dir / "auto_beq_weird.csv"
    _write_csv(
        csv_path,
        ["title", "loss_db", "verdict"],
        [
            ["A", "1.0", "PASS"],
            ["B", "NaN", "PASS"],   # NaN parses as a valid float in Python
            ["C", "oops", "PASS"],  # not a float — skipped
        ],
    )
    m = perf.parse_metric_csv(csv_path)
    assert m is not None
    assert m.row_count == 3
    assert m.pass_count == 3


# --- collect_csv_metrics ----------------------------------------------------


def test_collect_csv_metrics_only_auto_beq_prefix(cache_dir: Path) -> None:
    _write_csv(cache_dir / "auto_beq_a.csv", ["title", "loss_db", "verdict"], [["A", "1.0", "PASS"]])
    _write_csv(cache_dir / "auto_beq_b.csv", ["title", "loss_db", "verdict"], [["B", "2.0", "PASS"]])
    _write_csv(cache_dir / "unrelated.csv", ["title", "loss_db", "verdict"], [["C", "9.0", "FAIL"]])

    metrics = perf.collect_csv_metrics(cache_dir)
    assert set(metrics) == {"auto_beq_a.csv", "auto_beq_b.csv"}


def test_collect_csv_metrics_missing_dir(tmp_path: Path) -> None:
    assert perf.collect_csv_metrics(tmp_path / "nope") == {}


# --- format_report ----------------------------------------------------------


def test_format_report_no_data_emits_marker_and_message() -> None:
    buf = io.StringIO()
    perf.format_report([], {}, None, output=buf)
    out = buf.getvalue()
    assert perf.COMMENT_MARKER in out
    assert "grumpy CI" in out
    assert "No `.pytest_cache/auto_beq_" in out


def test_format_report_shows_suite_status_rows() -> None:
    suites = [
        perf.SuiteStatus(
            name="spike-unit", passed=42, failed=0, errors=0, skipped=1,
            duration_s=12.34, log_path=Path("x"), present=True,
        ),
        perf.SuiteStatus(
            name="spike-integration", passed=0, failed=0, errors=0, skipped=0,
            duration_s=0.0, log_path=Path("y"), present=False,
        ),
    ]
    buf = io.StringIO()
    perf.format_report(suites, {}, None, output=buf)
    out = buf.getvalue()
    assert "| spike-unit | OK | 42 | 0 | 0 | 1 | 12.3s |" in out
    assert "| spike-integration | missing |" in out


def test_format_report_with_baseline_includes_delta_column() -> None:
    current = {
        "auto_beq_f.csv": perf.CsvMetric(
            name="auto_beq_f.csv", row_count=10,
            mean_loss_db=2.00, max_loss_db=4.5,
            pass_count=6, marginal_count=3, fail_count=1,
        ),
    }
    baseline = {
        "auto_beq_f.csv": perf.CsvMetric(
            name="auto_beq_f.csv", row_count=10,
            mean_loss_db=2.50, max_loss_db=5.0,
            pass_count=4, marginal_count=4, fail_count=2,
        ),
    }
    buf = io.StringIO()
    perf.format_report([], current, baseline, output=buf)
    out = buf.getvalue()
    assert "Δ mean dB" in out
    assert "-0.50" in out  # improved by 0.50 dB
    # And the "no baseline" disclaimer is NOT present.
    assert "No baseline metrics available" not in out


def test_format_report_new_csv_not_in_baseline() -> None:
    """A CSV that only exists in the current run shows '(new)' for the delta.

    Uses a non-empty baseline containing a *different* CSV name — simulating
    "the PR added a new experiment that doesn't exist on main yet", which is
    the specific path that should render ``(new)`` in the delta column.
    """
    current = {
        "auto_beq_new.csv": perf.CsvMetric(
            name="auto_beq_new.csv", row_count=3,
            mean_loss_db=1.0, max_loss_db=1.5,
            pass_count=3, marginal_count=0, fail_count=0,
        ),
    }
    baseline = {
        "auto_beq_legacy.csv": perf.CsvMetric(
            name="auto_beq_legacy.csv", row_count=5,
            mean_loss_db=2.5, max_loss_db=4.0,
            pass_count=3, marginal_count=1, fail_count=1,
        ),
    }
    buf = io.StringIO()
    perf.format_report([], current, baseline, output=buf)
    out = buf.getvalue()
    assert "| Δ mean dB |" in out   # header row is present → baseline mode
    assert "(new)" in out            # auto_beq_new.csv is not in baseline


def test_format_report_commit_and_run_url_header() -> None:
    buf = io.StringIO()
    perf.format_report(
        [], {}, None,
        commit_sha="deadbeefcafefeed1234",
        run_url="https://github.com/astubbs/beqdesigner/actions/runs/42",
        output=buf,
    )
    out = buf.getvalue()
    assert "`deadbeefcafe`" in out   # truncated to 12 chars
    assert "runs/42" in out


# --- main() end-to-end ------------------------------------------------------


def test_main_writes_output_file(cache_dir: Path, tmp_path: Path) -> None:
    _write_log(cache_dir / "spike_tests.log", "===== 3 passed in 1.1s =====")
    _write_csv(
        cache_dir / "auto_beq_unit.csv",
        ["title", "loss_db", "verdict"],
        [["A", "1.2", "PASS"], ["B", "2.5", "MARGINAL"]],
    )
    out_path = tmp_path / "report.md"
    rc = perf.main([
        "--cache-dir", str(cache_dir),
        "--output", str(out_path),
        "--commit-sha", "deadbeefcafefeed",
        "--run-url", "https://example/runs/1",
    ])
    assert rc == 0
    body = out_path.read_text()
    assert perf.COMMENT_MARKER in body
    assert "spike-unit" in body
    assert "auto_beq_unit.csv" in body
    assert "`deadbeefcafe`" in body


def test_main_empty_cache_dir_still_writes_valid_markdown(tmp_path: Path) -> None:
    """Smoke-test the "workflow failed early, no artifacts" path."""
    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    out_path = tmp_path / "report.md"
    rc = perf.main([
        "--cache-dir", str(empty_cache),
        "--output", str(out_path),
    ])
    assert rc == 0
    body = out_path.read_text()
    assert perf.COMMENT_MARKER in body
    assert "No `.pytest_cache/auto_beq_" in body


def test_main_baseline_empty_dir_treated_as_no_baseline(
    cache_dir: Path, tmp_path: Path,
) -> None:
    _write_csv(
        cache_dir / "auto_beq_x.csv",
        ["title", "loss_db", "verdict"],
        [["A", "1.0", "PASS"]],
    )
    empty_baseline = tmp_path / "baseline-dir"
    empty_baseline.mkdir()  # dir exists but has no CSVs

    out_path = tmp_path / "report.md"
    rc = perf.main([
        "--cache-dir", str(cache_dir),
        "--baseline", str(empty_baseline),
        "--output", str(out_path),
    ])
    assert rc == 0
    body = out_path.read_text()
    # Should still be "no baseline" rather than showing an empty delta column.
    # Check for the column header specifically — the "no baseline" disclaimer
    # text itself mentions "Δ mean dB" in prose, so the bare substring isn't
    # discriminating enough.
    assert "| Δ mean dB |" not in body
    assert "No baseline metrics available" in body
