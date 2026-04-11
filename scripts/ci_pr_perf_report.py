#!/usr/bin/env python3
"""Compact markdown summary of a grumpy CI run, for PR comments.

Parses what the local-integration workflow leaves behind in
``.pytest_cache/`` after running the full spike + experiment suite,
and emits a GitHub-flavoured markdown snippet suitable for posting
as a PR comment:

- ``spike_*.log`` files → pytest summary lines (pass/fail/skip counts
  + duration per suite).
- ``auto_beq_*.csv`` files → aggregate mean/max ``loss_db`` (or
  ``mean_err_db`` for the library sweep) + PASS/MARGINAL/FAIL verdict
  counts.

If ``--baseline`` points to a directory of CSVs from a previous run
(restored by the workflow from the Actions cache, keyed on the
latest successful ``main`` build), the per-CSV metric table gains a
"Δ mean dB" column so reviewers can see at a glance whether the PR
regressed or improved the auto-BEQ model.

Deliberately **stdlib-only**. Runs from the grumpy runner host with
whatever ``python3`` is on PATH — no poetry, no scipy, no Docker
shell-out. Matches the ``nn_f_experiment_report.py`` aggregation
style but stops short of per-title tables so the PR comment stays
short enough to scan.

Usage::

    python3 scripts/ci_pr_perf_report.py \\
        --cache-dir .pytest_cache \\
        --output ci-perf-report.md \\
        --commit-sha "$GITHUB_SHA" \\
        --run-url "$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

    # With a baseline restored from cache:
    python3 scripts/ci_pr_perf_report.py \\
        --cache-dir .pytest_cache \\
        --baseline ci-perf-baseline \\
        --output ci-perf-report.md
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Hidden HTML marker used by the workflow's PR-comment step to
# find-and-update the previous comment rather than appending a new
# one on every run. Keep in sync with
# .github/workflows/local-integration.yml.
COMMENT_MARKER = "<!-- grumpy-perf-report -->"

# Friendly names for each suite's log file. Ordered from cheapest to
# most expensive so the report reads top-down as the run progresses.
SUITE_LOGS: list[tuple[str, str]] = [
    ("spike-unit", "spike_tests.log"),
    ("spike-integration", "spike_integration.log"),
    ("spike-experiments", "spike_experiments.log"),
]


@dataclass
class SuiteStatus:
    """Pytest counts + duration extracted from a ``spike_*.log`` file."""

    name: str
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_s: float
    log_path: Path
    present: bool  # False if the log file didn't exist at all

    @property
    def status_label(self) -> str:
        if not self.present:
            return "missing"
        if self.failed or self.errors:
            return "FAIL"
        if self.passed == 0 and self.skipped == 0:
            return "no run"
        return "OK"


@dataclass
class CsvMetric:
    """Aggregated mean/max/verdicts for a single experiment CSV file."""

    name: str                          # basename (e.g. auto_beq_f_experiments.csv)
    row_count: int
    mean_loss_db: float | None
    max_loss_db: float | None
    pass_count: int
    marginal_count: int
    fail_count: int


# --- parsing ----------------------------------------------------------------


def parse_spike_log(path: Path, suite_name: str) -> SuiteStatus:
    """Return a :class:`SuiteStatus` for one ``spike_*.log`` file.

    Scans for the **last** pytest summary line in the file (pytest
    writes "===== N passed, M failed, ... in X.XXs =====" once per run,
    and the file only contains one run's output at a time, but the
    regex is forgiving in case future runners append summaries).
    """
    if not path.exists():
        return SuiteStatus(
            name=suite_name, passed=0, failed=0, errors=0, skipped=0,
            duration_s=0.0, log_path=path, present=False,
        )

    summary_line: str | None = None
    with path.open(errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("=") and " in " in stripped and stripped.endswith("="):
                # Typical shape:
                #   ===== 42 passed, 3 skipped in 12.34s =====
                #   ===== 5 failed, 10 passed in 120.5s =====
                #   ===== no tests ran in 0.12s =====
                summary_line = stripped

    if summary_line is None:
        return SuiteStatus(
            name=suite_name, passed=0, failed=0, errors=0, skipped=0,
            duration_s=0.0, log_path=path, present=True,
        )

    def _count(keyword_regex: str) -> int:
        m = re.search(rf"(\d+)\s+{keyword_regex}", summary_line)
        return int(m.group(1)) if m else 0

    dur_m = re.search(r"in\s+([\d.]+)s", summary_line)
    duration = float(dur_m.group(1)) if dur_m else 0.0

    return SuiteStatus(
        name=suite_name,
        passed=_count("passed"),
        failed=_count("failed"),
        errors=_count("errors?"),
        skipped=_count("skipped"),
        duration_s=duration,
        log_path=path,
        present=True,
    )


# Column-name aliases for "the number we care about". F/G/H/I
# experiments write ``loss_db``; the library sweep writes
# ``mean_err_db``; older NN reports sometimes use ``loss``.
_LOSS_COLUMNS = ("loss_db", "mean_err_db", "mean_loss_db", "loss")


def _loss_from_row(row: dict) -> float | None:
    for key in _LOSS_COLUMNS:
        val = row.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def parse_metric_csv(path: Path) -> CsvMetric | None:
    """Aggregate one experiment CSV. Returns ``None`` only on I/O error."""
    try:
        with path.open(errors="replace") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None

    losses: list[float] = []
    pc = mc = fc = 0
    for row in rows:
        loss = _loss_from_row(row)
        if loss is not None:
            losses.append(loss)
        verdict = (row.get("verdict") or "").strip().upper()
        if verdict == "PASS":
            pc += 1
        elif verdict == "MARGINAL":
            mc += 1
        elif verdict == "FAIL":
            fc += 1

    return CsvMetric(
        name=path.name,
        row_count=len(rows),
        mean_loss_db=(sum(losses) / len(losses)) if losses else None,
        max_loss_db=max(losses) if losses else None,
        pass_count=pc,
        marginal_count=mc,
        fail_count=fc,
    )


def collect_csv_metrics(cache_dir: Path) -> dict[str, CsvMetric]:
    """Return ``{csv_name: CsvMetric}`` for every ``auto_beq_*.csv`` in ``cache_dir``."""
    if not cache_dir.exists():
        return {}
    out: dict[str, CsvMetric] = {}
    for csv_path in sorted(cache_dir.glob("auto_beq_*.csv")):
        m = parse_metric_csv(csv_path)
        if m is not None:
            out[m.name] = m
    return out


# --- formatting -------------------------------------------------------------


def _fmt_float(val: float | None, places: int = 2) -> str:
    return f"{val:.{places}f}" if val is not None else "—"


def _fmt_delta(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "(new)"
    return f"{current - baseline:+.2f}"


def format_report(
    suites: list[SuiteStatus],
    current: dict[str, CsvMetric],
    baseline: dict[str, CsvMetric] | None,
    *,
    commit_sha: str | None = None,
    run_url: str | None = None,
    output=sys.stdout,
) -> None:
    """Write the full markdown report to ``output``."""
    w = lambda s="": print(s, file=output)

    w(COMMENT_MARKER)
    w("## grumpy CI: full spike + experiment suite")
    w()
    meta: list[str] = []
    if commit_sha:
        meta.append(f"**Commit**: `{commit_sha[:12]}`")
    if run_url:
        meta.append(f"**Run**: {run_url}")
    if meta:
        w("  \n".join(meta))
        w()

    # --- Suite status ---
    if suites:
        w("### Suite status")
        w()
        w("| Suite | Status | Passed | Failed | Errors | Skipped | Duration |")
        w("|---|---|---|---|---|---|---|")
        for s in suites:
            dur = f"{s.duration_s:.1f}s" if s.duration_s else "—"
            w(
                f"| {s.name} | {s.status_label} | {s.passed} | "
                f"{s.failed} | {s.errors} | {s.skipped} | {dur} |"
            )
        w()

    # --- Experiment metrics ---
    w("### Experiment metrics")
    w()
    if not current:
        w("_No `.pytest_cache/auto_beq_*.csv` files produced by this run._")
        w()
    else:
        has_baseline = bool(baseline)
        headers = ["CSV", "Titles", "Mean dB", "Max dB", "PASS", "MARG", "FAIL"]
        if has_baseline:
            headers.append("Δ mean dB")
        w("| " + " | ".join(headers) + " |")
        w("|" + "|".join(["---"] * len(headers)) + "|")
        for name in sorted(current):
            m = current[name]
            row = [
                f"`{name}`",
                str(m.row_count),
                _fmt_float(m.mean_loss_db),
                _fmt_float(m.max_loss_db),
                str(m.pass_count),
                str(m.marginal_count),
                str(m.fail_count),
            ]
            if has_baseline:
                b = baseline.get(name) if baseline else None
                row.append(_fmt_delta(m.mean_loss_db, b.mean_loss_db if b else None))
            w("| " + " | ".join(row) + " |")
        w()

        if not has_baseline:
            w(
                "_No baseline metrics available for this run — this is either "
                "the first run of this workflow, or the cache restore missed. "
                "A Δ mean dB column will appear once a baseline is populated "
                "by a successful run on `main`._"
            )
            w()

    w("---")
    w(
        "_Generated by `scripts/ci_pr_perf_report.py`. Full spike logs are "
        "attached to the run as the `spike-logs-<sha>` artifact._"
    )


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate a compact markdown summary of a grumpy CI run "
                    "for posting as a PR comment.",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=Path(".pytest_cache"),
        help="Directory containing spike_*.log + auto_beq_*.csv (default: .pytest_cache)",
    )
    ap.add_argument(
        "--baseline", type=Path, default=None,
        help="Directory with baseline auto_beq_*.csv files to diff against",
    )
    ap.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Markdown output file (default: stdout)",
    )
    ap.add_argument("--commit-sha", default=None)
    ap.add_argument("--run-url", default=None)
    args = ap.parse_args(argv)

    suites = [
        parse_spike_log(args.cache_dir / log_name, suite_name)
        for suite_name, log_name in SUITE_LOGS
    ]

    current = collect_csv_metrics(args.cache_dir)
    baseline = None
    if args.baseline is not None and args.baseline.exists():
        baseline = collect_csv_metrics(args.baseline)
        # Empty baseline dir is treated the same as "no baseline".
        if not baseline:
            baseline = None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            format_report(
                suites, current, baseline,
                commit_sha=args.commit_sha, run_url=args.run_url, output=f,
            )
    else:
        format_report(
            suites, current, baseline,
            commit_sha=args.commit_sha, run_url=args.run_url, output=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
