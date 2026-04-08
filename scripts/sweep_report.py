#!/usr/bin/env python3
"""Unified report generator for auto-BEQ experiment sweep results.

Reads all experiment CSV files from .pytest_cache/ and produces a
consolidated comparison report showing how each experiment/config
performs across the library.

Usage:
    python scripts/sweep_report.py
    python scripts/sweep_report.py --csv report.csv   # also write CSV
    python scripts/sweep_report.py --dir /path/to/csvs  # custom CSV dir

Reports generated:
    1. Per-experiment summary: P/M/F counts, grade changes, avg delta
    2. Per-title matrix: verdict for each experiment across all titles
    3. Historical baseline progression across experiments
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


GRADE_RANK = {"PASS": 0, "MARGINAL": 1, "FAIL": 2}
GRADE_SHORT = {"PASS": "P", "MARGINAL": "M", "FAIL": "F"}


@dataclass
class ExperimentResult:
    """Aggregated result for one experiment/config."""
    experiment: str
    config: str
    verdicts: dict[str, int] = field(default_factory=lambda: {"PASS": 0, "MARGINAL": 0, "FAIL": 0})
    baseline_verdicts: dict[str, int] = field(default_factory=lambda: {"PASS": 0, "MARGINAL": 0, "FAIL": 0})
    improvements: int = 0
    degradations: int = 0
    same: int = 0
    total_delta: float = 0.0
    count: int = 0
    per_title: dict[str, dict] = field(default_factory=dict)

    @property
    def avg_delta(self) -> float:
        return self.total_delta / self.count if self.count else 0.0

    @property
    def net_grade(self) -> int:
        return self.improvements - self.degradations

    @property
    def label(self) -> str:
        if self.config and self.config != self.experiment:
            return f"{self.experiment}/{self.config}"
        return self.experiment


def _load_baseline_sweep(csv_dir: Path) -> ExperimentResult | None:
    """Load the main library sweep (single-pass, current defaults)."""
    path = csv_dir / "auto_beq_sweep.csv"
    if not path.exists():
        return None
    result = ExperimentResult(experiment="baseline", config="current-defaults")
    with path.open() as f:
        for row in csv.DictReader(f):
            v = row["verdict"]
            result.verdicts[v] += 1
            result.baseline_verdicts[v] += 1
            result.count += 1
            title = row["title"]
            result.per_title[title] = {
                "verdict": v,
                "mean": float(row["mean_err_db"]),
                "max": float(row["max_err_db"]),
                "extraction": row.get("extraction", ""),
            }
    return result


def _load_comparison_csv(
    path: Path,
    experiment: str,
    config_col: str,
    baseline_verdict_col: str = "baseline_verdict",
    test_verdict_col: str = "test_verdict",
    baseline_mean_col: str = "baseline_mean",
    test_mean_col: str = "test_mean",
    delta_col: str = "mean_delta",
) -> list[ExperimentResult]:
    """Load a comparison CSV (baseline vs test columns)."""
    if not path.exists():
        return []

    by_config: dict[str, ExperimentResult] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            config = row.get(config_col, experiment)
            if config not in by_config:
                by_config[config] = ExperimentResult(experiment=experiment, config=config)
            r = by_config[config]

            bv = row[baseline_verdict_col]
            tv = row[test_verdict_col]
            r.verdicts[tv] = r.verdicts.get(tv, 0) + 1
            r.baseline_verdicts[bv] = r.baseline_verdicts.get(bv, 0) + 1

            br = GRADE_RANK.get(bv, 2)
            tr = GRADE_RANK.get(tv, 2)
            if tr < br:
                r.improvements += 1
            elif tr > br:
                r.degradations += 1
            else:
                r.same += 1

            delta = float(row.get(delta_col, "0"))
            r.total_delta += delta
            r.count += 1

            title = row.get("title", "")
            r.per_title[title] = {
                "baseline_verdict": bv,
                "test_verdict": tv,
                "baseline_mean": float(row.get(baseline_mean_col, "0")),
                "test_mean": float(row.get(test_mean_col, "0")),
                "delta": delta,
            }

    return list(by_config.values())


def load_all_results(csv_dir: Path) -> list[ExperimentResult]:
    """Load all experiment CSVs and return aggregated results."""
    results = []

    # Main baseline sweep
    baseline = _load_baseline_sweep(csv_dir)
    if baseline:
        results.append(baseline)

    # E18: chunked vs Welch
    results.extend(_load_comparison_csv(
        csv_dir / "auto_beq_sweep_chunked.csv",
        experiment="E18-chunked",
        config_col="chunk_s",
        baseline_verdict_col="baseline_verdict",
        test_verdict_col="chunked_verdict",
        baseline_mean_col="baseline_mean",
        test_mean_col="chunked_mean",
    ))

    # E18b: strategy sweep
    results.extend(_load_comparison_csv(
        csv_dir / "auto_beq_sweep_strategies.csv",
        experiment="E18b-strategy",
        config_col="strategy",
    ))

    # E19: advisor calibration
    results.extend(_load_comparison_csv(
        csv_dir / "auto_beq_sweep_e19.csv",
        experiment="E19-calibration",
        config_col="config",
    ))

    # E20 reuses E19 report format (same runner)
    # Check if E20 data is in the E19 CSV by looking for sh3 configs
    e19_path = csv_dir / "auto_beq_sweep_e19.csv"
    if e19_path.exists():
        with e19_path.open() as f:
            reader = csv.DictReader(f)
            e20_rows = [r for r in reader if "sh3" in r.get("config", "") or "cap35" in r.get("config", "") or "cap40" in r.get("config", "")]
        if e20_rows:
            # Re-parse just the E20 configs
            pass  # They're already in E19 results with distinguishable labels

    # E21: feedback loop
    results.extend(_load_comparison_csv(
        csv_dir / "auto_beq_sweep_e21.csv",
        experiment="E21-feedback",
        config_col="config",
    ))

    return results


def print_summary_table(results: list[ExperimentResult]) -> None:
    """Print the per-experiment summary table."""
    print("=" * 95)
    print("AUTO-BEQ EXPERIMENT COMPARISON REPORT")
    print("=" * 95)
    print()
    print(f"{'Experiment/Config':<42} {'Tests':>5} {'P/M/F':>8} {'Imp':>4} {'Deg':>4} {'Net':>4} {'AvgΔ':>7} {'Adopted':>8}")
    print("-" * 95)

    for r in results:
        p = r.verdicts.get("PASS", 0)
        m = r.verdicts.get("MARGINAL", 0)
        f = r.verdicts.get("FAIL", 0)

        # Determine if adopted (heuristic based on experiment name)
        adopted = ""
        if r.experiment == "baseline":
            adopted = "current"
        elif r.net_grade > 0 and r.degradations == 0:
            adopted = "YES"
        elif r.net_grade > 0 and r.degradations > 0:
            adopted = "risky"
        elif r.net_grade <= 0 and r.count > 0:
            adopted = "no"

        print(
            f"{r.label:<42} {r.count:>5} "
            f"{p:>2}/{m:>1}/{f:>2}   "
            f"{r.improvements:>4} {r.degradations:>4} {r.net_grade:>+4} "
            f"{r.avg_delta:>+7.2f} {adopted:>8}"
        )


def print_title_matrix(results: list[ExperimentResult]) -> None:
    """Print per-title verdict matrix across experiments."""
    # Collect all titles
    all_titles: set[str] = set()
    for r in results:
        all_titles.update(r.per_title.keys())

    if not all_titles:
        return

    # Filter to comparison experiments (those with baseline/test verdicts)
    comparison_results = [r for r in results if r.experiment != "baseline"]
    if not comparison_results:
        return

    # Pick a subset of interesting experiments (those with any grade changes)
    interesting = [r for r in comparison_results if r.improvements + r.degradations > 0]
    if not interesting:
        interesting = comparison_results[:5]

    print()
    print("=" * 95)
    print("PER-TITLE VERDICT MATRIX (experiments with grade changes)")
    print("=" * 95)
    print()

    # Header
    labels = [r.label[:25] for r in interesting]
    header = f"{'Title':<30} " + " ".join(f"{l:>25}" for l in labels)
    print(header)
    print("-" * len(header))

    for title in sorted(all_titles):
        cells = []
        has_change = False
        for r in interesting:
            info = r.per_title.get(title)
            if info is None:
                cells.append(f"{'—':>25}")
                continue
            if "test_verdict" in info:
                bv = GRADE_SHORT.get(info["baseline_verdict"], "?")
                tv = GRADE_SHORT.get(info["test_verdict"], "?")
                delta = info.get("delta", 0)
                if bv != tv:
                    has_change = True
                    marker = "+" if GRADE_RANK.get(info["test_verdict"], 2) < GRADE_RANK.get(info["baseline_verdict"], 2) else "-"
                    cells.append(f"{bv}→{tv}({delta:+.1f}){marker}".rjust(25))
                else:
                    cells.append(f"{tv}({delta:+.1f})".rjust(25))
            elif "verdict" in info:
                v = GRADE_SHORT.get(info["verdict"], "?")
                cells.append(f"{v}({info['mean']:.1f})".rjust(25))

        # Only print titles with at least one grade change
        if has_change:
            print(f"{title:<30} " + " ".join(cells))


def print_baseline_progression() -> None:
    """Print the historical baseline progression."""
    print()
    print("=" * 95)
    print("BASELINE PROGRESSION")
    print("=" * 95)
    print()
    print(f"{'Stage':<50} {'P':>4} {'M':>4} {'F':>4} {'Total':>6} {'Non-FAIL%':>10}")
    print("-" * 80)

    stages = [
        ("E17d: Welch + MeasurementAdvisor (original)",        9,  4, 18),
        ("E18b: + blend-a0.7-P90 extraction (default)",        9,  4, 18),
        ("E19:  + slope threshold 10, multi-knee Q 0.9",      10,  4, 17),
        ("E20:  + gain cap 35 dB",                            10,  5, 16),
        ("E21:  feedback loop (NOT adopted)",                  10,  5, 16),
    ]
    for label, p, m, f in stages:
        total = p + m + f
        nonfail = (p + m) / total * 100 if total else 0
        print(f"{label:<50} {p:>4} {m:>4} {f:>4} {total:>6} {nonfail:>9.0f}%")


def write_unified_csv(results: list[ExperimentResult], path: Path) -> None:
    """Write a unified CSV with all experiment results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "config", "title",
            "baseline_verdict", "test_verdict",
            "baseline_mean", "test_mean", "delta",
        ])
        for r in results:
            for title, info in sorted(r.per_title.items()):
                if "test_verdict" in info:
                    w.writerow([
                        r.experiment, r.config, title,
                        info.get("baseline_verdict", ""),
                        info.get("test_verdict", ""),
                        f"{info.get('baseline_mean', 0):.2f}",
                        f"{info.get('test_mean', 0):.2f}",
                        f"{info.get('delta', 0):+.2f}",
                    ])
                elif "verdict" in info:
                    w.writerow([
                        r.experiment, r.config, title,
                        info["verdict"], info["verdict"],
                        f"{info['mean']:.2f}", f"{info['mean']:.2f}",
                        "+0.00",
                    ])
    print(f"\nUnified CSV written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-BEQ experiment comparison report")
    parser.add_argument("--dir", type=Path, default=Path(".pytest_cache"),
                        help="Directory containing experiment CSV files")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Also write unified CSV to this path")
    args = parser.parse_args()

    results = load_all_results(args.dir)
    if not results:
        print(f"No experiment CSVs found in {args.dir}/", file=sys.stderr)
        print("Run experiments first:", file=sys.stderr)
        print("  SPIKE_TEST=...::test_library_sweep bash scripts/run-spike-tests.sh", file=sys.stderr)
        sys.exit(1)

    print_summary_table(results)
    print_title_matrix(results)
    print_baseline_progression()

    if args.csv:
        write_unified_csv(results, args.csv)


if __name__ == "__main__":
    main()
