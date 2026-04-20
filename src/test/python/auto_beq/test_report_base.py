"""Tests for cli.report_base shared infrastructure."""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import pytest

from cli.report_base import (
    classify_era,
    classify_format,
    create_report_argparser,
    dist_counts,
    dist_pct,
    run_report,
)


# ---------------------------------------------------------------------------
# classify_format
# ---------------------------------------------------------------------------


class TestClassifyFormat:
    def test_atmos(self):
        assert classify_format(["Dolby Atmos"]) == "atmos"

    def test_truehd(self):
        assert classify_format(["TrueHD 7.1"]) == "truehd"

    def test_dts_hd(self):
        assert classify_format(["DTS-HD MA 5.1"]) == "dts-hd"

    def test_dd_plus(self):
        assert classify_format(["DD+ 5.1"]) == "dd+"

    def test_other(self):
        assert classify_format(["PCM"]) == "other"

    def test_none_input(self):
        assert classify_format(None) == "other"

    def test_empty_list(self):
        assert classify_format([]) == "other"

    def test_atmos_takes_priority_over_truehd(self):
        assert classify_format(["TrueHD Atmos"]) == "atmos"


# ---------------------------------------------------------------------------
# classify_era
# ---------------------------------------------------------------------------


class TestClassifyEra:
    def test_pre1990(self):
        assert classify_era(1985) == "pre1990"

    def test_1990s_2000s(self):
        assert classify_era(1999) == "1990s-2000s"

    def test_2010s(self):
        assert classify_era(2015) == "2010s"

    def test_2020s(self):
        assert classify_era(2023) == "2020s"

    def test_boundary_1990(self):
        assert classify_era(1990) == "1990s-2000s"

    def test_boundary_2010(self):
        assert classify_era(2010) == "2010s"

    def test_boundary_2020(self):
        assert classify_era(2020) == "2020s"

    def test_string_year(self):
        assert classify_era("2015") == "2010s"

    def test_none(self):
        assert classify_era(None) == "unknown"

    def test_invalid(self):
        assert classify_era("abc") == "unknown"


# ---------------------------------------------------------------------------
# dist_pct / dist_counts
# ---------------------------------------------------------------------------


class TestDistHelpers:
    def test_dist_pct_basic(self):
        entries = [{"x": "a"}, {"x": "a"}, {"x": "b"}]
        result = dist_pct(entries, lambda e: e["x"])
        assert abs(result["a"] - 66.666) < 0.1
        assert abs(result["b"] - 33.333) < 0.1

    def test_dist_pct_empty(self):
        assert dist_pct([], lambda e: e) == {}

    def test_dist_counts_basic(self):
        entries = [{"x": "a"}, {"x": "a"}, {"x": "b"}]
        result = dist_counts(entries, lambda e: e["x"])
        assert result["a"] == 2
        assert result["b"] == 1


# ---------------------------------------------------------------------------
# create_report_argparser
# ---------------------------------------------------------------------------


class TestCreateReportArgparser:
    def test_has_output_flag(self):
        parser = create_report_argparser("test")
        args = parser.parse_args(["-o", "out.md"])
        assert args.output == Path("out.md")

    def test_output_defaults_to_none(self):
        parser = create_report_argparser("test")
        args = parser.parse_args([])
        assert args.output is None

    def test_extra_args(self):
        parser = create_report_argparser("test", extra_args=[
            (("-n", "--count"), {"type": int, "default": 10}),
        ])
        args = parser.parse_args(["-n", "5"])
        assert args.count == 5

    def test_extra_args_single_flag(self):
        parser = create_report_argparser("test", extra_args=[
            ("--verbose", {"action": "store_true"}),
        ])
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True


# ---------------------------------------------------------------------------
# run_report
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_stdout_output(self, capsys):
        def gen(output=None):
            print("hello", file=output or sys.stdout)

        args = argparse.Namespace(output=None)
        run_report(gen, args)
        assert "hello" in capsys.readouterr().out

    def test_file_output(self, tmp_path, capsys):
        def gen(output=None):
            print("file content", file=output or sys.stdout)

        out_file = tmp_path / "report.md"
        args = argparse.Namespace(output=out_file)
        run_report(gen, args)
        assert out_file.read_text().strip() == "file content"
        assert "Report written to" in capsys.readouterr().out

    def test_extra_kwargs_forwarded(self):
        received = {}

        def gen(output=None, n=0, label=""):
            received["n"] = n
            received["label"] = label

        args = argparse.Namespace(output=None)
        run_report(gen, args, extra_kwargs={"n": 42, "label": "test"})
        assert received == {"n": 42, "label": "test"}
