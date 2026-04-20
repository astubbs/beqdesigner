"""Shared infrastructure for BEQ report scripts.

Extracts the common patterns found across nn_report.py, nn_cache_bias_report.py,
nn_author_pattern_report.py, nn_acquisition_recommender.py, and
nn_f_experiment_report.py:

- argparse setup with ``-o/--output`` flag
- Output dispatch (write to file or stdout with confirmation message)
- Audio format and era classification helpers
- Percentage distribution computation
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def create_report_argparser(
    description: str,
    *,
    extra_args: list[tuple] | None = None,
) -> argparse.ArgumentParser:
    """Create an ArgumentParser pre-configured with the ``-o/--output`` flag.

    Parameters
    ----------
    description:
        One-line description shown in ``--help``.
    extra_args:
        Optional list of ``(flags, kwargs)`` tuples for additional arguments.
        Each tuple is ``(("-x", "--extra"), {"type": int, "default": 5})``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output file (markdown). Defaults to stdout.",
    )
    for flags, kwargs in (extra_args or []):
        if isinstance(flags, str):
            flags = (flags,)
        parser.add_argument(*flags, **kwargs)
    return parser


# ---------------------------------------------------------------------------
# Output dispatch
# ---------------------------------------------------------------------------


def run_report(
    generate_fn,
    args: argparse.Namespace,
    *,
    extra_kwargs: dict | None = None,
) -> None:
    """Run *generate_fn* with output directed to a file or stdout.

    If ``args.output`` is set, opens the file for writing and passes it as
    the ``output`` keyword argument to *generate_fn*.  Prints a confirmation
    message after writing.  Otherwise calls *generate_fn* with ``output=None``
    (which should default to ``sys.stdout``).

    *extra_kwargs* are forwarded to *generate_fn* alongside ``output``.
    """
    kwargs = dict(extra_kwargs or {})
    if args.output:
        with args.output.open("w") as f:
            generate_fn(output=f, **kwargs)
        print(f"Report written to {args.output}")
    else:
        generate_fn(**kwargs)


# ---------------------------------------------------------------------------
# Classification helpers (shared by cache_bias, author_pattern, acquisition)
# ---------------------------------------------------------------------------


def classify_format(audio_types) -> str:
    """Classify an ``audioTypes`` list into a format bucket."""
    j = " ".join(audio_types or []).lower()
    if "atmos" in j:
        return "atmos"
    if "truehd" in j:
        return "truehd"
    if "dts-hd" in j:
        return "dts-hd"
    if "dd+" in j:
        return "dd+"
    return "other"


def classify_era(year) -> str:
    """Classify a year into an era bucket."""
    try:
        y = int(year)
    except (ValueError, TypeError):
        return "unknown"
    if y < 1990:
        return "pre1990"
    if y < 2010:
        return "1990s-2000s"
    if y < 2020:
        return "2010s"
    return "2020s"


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def dist_pct(entries, key_fn) -> dict[str, float]:
    """Return percentage distribution of *entries* by *key_fn*."""
    c = Counter(key_fn(e) for e in entries)
    total = sum(c.values())
    if total == 0:
        return {}
    return {k: 100 * v / total for k, v in c.items()}


def dist_counts(entries, key_fn) -> Counter:
    """Return raw counts of *entries* by *key_fn*."""
    return Counter(key_fn(e) for e in entries)
