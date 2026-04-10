#!/usr/bin/env python3
"""WAV cache bias report — compare our cache to the full BEQ catalogue.

Generates a markdown report showing how the local WAV cache distribution
differs from the full catalogue across every metadata dimension (author,
audio format, era, content type, studio family, country).

The cache reflects the user's personal media library and is therefore
biased toward their viewing taste.  This report exposes the bias so we
know which dimensions need correction via targeted acquisition.

Usage::

    poetry run python3 scripts/nn_cache_bias_report.py
    poetry run python3 scripts/nn_cache_bias_report.py -o docs/wav_cache_bias.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running from project root.
sys.path.insert(0, "src/main/python")
sys.path.insert(0, "src/test/python")


def _classify_format(audio_types) -> str:
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


def _classify_era(year) -> str:
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


def _dist(entries, key_fn) -> dict[str, float]:
    c = Counter(key_fn(e) for e in entries)
    total = sum(c.values())
    if total == 0:
        return {}
    return {k: 100 * v / total for k, v in c.items()}


def _bias_table(
    pr,
    title: str,
    cat_dist: dict[str, float],
    have_dist: dict[str, float],
    cat_counts: dict[str, int],
) -> None:
    pr(f"### {title}")
    pr()
    pr("| Bucket | Catalogue % | Cache % | Bias | Catalogue entries |")
    pr("|---|---:|---:|---:|---:|")
    keys = sorted(set(cat_dist) | set(have_dist), key=lambda k: -cat_dist.get(k, 0))
    for k in keys:
        c = cat_dist.get(k, 0)
        h = have_dist.get(k, 0)
        bias = h - c
        sign = "+" if bias > 0 else ""
        n = cat_counts.get(k, 0)
        pr(f"| {k} | {c:.1f}% | {h:.1f}% | {sign}{bias:+.1f} | {n:,} |")
    pr()


def generate_report(output=None) -> None:
    pr = lambda s="": print(s, file=output or sys.stdout)

    from model.auto_beq_catalogue import _fetch_or_cache
    from spike._auto_beq_helpers import discover_wav_catalogue_pairs

    catalogue = _fetch_or_cache()
    trainable = [e for e in catalogue if e.get("filters")]
    pairs = discover_wav_catalogue_pairs()

    # Match WAV pairs back to catalogue entries.
    wav_tmdb_ids = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    have_entries = [
        e for e in trainable
        if str(e.get("theMovieDB", "")).strip() in wav_tmdb_ids
    ]

    pr("# WAV Cache Bias Report")
    pr()
    pr(f"**Catalogue (trainable)**: {len(trainable):,} entries")
    pr(f"**Cache match**: {len(have_entries):,} trainable entries "
       f"({len(pairs)} WAV files, {len(wav_tmdb_ids)} unique tmdb IDs)")
    pr(f"**Coverage**: {100 * len(have_entries) / len(trainable):.2f}% of catalogue")
    pr()
    pr("This report compares the local WAV cache distribution to the full "
       "BEQ catalogue across metadata dimensions.  Positive bias means "
       "the cache **over-represents** that bucket; negative means **under-"
       "represents**.")
    pr()
    pr("---")
    pr()
    pr("## Distributions by dimension")
    pr()

    dims = [
        ("Author", lambda e: str(e.get("author", "unknown")).strip().lower()),
        ("Audio format", lambda e: _classify_format(e.get("audioTypes", []))),
        ("Era", lambda e: _classify_era(e.get("year"))),
        ("Content type", lambda e: e.get("content_type", "film")),
        ("Source", lambda e: e.get("source", "unknown")),
    ]
    for name, key_fn in dims:
        cat_dist = _dist(trainable, key_fn)
        have_dist = _dist(have_entries, key_fn)
        cat_counts = Counter(key_fn(e) for e in trainable)
        _bias_table(pr, name, cat_dist, have_dist, cat_counts)

    # Top-N studios is more involved (long tail).
    pr("### Top 15 studios")
    pr()
    studio_counts = Counter()
    for e in trainable:
        s = (e.get("studio") or "unknown").strip()
        studio_counts[s] += 1
    have_studio_counts = Counter()
    for e in have_entries:
        s = (e.get("studio") or "unknown").strip()
        have_studio_counts[s] += 1

    pr("| Studio | Catalogue | Cache | Cache coverage |")
    pr("|---|---:|---:|---:|")
    for s, n in studio_counts.most_common(15):
        h = have_studio_counts.get(s, 0)
        cov = 100 * h / n if n else 0
        pr(f"| {s} | {n:,} | {h} | {cov:.1f}% |")
    pr()

    # Summary: top biases.
    pr("## Top bias issues")
    pr()
    pr("Sorted by absolute bias.  Positive = overrepresented in cache, "
       "negative = underrepresented.")
    pr()
    pr("| Dimension | Bucket | Catalogue % | Cache % | Bias |")
    pr("|---|---|---:|---:|---:|")
    rows = []
    for name, key_fn in dims:
        cat_dist = _dist(trainable, key_fn)
        have_dist = _dist(have_entries, key_fn)
        for k in set(cat_dist) | set(have_dist):
            bias = have_dist.get(k, 0) - cat_dist.get(k, 0)
            rows.append((abs(bias), name, k, cat_dist.get(k, 0),
                        have_dist.get(k, 0), bias))
    rows.sort(reverse=True)
    for _, name, k, c, h, bias in rows[:15]:
        sign = "+" if bias > 0 else ""
        pr(f"| {name} | {k} | {c:.1f}% | {h:.1f}% | {sign}{bias:+.1f} |")
    pr()

    pr("---")
    pr("Generated by `scripts/nn_cache_bias_report.py`")


def main():
    parser = argparse.ArgumentParser(description="WAV cache bias report")
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    if args.output:
        with args.output.open("w") as f:
            generate_report(output=f)
        print(f"Report written to {args.output}")
    else:
        generate_report()


if __name__ == "__main__":
    main()
