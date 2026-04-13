#!/usr/bin/env python3
"""WAV cache bias report — compare our cache to the full BEQ catalogue.

Generates a markdown report showing how the local WAV cache distribution
differs from the full catalogue across every metadata dimension (author,
audio format, era, content type, studio family, country).

The cache reflects the user's personal media library and is therefore
biased toward their viewing taste.  This report exposes the bias so we
know which dimensions need correction via targeted acquisition.

Also surfaces "missing-ID" media files (library files without a
``[tmdb-NNN]`` tag in the filename) when ``media_inventory.json`` is
present, so the user knows which files need their tags fixed.

Usage::

    bin/beq-designer cache-bias-report
    bin/beq-designer cache-bias-report -o docs/wav_cache_bias.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger("nn_cache_bias_report")


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


def _load_inventory(inventory_path: Path | None) -> dict | None:
    """Load media_inventory.json if present.

    **Fail-fast semantics**: if the caller explicitly passes an
    ``inventory_path``, the file must exist.  Only the default fallback
    path (``{beq_dir}/media_inventory.json``) is tolerated as missing
    (returns ``None``).
    """
    explicit = inventory_path is not None
    if inventory_path is None:
        try:
            from spike._auto_beq_helpers import beq_dir
            _beq = beq_dir()
        except RuntimeError:
            _beq = Path.home() / "Downloads" / "beqdesigner"
        inventory_path = _beq / "media_inventory.json"
    if not inventory_path.exists():
        if explicit:
            raise FileNotFoundError(
                f"configured media inventory does not exist: {inventory_path}. "
                f"Check that the path is correct and, if on a network mount, "
                f"that the mount is active.",
            )
        return None
    try:
        return json.loads(inventory_path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("failed to parse %s: %s — skipping inventory", inventory_path, exc)
        return None
    except OSError as exc:
        log.warning("could not read %s: %s", inventory_path, exc)
        return None


def generate_report(output=None, inventory_path: Path | None = None) -> None:
    pr = lambda s="": print(s, file=output or sys.stdout)

    from model.auto_beq_catalogue import _fetch_or_cache
    from spike._auto_beq_helpers import discover_wav_catalogue_pairs_cached

    catalogue = _fetch_or_cache()
    trainable = [e for e in catalogue if e.get("filters")]
    pairs = discover_wav_catalogue_pairs_cached()

    # Match WAV pairs back to catalogue entries.
    wav_tmdb_ids = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    have_entries = [
        e for e in trainable
        if str(e.get("theMovieDB", "")).strip() in wav_tmdb_ids
    ]

    inventory = _load_inventory(inventory_path)

    pr("# WAV Cache Bias Report")
    pr()
    pr(f"**Catalogue (trainable)**: {len(trainable):,} entries")
    pr(f"**Cache match**: {len(have_entries):,} trainable entries "
       f"({len(pairs)} WAV files, {len(wav_tmdb_ids)} unique tmdb IDs)")
    pr(f"**Coverage**: {100 * len(have_entries) / len(trainable):.2f}% of catalogue")
    if inventory is not None:
        pr(f"**Library inventory**: {inventory.get('n_total_with_ids', 0):,} "
           f"files with DB IDs, {inventory.get('n_catalogue_matched', 0):,} "
           f"catalogue-matched, {inventory.get('n_missing_ids', 0):,} "
           f"missing IDs")
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

    # Missing IDs section — files in the library without a [tmdb-NNN] tag.
    if inventory is not None:
        missing_ids = inventory.get("missing_ids", [])
        if missing_ids:
            pr("## Missing media DB IDs")
            pr()
            pr(f"**{len(missing_ids):,} media files** in your library are "
               "missing a `[tmdb-NNN]`, `[tvdb-NNN]`, or `[imdb-NNN]` tag in "
               "their filename.  These are excluded from extraction and "
               "from this analysis.  Renaming them to include the correct "
               "tag would let the model learn from them.")
            pr()
            pr("First 30 files needing tags:")
            pr()
            for path in missing_ids[:30]:
                # Show only the filename, not the full path.
                pr(f"- `{Path(path).name}`")
            pr()
            if len(missing_ids) > 30:
                pr(f"... and {len(missing_ids) - 30:,} more (full list in "
                   "`media_inventory.json`).")
                pr()
    else:
        pr("## Missing media DB IDs")
        pr()
        pr("`media_inventory.json` not found.  Run `bin/beq-designer extract` "
           "to scan the library and populate it.  After that, this section "
           "will list any media files missing their tmdb/tvdb/imdb tags.")
        pr()

    pr("---")
    pr("Generated by `bin/beq-designer cache-bias-report`")


def main():
    parser = argparse.ArgumentParser(description="WAV cache bias report")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None,
                        help="Path to media_inventory.json (default: "
                             "~/Downloads/beqdesigner/media_inventory.json)")
    args = parser.parse_args()

    if args.output:
        with args.output.open("w") as f:
            generate_report(output=f, inventory_path=args.inventory)
        print(f"Report written to {args.output}")
    else:
        generate_report(inventory_path=args.inventory)


if __name__ == "__main__":
    main()
