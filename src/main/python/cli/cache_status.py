#!/usr/bin/env python3
"""WAV cache status — summarise what's available for NN training.

Reports total WAVs, catalogue-matched counts, movie/TV breakdown,
unique titles, and growth since last check.

Usage: bin/beq-designer cache-status
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "main" / "python"
_TEST = _REPO_ROOT / "src" / "test" / "python"
for p in (_SRC, _TEST):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import json
import logging

logging.basicConfig(level=logging.WARNING)  # suppress noisy logs

from spike._auto_beq_helpers import discover_wav_catalogue_pairs, wav_cache_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarise WAV cache training set status.")
    parser.add_argument("cache_dir", nargs="?", type=Path, default=None,
                        help="WAV cache directory. Defaults to BEQ_WAV_CACHE env var or settings.json.")
    args = parser.parse_args()

    if args.cache_dir:
        # Override the env var so discover_wav_catalogue_pairs() uses it too.
        import os
        os.environ["BEQ_WAV_CACHE"] = str(args.cache_dir.resolve())
    cache = wav_cache_dir()
    print(f"WAV cache: {cache}")
    print()

    # Raw WAV counts -- walk bucket dirs with progress instead of rglob.
    import os as _os
    from model.media_utils import ProgressLogger
    from model.wav_cache import WAV_SUFFIX

    log = logging.getLogger("cache_status")
    log.info("scanning WAV cache at %s ...", cache)
    bucket_dirs = sorted(
        e.path for e in _os.scandir(cache)
        if e.is_dir()
    )
    progress = ProgressLogger(len(bucket_dirs), logger=log, min_interval_s=5)
    all_wavs: list[Path] = []
    for i, bucket_path in enumerate(bucket_dirs):
        for dirpath, _dirnames, filenames in _os.walk(bucket_path):
            for f in filenames:
                if f.endswith(WAV_SUFFIX):
                    all_wavs.append(Path(dirpath) / f)
        progress.update(i + 1, label=_os.path.basename(bucket_path))

    print(f"Raw WAV files:        {len(all_wavs)}")
    print()

    # Catalogue-matched (usable for training).
    pairs = discover_wav_catalogue_pairs()
    titles = set()
    matched_movies = 0
    matched_tv = 0
    authors = {}
    for p in pairs:
        entry = p["catalogue_entry"]
        titles.add(entry.get("title", ""))
        ct = entry.get("content_type", "film")
        if ct == "TV":
            matched_tv += 1
        else:
            matched_movies += 1
        author = entry.get("author", "unknown")
        authors[author] = authors.get(author, 0) + 1

    print(f"Catalogue-matched:    {len(pairs)} WAVs across {len(titles)} unique titles")
    print(f"  Movies:             {matched_movies}")
    print(f"  TV episodes:        {matched_tv}")
    print()

    # Author breakdown.
    if authors:
        print("By BEQ author:")
        for author, count in sorted(authors.items(), key=lambda x: -x[1]):
            print(f"  {author:20s} {count:4d}")
        print()

    # Check for growth — save/load last count in local config dir.
    # NEVER write to the WAV cache directory itself (would pollute the
    # cache and break mtime-based change detection).
    from spike._auto_beq_helpers import beq_config_dir
    status_file = beq_config_dir() / "cache_status_last.json"
    last = {}
    if status_file.exists():
        try:
            last = json.loads(status_file.read_text())
        except Exception:
            pass

    if last:
        delta_wavs = len(pairs) - last.get("matched_wavs", 0)
        delta_titles = len(titles) - last.get("unique_titles", 0)
        if delta_wavs != 0 or delta_titles != 0:
            print(f"Since last check:     {delta_wavs:+d} WAVs, {delta_titles:+d} titles")
            print()

    # Save current counts.
    status_file.write_text(json.dumps({
        "matched_wavs": len(pairs),
        "unique_titles": len(titles),
        "raw_wavs": len(all_wavs),
    }, indent=2) + "\n")

    # Summary line for quick glance.
    # Count unique TV shows from catalogue matches.
    tv_shows = {e["catalogue_entry"].get("title", "") for e in pairs
                if e["catalogue_entry"].get("content_type") == "TV"}
    print(f"Training-ready: {len(pairs)} WAVs / {len(titles)} titles "
          f"({matched_movies} movies + {matched_tv} TV eps from {len(tv_shows)} shows)")

    # Usage hints.
    print()
    print("Next steps:")
    print(f"  Repair cache:      bin/beq-designer verify")
    print(f"  Extract more:      bin/beq-designer extract")
    print(f"  Generate profile:  bin/beq-designer profile")


if __name__ == "__main__":
    main()
