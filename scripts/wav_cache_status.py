#!/usr/bin/env python3
"""WAV cache status — summarise what's available for NN training.

Reports total WAVs, catalogue-matched counts, movie/TV breakdown,
unique titles, and growth since last check.

Usage:
    poetry run python3 scripts/wav_cache_status.py
    BEQ_WAV_CACHE=/Volumes/nas/beqdesigner/wav-cache poetry run python3 scripts/wav_cache_status.py
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
    cache = wav_cache_dir()
    print(f"WAV cache: {cache}")
    print()

    # Raw WAV counts.
    all_wavs = sorted(cache.rglob("*.lfe-1000hz.wav"))
    movies_dir = cache / "Movies"
    tv_dir = cache / "TV"
    raw_movies = len(list(movies_dir.rglob("*.lfe-1000hz.wav"))) if movies_dir.exists() else 0
    raw_tv = len(list(tv_dir.rglob("*.lfe-1000hz.wav"))) if tv_dir.exists() else 0

    # TV show count (unique show dirs under TV/Letter/ShowName/).
    tv_shows = set()
    if tv_dir.exists():
        for letter_dir in tv_dir.iterdir():
            if letter_dir.is_dir():
                for show_dir in letter_dir.iterdir():
                    if show_dir.is_dir():
                        tv_shows.add(show_dir.name)

    print(f"Raw WAV files:        {len(all_wavs)}")
    print(f"  Movies:             {raw_movies}")
    print(f"  TV episodes:        {raw_tv}")
    print(f"  TV shows:           {len(tv_shows)}")
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

    # Check for growth — save/load last count.
    status_file = cache / ".status_last.json"
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
    print(f"Training-ready: {len(pairs)} WAVs / {len(titles)} titles "
          f"({matched_movies} movies + {matched_tv} TV eps from {len(tv_shows)} shows)")


if __name__ == "__main__":
    main()
