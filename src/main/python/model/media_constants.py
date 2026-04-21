"""Shared constants for media discovery and filtering.

Stdlib-only — no scipy, numpy, or PyQt dependencies. Safe to import
from standalone scripts (Docker containers, NAS extraction, etc.).
"""

from __future__ import annotations

import re

# Subdirectory names that contain non-feature content (samples, trailers, etc.).
# Media files found inside these directories are excluded from discovery.
JUNK_SUBDIRS = frozenset({
    "sample", "samples", "featurettes", "featurette", "extras", "extra",
    "backdrops", "behind the scenes", "deleted scenes", "trailers", "trailer",
    "interviews", "shorts",
})

# Minimum file size for a feature-length media file. Anything smaller is
# likely a sample, trailer, or theme file.
MIN_FEATURE_SIZE_BYTES = 500_000_000  # 500 MB

# Media file extensions to scan.
MEDIA_EXTENSIONS = frozenset({".mkv", ".iso", ".mp4", ".m2ts", ".ts", ".avi"})

# Matches [tmdb-NNN], [tvdb-NNN], or [imdb-ttNNN] in a path string.
ID_RE = re.compile(r"\[(tmdb|tvdb|imdb)-([^\]]+)\]")

# Title (Year) from directory names.
TITLE_YEAR_RE = re.compile(r"^(.+?)\s*\((\d{4})\)")

# Episode extraction: S01E02 etc.
EPISODE_RE = re.compile(r"S(\d+)E(\d+)", re.IGNORECASE)

# BEQ catalogue URL.
CATALOGUE_URL = (
    "https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json"
)
