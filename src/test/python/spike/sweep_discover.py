"""Discovery CLI for the auto-BEQ library sweep.

Walks one or more media library roots, parses Plex-style filenames
(``Title (YEAR) [tmdb-NNNNN]``), cross-references every (title, year)
against the full BEQ catalogue, and persists all matches to
``~/.config/beqdesigner/auto_beq_sweep.json``.

Discovery is fast (in-memory filename parsing + dict lookups) and maps
the ENTIRE catalogue-matched library — the pytest sweep test applies a
limit at runtime.

Run with::

    poetry run python -m spike.sweep_discover --library /path/to/movies

Or via ``.env`` / env var::

    AUTO_BEQ_LIBRARY_ROOTS=/Volumes/A:/Volumes/B poetry run python -m spike.sweep_discover --yes
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from spike._auto_beq_helpers import beq_config_dir

log = logging.getLogger("auto_beq_sweep")

_DEFAULT_CATALOGUE_URL = (
    "https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json"
)
_DEFAULT_TEST_LIMIT = 10
_CATALOGUE_CACHE_MAX_AGE_HOURS = 24
_SCHEMA_VERSION = 1

# Pattern 1 — Plex/Jellyfin: "Title (YEAR)" optionally followed by " [tmdb-NNN]".
_PLEX_RE = re.compile(
    r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)"
    r"(?:\s*\[(?:tmdb|imdb)-[^\]]+\])?\s*$"
)

# Pattern 2 — Scene-style: "Title.Name.YEAR.codec.source..." with dots as separators.
# Matches the year as the first 4-digit group that looks like a plausible release year
# (1920-2039). Everything before the year (with dots replaced by spaces) is the title.
_SCENE_RE = re.compile(
    r"^(?P<title>.+?)\.(?P<year>(?:19|20)\d{2})\."
)

_MEDIA_EXTENSIONS = (".mkv",)


# ---------------------------------------------------------------------------
# .env loader (tiny, no deps)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read KEY=value lines from a ``.env`` file, set os.environ defaults.

    Returns the parsed mapping. Does not overwrite existing env vars.
    Supports quoted values, comments (``#``), and blank lines. Quotes
    are stripped from values; no shell interpolation.
    """
    if path is None:
        # Walk up from cwd looking for a .env file.
        cur = Path.cwd()
        for candidate in [cur, *cur.parents]:
            p = candidate / ".env"
            if p.is_file():
                path = p
                break
    if path is None or not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        parsed[key] = value
        os.environ.setdefault(key, value)
    return parsed


# ---------------------------------------------------------------------------
# Filename parsing + catalogue matching
# ---------------------------------------------------------------------------

def parse_plex_filename(path: Path) -> tuple[str, int] | None:
    """Extract (title, year) from a media file's directory or stem name.

    Supports two naming conventions:
      - Plex/Jellyfin: ``Title (YEAR) [tmdb-NNN]``
      - Scene-style: ``Title.Name.YEAR.codec.source.mkv``

    Tries the enclosing directory name first (Plex convention), then
    the file stem, for each pattern.
    """
    for candidate in (path.parent.name, path.stem):
        # Plex/Jellyfin pattern first (more specific).
        m = _PLEX_RE.match(candidate)
        if m:
            return (m.group("title").strip(), int(m.group("year")))
        # Scene-style with dots.
        m = _SCENE_RE.match(candidate)
        if m:
            title = m.group("title").replace(".", " ").strip()
            return (title, int(m.group("year")))
    return None


def normalise_title(s: str) -> str:
    """Lowercase, strip non-alphanumeric. Used for match keys only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_catalogue(catalogue: list[dict], title: str, year: int) -> dict | None:
    """Find the catalogue entry for (title, year).

    Requires exact normalised-title + year match. Requires non-empty
    ``filters``. Ties broken by filter-count (richest wins — tends to
    be the most comprehensive variant).
    """
    norm = normalise_title(title)
    candidates = []
    for entry in catalogue:
        if not entry.get("filters"):
            continue
        if normalise_title(entry.get("title", "")) != norm:
            continue
        try:
            entry_year = int(entry.get("year", 0))
        except (TypeError, ValueError):
            continue
        if entry_year != year:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: len(e["filters"]), reverse=True)
    return candidates[0]


def extract_rating(entry: dict) -> float | None:
    """Try to parse the ``rating`` field as a numeric score.

    The committed snapshot has MPAA strings like ``"PG-13"`` which
    return None. A fresh fetch MAY have numeric scores for some
    entries. Non-numeric ratings sort last.
    """
    r = entry.get("rating")
    try:
        return float(r) if r is not None else None
    except (TypeError, ValueError):
        return None


def bucket_rating(rating: float | None) -> float:
    """Bucket to 0.5 granularity for sorting. None → -1.0 (sorts last)."""
    if rating is None:
        return -1.0
    return float(int(rating * 2)) / 2.0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class SweepFilm:
    path: str
    library_root: str
    title: str
    year: int
    rating: float | None
    catalogue_entry: dict = field(repr=False)

    @property
    def rating_bucket(self) -> float:
        return bucket_rating(self.rating)


def discover_matches(
    library_root: Path,
    catalogue: list[dict],
) -> list[SweepFilm]:
    """Walk a library root and return all catalogue-matched films."""
    matches: list[SweepFilm] = []
    scanned = 0
    for ext in _MEDIA_EXTENSIONS:
        for media_path in library_root.rglob(f"*{ext}"):
            scanned += 1
            parsed = parse_plex_filename(media_path)
            if parsed is None:
                continue
            title, year = parsed
            entry = match_catalogue(catalogue, title, year)
            if entry is None:
                continue
            matches.append(SweepFilm(
                path=str(media_path),
                library_root=str(library_root),
                title=title,
                year=year,
                rating=extract_rating(entry),
                catalogue_entry=entry,
            ))
    log.info("root %s: scanned=%d matched=%d", library_root, scanned, len(matches))
    return matches


def sort_matches(matches: list[SweepFilm]) -> list[SweepFilm]:
    """Sort by (rating bucket desc, year desc)."""
    return sorted(
        matches,
        key=lambda f: (-bucket_rating(f.rating), -(f.year or 0)),
    )


# ---------------------------------------------------------------------------
# Catalogue fetch + cache
# ---------------------------------------------------------------------------

def _catalogue_cache_path() -> Path:
    return beq_config_dir() / "beq_catalogue_cache.json"


def _is_cache_fresh(cache_path: Path, max_age_hours: int) -> bool:
    if not cache_path.exists():
        return False
    age_s = dt.datetime.now().timestamp() - cache_path.stat().st_mtime
    return age_s < max_age_hours * 3600


def load_or_fetch_catalogue(
    url: str = _DEFAULT_CATALOGUE_URL,
    force_refresh: bool = False,
    max_age_hours: int = _CATALOGUE_CACHE_MAX_AGE_HOURS,
) -> list[dict]:
    """Load the BEQ catalogue from cache, or fetch fresh from upstream."""
    cache_path = _catalogue_cache_path()
    if not force_refresh and _is_cache_fresh(cache_path, max_age_hours):
        log.info("using cached catalogue: %s", cache_path)
        with cache_path.open() as f:
            return json.load(f)
    log.info("fetching catalogue: %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        data = resp.read()
    cache_path.write_bytes(data)
    log.info("wrote catalogue cache: %s (%d bytes)", cache_path, len(data))
    return json.loads(data)


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    return beq_config_dir() / "auto_beq_sweep.json"


def write_config(
    films: list[SweepFilm],
    library_roots: list[Path],
    test_limit: int,
    output: Path | None = None,
    catalogue_url: str = _DEFAULT_CATALOGUE_URL,
) -> Path:
    """Write the discovered films to the sweep config JSON."""
    path = output or _default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": now,
        "library_roots": [str(r) for r in library_roots],
        "catalogue_url": catalogue_url,
        "catalogue_fetched_at": now,
        "test_limit": test_limit,
        "films": [asdict(f) for f in films],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_config(path: Path | None = None) -> dict | None:
    """Load the sweep config. Returns None if missing."""
    p = path or _default_config_path()
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_library_roots(args: argparse.Namespace) -> list[Path]:
    """Resolve library roots from (in order) CLI flags, env var, prompt."""
    if args.library:
        return [Path(p).expanduser() for p in args.library]
    env = os.environ.get("AUTO_BEQ_LIBRARY_ROOTS")
    if env:
        return [Path(p.strip()).expanduser() for p in env.split(":") if p.strip()]
    # Interactive prompt.
    raw = input("Library path(s), colon-separated: ").strip()
    if not raw:
        raise SystemExit("no library root provided")
    return [Path(p.strip()).expanduser() for p in raw.split(":") if p.strip()]


def _print_summary(
    matches: list[SweepFilm],
    library_roots: list[Path],
    top_n: int = 20,
) -> None:
    print()
    print(f"Library roots scanned: {len(library_roots)}")
    for root in library_roots:
        n = sum(1 for m in matches if m.library_root == str(root))
        print(f"  {root}  →  {n} catalogue matches")
    print(f"Total matches: {len(matches)}")
    buckets = Counter(m.rating_bucket for m in matches)
    print("Rating buckets:")
    for bucket in sorted(buckets, reverse=True):
        label = f"{bucket:.1f}" if bucket >= 0 else "(none)"
        print(f"  {label}: {buckets[bucket]}")
    print()
    print(f"Top {min(top_n, len(matches))} by (rating desc, year desc):")
    for i, m in enumerate(matches[:top_n], 1):
        r_label = f"{m.rating:.1f}" if m.rating is not None else "—"
        print(f"  {i:3d}. [r={r_label} y={m.year}] {m.title}")
    print()


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [Y/n]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("", "y", "yes")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sweep_discover",
        description="Discover catalogue-matched films in the user's library.",
    )
    p.add_argument(
        "--library", action="append", default=None,
        help="Library root (repeatable). Falls back to $AUTO_BEQ_LIBRARY_ROOTS or prompt.",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch catalogue even if the cached copy is fresh.",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompt (non-interactive mode).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help=f"Override config output path (default: {_default_config_path()}).",
    )
    p.add_argument(
        "--test-limit", type=int, default=_DEFAULT_TEST_LIMIT,
        help=f"Default test-run limit persisted in config (default: {_DEFAULT_TEST_LIMIT}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args = _parse_args(argv)

    try:
        library_roots = _resolve_library_roots(args)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    for root in library_roots:
        if not root.exists():
            print(f"error: library root does not exist: {root}", file=sys.stderr)
            return 2

    catalogue = load_or_fetch_catalogue(force_refresh=args.refresh)
    log.info("catalogue entries: %d", len(catalogue))

    matches: list[SweepFilm] = []
    for root in library_roots:
        matches.extend(discover_matches(root, catalogue))
    matches = sort_matches(matches)

    _print_summary(matches, library_roots)

    if not matches:
        print("No catalogue-matched films found.", file=sys.stderr)
        return 1

    if not args.yes:
        output_path = args.output or _default_config_path()
        if not _confirm(f"Save {len(matches)} films to {output_path}?"):
            print("aborted (user declined)")
            return 1

    path = write_config(
        films=matches,
        library_roots=library_roots,
        test_limit=args.test_limit,
        output=args.output,
    )
    print(f"Wrote {len(matches)} films to {path}")
    print(f"Test will run top {args.test_limit} by default "
          "(override via AUTO_BEQ_SWEEP_LIMIT).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
