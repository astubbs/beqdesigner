"""Discovery CLI for the auto-BEQ library sweep.

Walks one or more media library roots, parses standard media filenames
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

# Pattern 1 — Standard directory: "Title (YEAR)" optionally followed by " [tmdb-NNN]",
# " [imdb-ttNNN]", or " [tvdb-NNN]". End-of-string anchored.
_TITLE_YEAR_DIR_RE = re.compile(
    r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)"
    r"(?:\s*\[(?:tmdb|imdb|tvdb)-[^\]]+\])?\s*$"
)

# Pattern 1b — File stem: "Title (YEAR) - S01E01 - Episode Name [tags]".
# More lenient than the dir pattern: allows arbitrary content after year+optional-tag.
_TITLE_YEAR_STEM_RE = re.compile(
    r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)"
    r"(?:\s*\[(?:tmdb|imdb|tvdb)-[^\]]+\])?"
    r"(?:\s*-\s*S(?P<season>\d+)E(?P<episode>\d+))?"
)

# Pattern 2 — Scene-style: "Title.Name.YEAR.codec.source..." with dots as separators.
_SCENE_RE = re.compile(
    r"^(?P<title>.+?)\.(?P<year>(?:19|20)\d{2})\."
)

# Extract S01E01 from an episode filename.
_EPISODE_RE = re.compile(r"S(?P<season>\d+)E(?P<episode>\d+)", re.IGNORECASE)

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

@dataclass(frozen=True)
class ParsedMedia:
    """Result of parsing a media filename."""
    title: str
    year: int
    season: int | None = None
    episode: int | None = None


def parse_media_filename(path: Path) -> ParsedMedia | None:
    """Extract (title, year, season, episode) from a media file path.

    Supports:
      - Standard dirs: ``Title (YEAR) [tmdb-NNN]`` / ``[tvdb-NNN]``
      - File stems: ``Title (YEAR) - S01E01 - Episode Name [tags]``
      - Scene-style: ``Title.Name.YEAR.codec.source.mkv``

    Checks (in order): parent directory, grandparent directory (for TV
    shows with ``Season N/`` subdirs), then the file stem. Episode info
    (S01E01) is extracted from the file stem when present.
    """
    title: str | None = None
    year: int | None = None

    # Try directory names first (strict end-of-string anchored).
    for dirname in (path.parent.name, path.parent.parent.name):
        m = _TITLE_YEAR_DIR_RE.match(dirname)
        if m:
            title = m.group("title").strip()
            year = int(m.group("year"))
            break

    # If no dir matched, try the file stem (lenient — allows episode
    # info and codec tags after the year).
    if title is None:
        m = _TITLE_YEAR_STEM_RE.match(path.stem)
        if m:
            title = m.group("title").strip()
            year = int(m.group("year"))
        else:
            # Scene-style with dots.
            for candidate in (path.parent.name, path.parent.parent.name, path.stem):
                m = _SCENE_RE.match(candidate)
                if m:
                    title = m.group("title").replace(".", " ").strip()
                    year = int(m.group("year"))
                    break

    if title is None or year is None:
        return None

    # Extract episode info from the file stem (always, even if title
    # came from a directory — the stem has the episode number).
    season: int | None = None
    episode: int | None = None
    ep_match = _EPISODE_RE.search(path.stem)
    if ep_match:
        season = int(ep_match.group("season"))
        episode = int(ep_match.group("episode"))

    return ParsedMedia(title=title, year=year, season=season, episode=episode)


def normalise_title(s: str) -> str:
    """Lowercase, strip non-alphanumeric. Used for match keys only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _parse_catalogue_episodes(entry: dict) -> set[tuple[int, int]]:
    """Parse the season/episode fields from a catalogue entry.

    Returns a set of (season, episode) tuples. The catalogue uses
    inconsistent formats:
      - season="1", episode="1,7,9,11"  → {(1,1),(1,7),(1,9),(1,11)}
      - season="27E01"                  → {(27,1)}
      - season="01E01, 04"              → {(1,1),(1,4)}
      - season="S01E06"                 → {(1,6)}
      - season="01"                     → all episodes in season 1
    """
    episodes: set[tuple[int, int]] = set()
    season_raw = str(entry.get("season", "")).strip()
    episode_raw = str(entry.get("episode", "")).strip()

    if not season_raw:
        return episodes

    # Try "S01E06" or "01E01, 04" patterns in the season field.
    se_matches = re.findall(r"S?(\d+)E(\d+)", season_raw, re.IGNORECASE)
    if se_matches:
        for s, e in se_matches:
            episodes.add((int(s), int(e)))
        # Also check for trailing bare episode numbers like "01E01, 04"
        # where "04" is another episode in the same season.
        trailing = re.findall(r",\s*(\d+)(?!\d*E)", season_raw, re.IGNORECASE)
        if se_matches and trailing:
            base_season = int(se_matches[0][0])
            for ep in trailing:
                episodes.add((base_season, int(ep)))
        return episodes

    # Plain season number + episode list.
    try:
        season_num = int(season_raw)
    except ValueError:
        return episodes

    if episode_raw:
        for ep_str in episode_raw.split(","):
            ep_str = ep_str.strip()
            if ep_str.isdigit():
                episodes.add((season_num, int(ep_str)))
    else:
        # Season-wide entry (no specific episodes).
        episodes.add((season_num, 0))  # 0 = "all episodes"

    return episodes


def match_catalogue(
    catalogue: list[dict],
    title: str,
    year: int,
    season: int | None = None,
    episode: int | None = None,
) -> dict | None:
    """Find the best catalogue entry for (title, year[, season, episode]).

    Requires exact normalised-title + year match. Requires non-empty
    ``filters``.

    For TV shows with per-episode profiles: if season+episode are
    provided, prefer an entry whose episode field covers that specific
    episode. Falls back to a season-wide or show-wide entry.

    Ties broken by filter-count (richest wins).
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

    # If we have episode info, try to find an episode-specific match.
    if season is not None and episode is not None:
        exact_ep = []
        season_wide = []
        no_ep_info = []
        for entry in candidates:
            eps = _parse_catalogue_episodes(entry)
            if not eps:
                no_ep_info.append(entry)
            elif (season, episode) in eps:
                exact_ep.append(entry)
            elif (season, 0) in eps:
                season_wide.append(entry)
        # Prefer: exact episode match > season-wide > no episode info.
        pool = exact_ep or season_wide or no_ep_info or candidates
        pool.sort(key=lambda e: len(e["filters"]), reverse=True)
        return pool[0]

    # No episode info — pick the richest entry.
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
    season: int | None = None
    episode: int | None = None
    catalogue_entry: dict = field(default_factory=dict, repr=False)

    @property
    def rating_bucket(self) -> float:
        return bucket_rating(self.rating)


def discover_matches(
    library_root: Path,
    catalogue: list[dict],
) -> tuple[list[SweepFilm], int]:
    """Walk a library root and return (matches, total_scanned)."""
    # Collect all media files first so we can show progress.
    media_files: list[Path] = []
    for ext in _MEDIA_EXTENSIONS:
        media_files.extend(library_root.rglob(f"*{ext}"))
    total = len(media_files)
    log.info("root %s: found %d media files to scan", library_root, total)

    matches: list[SweepFilm] = []
    for i, media_path in enumerate(media_files):
        # Progress — print to stderr so it's visible even without -v.
        pct = ((i + 1) * 100) // total if total else 100
        print(f"\r  scanning [{pct:3d}%] {i + 1}/{total}", end="", flush=True)
        log.debug("  %s", media_path.relative_to(library_root))

        parsed = parse_media_filename(media_path)
        if parsed is None:
            log.debug("    → no title/year parsed, skipping")
            continue
        ep_label = f" S{parsed.season:02d}E{parsed.episode:02d}" if parsed.episode else ""
        log.debug("    → parsed: %r (%d)%s", parsed.title, parsed.year, ep_label)
        entry = match_catalogue(
            catalogue, parsed.title, parsed.year,
            season=parsed.season, episode=parsed.episode,
        )
        if entry is None:
            log.debug("    → no catalogue match")
            continue
        log.debug("    → MATCHED catalogue entry: %r", entry.get("title"))
        matches.append(SweepFilm(
            path=str(media_path),
            library_root=str(library_root),
            title=parsed.title,
            year=parsed.year,
            rating=extract_rating(entry),
            season=parsed.season,
            episode=parsed.episode,
            catalogue_entry=entry,
        ))
    print()  # newline after progress bar
    log.info("root %s: scanned=%d matched=%d", library_root, total, len(matches))
    return matches, total


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
    """Resolve library roots from (in order) CLI flags, env var, previous config, prompt."""
    if args.library:
        return [Path(p).expanduser() for p in args.library]
    env = os.environ.get("AUTO_BEQ_LIBRARY_ROOTS")
    if env:
        return _split_paths(env)

    # Check if a previous run saved library roots in the config.
    existing = load_config()
    cached_roots = existing.get("library_roots", []) if existing else []
    if cached_roots:
        cached_display = ", ".join(cached_roots)
        print(f"Previous library roots: {cached_display}")
        raw = input("Library path(s) [Enter to reuse, or new comma-separated paths]: ").strip()
        if not raw:
            return [Path(p).expanduser() for p in cached_roots]
        return _split_paths(raw)

    # No previous config, no env var — prompt.
    raw = input("Library path(s), comma-separated: ").strip()
    if not raw:
        raise SystemExit("no library root provided")
    return _split_paths(raw)


def _split_paths(raw: str) -> list[Path]:
    """Split a string of paths by comma, strip whitespace."""
    return [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]


def _print_summary(
    matches: list[SweepFilm],
    library_roots: list[Path],
    total_scanned: int,
) -> None:
    print()
    print(f"Library roots scanned: {len(library_roots)}")
    for root in library_roots:
        n = sum(1 for m in matches if m.library_root == str(root))
        print(f"  {root}  →  {n} catalogue matches")

    # Group by (title, year) to collapse episodes into one line.
    from collections import OrderedDict
    grouped: OrderedDict[tuple[str, int | None], list[SweepFilm]] = OrderedDict()
    for m in matches:
        key = (m.title, m.year)
        grouped.setdefault(key, []).append(m)

    unique_titles = len(grouped)
    total_files = len(matches)
    match_pct = (total_files * 100 // total_scanned) if total_scanned else 0
    print(f"Total: {unique_titles} titles ({total_files} files matched out of {total_scanned} scanned — {match_pct}%)")

    buckets = Counter(bucket_rating(films[0].rating) for films in grouped.values())
    print("Rating buckets:")
    for bucket in sorted(buckets, reverse=True):
        label = f"{bucket:.1f}" if bucket >= 0 else "(none)"
        print(f"  {label}: {buckets[bucket]}")

    print()
    print(f"All {unique_titles} matched titles (rating desc, year desc):")
    for i, ((title, year), films) in enumerate(grouped.items(), 1):
        r_label = f"{films[0].rating:.1f}" if films[0].rating is not None else "—"
        ep_count = len(films)
        if ep_count == 1:
            suffix = ""
        else:
            # Group episodes by season for a compact display.
            seasons: dict[str, int] = Counter()
            for f in films:
                # Try to extract "Season N" from the path.
                parts = Path(f.path).parts
                for part in parts:
                    if part.lower().startswith("season"):
                        seasons[part] += 1
                        break
                else:
                    seasons["(no season)"] += 1
            season_str = ", ".join(
                f"{s}: {c} ep" for s, c in sorted(seasons.items())
            )
            suffix = f"  ({ep_count} episodes — {season_str})"
        print(f"  {i:3d}. [r={r_label} y={year}] {title}{suffix}")
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
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
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
    total_scanned = 0
    for root in library_roots:
        root_matches, root_scanned = discover_matches(root, catalogue)
        matches.extend(root_matches)
        total_scanned += root_scanned
    matches = sort_matches(matches)

    _print_summary(matches, library_roots, total_scanned)

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
