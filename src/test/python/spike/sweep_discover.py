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
from dataclasses import dataclass, field
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

# Subdirectory names that contain non-feature content (samples, trailers, etc.).
# Media files found inside these directories are excluded from inventory.
_JUNK_SUBDIRS = {
    "sample", "samples", "featurettes", "featurette", "extras", "extra",
    "backdrops", "behind the scenes", "deleted scenes", "trailers", "trailer",
    "interviews", "shorts",
}

# Minimum file size for a feature-length media file. Anything smaller is
# likely a sample, trailer, or theme file.
_MIN_FEATURE_SIZE_BYTES = 500_000_000  # 500 MB


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
        # Prefer: exact episode match > season-wide > entries without
        # episode info (show-wide). If ALL entries have episode info
        # but none cover this episode, return None — don't match a
        # wrong profile.
        pool = exact_ep or season_wide or no_ep_info
        if not pool:
            return None
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
    size_bytes: int = 0

    @property
    def rating_bucket(self) -> float:
        return bucket_rating(self.rating)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


@dataclass
class DiscoveryResult:
    """Result of scanning one library root."""
    matches: list[SweepFilm]
    total_scanned: int
    # Counts of all parsed files per (title, year), matched or not.
    # Used for "X of Y episodes matched" in the summary.
    parsed_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    # Media files missing a [tmdb-NNN], [imdb-ttNNN], or [tvdb-NNN] tag.
    missing_ids: list[str] = field(default_factory=list)


def _find_media_dirs(library_root: Path) -> list[Path]:
    """Find directories at the depth where media files live.

    Finds the first ``.mkv`` file, determines its parent's depth
    relative to the root, then lists all directories at that depth.
    This adapts to any layout (flat, one-level, genre-grouped, etc.).
    Falls back to immediate children if no media files found.
    """
    # Find one media file to determine the depth. Check each top-level
    # child individually so we don't traverse the entire tree on a slow
    # network volume — we stop as soon as we find the first file.
    sample: Path | None = None
    for child in library_root.iterdir():
        if child.is_dir():
            for ext in _MEDIA_EXTENSIONS:
                for p in child.rglob(f"*{ext}"):
                    sample = p
                    break
                if sample:
                    break
        elif any(child.suffix == ext for ext in _MEDIA_EXTENSIONS):
            sample = child
        if sample:
            break
    if sample is None:
        return sorted(p for p in library_root.iterdir() if p.is_dir())

    # Walk up from the media file's parent to find the title directory —
    # the highest ancestor (below root) whose name contains "(YEAR)".
    # For "root/Show (2023)/Season 1/file.mkv" that's "Show (2023)" at depth 1.
    # For "root/file.mkv" that's root itself.
    _HAS_YEAR = re.compile(r"\(\d{4}\)")
    title_dir: Path | None = None
    cur = sample.parent
    while cur != library_root and cur != cur.parent:
        if _HAS_YEAR.search(cur.name):
            title_dir = cur
        cur = cur.parent

    if title_dir is None:
        # No dir with (YEAR) found — use immediate children of root.
        return sorted(p for p in library_root.iterdir() if p.is_dir())

    # The title dir's parent is the level we want to list.
    level_parent = title_dir.parent
    return sorted(p for p in level_parent.iterdir() if p.is_dir())


def inventory_root(library_root: Path) -> list[Path]:
    """Phase 1: find all media files under a library root.

    Returns a list of media file paths. No catalogue matching — just
    filesystem walk + file counting.
    """
    log.info("root %s: walking filesystem...", library_root)
    media_dirs = _find_media_dirs(library_root)
    n_dirs = len(media_dirs) or 1
    print(f"  root {library_root}: {n_dirs} media directories found")

    media_files: list[Path] = []
    for idx, child in enumerate(media_dirs):
        pct = (idx + 1) * 100 // n_dirs
        print(
            f"\r\033[2K    inventorying [{pct:3d}%] {idx + 1}/{n_dirs} "
            f"({len(media_files)} files) — {child.name}",
            end="", flush=True,
        )
        for ext in _MEDIA_EXTENSIONS:
            media_files.extend(child.rglob(f"*{ext}"))
    # Also check for media files directly in the root (not in subdirs).
    for ext in _MEDIA_EXTENSIONS:
        media_files.extend(library_root.glob(f"*{ext}"))

    # Filter out samples, trailers, featurettes: exclude files in junk
    # subdirectories or below minimum size for a feature.
    before = len(media_files)
    media_files = [
        f for f in media_files
        if not any(part.lower() in _JUNK_SUBDIRS for part in f.parts)
        and f.stat().st_size >= _MIN_FEATURE_SIZE_BYTES
    ]
    skipped = before - len(media_files)
    if skipped:
        log.info("filtered %d samples/trailers/featurettes (<%d MB or junk dir)",
                 skipped, _MIN_FEATURE_SIZE_BYTES // 1_000_000)
    print(f"\r\033[2K    {len(media_files)} media files across {n_dirs} directories.")
    return media_files


def match_media_files(
    all_files: list[tuple[Path, Path]],
    catalogue: list[dict],
) -> DiscoveryResult:
    """Phase 2: parse + catalogue-match all media files across all roots.

    ``all_files`` is a list of ``(media_path, library_root)`` pairs.
    Progress is reported as a single global counter across all roots.
    """
    _ID_TAG_RE = re.compile(r"\[(tmdb|imdb|tvdb)-[^\]]+\]")

    total = len(all_files)
    matches: list[SweepFilm] = []
    missing_ids: list[str] = []
    parsed_counts: dict[tuple[str, int], int] = Counter()
    for i, (media_path, library_root) in enumerate(all_files):
        pct = ((i + 1) * 100) // total if total else 100
        try:
            rel = media_path.relative_to(library_root)
        except ValueError:
            rel = media_path
        print(
            f"\r\033[2K  matching [{pct:3d}%] {i + 1}/{total}  {rel}",
            end="", flush=True,
        )

        parsed = parse_media_filename(media_path)
        if parsed is None:
            log.debug("    → no title/year parsed, skipping")
            continue
        ep_label = f" S{parsed.season:02d}E{parsed.episode:02d}" if parsed.episode else ""
        print(
            f"\r\033[2K  matching [{pct:3d}%] {i + 1}/{total}  {rel}",
        )
        print(f"    → parsed: {parsed.title!r} ({parsed.year}){ep_label}")
        # Flag media missing a [tmdb-NNN]/[imdb-ttNNN]/[tvdb-NNN] tag.
        if not _ID_TAG_RE.search(str(media_path)):
            missing_ids.append(str(media_path))

        parsed_counts[(parsed.title, parsed.year)] += 1
        entry = match_catalogue(
            catalogue, parsed.title, parsed.year,
            season=parsed.season, episode=parsed.episode,
        )
        if entry is None:
            print("    → no catalogue match")
            continue
        cat_season = entry.get("season", "")
        cat_episode = entry.get("episode", "")
        cat_label = f" (catalogue: season={cat_season} episode={cat_episode})" if cat_season else ""
        print(f"    → MATCHED: {entry.get('title')!r} ({parsed.year}){cat_label}")
        try:
            file_size = media_path.stat().st_size
        except OSError:
            file_size = 0
        matches.append(SweepFilm(
            path=str(media_path),
            library_root=str(library_root),
            title=parsed.title,
            year=parsed.year,
            rating=extract_rating(entry),
            season=parsed.season,
            episode=parsed.episode,
            catalogue_entry=entry,
            size_bytes=file_size,
        ))

    print()  # newline after progress
    match_pct = (len(matches) * 100 // total) if total else 0
    log.info("matched=%d/%d (%d%%)", len(matches), total, match_pct)

    # Save missing-ID list to a file for the user to fix.
    if missing_ids:
        missing_file = Path.home() / ".config" / "beqdesigner" / "media_missing_ids.txt"
        missing_file.parent.mkdir(parents=True, exist_ok=True)
        missing_file.write_text("\n".join(sorted(set(missing_ids))) + "\n")
        log.info("%d media files missing [tmdb/imdb/tvdb] ID tags → %s",
                 len(set(missing_ids)), missing_file)

    return DiscoveryResult(
        matches=matches, total_scanned=total, parsed_counts=dict(parsed_counts),
        missing_ids=missing_ids,
    )


def discover_matches(
    library_root: Path,
    catalogue: list[dict],
) -> DiscoveryResult:
    """Legacy single-root entry point. Calls inventory + match."""
    files = inventory_root(library_root)
    all_files = [(f, library_root) for f in files]
    return match_media_files(all_files, catalogue)


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


def _catalogue_digest(entry: dict) -> str:
    """Return the catalogue entry's unique digest hash."""
    return entry.get("digest", "")


def write_config(
    films: list[SweepFilm],
    library_roots: list[Path],
    test_limit: int,
    output: Path | None = None,
    catalogue_url: str = _DEFAULT_CATALOGUE_URL,
) -> Path:
    """Write the discovered films to the sweep config JSON.

    Each film stores only the catalogue entry's ``digest`` hash.
    The full entry is looked up from the cached catalogue at test time.
    This keeps the config small regardless of how many episodes match.
    """
    path = output or _default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()

    film_records = []
    for f in films:
        film_records.append({
            "path": f.path,
            "library_root": f.library_root,
            "title": f.title,
            "year": f.year,
            "rating": f.rating,
            "season": f.season,
            "episode": f.episode,
            "catalogue_digest": _catalogue_digest(f.catalogue_entry),
        })

    payload = {
        "schema_version": 3,
        "generated_at": now,
        "library_roots": [str(r) for r in library_roots],
        "catalogue_url": catalogue_url,
        "catalogue_fetched_at": now,
        "test_limit": test_limit,
        "films": film_records,
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


def load_catalogue_by_digest(
    catalogue_path: Path | None = None,
) -> dict[str, dict]:
    """Load the cached catalogue and index by digest hash.

    Returns ``{digest: entry_dict}``. Used at test time to resolve
    the ``catalogue_digest`` stored in the sweep config.
    """
    path = catalogue_path or _catalogue_cache_path()
    if not path.exists():
        return {}
    with path.open() as f:
        entries = json.load(f)
    return {e.get("digest", ""): e for e in entries if e.get("digest")}


def _save_library_roots(library_roots: list[Path], output: Path | None = None) -> None:
    """Persist library roots to config immediately (survives Ctrl-C during scan).

    Merges into existing config if present, otherwise creates a minimal stub.
    """
    path = output or _default_config_path()
    existing = load_config(path) or {}
    existing["library_roots"] = [str(r) for r in library_roots]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_library_roots(args: argparse.Namespace) -> list[Path]:
    """Resolve library roots from (in order) CLI flags, env var, previous config, prompt."""
    if args.library:
        return _split_paths(",".join(args.library))
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
    """Split a string of paths by comma, strip whitespace and shell escapes.

    Users often paste paths with backslash-escaped spaces (e.g. from
    shell tab-completion or drag-and-drop): ``/Volumes/DMZ\\ Storage``.
    Interactive input isn't shell-parsed, so we strip those escapes.
    """
    parts = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        # Remove backslash escapes: "DMZ\ Storage" -> "DMZ Storage"
        p = p.replace("\\ ", " ")
        # Also handle double-backslash from some terminals
        p = p.replace("\\\\", "\\")
        parts.append(Path(p).expanduser())
    return parts


def _print_summary(
    matches: list[SweepFilm],
    library_roots: list[Path],
    total_scanned: int,
    parsed_counts: dict[tuple[str, int], int] | None = None,
    per_root_scanned: dict[str, int] | None = None,
) -> None:
    if parsed_counts is None:
        parsed_counts = {}
    if per_root_scanned is None:
        per_root_scanned = {}

    print()
    print("\n── Summary ──")
    print(f"Library roots scanned: {len(library_roots)}")
    for root in library_roots:
        n_matched = sum(1 for m in matches if m.library_root == str(root))
        n_scanned = per_root_scanned.get(str(root), 0)
        pct = (n_matched * 100 // n_scanned) if n_scanned else 0
        print(f"  {root}  →  {n_matched} catalogue matches / {n_scanned} files ({pct}%)")

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
        label = f"{bucket:.1f}" if bucket >= 0 else "n/a"
        print(f"  {label}: {buckets[bucket]}")

    print()
    print(f"All {unique_titles} matched titles (rating desc, year desc):")
    for i, ((title, year), films) in enumerate(grouped.items(), 1):
        rating = films[0].rating
        r_part = f" [r={rating:.1f}]" if rating is not None else ""
        matched_count = len(films)
        total_for_title = parsed_counts.get((title, year), matched_count)
        has_episodes = any(f.season is not None for f in films)

        if not has_episodes and total_for_title <= 1:
            suffix = ""
        else:
            # Group matched episodes by season.
            seasons: dict[str, int] = Counter()
            for f in films:
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
            pct = (matched_count * 100 // total_for_title) if total_for_title else 100
            suffix = (
                f"  ({matched_count}/{total_for_title} episodes matched "
                f"— {pct}% — {season_str})"
            )
        print(f"  {i:3d}. [{year}]{r_part} {title}{suffix}")
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
        "--clean", action="store_true",
        help="Delete existing config and start fresh.",
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

    if args.clean:
        config_path = args.output or _default_config_path()
        if config_path.exists():
            config_path.unlink()
            print(f"Deleted {config_path}")
        else:
            print(f"No config to clean at {config_path}")

    try:
        library_roots = _resolve_library_roots(args)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    for root in library_roots:
        if not root.exists():
            print(f"error: library root does not exist: {root}", file=sys.stderr)
            return 2

    # Persist library roots immediately so they survive Ctrl-C during scan.
    _save_library_roots(library_roots, output=args.output)

    catalogue = load_or_fetch_catalogue(force_refresh=args.refresh)
    log.info("catalogue entries: %d", len(catalogue))

    # ---- Phase 1: Inventory ----
    # Scan all roots first to get a global file count for progress.
    print("\n── Phase 1: Inventory ──")
    all_files: list[tuple[Path, Path]] = []
    per_root_scanned: dict[str, int] = {}
    for root in library_roots:
        files = inventory_root(root)
        all_files.extend((f, root) for f in files)
        per_root_scanned[str(root)] = len(files)
    print(f"\nTotal media files across all roots: {len(all_files)}")

    # ---- Phase 2: Matching ----
    print("\n── Phase 2: Catalogue matching ──")
    result = match_media_files(all_files, catalogue)
    matches = sort_matches(result.matches)
    total_scanned = result.total_scanned
    parsed_counts = result.parsed_counts

    _print_summary(matches, library_roots, total_scanned, parsed_counts,
                    per_root_scanned=per_root_scanned)

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
    print(f"\nWrote {len(matches)} films to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
