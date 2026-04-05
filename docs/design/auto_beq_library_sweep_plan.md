# Dynamic catalogue-sweep testing harness

## Quick start

The sweep has two phases: **discover** (one-time, fast) then **run**
(per-session, slow on first run per film).

### 1. Discover your library

```bash
bash scripts/run-sweep-discover.sh --library /path/to/movies
```

This walks the library, fetches the full BEQ catalogue (14,788
entries, cached for 24 hours), matches every film by title+year, and
saves ALL matches to `~/.config/beqdesigner/auto_beq_sweep.json`.

Supports multiple library roots and two filename conventions:
- Plex/Jellyfin: `Title (YEAR) [tmdb-NNN]`
- Scene-style: `Title.Name.YEAR.codec.source.mkv`

Options:
```
--library PATH    repeatable; falls back to $AUTO_BEQ_LIBRARY_ROOTS (colon-separated) or .env
--refresh         re-fetch catalogue even if the cached copy is <24h old
--yes             skip confirmation prompt
--test-limit N    default number of films the sweep test runs (default: 10)
--output PATH     override config file location
```

Re-run discovery any time your library changes or you want to refresh
the catalogue.

### 2. Run the sweep

```bash
AUTO_BEQ_ADVISOR=measurement \
SPIKE_TEST=src/test/python/spike/test_auto_beq_library_sweep.py \
  bash scripts/run-spike-tests.sh
```

This runs the full pipeline (extract LFE → smooth → advisor →
propose filters → grade vs catalogue ground truth) on the top N
films from your config (N = `test_limit` in config, default 10).

First run per film is slow (~30-60s for LFE extraction via ffmpeg).
Subsequent runs are fast (WAV is cached next to the source file as
`<stem>.lfe-1000hz.wav`).

Override the limit for faster iteration:

```bash
AUTO_BEQ_SWEEP_LIMIT=3 AUTO_BEQ_ADVISOR=measurement \
SPIKE_TEST=src/test/python/spike/test_auto_beq_library_sweep.py \
  bash scripts/run-spike-tests.sh
```

### 3. View results

Each run appends to a CSV report:

```bash
column -t -s, .pytest_cache/auto_beq_sweep.csv
```

Columns: `rating_bucket`, `rating`, `release_year`, `title`,
`catalogue_filter_count`, `advisor`, `verdict`, `mean_err_db`,
`max_err_db`, `summed_catalogue_gain_db`.

Delete the CSV file before running to start fresh.

### Environment variables

| Var | Purpose | Default |
|---|---|---|
| `AUTO_BEQ_ADVISOR` | advisor: heuristic / mock / ollama / measurement | `mock` (via run-spike-tests.sh) |
| `AUTO_BEQ_SWEEP_LIMIT` | override test_limit from config | config value (default 10) |
| `AUTO_BEQ_SWEEP_CONFIG` | override config file path | `~/.config/beqdesigner/auto_beq_sweep.json` |
| `AUTO_BEQ_SWEEP_REPORT` | override CSV output path | `.pytest_cache/auto_beq_sweep.csv` |
| `AUTO_BEQ_LIBRARY_ROOTS` | colon-separated library paths (discovery) | unset |
| `SPIKE_VERBOSE` | `1` = show stdout during tests | `0` |

---

## Design context

### Architecture

Two-phase workflow:
- **Phase A — Discovery** (`scripts/run-sweep-discover.sh`): maps
  the entire library, persists all matches to user config. Fast,
  one-time, interactive.
- **Phase B — Sweep test** (`test_auto_beq_library_sweep.py`): reads
  persisted config, applies a limit, runs the pipeline. Slow (LFE
  extraction), informational (no assertions).

### File layout

| File | Role |
|---|---|
| `scripts/run-sweep-discover.sh` | Discovery wrapper (sets PYTHONPATH) |
| `src/test/python/spike/sweep_discover.py` | Discovery CLI implementation |
| `src/test/python/spike/test_auto_beq_library_sweep.py` | Parametrised sweep test |
| `src/test/python/spike/test_sweep_discover.py` | 42 unit tests for discovery |
| `src/test/python/spike/_auto_beq_helpers.py` | Shared helpers (extracted from test_auto_beq.py) |
| `src/test/resources/auto_beq/fake_library/` | Committed fixture for discovery tests |
| `~/.config/beqdesigner/auto_beq_sweep.json` | Persisted discovery config (user-local) |
| `~/.config/beqdesigner/beq_catalogue_cache.json` | Cached full catalogue (24h TTL) |

### Original motivation

The auto-BEQ spike currently tests 3 hand-picked films from a JSON
manifest at `~/.config/beqdesigner/auto_beq_media.json`. That's too
narrow a sample to evaluate how well the pipeline generalises across
the user's real library.

The user has a media library at `/Volumes/DMZ Storage .../Movies/`
where films are tagged with standard Plex/Jellyfin filename
conventions like `Title (YEAR) [tmdb-NNNNN]`. The BEQ catalogue has
~14,700 entries; many of those titles exist in the user's library.

We need a **parametrised pytest test that auto-discovers real films
from the library, cross-references them against the catalogue, and
runs the full auto-BEQ pipeline on each match**, producing a CSV
report sorted by catalogue `rating` (bucketed to 0.5) and release
year (newest first).

This gives statistics across dozens of films instead of just 3, so
we can honestly answer "how often does the measurement-only advisor
produce a passable chain" across varied real content.

### Parallel-session note (historical)

This plan is designed to run in a **separate session** alongside
ongoing work in `auto_beq_advisor.py` / `MeasurementAdvisor`. The
sweep test reuses the `Advisor` interface as-is; it doesn't need to
know which advisor impl is active (selection via `AUTO_BEQ_ADVISOR`
env var, same as the existing real-media test).

Avoid touching:
- `src/main/python/model/auto_beq_advisor.py` (being actively
  edited)
- `src/main/python/model/auto_beq.py`
- `src/test/python/spike/test_auto_beq.py` (except the small helper
  extraction below)

### Original design goal

A new parametrised test in
`src/test/python/spike/test_auto_beq_library_sweep.py` that:

1. On collection time (NOT at test-runtime), walks
   `AUTO_BEQ_LIBRARY_ROOT` (env var), parses filenames for title
   and year using Plex/Jellyfin conventions.
2. Cross-references each parsed (title, year) against the BEQ
   catalogue (reuse `load_catalogue` from `model.catalogue` or the
   committed snapshot for offline deterministic runs).
3. Filters to titles that HAVE a catalogue entry AND the media file
   exists on disk.
4. Sorts matches by rating bucket desc, release year desc.
5. Emits one parametrised test per match.
6. Each test runs the full extract-smooth-advise-fit-grade pipeline
   (same as `test_real_media_roundtrip`) and appends a row to a CSV
   report.
7. Skipped when `AUTO_BEQ_LIBRARY_ROOT` is unset.

## Implementation plan (historical)

### File: `src/test/python/spike/test_auto_beq_library_sweep.py` (new)

```python
"""Library sweep: run the auto-BEQ pipeline over every film in the
user's library that has a matching catalogue entry.

Activates when AUTO_BEQ_LIBRARY_ROOT env var points to a directory
of media files tagged with Plex-style filenames
('Title (YEAR) [tmdb-NNNNN]'). Finds all films that also have a BEQ
catalogue entry, runs the full pipeline, and writes a CSV report
summarising per-film results.

This complements test_real_media_roundtrip's 3 hand-picked fixtures
with statistics across dozens of films to show how the advisor
generalises beyond the fixture set.
"""
```

### Module-level collection fixture

Build the parametrised list at collection time (before any test
runs), so pytest's `parametrize(ids=...)` can print per-title IDs:

```python
_LIBRARY_ROOT = os.environ.get("AUTO_BEQ_LIBRARY_ROOT")
_SWEEP_FILMS = _discover_sweep_films() if _LIBRARY_ROOT else []


def _discover_sweep_films() -> list[SweepFilm]:
    """Walk library, parse filenames, match catalogue."""
    root = Path(os.environ["AUTO_BEQ_LIBRARY_ROOT"])
    catalogue = _load_catalogue_snapshot_or_fresh()
    matches: list[SweepFilm] = []
    for media_path in root.rglob("*.mkv"):
        parsed = _parse_plex_filename(media_path)
        if parsed is None:
            continue
        title, year = parsed
        entry = _match_catalogue(catalogue, title, year)
        if entry is None:
            continue
        matches.append(SweepFilm(
            path=media_path,
            title=title,
            year=year,
            entry=entry,
            rating=_extract_rating(entry),
        ))
    # Sort: rating bucket desc, year desc.
    matches.sort(key=lambda f: (-_bucket_rating(f.rating), -(f.year or 0)))
    return matches
```

### Filename parsing

`Title (YEAR) [tmdb-NNNNN]` is common; also accept `[imdb-ttNNNNN]`
and fallback to `Title (YEAR)`:

```python
_FILENAME_RE = re.compile(
    r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)"
    r"(?:\s*\[(?:tmdb|imdb)-[^\]]+\])?"
)

def _parse_plex_filename(path: Path) -> tuple[str, int] | None:
    # Use the PARENT directory name if it parses, else the filename
    # stem (Plex uses directory naming).
    for candidate in (path.parent.name, path.stem):
        m = _FILENAME_RE.match(candidate)
        if m:
            return (m.group("title").strip(), int(m.group("year")))
    return None
```

### Catalogue matching

Match by normalised title + year (exact year match required; fuzzy
title match optional, log warnings on multi-match):

```python
def _normalise_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_catalogue(catalogue: list[dict], title: str, year: int) -> dict | None:
    norm = _normalise_title(title)
    candidates = [
        e for e in catalogue
        if _normalise_title(e.get("title", "")) == norm
        and e.get("year") == year
        and e.get("filters")
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        # Multiple catalogue entries for the same title+year
        # (e.g. 5-filter vs 7-filter variants). Pick the one with
        # the most filters - tends to be the most comprehensive.
        candidates.sort(key=lambda e: len(e["filters"]), reverse=True)
    return candidates[0]
```

### Rating bucketing

```python
def _extract_rating(entry: dict) -> float | None:
    r = entry.get("rating")
    try:
        return float(r) if r is not None else None
    except (TypeError, ValueError):
        return None


def _bucket_rating(rating: float | None) -> float:
    """Bucket to 0.5 granularity. None -> -1 (sorts last)."""
    if rating is None:
        return -1.0
    return float(int(rating * 2)) / 2.0
```

### The parametrised test

```python
@pytest.mark.skipif(not _SWEEP_FILMS, reason="no AUTO_BEQ_LIBRARY_ROOT matches")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[
        f"rating-{_bucket_rating(f.rating):.1f}|{f.year}|{f.title}"
        for f in _SWEEP_FILMS
    ],
)
def test_library_sweep(film: SweepFilm, caplog):
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    # 1. Extract LFE (cached).
    wav_path = _extract_lfe_wav(film.path, fs=1000)
    # 2. Load signal, compute avg spectrum, interp, normalise, smooth.
    measured_on_grid = _load_and_smooth(wav_path)
    # 3. Advisor.
    advisor = get_advisor()  # honors AUTO_BEQ_ADVISOR env var
    metadata = MediaMetadata(title=film.title, year=film.year)
    proposed = propose_filters_from_measured(
        measured_on_grid, DEFAULT_GRID, fs=1000,
        advisor=advisor, metadata=metadata,
    )
    # 4. Ground truth + grade.
    ground_truth = evaluate_filter_chain(
        film.entry["filters"], DEFAULT_GRID, fs=1000,
    )
    metrics = compute_match_metrics(-ground_truth, proposed, DEFAULT_GRID, fs=1000)
    # 5. Append to CSV report.
    _append_sweep_report(film, advisor.name, metrics, len(film.entry["filters"]))
    # Log PASS/MARGINAL/FAIL but do NOT assert (sweep is
    # informational - failures are the point of running it).
```

### CSV report

```python
_SWEEP_REPORT_PATH = Path(os.environ.get(
    "AUTO_BEQ_SWEEP_REPORT", ".pytest_cache/auto_beq_sweep.csv",
))


def _append_sweep_report(film, advisor_name, metrics, catalogue_filter_count):
    # Create header if file doesn't exist.
    new = not _SWEEP_REPORT_PATH.exists()
    _SWEEP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SWEEP_REPORT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "rating_bucket", "rating", "release_year", "title",
                "catalogue_filter_count", "advisor", "verdict",
                "mean_err_db", "max_err_db", "summed_catalogue_gain_db",
            ])
        w.writerow([
            f"{_bucket_rating(film.rating):.1f}",
            f"{film.rating:.2f}" if film.rating else "",
            film.year or "", film.title,
            catalogue_filter_count, advisor_name,
            metrics.verdict,
            f"{metrics.mean_abs_err_db:.2f}",
            f"{metrics.max_abs_err_db:.2f}",
            f"{_summed_low_shelf_gain(film.entry):.1f}",
        ])
```

### Helpers to extract from `test_auto_beq.py`

Several functions need to be shared. **Extract them to a new
helper module** `src/test/python/spike/_auto_beq_helpers.py`:

- `_have_tool(name: str) -> bool` (existing in test_auto_beq.py)
- `_probe_audio_stream(media_path: Path) -> dict` (existing)
- `_extract_lfe_wav(media_path: Path, fs: int) -> Path` (existing)
- NEW: `load_and_smooth(wav_path: Path, fs: int, freqs: np.ndarray) -> np.ndarray`
  that wraps the Signal-load, interp-to-grid, normalise-to-80Hz,
  1/6-octave-smooth pipeline.

Update `test_auto_beq.py` to import these from the helper module.
Don't change the existing test logic.

## Environment variables

| Var | Purpose | Default |
|---|---|---|
| `AUTO_BEQ_LIBRARY_ROOT` | Path to media library | unset → skip |
| `AUTO_BEQ_SWEEP_REPORT` | Where to write CSV | `.pytest_cache/auto_beq_sweep.csv` |
| `AUTO_BEQ_ADVISOR` | Advisor to use | `measurement` (same default as test_real_media_roundtrip) |
| `AUTO_BEQ_SWEEP_LIMIT` | Max films to test (for debugging) | unset → all |
| `AUTO_BEQ_SWEEP_MIN_RATING` | Skip films below this rating | unset → no filter |

## File structure

### New files
- `src/test/python/spike/test_auto_beq_library_sweep.py` — parametrised sweep test
- `src/test/python/spike/_auto_beq_helpers.py` — shared test helpers

### Modified files
- `src/test/python/spike/test_auto_beq.py` — import helpers from new
  module; remove the copies. Keep all test logic unchanged.

### Do NOT modify
- `src/main/python/model/auto_beq.py`
- `src/main/python/model/auto_beq_advisor.py` (being actively edited)
- `src/test/resources/auto_beq/database.json` (keep the committed
  snapshot; optionally offer a fresh-fetch mode via env var)

## Catalogue source

**Default**: load from the committed snapshot
`src/test/resources/auto_beq/database.json` (18 entries, enough for
small-library matches plus the fixture films).

**Opt-in fresh fetch** via `AUTO_BEQ_SWEEP_FRESH_CATALOGUE=1`:
download from
`https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json`
at collection time, cache to
`.pytest_cache/beq_catalogue_full.json` with a 24-hour TTL. This
gives access to the full ~14,700 entries when the user wants broad
coverage.

## Verification

1. `AUTO_BEQ_LIBRARY_ROOT=/Volumes/... AUTO_BEQ_ADVISOR=measurement bash scripts/run-spike-tests.sh`
   (with `SPIKE_TEST=src/test/python/spike/test_auto_beq_library_sweep.py`)
   → discovers N matching films, runs N tests, writes CSV.
2. Without the env var: test collection yields 0 cases, reported
   as "skipped" by pytest, CI unaffected.
3. Without ffmpeg: all cases SKIPPED with clear reason.
4. `column -t -s, .pytest_cache/auto_beq_sweep.csv | sort -k1,1 -k3,3 -r | head -30`
   → readable sorted output of per-film results.
5. `poetry run ruff check src/test/python/spike/` → clean.

## Success criteria

- Test collection runs in <5 seconds (parsing filenames + catalogue
  matches is all in-memory lookup).
- Each film's test runs in <5 seconds once LFE WAV is cached
  (extraction is the bottleneck, but cached on second run).
- CSV report is sortable, readable, and has one row per discovered
  film.
- The report lets the user answer:
  - What fraction of my library's matched films grade PASS with
    `MeasurementAdvisor`?
  - Does the pass rate correlate with rating / release year /
    catalogue-filter-count?

## Risks / known issues

1. **First-run extraction is slow** — feature-length 5.1/7.1 LFE
   extraction at 1 kHz typically takes 30-60s per film. With a
   library of 100+ matches, first run takes hours. Subsequent runs
   are fast (cached WAV per film). Mitigate: `AUTO_BEQ_SWEEP_LIMIT`
   env var lets the user start small.
2. **Filename parsing misses** — films with non-standard names won't
   match. Log INFO per skipped file so the user can audit.
3. **Catalogue duplicates** — some titles have multiple catalogue
   entries (different releases). The matcher picks the richest
   (most filters) entry; this is arbitrary. Could add
   `AUTO_BEQ_SWEEP_FILTER_COUNT=N` to target a specific variant.
4. **Network volumes unmount** — existing test skips with
   `assert media_path.exists()`. Sweep should skip rather than fail
   on missing paths (the match list is built at collection time,
   but paths may disappear between collection and test runtime).
5. **Report CSV grows indefinitely** — each test run appends. For a
   clean slate, delete the file before running. Consider a flag
   `AUTO_BEQ_SWEEP_CLEAN_REPORT=1` to truncate at collection time.

## Non-goals

- No pass/fail assertions on individual sweep tests (they're
  informational — CI won't run them anyway without the env var).
- No TMDB API integration — the catalogue's `rating` field is the
  truth source.
- No automatic library-scanning recursion into nested subfolders
  beyond what `rglob` gives us.
- No UI / GUI integration.
