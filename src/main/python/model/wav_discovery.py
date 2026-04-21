"""WAV-catalogue matching, settings, and directory configuration.

Handles discovering WAV files in the cache that match BEQ catalogue entries,
finding unmatched WAVs, managing settings persistence, and providing the
canonical directory paths for the BEQ shared directory, WAV cache, and
config directory. All functions are safe for CLI and Docker use - no PyQt6
or Qt imports.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory and settings helpers
# ---------------------------------------------------------------------------

def audio_cache_dir() -> Path:
    """Return the audio cache directory, creating it if needed.

    .. deprecated::
        Use ``wav_cache_dir()`` and ``cli.extract.cache_path()`` instead.
        This function builds a mirrored-path layout which is being replaced
        by the two-letter bucket layout. Kept for backward compatibility
        with callers that have not migrated yet.

    **Required config** - set ``AUTO_BEQ_AUDIO_CACHE`` env var or
    ``audio_cache_dir`` (deprecated) / ``shared_beq_dir`` + ``/wav-cache``
    in ``~/.config/beqdesigner/settings.json``.
    We never write WAV files next to source media; all extractions go
    into this cache dir with a mirrored path structure.

    Raises RuntimeError if not configured.
    """
    raw = os.environ.get("AUTO_BEQ_AUDIO_CACHE")
    if not raw:
        cfg_path = _settings_path()
        if cfg_path.exists():
            with cfg_path.open() as _f:
                try:
                    data = json.load(_f)
                    raw = data.get("audio_cache_dir")
                    if not raw and data.get("shared_beq_dir"):
                        # Derive from shared_beq_dir + /wav-cache.
                        raw = str(Path(data["shared_beq_dir"]) / "wav-cache")
                except (json.JSONDecodeError, ValueError) as exc:
                    log.warning("failed to parse %s: %s", cfg_path, exc)
    if not raw:
        raise RuntimeError(
            "audio cache dir not configured. Set AUTO_BEQ_AUDIO_CACHE env var "
            "or add {\"audio_cache_dir\": \"/path/to/cache\"} "
            "to ~/.config/beqdesigner/settings.json"
        )
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def beq_config_dir() -> Path:
    """Return the BEQ designer user-config directory, creating it if needed."""
    path = Path.home() / ".config" / "beqdesigner"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings_path() -> Path:
    """Return the path to settings.json via beq_config_dir()."""
    return beq_config_dir() / "settings.json"


# Backward-compat module-level alias for tests that monkeypatch it.
_SETTINGS_PATH = _settings_path


def load_settings() -> dict:
    """Load the shared settings.json, returning {} on any error."""
    path = _settings_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("failed to parse %s: %s - returning empty settings", path, exc)
        except OSError as exc:
            log.warning("could not read %s: %s", path, exc)
    return {}


def save_settings(settings: dict) -> None:
    """Persist the shared settings.json (merges with existing)."""
    existing = load_settings()
    existing.update(settings)
    path = _settings_path()
    path.write_text(json.dumps(existing, indent=2) + "\n")


def beq_shared_dir() -> Path:
    """Return the BEQ shared working directory.

    Holds wav-cache/, beq_catalogue.json, media_inventory.json, and
    production_model.joblib.  This is the *portable* shared directory
    (e.g. a NAS mount) - machine-specific config lives in
    ``beq_config_dir()`` (~/.config/beqdesigner/).

    Resolution order:
    1. ``BEQ_SHARED_DIR`` env var
    2. ``shared_beq_dir`` in ``~/.config/beqdesigner/settings.json``

    Raises RuntimeError if not configured.
    """
    raw = os.environ.get("BEQ_SHARED_DIR")
    if raw:
        p = Path(raw).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    cfg_path = _settings_path()
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
            raw = data.get("shared_beq_dir")
            if raw:
                p = Path(raw).expanduser()
                p.mkdir(parents=True, exist_ok=True)
                return p
        except Exception:
            pass

    raise RuntimeError(
        "BEQ shared directory not configured. "
        "Set BEQ_SHARED_DIR env var or add "
        '{"shared_beq_dir": "/path/to/dir"} '
        "to ~/.config/beqdesigner/settings.json"
    )


def wav_cache_dir() -> Path:
    """Return the portable WAV cache directory.

    Single source of truth for where extracted LFE WAVs live.  Resolution
    order: ``wav_cache_dir`` in ``~/.config/beqdesigner/settings.json``
    -> ``shared_beq_dir`` + ``/wav-cache`` -> ``audio_cache_dir``
    (backward compat) -> error.

    Raises RuntimeError if not configured.
    """
    explicit_source: str | None = None
    raw: str | None = None
    cfg_path = _settings_path()
    if cfg_path.exists():
        with cfg_path.open() as _f:
            try:
                data = json.load(_f)
                raw = data.get("wav_cache_dir")
                if raw:
                    explicit_source = f"wav_cache_dir in {cfg_path}"
                elif data.get("shared_beq_dir"):
                    # Derive from shared_beq_dir + /wav-cache.
                    raw = str(Path(data["shared_beq_dir"]) / "wav-cache")
                    explicit_source = f"shared_beq_dir in {cfg_path}"
                elif data.get("audio_cache_dir"):
                    # Fall back to audio_cache_dir if wav_cache_dir not set.
                    # Deprecated - prefer shared_beq_dir.
                    raw = data["audio_cache_dir"]
                    explicit_source = f"audio_cache_dir in {cfg_path}"
            except Exception:
                pass
    # Derive from beq_shared_dir() as last resort.
    if not raw:
        try:
            raw = str(beq_shared_dir() / "wav-cache")
            explicit_source = None  # auto-derived, will auto-create
        except Exception:
            pass
    if not raw:
        raise RuntimeError(
            "WAV cache dir not configured. Set BEQ_SHARED_DIR env var "
            'or add {"wav_cache_dir": "/path/to/cache"} '
            "to ~/.config/beqdesigner/settings.json"
        )

    path = Path(raw).expanduser()

    if explicit_source is not None:
        # User explicitly configured this path - fail fast if it doesn't
        # exist (e.g. network mount dropped, typo in the path).
        if not path.exists():
            raise FileNotFoundError(
                f"configured WAV cache does not exist: {path} "
                f"(from {explicit_source}). "
                f"Check that the path is correct and, if on a network "
                f"mount, that the mount is active.",
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"configured WAV cache path is not a directory: {path} "
                f"(from {explicit_source}).",
            )
    else:
        # Default fallback path - auto-create for first-time users.
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Disk-backed discovery cache
# ---------------------------------------------------------------------------

# Matches [tmdb-NNN], [tvdb-NNN], or [imdb-NNN] in a path string.
_ID_RE = __import__("re").compile(r"\[(tmdb|tvdb|imdb)-([^\]]+)\]")

_DISCOVERY_CACHE_ENABLED_ENV = "AUTO_BEQ_DISCOVERY_CACHE"
_DISCOVERY_CACHE_FILENAME = "discovered_pairs.pkl"
_CATALOGUE_CACHE_FILENAME = "beq_catalogue.json"
_UNMATCHED_CACHE_FILENAME = "unmatched_wavs.pkl"


def _cache_enabled(env_var: str) -> bool:
    """Cache helpers honour env var opt-out. Default: enabled."""
    return os.environ.get(env_var, "1") != "0"


def _latest_wav_mtime(cache_root: Path) -> float:
    """Return the max mtime across top-level dirs in the cache, or 0.0.

    Uses directory mtimes (cheap - one scandir) rather than statting
    every WAV file (slow on NFS). A directory's mtime changes when
    files are added/removed inside it, which is good enough for
    staleness detection.
    """
    latest = 0.0
    try:
        latest = max(latest, cache_root.stat().st_mtime)
        for entry in os.scandir(cache_root):
            if entry.is_dir():
                try:
                    m = entry.stat().st_mtime
                    if m > latest:
                        latest = m
                except OSError:
                    pass
    except OSError:
        pass
    return latest


def ensure_analysis_reports_current(
    repo_root: Path | None = None,
    force: bool = False,
) -> list[str]:
    """Regenerate analysis reports when they're stale vs the WAV cache.

    Reports are written to ``{beq_shared_dir}/reports/`` (not checked
    into the repo). Regenerates (via subprocess) any report whose mtime
    is older than the newest WAV.

    Called at the start of experiment harness runs so downstream analysis
    always reflects the current training set without a manual step.

    Returns the list of reports that were (re)generated.
    """
    import subprocess

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    cache_root = wav_cache_dir()
    cache_mtime = _latest_wav_mtime(cache_root)
    if cache_mtime == 0.0:
        log.info("WAV cache is empty; skipping analysis report refresh")
        return []

    try:
        report_dir = beq_shared_dir() / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
    except RuntimeError:
        log.info("BEQ shared dir not configured; skipping report refresh")
        return []
    except OSError as exc:
        log.warning("could not create report output dir: %s; skipping report refresh", exc)
        return []

    reports = [
        (
            report_dir / "wav_cache_bias.md",
            repo_root / "src" / "main" / "python" / "cli" / "nn_cache_bias_report.py",
            ["-o"],
        ),
        (
            report_dir / "author_patterns.md",
            repo_root / "src" / "main" / "python" / "cli" / "nn_author_pattern_report.py",
            ["-o"],
        ),
        (
            report_dir / "acquisition_recommendations.md",
            repo_root / "src" / "main" / "python" / "cli" / "nn_acquisition_recommender.py",
            ["-n", "50", "-o"],
        ),
    ]

    regenerated: list[str] = []
    for report_path, script_path, extra_args in reports:
        stale = (
            force
            or not report_path.exists()
            or report_path.stat().st_mtime < cache_mtime
        )
        if not stale:
            log.debug("report up to date: %s", report_path.name)
            continue

        log.info("regenerating stale report: %s", report_path.name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3", str(script_path),
            *extra_args, str(report_path),
        ]
        env = {**os.environ}
        env.setdefault(
            "PYTHONPATH",
            f"{repo_root / 'src/main/python'}:{repo_root / 'src/test/python'}",
        )
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            subprocess.run(
                cmd, check=True, env=env, cwd=str(repo_root),
                capture_output=True, text=True, timeout=120,
            )
            regenerated.append(report_path.name)
        except subprocess.CalledProcessError as e:
            log.warning(
                "failed to regenerate %s: %s\nstderr:\n%s",
                report_path.name, e, e.stderr[-500:] if e.stderr else "",
            )
        except subprocess.TimeoutExpired:
            log.warning("timeout regenerating %s", report_path.name)

    return regenerated


def _count_wav_files(cache_root: Path) -> int:
    """Count WAV files via os.scandir at each level (NFS-safe).

    Walks bucket/shard/file structure counting files matching WAV_SUFFIX.
    O(buckets * shards) scandir calls - fast on NFS, no per-file stat.
    """
    count = 0
    try:
        for bucket in os.scandir(cache_root):
            if not bucket.is_dir():
                continue
            try:
                for shard in os.scandir(bucket.path):
                    if not shard.is_dir():
                        continue
                    try:
                        for entry in os.scandir(shard.path):
                            if entry.is_file() and entry.name.endswith(".wav"):
                                count += 1
                    except OSError:
                        pass
            except OSError:
                pass
    except OSError:
        pass
    return count


def _discovery_cache_signature() -> dict | None:
    """Lightweight signature using WAV count + catalogue mtime.

    Invalidation sources:
      * wav_count - changes when WAVs are added or removed anywhere in
        the cache tree, including inside existing subdirectories (which
        the old mtime-only check missed).
      * catalogue cache file mtime - bumped when the BEQ catalogue is refetched.

    Returns None if stats fail (cache miss forced).
    """
    try:
        wav_root = wav_cache_dir()
        if not wav_root.exists():
            return None
        wav_count = _count_wav_files(wav_root)
    except Exception:
        return None

    catalogue_mtime = 0.0
    catalogue_cache = beq_shared_dir() / _CATALOGUE_CACHE_FILENAME
    if catalogue_cache.exists():
        try:
            catalogue_mtime = catalogue_cache.stat().st_mtime
        except Exception:
            catalogue_mtime = 0.0

    return {
        "wav_count": wav_count,
        "catalogue_mtime": float(catalogue_mtime),
    }


def _match_wavs_to_catalogue(
    wav_files: list[Path],
) -> tuple[list[dict], list[Path]]:
    """Match WAV files against the BEQ catalogue.

    Shared matching logic used by both ``discover_wav_catalogue_pairs``
    and ``discover_unmatched_wavs``. Matches by media DB ID extracted
    from the WAV path, with title+year fallback from directory names.

    Returns (matched_pairs, unmatched_wavs).
    """
    import re as _re

    from model.auto_beq_catalogue import _fetch_or_cache
    catalogue = _fetch_or_cache()

    # Index by tmdb ID (primary) and title+year (fallback for tvdb/imdb).
    by_tmdb: dict[str, list[dict]] = {}
    by_title_year: dict[tuple[str, str], list[dict]] = {}
    for e in catalogue:
        tid = str(e.get("theMovieDB", "")).strip()
        if tid:
            by_tmdb.setdefault(tid, []).append(e)
        key = (e.get("title", "").lower().strip(), str(e.get("year", "")))
        by_title_year.setdefault(key, []).append(e)

    _title_year_re = _re.compile(r"^(.+?)\s*\((\d{4})\)")

    matched: list[dict] = []
    unmatched: list[Path] = []

    for wav in wav_files:
        m = _ID_RE.search(str(wav))
        if not m:
            continue

        id_type, id_value = m.group(1), m.group(2)

        # Look up catalogue entry: tmdb direct, tvdb/imdb via title+year.
        entry = None
        if id_type == "tmdb":
            entries = by_tmdb.get(id_value)
            if entries:
                entry = entries[0]
        if entry is None:
            # Fallback: title+year from directory name.
            for dirname in (wav.parent.name, wav.parent.parent.name, wav.parent.parent.parent.name):
                m2 = _title_year_re.match(dirname)
                if m2:
                    key = (m2.group(1).strip().lower(), m2.group(2))
                    entries = by_title_year.get(key)
                    if entries:
                        entry = entries[0]
                        break

        if entry is None:
            unmatched.append(wav)
            continue

        matched.append({
            "wav_path": wav,
            "catalogue_entry": entry,
            "tmdb_id": str(entry.get("theMovieDB", "")),
            "media_id": f"{id_type}-{id_value}",
        })

    return matched, unmatched


def discover_wav_catalogue_pairs() -> list[dict]:
    """Find all cached LFE WAVs that match a BEQ catalogue entry.

    Searches the portable WAV cache (single source of truth). Matches
    by media DB ID ([tmdb-NNN], [tvdb-NNN], [imdb-NNN]) extracted from
    the WAV path. Looks up catalogue by tmdb ID; for tvdb/imdb WAVs,
    the catalogue match requires the catalogue to also carry that ID type.

    Returns list of dicts with keys: wav_path, catalogue_entry, tmdb_id.
    Each WAV is a separate entry (TV episodes are NOT deduplicated).
    """
    cache_root = wav_cache_dir()
    if not cache_root.exists():
        log.warning("WAV cache does not exist: %s", cache_root)
        return []

    from model.wav_cache import iter_cached_wavs
    wav_files = iter_cached_wavs(cache_root)
    log.info("found %d WAV files in cache", len(wav_files))

    pairs, _unmatched = _match_wavs_to_catalogue(wav_files)

    unique_titles = {p["catalogue_entry"].get("title", "") for p in pairs}
    unique_ids = {p["media_id"] for p in pairs}
    log.info(
        "discovered %d WAV-catalogue pairs (%d unique titles, %d total media files) "
        "from %d WAV files in %s",
        len(pairs), len(unique_titles), len(unique_ids),
        len(wav_files), cache_root,
    )
    return pairs


def discover_wav_catalogue_pairs_cached() -> list[dict]:
    """Cached wrapper around ``discover_wav_catalogue_pairs``.

    On a warm cache, returns the previous result after two stat calls
    (no directory walk over the WAV cache root). On a cold cache or
    mismatched signature, walks the cache and writes a fresh pickle.

    Opt-out via ``AUTO_BEQ_DISCOVERY_CACHE=0``.

    Signature captures the WAV file count and the catalogue cache file
    mtime. Adding a new WAV anywhere in the cache tree invalidates the
    signature (count changes). The ``AUTO_BEQ_DISCOVERY_CACHE=0`` env
    var bypasses the cache entirely.
    """
    if not _cache_enabled(_DISCOVERY_CACHE_ENABLED_ENV):
        log.info("discovery cache disabled via %s=0", _DISCOVERY_CACHE_ENABLED_ENV)
        return discover_wav_catalogue_pairs()

    import pickle as _pickle
    try:
        cache_file = beq_shared_dir() / _DISCOVERY_CACHE_FILENAME
    except Exception as exc:
        log.warning("beq_shared_dir unavailable for discovery cache: %s", exc)
        return discover_wav_catalogue_pairs()

    signature = _discovery_cache_signature()
    if signature is not None and cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                blob = _pickle.load(f)
            if isinstance(blob, dict) and blob.get("signature") == signature:
                pairs = blob.get("pairs", [])
                log.info(
                    "discover_wav_catalogue_pairs: cache hit - %d pairs "
                    "(wav_count=%d, catalogue_mtime=%.0f)",
                    len(pairs), signature.get("wav_count", 0), signature.get("catalogue_mtime", 0),
                )
                return pairs
            log.info("discover_wav_catalogue_pairs: cache signature mismatch, re-walking")
        except Exception as exc:
            log.warning("corrupt discovery cache %s: %s - re-walking", cache_file, exc)

    pairs = discover_wav_catalogue_pairs()

    if signature is not None:
        tmp_file = cache_file.with_suffix(".pkl.tmp")
        try:
            with tmp_file.open("wb") as f:
                _pickle.dump(
                    {"signature": signature, "pairs": pairs},
                    f,
                    protocol=_pickle.HIGHEST_PROTOCOL,
                )
            tmp_file.replace(cache_file)
            log.info("discover_wav_catalogue_pairs: wrote cache (%d pairs)", len(pairs))
        except Exception as exc:
            log.warning("failed to write discovery cache %s: %s", cache_file, exc)
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass

    return pairs


def discover_unmatched_wavs() -> list[Path]:
    """Find all cached LFE WAVs that have NO matching BEQ catalogue entry.

    Inverse of ``discover_wav_catalogue_pairs``: walks the same cache
    and returns the WAVs that fall through without a catalogue match.
    These are titles we have audio for but no ground-truth filter
    chain - usable as unlabelled data for E84 self-training
    (``pseudo_label_unmatched`` in ``auto_beq_nn``).

    Returns sorted list of WAV paths (no wrapping dict - these rows
    have no catalogue entry by definition).
    """
    cache_root = wav_cache_dir()
    if not cache_root.exists():
        log.warning("WAV cache does not exist: %s", cache_root)
        return []

    from model.wav_cache import iter_cached_wavs
    wav_files = iter_cached_wavs(cache_root)
    matched, unmatched = _match_wavs_to_catalogue(wav_files)

    log.info(
        "discovered %d unmatched WAVs (no catalogue entry) from %d WAVs in %s",
        len(unmatched), len(wav_files), cache_root,
    )
    return unmatched


def discover_unmatched_wavs_cached() -> list[Path]:
    """Cached wrapper around ``discover_unmatched_wavs``.

    Uses the same lightweight signature (wav-cache root mtime +
    catalogue cache mtime) as ``discover_wav_catalogue_pairs_cached``.
    Both caches are independent but share invalidation triggers, so
    they refresh in lockstep after a fresh extract_lfe.py run or a
    catalogue refetch.
    """
    if not _cache_enabled(_DISCOVERY_CACHE_ENABLED_ENV):
        log.info("discovery cache disabled via %s=0", _DISCOVERY_CACHE_ENABLED_ENV)
        return discover_unmatched_wavs()

    import pickle as _pickle
    try:
        cache_file = beq_shared_dir() / _UNMATCHED_CACHE_FILENAME
    except Exception as exc:
        log.warning("beq_shared_dir unavailable for unmatched cache: %s", exc)
        return discover_unmatched_wavs()

    signature = _discovery_cache_signature()
    if signature is not None and cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                blob = _pickle.load(f)
            if isinstance(blob, dict) and blob.get("signature") == signature:
                wavs = blob.get("wavs", [])
                log.info(
                    "discover_unmatched_wavs: cache hit - %d unmatched WAVs",
                    len(wavs),
                )
                return wavs
            log.info("discover_unmatched_wavs: cache signature mismatch, re-walking")
        except Exception as exc:
            log.warning("corrupt unmatched cache %s: %s - re-walking", cache_file, exc)

    wavs = discover_unmatched_wavs()

    if signature is not None:
        tmp_file = cache_file.with_suffix(".pkl.tmp")
        try:
            with tmp_file.open("wb") as f:
                _pickle.dump(
                    {"signature": signature, "wavs": wavs},
                    f,
                    protocol=_pickle.HIGHEST_PROTOCOL,
                )
            tmp_file.replace(cache_file)
            log.info("discover_unmatched_wavs: wrote cache (%d WAVs)", len(wavs))
        except Exception as exc:
            log.warning("failed to write unmatched cache %s: %s", cache_file, exc)
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass

    return wavs


def check_production_model() -> bool:
    """Return True if a production model exists (fast - no loading).

    Checks env vars and the shared beq_shared_dir for model files. Does NOT
    import any Qt, scipy, or heavy dependencies - safe for headless
    Docker containers and CLI startup.
    """
    model_env = os.environ.get("AUTO_BEQ_MODEL_PATH")
    if model_env:
        return Path(model_env).exists()
    if os.environ.get("AUTO_BEQ_ADVISOR", "").lower() == "torch_differentiable":
        try:
            return (beq_shared_dir() / "e85_torch_filter.pt").exists()
        except Exception:
            return False
    try:
        return (beq_shared_dir() / "production_model.joblib").exists()
    except Exception:
        return False
