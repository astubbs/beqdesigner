"""Shared helpers for auto-BEQ spike tests and discovery CLI.

Extracted from ``test_auto_beq.py`` so the discovery CLI and the library
sweep test can reuse them without importing a test module. Keep this
module dependency-light — it is imported at both test-collection time
and CLI startup time.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

log = logging.getLogger("auto_beq_spike")


# ---------------------------------------------------------------------------
# Extraction strategy — selects how the measured LFE curve is computed.
# ---------------------------------------------------------------------------

class ExtractionMethod(Enum):
    """How raw LFE samples are turned into a magnitude-vs-frequency curve."""
    WELCH = "welch"
    CHUNKED = "chunked"
    BLENDED = "blended"


@dataclass(frozen=True)
class ExtractionStrategy:
    """Full specification of an extraction approach.

    Attributes
    ----------
    method : ExtractionMethod
        Core algorithm (Welch average, chunked-percentile, or blended).
    chunk_s : float
        Chunk length in seconds (ignored for WELCH).
    percentile : float
        Percentile across chunks (ignored for WELCH).
    alpha : float
        Welch weight in blended mode (0=pure chunked, 1=pure Welch).
        Ignored for non-BLENDED methods.
    """
    method: ExtractionMethod
    chunk_s: float = 60.0
    percentile: float = 90.0
    alpha: float = 0.7

    @property
    def label(self) -> str:
        if self.method == ExtractionMethod.WELCH:
            return "welch"
        if self.method == ExtractionMethod.BLENDED:
            return f"blend-a{self.alpha:.1f}-P{self.percentile:.0f}-{self.chunk_s:.0f}s"
        return f"chunked-P{self.percentile:.0f}-{self.chunk_s:.0f}s"

    def __str__(self) -> str:
        return self.label


# Pre-defined strategies.
STRATEGY_WELCH = ExtractionStrategy(ExtractionMethod.WELCH)
STRATEGY_BLENDED_07 = ExtractionStrategy(
    ExtractionMethod.BLENDED, chunk_s=60.0, percentile=90.0, alpha=0.7,
)
STRATEGY_BLENDED_03 = ExtractionStrategy(
    ExtractionMethod.BLENDED, chunk_s=60.0, percentile=90.0, alpha=0.3,
)
STRATEGY_CHUNKED_P90 = ExtractionStrategy(
    ExtractionMethod.CHUNKED, chunk_s=60.0, percentile=90.0,
)

# Default for production use — E18b showed blend-a0.7-P90 is the safest
# (2 grade improvements, 0 degradations across 31 test cases).
DEFAULT_STRATEGY = STRATEGY_BLENDED_07


def _strategy_from_env() -> ExtractionStrategy:
    """Read extraction strategy from AUTO_BEQ_EXTRACTION env var.

    Values: "welch", "blended" (default), "blended-0.3", "chunked".
    Falls back to DEFAULT_STRATEGY.
    """
    raw = os.environ.get("AUTO_BEQ_EXTRACTION", "").strip().lower()
    if not raw or raw == "blended":
        return DEFAULT_STRATEGY
    if raw == "welch":
        return STRATEGY_WELCH
    if raw == "blended-0.3":
        return STRATEGY_BLENDED_03
    if raw == "chunked":
        return STRATEGY_CHUNKED_P90
    log.warning("unknown AUTO_BEQ_EXTRACTION=%r, using default", raw)
    return DEFAULT_STRATEGY

def audio_cache_dir() -> Path:
    """Return the audio cache directory, creating it if needed.

    **Required config** — set ``AUTO_BEQ_AUDIO_CACHE`` env var or
    ``audio_cache_dir`` in ``~/.config/beqdesigner/settings.json``.
    We never write WAV files next to source media; all extractions go
    into this cache dir with a mirrored path structure.

    Raises RuntimeError if not configured.
    """
    raw = os.environ.get("AUTO_BEQ_AUDIO_CACHE")
    if not raw:
        cfg_path = Path.home() / ".config" / "beqdesigner" / "settings.json"
        if cfg_path.exists():
            with cfg_path.open() as _f:
                try:
                    data = json.load(_f)
                    raw = data.get("audio_cache_dir")
                except Exception:
                    pass
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


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def probe_audio_stream(media_path: Path) -> dict:
    """Probe the first audio stream with ffprobe. Returns stream info dict."""
    log.info("probing audio streams via ffprobe: %s", media_path)
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index,codec_name,channels,channel_layout,sample_rate",
            "-of", "json",
            str(media_path),
        ],
        capture_output=True, check=True, text=True, timeout=30,
    )
    info = json.loads(result.stdout)["streams"][0]
    log.info(
        "audio stream: codec=%s channels=%d layout=%s sample_rate=%s",
        info.get("codec_name"), info.get("channels"),
        info.get("channel_layout"), info.get("sample_rate"),
    )
    return info


def extract_lfe_wav(
    media_path: Path,
    target_fs: int,
    trim_start_s: float | None = None,
    trim_end_s: float | None = None,
) -> Path:
    """Extract the LFE channel to a cached WAV in the audio cache dir.

    Cache is stored under ``audio_cache_dir()`` with a mirrored path
    structure so source media directories stay clean. Example:
      source: /Volumes/NAS/Movies/Dune (2021)/Dune.mkv
      cache:  ~/Downloads/beqdesigner/audio-cache/Volumes/NAS/Movies/Dune (2021)/Dune.lfe-1000hz.wav

    ``trim_start_s`` / ``trim_end_s`` inject ``-ss`` / ``-to`` before
    ``-i`` (keyframe-accurate-fast seek). Either may be None.
    """
    cache_root = audio_cache_dir()
    # Mirror the source path under the cache root. Strip the leading /
    # so it nests cleanly: /Volumes/X/Y.mkv -> cache_root/Volumes/X/Y
    relative = Path(str(media_path.resolve()).lstrip("/"))
    stem = relative.with_suffix("").name
    trim_suffix = ""
    if trim_start_s is not None or trim_end_s is not None:
        start_tag = f"{trim_start_s:g}" if trim_start_s is not None else "0"
        end_tag = f"{trim_end_s:g}" if trim_end_s is not None else "end"
        trim_suffix = f"-t{start_tag}-{end_tag}"
    cache_dir = cache_root / relative.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{stem}.lfe-{target_fs}hz{trim_suffix}.wav"

    # Clean up orphaned .tmp from interrupted previous extractions.
    tmp_orphan = cache_path.with_suffix(".tmp")
    if tmp_orphan.exists():
        log.info("cleaning up interrupted extraction: %s", tmp_orphan)
        tmp_orphan.unlink()

    if cache_path.exists() and cache_path.stat().st_size > 0:
        log.info("cached LFE WAV found, skipping extraction: %s (%d bytes)",
                 cache_path, cache_path.stat().st_size)
        return cache_path

    stream = probe_audio_stream(media_path)
    layout = stream.get("channel_layout", "")
    has_lfe = "LFE" in layout.upper() or any(
        layout.lower().startswith(p) for p in ("5.1", "6.1", "7.1")
    )
    if has_lfe:
        af_filter = "pan=mono|c0=LFE"
        log.info("LFE channel detected in layout %r", layout)
    else:
        # Post-BM fallback: no discrete LFE (stereo, mono, etc).
        # Mix all channels to mono — per docs/workflow/beq.md this is
        # the "Post BM BEQ" approach (Mix to Mono checked).
        af_filter = "aresample"  # -ac 1 handles the downmix
        log.info(
            "no LFE channel in layout %r — falling back to Post-BM "
            "mono downmix (all channels mixed)",
            layout,
        )

    if trim_start_s is not None or trim_end_s is not None:
        log.info(
            "APPLYING TRIM: start=%ss end=%ss - this is NOT the full film",
            trim_start_s, trim_end_s,
        )
    log.info("extracting %s -> %s (fs=%d)",
             "LFE" if has_lfe else "mono-mix", cache_path, target_fs)
    log.info("this may take 1-3 minutes for a feature-length movie...")

    # Atomic write: extract to .tmp, rename only on success.
    # Prevents corrupt WAVs from interrupted extractions polluting the cache.
    tmp_path = cache_path.with_suffix(".tmp")
    start = time.time()
    ff_args: list[str] = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "warning",
    ]
    if trim_start_s is not None:
        ff_args += ["-ss", str(trim_start_s)]
    if trim_end_s is not None:
        ff_args += ["-to", str(trim_end_s)]
    ff_args += [
        "-i", str(media_path),
        "-af", af_filter,
        "-ar", str(target_fs),
        "-ac", "1",
        str(tmp_path),
    ]
    proc = subprocess.run(ff_args, capture_output=True, text=True)
    elapsed = time.time() - start
    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        log.error("ffmpeg stderr:\n%s", proc.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")

    # Validate before committing to cache.
    from model.wav_integrity import validate_wav_header
    ok, reason = validate_wav_header(tmp_path)
    if not ok:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"extracted WAV failed integrity check: {reason}")

    tmp_path.rename(cache_path)
    size = cache_path.stat().st_size
    log.info("extracted %d bytes in %.1fs -> %s", size, elapsed, cache_path)
    return cache_path


def load_and_smooth(
    wav_path: Path,
    fs: int,
    freqs: np.ndarray,
    expected_runtime_min: float = 0,
    return_absolute: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load a WAV file, compute avg spectrum, interp to grid, smooth to 1/6-octave.

    Pipeline: validate WAV integrity → WAV → Welch avg spectrum → interp
    to log grid → normalise to 80 Hz anchor → 1/6-octave smooth → re-anchor.

    When *return_absolute* is True (F3/E42), also returns absolute dBFS
    levels at the 9 Option A frequency bins **before** 80 Hz normalisation.
    This captures mastering-level information that normalisation strips out.

    Raises RuntimeError if the WAV fails integrity checks.
    """
    from model.auto_beq import smooth_fractional_octave
    from model.auto_beq_nn import OPTION_A_BINS_HZ
    from model.signal import Signal, read_wav_data
    from model.wav_integrity import validate_wav

    # Integrity gate — every WAV is validated before use.
    ok, reason = validate_wav(wav_path, expected_runtime_min=expected_runtime_min)
    if not ok:
        raise RuntimeError(f"corrupt WAV, skipping: {wav_path.name} — {reason}")

    samples, read_fs, _ = read_wav_data(str(wav_path))
    assert read_fs == fs, f"expected fs={fs}, got {read_fs}"
    mono = samples[:, 0] if samples.ndim > 1 else samples
    duration_s = len(mono) / fs
    log.info("loaded %d samples (%.1f s = %.1f min)", len(mono), duration_s, duration_s / 60)
    sig = Signal(wav_path.stem, mono, fs=fs)

    log.info("computing average spectrum (Welch)")
    measured_freqs, measured_db = sig.avg_spectrum()
    log.info("raw spectrum: %d bins from %.1f to %.1f Hz",
             len(measured_freqs), measured_freqs[0], measured_freqs[-1])

    measured_on_grid = np.interp(freqs, measured_freqs, measured_db)

    # F3/E42: capture absolute dBFS at Option A bins BEFORE normalisation.
    absolute_at_bins: np.ndarray | None = None
    if return_absolute:
        absolute_at_bins = np.interp(OPTION_A_BINS_HZ, freqs, measured_on_grid)

    anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
    measured_on_grid -= measured_on_grid[anchor_idx]
    measured_on_grid = smooth_fractional_octave(measured_on_grid, freqs, octaves=1.0 / 6.0)
    measured_on_grid -= measured_on_grid[anchor_idx]

    if return_absolute:
        return measured_on_grid, absolute_at_bins
    return measured_on_grid


def load_and_smooth_chunked(
    wav_path: Path,
    fs: int,
    freqs: np.ndarray,
    chunk_s: float = 60.0,
    percentile: float = 90.0,
    expected_runtime_min: float = 0,
    return_absolute: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Chunked-percentile spectrum: chunks → STFT peak per chunk → Nth-percentile.

    Pipeline: validate WAV integrity → WAV → split into fixed-length chunks → STFT peak curve
    per chunk → stack [n_chunks × n_freq_bins] → Nth-percentile across
    chunks → interp to log grid → normalise to 80 Hz anchor → 1/6-oct
    smooth → re-anchor.

    The percentile across chunk peaks is a more robust rolloff ceiling
    estimate than a whole-film Welch average, especially for short
    content where a single outlier scene can dominate the whole-film
    statistic (see E15c: EoT showcase scenes inflate 10 Hz by 16-19 dB).

    When *return_absolute* is True (F3/E42), also returns absolute dBFS
    levels at the 9 Option A bins before normalisation.

    Parameters
    ----------
    wav_path : Path
        Extracted LFE WAV (mono, resampled to *fs*).
    fs : int
        Sample rate of the WAV (typically 1000 Hz).
    freqs : np.ndarray
        Target log-frequency grid (e.g. ``DEFAULT_GRID``).
    chunk_s : float
        Chunk length in seconds (sub-experiment: 30/60/90).
    percentile : float
        Percentile across chunks (default 90th).
    """
    import scipy.signal as ss

    from model.auto_beq import smooth_fractional_octave
    from model.auto_beq_nn import OPTION_A_BINS_HZ
    from model.signal import read_wav_data
    from model.wav_integrity import validate_wav

    # Integrity gate — every WAV is validated before use.
    ok, reason = validate_wav(wav_path, expected_runtime_min=expected_runtime_min)
    if not ok:
        raise RuntimeError(f"corrupt WAV, skipping: {wav_path.name} — {reason}")

    samples, read_fs, _ = read_wav_data(str(wav_path))
    assert read_fs == fs, f"expected fs={fs}, got {read_fs}"
    mono = samples[:, 0] if samples.ndim > 1 else samples
    duration_s = len(mono) / fs
    log.info(
        "loaded %d samples (%.1f s = %.1f min) for chunked analysis",
        len(mono), duration_s, duration_s / 60,
    )

    chunk_samples = int(chunk_s * fs)
    min_chunk_samples = chunk_samples // 2  # drop trailing runt < 50% full
    chunks = [
        mono[i : i + chunk_samples]
        for i in range(0, len(mono), chunk_samples)
        if len(mono[i : i + chunk_samples]) >= min_chunk_samples
    ]
    log.info(
        "chunked into %d chunks of %.0f s (fs=%d, percentile=%.0f)",
        len(chunks), chunk_s, fs, percentile,
    )

    # STFT parameters: nperseg=1024 at 1000 Hz → ~1 Hz resolution,
    # consistent with the Welch resolution in load_and_smooth().
    nperseg = min(1024, chunk_samples)
    noverlap = nperseg // 2

    chunk_peaks = []
    for idx, chunk in enumerate(chunks):
        f_stft, _t, Zxx = ss.stft(
            chunk, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann",
        )
        # Peak amplitude at each frequency across all time frames in this chunk.
        peak_mag = np.max(np.abs(Zxx), axis=-1)
        peak_db = 20.0 * np.log10(peak_mag + 1e-12)
        # Interpolate to our standard log grid immediately.
        peak_on_grid = np.interp(freqs, f_stft, peak_db)
        chunk_peaks.append(peak_on_grid)
        if idx < 3 or idx == len(chunks) - 1:
            log.debug(
                "chunk %d/%d: peak 20Hz=%.1f 80Hz=%.1f dB",
                idx + 1, len(chunks),
                peak_on_grid[int(np.argmin(np.abs(freqs - 20.0)))],
                peak_on_grid[int(np.argmin(np.abs(freqs - 80.0)))],
            )

    # [n_chunks × n_freq_bins] → percentile across chunks at each freq bin.
    matrix = np.stack(chunk_peaks, axis=0)
    aggregated = np.percentile(matrix, percentile, axis=0)

    # F3/E42: capture absolute dBFS at Option A bins BEFORE normalisation.
    absolute_at_bins: np.ndarray | None = None
    if return_absolute:
        absolute_at_bins = np.interp(OPTION_A_BINS_HZ, freqs, aggregated)

    # Same normalisation pipeline as load_and_smooth().
    anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
    aggregated -= aggregated[anchor_idx]
    aggregated = smooth_fractional_octave(aggregated, freqs, octaves=1.0 / 6.0)
    aggregated -= aggregated[anchor_idx]
    log.info(
        "chunked-percentile curve: 10Hz=%.1f 20Hz=%.1f 80Hz=%.1f dB",
        aggregated[0],
        aggregated[int(np.argmin(np.abs(freqs - 20.0)))],
        aggregated[anchor_idx],
    )
    if return_absolute:
        return aggregated, absolute_at_bins
    return aggregated


def load_and_smooth_blended(
    wav_path: Path,
    fs: int,
    freqs: np.ndarray,
    chunk_s: float = 60.0,
    percentile: float = 90.0,
    alpha: float = 0.5,
    return_absolute: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Blend Welch average and chunked-percentile curves.

    ``alpha`` controls the blend: 0.0 = pure chunked, 1.0 = pure Welch.
    Default 0.5 = equal weight.

    Both curves are computed independently (each normalised to 80 Hz
    anchor and 1/6-oct smoothed), then blended in dB domain.

    When *return_absolute* is True (F3/E42), returns absolute dBFS from
    the Welch extraction (the more stable of the two).
    """
    if return_absolute:
        welch, absolute_at_bins = load_and_smooth(
            wav_path, fs, freqs, return_absolute=True,
        )
    else:
        welch = load_and_smooth(wav_path, fs, freqs)
        absolute_at_bins = None
    chunked = load_and_smooth_chunked(
        wav_path, fs, freqs, chunk_s=chunk_s, percentile=percentile,
    )
    blended = alpha * welch + (1.0 - alpha) * chunked
    log.info(
        "blended curve (alpha=%.2f): 10Hz=%.1f 20Hz=%.1f 80Hz=%.1f dB",
        alpha, blended[0],
        blended[int(np.argmin(np.abs(freqs - 20.0)))],
        blended[int(np.argmin(np.abs(freqs - 80.0)))],
    )
    if return_absolute:
        return blended, absolute_at_bins
    return blended


def load_measured(
    wav_path: Path,
    fs: int,
    freqs: np.ndarray,
    strategy: ExtractionStrategy | None = None,
    return_absolute: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate extraction function based on strategy.

    If *strategy* is None, uses ``_strategy_from_env()`` (which defaults
    to ``DEFAULT_STRATEGY`` = blend-a0.7-P90).

    When *return_absolute* is True (F3/E42), also returns absolute dBFS
    levels at the 9 Option A bins before normalisation.
    """
    if strategy is None:
        strategy = _strategy_from_env()
    log.info("extraction strategy: %s", strategy.label)

    if strategy.method == ExtractionMethod.WELCH:
        return load_and_smooth(wav_path, fs, freqs, return_absolute=return_absolute)
    if strategy.method == ExtractionMethod.CHUNKED:
        return load_and_smooth_chunked(
            wav_path, fs, freqs,
            chunk_s=strategy.chunk_s,
            percentile=strategy.percentile,
            return_absolute=return_absolute,
        )
    if strategy.method == ExtractionMethod.BLENDED:
        return load_and_smooth_blended(
            wav_path, fs, freqs,
            chunk_s=strategy.chunk_s,
            percentile=strategy.percentile,
            alpha=strategy.alpha,
            return_absolute=return_absolute,
        )
    raise ValueError(f"unknown extraction method: {strategy.method}")


# ---------------------------------------------------------------------------
# Shared helpers for NN training experiments
# ---------------------------------------------------------------------------

# Matches [tmdb-NNN], [tvdb-NNN], or [imdb-NNN] in a path string.
_ID_RE = __import__("re").compile(r"\[(tmdb|tvdb|imdb)-([^\]]+)\]")


def wav_cache_dir() -> Path:
    """Return the portable WAV cache directory.

    Single source of truth for where extracted LFE WAVs live. Reads from
    ``wav_cache_dir`` in settings.json, falls back to
    ``~/Downloads/beqdesigner/wav-cache``.
    """
    raw = os.environ.get("BEQ_WAV_CACHE")
    if not raw:
        cfg_path = Path.home() / ".config" / "beqdesigner" / "settings.json"
        if cfg_path.exists():
            with cfg_path.open() as _f:
                try:
                    data = __import__("json").load(_f)
                    raw = data.get("wav_cache_dir")
                except Exception:
                    pass
    if not raw:
        raw = str(Path.home() / "Downloads" / "beqdesigner" / "wav-cache")
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    _title_year_re = __import__("re").compile(r"^(.+?)\s*\((\d{4})\)")

    wav_files = sorted(cache_root.rglob("*.lfe-1000hz.wav"))
    pairs = []

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
            continue  # no catalogue match — skip

        pairs.append({
            "wav_path": wav,
            "catalogue_entry": entry,
            "tmdb_id": str(entry.get("theMovieDB", "")),
            "media_id": f"{id_type}-{id_value}",
        })

    log.info("discovered %d WAV-catalogue pairs from %d WAV files in %s",
             len(pairs), len(wav_files), cache_root)
    return pairs


def synthetic_features(entry: dict, freqs_hz: np.ndarray):
    """Compute CurveFeatures from the inverse of an entry's filter chain.

    rolloff = -evaluate_filter_chain(entry["filters"]) represents what the
    LFE would look like before BEQ correction.
    """
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_advisor import extract_curve_features

    correction = evaluate_filter_chain(entry["filters"], freqs_hz, fs=1000)
    rolloff = -correction
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    rolloff_norm = rolloff - rolloff[anchor_idx]
    return extract_curve_features(rolloff_norm, freqs_hz)


def build_training_dataset(freqs_hz: np.ndarray):
    """Load full catalogue, dedup, enrich with TMDb cache, build (X, Y, entries).

    Uses the local TMDb cache (expected to be fully populated). Does NOT
    call fetch_metadata_batch — if a TMDb ID is missing from the cache,
    its metadata fields will simply be empty/zero.

    Returns (X, Y, entries, tmdb_cache) where X is float32 [N, 60],
    Y is float32 [N, 16], entries is the list of catalogue dicts used.
    """
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import enrich_media_metadata, load_cache
    from model.auto_beq_nn import build_feature_vector, catalogue_entry_to_labels, deduplicate_by_title

    catalogue = _fetch_or_cache()
    log.info("full catalogue: %d entries", len(catalogue))

    deduped = deduplicate_by_title(catalogue)
    deduped = [e for e in deduped if e.get("filters")]
    log.info("after dedup + filter: %d unique titles with filters", len(deduped))

    tmdb_cache = load_cache()
    log.info("TMDb cache: %d entries", len(tmdb_cache))

    X_list, Y_list, used = [], [], []
    for e in deduped:
        features = synthetic_features(e, freqs_hz)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_list.append(build_feature_vector(features, metadata))
        Y_list.append(catalogue_entry_to_labels(e))
        used.append(e)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    log.info("training dataset: X=%s Y=%s", X.shape, Y.shape)
    return X, Y, used, tmdb_cache


def extract_features_with_strategy(
    wav_path,
    freqs_hz: np.ndarray,
    fs: int,
    strategy: ExtractionStrategy | None = None,
):
    """Load WAV → extract spectrum using strategy → extract_curve_features.

    Uses ``load_measured()`` which dispatches to Welch, chunked, or blended.
    If strategy is None, uses DEFAULT_STRATEGY.
    """
    from model.auto_beq_advisor import extract_curve_features

    curve = load_measured(wav_path, fs, freqs_hz, strategy)
    return extract_curve_features(curve, freqs_hz)


# Backwards-compatible aliases (the existing test_auto_beq.py uses
# underscore-prefixed names).
_have_tool = have_tool
_probe_audio_stream = probe_audio_stream
_extract_lfe_wav = extract_lfe_wav
_load_and_smooth_chunked = load_and_smooth_chunked
