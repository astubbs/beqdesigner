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

    .. deprecated::
        Use ``wav_cache_dir()`` and ``cli.extract.cache_path()`` instead.
        This function builds a mirrored-path layout which is being replaced
        by the two-letter bucket layout. Kept for backward compatibility
        with callers that have not migrated yet.

    **Required config** — set ``AUTO_BEQ_AUDIO_CACHE`` env var or
    ``audio_cache_dir`` (deprecated) / ``shared_beq_dir`` + ``/wav-cache``
    in ``~/.config/beqdesigner/settings.json``.
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


_SETTINGS_PATH = Path.home() / ".config" / "beqdesigner" / "settings.json"


def load_settings() -> dict:
    """Load the shared settings.json, returning {} on any error."""
    if _SETTINGS_PATH.exists():
        try:
            return json.loads(_SETTINGS_PATH.read_text())
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("failed to parse %s: %s — returning empty settings",
                        _SETTINGS_PATH, exc)
        except OSError as exc:
            log.warning("could not read %s: %s", _SETTINGS_PATH, exc)
    return {}


def save_settings(settings: dict) -> None:
    """Persist the shared settings.json (merges with existing)."""
    existing = load_settings()
    existing.update(settings)
    beq_config_dir()  # ensure directory exists
    _SETTINGS_PATH.write_text(json.dumps(existing, indent=2) + "\n")


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
    target_path: Path | None = None,
) -> Path:
    """Extract the LFE channel to a cached WAV in the audio cache dir.

    If ``target_path`` is provided, the WAV is written there (unified
    cache layout). Otherwise falls back to the legacy mirrored-path
    layout under ``audio_cache_dir()``.

    ``trim_start_s`` / ``trim_end_s`` inject ``-ss`` / ``-to`` before
    ``-i`` (keyframe-accurate-fast seek). Either may be None.
    """
    if target_path is not None:
        cache_path = target_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Legacy fallback — mirrored path (deprecated, will be removed).
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
        "-sample_fmt", "s16",
        "-f", "wav",
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
    log.debug("loaded %d samples (%.1f s = %.1f min)", len(mono), duration_s, duration_s / 60)
    sig = Signal(wav_path.stem, mono, fs=fs)

    log.debug("computing average spectrum (Welch)")
    measured_freqs, measured_db = sig.avg_spectrum()
    log.debug("raw spectrum: %d bins from %.1f to %.1f Hz",
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


def detect_music_chunks(
    mono: np.ndarray,
    fs: int,
    chunk_s: float = 60.0,
    periodicity_threshold: float = 0.4,
) -> np.ndarray:
    """Detect music-dominated chunks via onset regularity (F4/E44).

    Musical content has periodic onset patterns at typical tempos
    (60-200 BPM = 0.3-1.0 s lag).  Impact/effects audio is aperiodic.
    By computing the autocorrelation of the spectral-flux onset envelope
    per chunk and checking for strong peaks in the musical tempo range,
    we can flag chunks where score/soundtrack dominates.

    Excluding or down-weighting these chunks improves rolloff detection
    because music bass is intentionally mixed and doesn't reflect the
    rolloff ceiling the same way effects do.

    Returns boolean array of shape ``(n_chunks,)`` where True = music detected.
    """
    import scipy.signal as ss

    chunk_samples = int(chunk_s * fs)
    min_chunk = chunk_samples // 2
    chunks = [
        mono[i : i + chunk_samples]
        for i in range(0, len(mono), chunk_samples)
        if len(mono[i : i + chunk_samples]) >= min_chunk
    ]
    is_music = np.zeros(len(chunks), dtype=bool)

    # Tempo range: 60-200 BPM → period 0.3-1.0 s → lag in samples.
    min_lag = int(0.3 * fs)
    max_lag = min(int(1.0 * fs), chunk_samples // 2)
    if max_lag <= min_lag:
        return is_music  # chunk too short for tempo detection

    for i, chunk in enumerate(chunks):
        # Spectral flux as onset strength proxy.
        nperseg = min(256, len(chunk))
        _, _, Zxx = ss.stft(chunk, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        mag = np.abs(Zxx)
        # Half-wave rectified difference between successive frames.
        flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
        if len(flux) < max_lag + 1:
            continue

        # Normalised autocorrelation in the tempo lag range.
        flux_centered = flux - flux.mean()
        norm = np.dot(flux_centered, flux_centered)
        if norm < 1e-12:
            continue
        acorr = np.correlate(flux_centered, flux_centered, mode="full")
        acorr = acorr[len(flux_centered) - 1 :]  # positive lags only
        acorr /= norm

        tempo_region = acorr[min_lag : max_lag + 1]
        if len(tempo_region) > 0 and tempo_region.max() > periodicity_threshold:
            is_music[i] = True

    n_flagged = int(is_music.sum())
    if n_flagged > 0:
        log.info("music detection: %d/%d chunks flagged as music", n_flagged, len(chunks))
    return is_music


def _weighted_percentile(
    matrix: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Weighted percentile across axis 0 of a 2-D matrix.

    Rows with weight 0 are excluded.  Falls back to ``np.percentile``
    when all weights are equal.
    """
    mask = weights > 0
    if mask.all() and np.allclose(weights, weights[0]):
        return np.percentile(matrix, percentile, axis=0)
    if not mask.any():
        return np.percentile(matrix, percentile, axis=0)

    m = matrix[mask]
    w = weights[mask]
    result = np.empty(m.shape[1], dtype=np.float64)
    for col in range(m.shape[1]):
        sorted_idx = np.argsort(m[:, col])
        sorted_vals = m[sorted_idx, col]
        sorted_w = w[sorted_idx]
        cumw = np.cumsum(sorted_w)
        cutoff = percentile / 100.0 * cumw[-1]
        idx = int(np.searchsorted(cumw, cutoff))
        idx = min(idx, len(sorted_vals) - 1)
        result[col] = sorted_vals[idx]
    return result


def extract_chunk_stats(
    chunk_matrix: np.ndarray,
    freqs: np.ndarray,
    bins_hz: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin chunk statistics (F2/E43 — Option B features).

    From the ``[n_chunks x n_freq_bins]`` dB matrix, compute at each of
    the target frequency bins:

    1. **Standard deviation** across chunks — captures content variability.
       High stddev = variable bass (showcase scenes inflate the average);
       low stddev = consistent rolloff (high confidence in the ceiling).
    2. **Ceiling fraction** — proportion of chunks within 3 dB of the 90th
       percentile.  High fraction = most chunks agree on the rolloff level.

    These 18 values (9 stddev + 9 ceiling_frac) supplement the 9 Option A
    percentile values, forming the 27-dim Option B feature vector.

    Returns ``(stddev_9, ceiling_frac_9)`` each of shape ``(len(bins_hz),)``.
    """
    n_bins = len(bins_hz)
    stddev = np.empty(n_bins, dtype=np.float32)
    ceiling_frac = np.empty(n_bins, dtype=np.float32)

    for i, target_hz in enumerate(bins_hz):
        col_idx = int(np.argmin(np.abs(freqs - target_hz)))
        column = chunk_matrix[:, col_idx]
        stddev[i] = float(np.std(column))
        p90 = float(np.percentile(column, 90))
        ceiling_frac[i] = float(np.mean(column >= p90 - 3.0))

    return stddev, ceiling_frac


def load_and_smooth_chunked(
    wav_path: Path,
    fs: int,
    freqs: np.ndarray,
    chunk_s: float = 60.0,
    percentile: float = 90.0,
    expected_runtime_min: float = 0,
    return_absolute: bool = False,
    return_chunk_stats: bool = False,
    chunk_weights: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | dict:
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
    log.debug(
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
    log.debug(
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
    if chunk_weights is not None:
        aggregated = _weighted_percentile(matrix, chunk_weights, percentile)
    else:
        aggregated = np.percentile(matrix, percentile, axis=0)

    # F2/E43 (Option B): per-bin chunk statistics for the 9 Option A bins.
    # stddev captures content variability; ceiling_frac captures how many
    # chunks are near the rolloff ceiling (high = confident estimate).
    chunk_stddev: np.ndarray | None = None
    chunk_ceiling_frac: np.ndarray | None = None
    if return_chunk_stats:
        chunk_stddev, chunk_ceiling_frac = extract_chunk_stats(
            matrix, freqs, OPTION_A_BINS_HZ,
        )

    # F3/E42: capture absolute dBFS at Option A bins BEFORE normalisation.
    absolute_at_bins: np.ndarray | None = None
    if return_absolute:
        absolute_at_bins = np.interp(OPTION_A_BINS_HZ, freqs, aggregated)

    # Same normalisation pipeline as load_and_smooth().
    anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
    aggregated -= aggregated[anchor_idx]
    aggregated = smooth_fractional_octave(aggregated, freqs, octaves=1.0 / 6.0)
    aggregated -= aggregated[anchor_idx]
    log.debug(
        "chunked-percentile curve: 10Hz=%.1f 20Hz=%.1f 80Hz=%.1f dB",
        aggregated[0],
        aggregated[int(np.argmin(np.abs(freqs - 20.0)))],
        aggregated[anchor_idx],
    )
    result: dict = {"curve": aggregated}
    if return_absolute:
        result["absolute"] = absolute_at_bins
    if return_chunk_stats:
        result["chunk_stddev"] = chunk_stddev
        result["chunk_ceiling_frac"] = chunk_ceiling_frac

    # Backward-compatible return: if no extras requested, return just the curve.
    if not return_absolute and not return_chunk_stats:
        return aggregated
    if return_absolute and not return_chunk_stats:
        return aggregated, absolute_at_bins
    return result


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
    log.debug(
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
    log.debug("extraction strategy: %s", strategy.label)

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


def _latest_wav_mtime(cache_root: Path) -> float:
    """Return the max mtime across all WAV files in the cache, or 0.0."""
    latest = 0.0
    for p in cache_root.rglob("*.lfe-1000hz.wav"):
        try:
            m = p.stat().st_mtime
            if m > latest:
                latest = m
        except OSError:
            pass
    return latest


def ensure_analysis_reports_current(
    repo_root: Path | None = None,
    force: bool = False,
) -> list[str]:
    """Regenerate analysis reports when they're stale vs the WAV cache.

    Checks ``docs/wav_cache_bias.md``, ``docs/acquisition_recommendations.md``
    and ``docs/author_patterns.md`` against the max mtime of any WAV in
    the cache.  Regenerates (via subprocess) any report whose mtime is
    older than the newest WAV.

    Called at the start of experiment harness runs so downstream analysis
    docs always reflect the current training set without a manual step.

    Returns the list of reports that were (re)generated — useful for
    logging and tests.
    """
    import subprocess

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    cache_root = wav_cache_dir()
    cache_mtime = _latest_wav_mtime(cache_root)
    if cache_mtime == 0.0:
        log.info("WAV cache is empty; skipping analysis report refresh")
        return []

    reports = [
        (
            repo_root / "docs" / "wav_cache_bias.md",
            repo_root / "scripts" / "nn_cache_bias_report.py",
            ["-o"],
        ),
        (
            repo_root / "docs" / "author_patterns.md",
            repo_root / "scripts" / "nn_author_pattern_report.py",
            ["-o"],
        ),
        (
            repo_root / "docs" / "acquisition_recommendations.md",
            repo_root / "scripts" / "nn_acquisition_recommender.py",
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


def beq_dir() -> Path:
    """Return the BEQ shared working directory.

    Holds wav-cache/, beq_catalogue.json, media_inventory.json, and
    production_model.joblib.  This is the *portable* shared directory
    (e.g. a NAS mount) — machine-specific config lives in
    ``beq_config_dir()`` (~/.config/beqdesigner/).

    Resolution order:
    1. ``BEQ_SHARED_DIR`` env var (preferred)
    2. ``shared_beq_dir`` in ``~/.config/beqdesigner/settings.json``
    3. ``BEQ_DIR`` env var (backward compat)
    4. ``audio_cache_dir`` in settings.json (backward compat — parent of wav-cache)
    5. Derived from ``wav_cache_dir().parent`` (last resort)
    """
    # 1. New env var.
    raw = os.environ.get("BEQ_SHARED_DIR")
    if raw:
        p = Path(raw).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. New settings key.
    cfg_path = Path.home() / ".config" / "beqdesigner" / "settings.json"
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

    # 3. Old env var (backward compat).
    raw = os.environ.get("BEQ_DIR")
    if raw:
        log.debug("BEQ_DIR is deprecated — use BEQ_SHARED_DIR instead")
        p = Path(raw).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 4. Old settings key (backward compat) — audio_cache_dir points to
    #    wav-cache, so the shared dir is its parent.
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
            raw = data.get("audio_cache_dir")
            if raw:
                log.debug("deriving beq_dir from deprecated audio_cache_dir setting")
                p = Path(raw).expanduser().parent
                p.mkdir(parents=True, exist_ok=True)
                return p
        except Exception:
            pass

    # 5. Fall back to wav_cache_dir parent.
    return wav_cache_dir().parent


def wav_cache_dir() -> Path:
    """Return the portable WAV cache directory.

    Single source of truth for where extracted LFE WAVs live.  Resolution
    order: ``BEQ_WAV_CACHE`` env var → ``wav_cache_dir`` in
    ``~/.config/beqdesigner/settings.json`` → ``shared_beq_dir`` + ``/wav-cache``
    → ``audio_cache_dir`` (backward compat) → error.

    Raises RuntimeError if not configured.
    """
    explicit_source: str | None = None
    raw = os.environ.get("BEQ_WAV_CACHE")
    if raw:
        explicit_source = "BEQ_WAV_CACHE env var"
    else:
        cfg_path = Path.home() / ".config" / "beqdesigner" / "settings.json"
        if cfg_path.exists():
            with cfg_path.open() as _f:
                try:
                    data = __import__("json").load(_f)
                    raw = data.get("wav_cache_dir")
                    if raw:
                        explicit_source = f"wav_cache_dir in {cfg_path}"
                    elif data.get("shared_beq_dir"):
                        # Derive from shared_beq_dir + /wav-cache.
                        raw = str(Path(data["shared_beq_dir"]) / "wav-cache")
                        explicit_source = f"shared_beq_dir in {cfg_path}"
                    elif data.get("audio_cache_dir"):
                        # Fall back to audio_cache_dir if wav_cache_dir not set.
                        # Deprecated — prefer shared_beq_dir.
                        raw = data["audio_cache_dir"]
                        explicit_source = f"audio_cache_dir in {cfg_path}"
                except Exception:
                    pass
    if not raw:
        raise RuntimeError(
            "WAV cache dir not configured. Set BEQ_WAV_CACHE env var "
            "or add {\"wav_cache_dir\": \"/path/to/cache\"} "
            "to ~/.config/beqdesigner/settings.json"
        )

    path = Path(raw).expanduser()

    if explicit_source is not None:
        # User explicitly configured this path — fail fast if it doesn't
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
        # Default fallback path — auto-create for first-time users.
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Disk-backed caches for expensive NAS / feature-extraction work
# ---------------------------------------------------------------------------
#
# Two small pickle caches to keep iteration fast when re-running experiments
# against the same WAV cache + catalogue:
#
#   1. curve-features cache: one pickle per WAV, keyed on
#      (stem, size, mtime, strategy.label). Skips ~2-3 min of Welch +
#      chunked-percentile work per experiment re-run.
#   2. discovery cache: single pickle holding the `discover_wav_catalogue_pairs`
#      output, keyed on (wav_cache_root_mtime, catalogue_cache_mtime) so any
#      fresh extract_lfe.py run or catalogue refresh invalidates it.
#
# Both caches are override-able via env var for safety.


_CURVE_FEATURES_CACHE_ENABLED_ENV = "AUTO_BEQ_FEATURE_CACHE"
_DISCOVERY_CACHE_ENABLED_ENV = "AUTO_BEQ_DISCOVERY_CACHE"
_DISCOVERY_CACHE_FILENAME = "discovered_pairs.pkl"
_CATALOGUE_CACHE_FILENAME = "beq_catalogue.json"


def _cache_enabled(env_var: str) -> bool:
    """Cache helpers honour env var opt-out. Default: enabled."""
    return os.environ.get(env_var, "1") != "0"


def _curve_features_cache_dir(strategy_label: str) -> Path:
    """Per-strategy directory under ``{beq-dir}/curve-features/``."""
    return beq_dir() / "curve-features" / strategy_label


def _curve_features_cache_key(wav_path: Path) -> str | None:
    """File-identity cache key — ``{stem}-{size}-{mtime_int}``.

    Returns None if the WAV can't be stat'd (missing file). File-identity
    keys auto-invalidate when extract_lfe.py re-extracts a WAV.
    """
    try:
        st = wav_path.stat()
    except FileNotFoundError:
        return None
    return f"{wav_path.stem}-{st.st_size}-{int(st.st_mtime)}"


def cached_extract_features_with_strategy(
    wav_path: Path,
    freqs_hz: np.ndarray,
    fs: int,
    strategy: "ExtractionStrategy",
):
    """Disk-cached wrapper around ``extract_features_with_strategy``.

    Cache hit path: one ``pickle.load`` from
    ``{beq-dir}/curve-features/<strategy.label>/<key>.pkl`` — no WAV read,
    no Welch, no chunked percentile, ~1 ms per call.

    Cache miss path: runs the uncached extractor and atomically writes the
    result to the cache. Subsequent callers see the hit.

    Opt-out via ``AUTO_BEQ_FEATURE_CACHE=0``.
    """
    if not _cache_enabled(_CURVE_FEATURES_CACHE_ENABLED_ENV):
        return extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=strategy)

    import pickle as _pickle
    key = _curve_features_cache_key(wav_path)
    if key is None:
        return extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=strategy)

    try:
        cache_dir = _curve_features_cache_dir(strategy.label)
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("curve-features cache dir unavailable: %s", exc)
        return extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=strategy)

    cache_file = cache_dir / f"{key}.pkl"
    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                return _pickle.load(f)
        except Exception as exc:
            log.warning("corrupt curve-features cache %s: %s — recomputing", cache_file, exc)

    features = extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=strategy)

    # Atomic write: temp file + rename so concurrent workers never see half-written pickles.
    tmp_file = cache_file.with_suffix(".pkl.tmp")
    try:
        with tmp_file.open("wb") as f:
            _pickle.dump(features, f, protocol=_pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(cache_file)
    except Exception as exc:
        log.warning("failed to write curve-features cache %s: %s", cache_file, exc)
        try:
            tmp_file.unlink(missing_ok=True)
        except Exception:
            pass
    return features


def _discovery_cache_signature() -> dict | None:
    """Lightweight signature: two stat calls, no directory walk.

    Invalidation sources:
      * wav cache root mtime — bumped when extract_lfe.py writes
        ``.status_last.json``, or when new top-level subdirs are added.
      * catalogue cache file mtime — bumped when the BEQ catalogue is refetched.

    Returns None if stats fail (cache miss forced).
    """
    try:
        wav_root = wav_cache_dir()
        if not wav_root.exists():
            return None
        root_mtime = wav_root.stat().st_mtime
    except Exception:
        return None

    catalogue_mtime = 0.0
    catalogue_cache = beq_dir() / _CATALOGUE_CACHE_FILENAME
    if catalogue_cache.exists():
        try:
            catalogue_mtime = catalogue_cache.stat().st_mtime
        except Exception:
            catalogue_mtime = 0.0

    return {
        "wav_root_mtime": float(root_mtime),
        "catalogue_mtime": float(catalogue_mtime),
    }


def discover_wav_catalogue_pairs_cached() -> list[dict]:
    """Cached wrapper around ``discover_wav_catalogue_pairs``.

    On a warm cache, returns the previous result after two stat calls
    (no directory walk over the WAV cache root). On a cold cache or
    mismatched signature, walks the cache and writes a fresh pickle.

    Opt-out via ``AUTO_BEQ_DISCOVERY_CACHE=0``.

    Signature captures the WAV cache root mtime and the catalogue cache
    file mtime. Adding a new WAV inside an existing subdirectory does
    NOT invalidate (root mtime unchanged), so if the user manually drops
    a WAV deep in the tree they need to either re-run extract_lfe.py
    (which refreshes ``.status_last.json`` in the root) or set
    ``AUTO_BEQ_DISCOVERY_CACHE=0`` once.
    """
    if not _cache_enabled(_DISCOVERY_CACHE_ENABLED_ENV):
        log.info("discovery cache disabled via %s=0", _DISCOVERY_CACHE_ENABLED_ENV)
        return discover_wav_catalogue_pairs()

    import pickle as _pickle
    try:
        cache_file = beq_dir() / _DISCOVERY_CACHE_FILENAME
    except Exception as exc:
        log.warning("beq_dir unavailable for discovery cache: %s", exc)
        return discover_wav_catalogue_pairs()

    signature = _discovery_cache_signature()
    if signature is not None and cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                blob = _pickle.load(f)
            if isinstance(blob, dict) and blob.get("signature") == signature:
                pairs = blob.get("pairs", [])
                log.info(
                    "discover_wav_catalogue_pairs: cache hit — %d pairs "
                    "(wav_root_mtime=%.0f, catalogue_mtime=%.0f)",
                    len(pairs), signature["wav_root_mtime"], signature["catalogue_mtime"],
                )
                return pairs
            log.info("discover_wav_catalogue_pairs: cache signature mismatch, re-walking")
        except Exception as exc:
            log.warning("corrupt discovery cache %s: %s — re-walking", cache_file, exc)

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
    chain — usable as unlabelled data for E84 self-training
    (``pseudo_label_unmatched`` in ``auto_beq_nn``).

    Returns sorted list of WAV paths (no wrapping dict — these rows
    have no catalogue entry by definition).
    """
    cache_root = wav_cache_dir()
    if not cache_root.exists():
        log.warning("WAV cache does not exist: %s", cache_root)
        return []

    from model.auto_beq_catalogue import _fetch_or_cache
    catalogue = _fetch_or_cache()

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
    unmatched: list[Path] = []

    for wav in wav_files:
        m = _ID_RE.search(str(wav))
        if not m:
            # No ID tag at all — not a training candidate either way.
            continue
        id_type, id_value = m.group(1), m.group(2)

        # Replicate the lookup logic from discover_wav_catalogue_pairs.
        entry = None
        if id_type == "tmdb":
            entries = by_tmdb.get(id_value)
            if entries:
                entry = entries[0]
        if entry is None:
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

    log.info(
        "discovered %d unmatched WAVs (no catalogue entry) from %d WAVs in %s",
        len(unmatched), len(wav_files), cache_root,
    )
    return unmatched


_UNMATCHED_CACHE_FILENAME = "unmatched_wavs.pkl"


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
        cache_file = beq_dir() / _UNMATCHED_CACHE_FILENAME
    except Exception as exc:
        log.warning("beq_dir unavailable for unmatched cache: %s", exc)
        return discover_unmatched_wavs()

    signature = _discovery_cache_signature()
    if signature is not None and cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                blob = _pickle.load(f)
            if isinstance(blob, dict) and blob.get("signature") == signature:
                wavs = blob.get("wavs", [])
                log.info(
                    "discover_unmatched_wavs: cache hit — %d unmatched WAVs",
                    len(wavs),
                )
                return wavs
            log.info("discover_unmatched_wavs: cache signature mismatch, re-walking")
        except Exception as exc:
            log.warning("corrupt unmatched cache %s: %s — re-walking", cache_file, exc)

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
