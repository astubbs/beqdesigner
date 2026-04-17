"""WAV file integrity validation for the LFE audio cache.

Provides cheap checks to detect truncated or corrupt WAV files without
reading the full audio data:

1. **Header vs file size**: WAV header declares data size; actual file
   should match. A truncated file has fewer bytes than declared.
2. **Duration vs expected runtime**: At 1000 Hz / 16-bit mono, a 90-min
   movie = ~10.8 MB. If the WAV is less than 80% of the expected
   duration, it's corrupt.

Used by the extraction script to verify on write, and by the training
pipeline to validate on load.
"""

from __future__ import annotations

import logging
import struct
import wave
from pathlib import Path

log = logging.getLogger("wav_integrity")

# BEQ LFE extraction parameters (hardcoded, coupled to algorithm).
LFE_SAMPLE_RATE = 1000
LFE_SAMPLE_WIDTH = 2  # 16-bit
LFE_CHANNELS = 1      # mono

# Duration check threshold: WAV must be at least this fraction of the
# expected runtime to be considered valid.
_MIN_DURATION_FRACTION = 0.8


def validate_wav_header(wav_path: Path) -> tuple[bool, str]:
    """Check that a WAV file's header matches its actual file size.

    Returns (is_valid, reason). A valid file has actual data bytes >=
    what the header declares. A truncated file has fewer.
    """
    try:
        file_size = wav_path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat: {exc}"

    if file_size == 0:
        return False, "empty file (0 bytes)"

    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
    except (wave.Error, EOFError, struct.error) as exc:
        return False, f"invalid WAV header: {exc}"

    expected_data_bytes = n_frames * n_channels * sample_width
    # WAV header is typically 44 bytes; actual overhead varies slightly.
    header_overhead = file_size - expected_data_bytes
    if header_overhead < 0:
        return False, (
            f"truncated: header declares {n_frames} frames "
            f"({expected_data_bytes} data bytes) but file is only "
            f"{file_size} bytes ({-header_overhead} bytes short)"
        )

    return True, (
        f"ok: {n_frames} frames, {frame_rate} Hz, "
        f"{n_channels}ch × {sample_width * 8}-bit, "
        f"{n_frames / frame_rate:.1f}s"
    )


def validate_wav_duration(
    wav_path: Path,
    expected_runtime_min: float,
) -> tuple[bool, str]:
    """Check that a WAV file's duration is consistent with the expected runtime.

    At 1000 Hz / 16-bit mono, a 90-minute movie produces a ~10.8 MB WAV.
    If the WAV is less than 80% of the expected duration, it's likely
    truncated or extracted from a trailer/sample rather than the feature.

    ``expected_runtime_min`` is the catalogue's runtime field (minutes).
    """
    if expected_runtime_min <= 0:
        return True, "no runtime to check against"

    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            frame_rate = wf.getframerate()
    except (wave.Error, EOFError, struct.error) as exc:
        return False, f"cannot read WAV: {exc}"

    if frame_rate == 0:
        return False, "frame rate is 0"

    wav_duration_s = n_frames / frame_rate
    expected_duration_s = expected_runtime_min * 60.0
    fraction = wav_duration_s / expected_duration_s

    if fraction < _MIN_DURATION_FRACTION:
        return False, (
            f"too short: WAV is {wav_duration_s:.1f}s but expected "
            f"~{expected_duration_s:.0f}s ({expected_runtime_min:.0f} min). "
            f"Only {fraction:.0%} of expected duration"
        )

    return True, (
        f"ok: {wav_duration_s:.1f}s vs expected ~{expected_duration_s:.0f}s "
        f"({fraction:.0%})"
    )


def validate_wav(
    wav_path: Path,
    expected_runtime_min: float = 0,
) -> tuple[bool, str]:
    """Run all validation checks on a WAV file.

    Returns (is_valid, reason). Fails fast on first check failure.
    """
    ok, reason = validate_wav_header(wav_path)
    if not ok:
        return False, reason

    if expected_runtime_min > 0:
        ok, reason = validate_wav_duration(wav_path, expected_runtime_min)
        if not ok:
            return False, reason

    return True, "all checks passed"


def verify_cache(cache_root: Path) -> tuple[list[Path], list[Path]]:
    """Scan a WAV cache directory and validate all files.

    Returns (valid_files, corrupt_files). Corrupt files can be deleted
    to trigger re-extraction.
    """
    import os as _os
    from model.media_utils import ProgressLogger
    from model.wav_cache import WAV_SUFFIX

    # Walk bucket dirs instead of rglob (faster on NFS).
    bucket_dirs = sorted(
        e.path for e in _os.scandir(cache_root)
        if e.is_dir()
    )
    progress = ProgressLogger(len(bucket_dirs), logger=log, min_interval_s=5)
    wav_files: list[Path] = []
    for i, bucket_path in enumerate(bucket_dirs):
        for dirpath, _dirnames, filenames in _os.walk(bucket_path):
            for f in filenames:
                if f.endswith(WAV_SUFFIX):
                    wav_files.append(Path(dirpath) / f)
        progress.update(i + 1, label=_os.path.basename(bucket_path))

    valid = []
    corrupt = []
    verify_progress = ProgressLogger(len(wav_files), logger=log, min_interval_s=5)
    for i, wav in enumerate(wav_files):
        ok, reason = validate_wav_header(wav)
        if ok:
            valid.append(wav)
        else:
            log.warning("corrupt WAV: %s -- %s", wav, reason)
            corrupt.append(wav)
        verify_progress.update(i + 1)
    log.info("cache verification: %d valid, %d corrupt out of %d total",
             len(valid), len(corrupt), len(wav_files))
    return valid, corrupt
