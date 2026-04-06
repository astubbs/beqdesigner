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
from pathlib import Path

import numpy as np
import pytest

log = logging.getLogger("auto_beq_spike")

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

    if cache_path.exists() and cache_path.stat().st_size > 0:
        log.info("cached LFE WAV found, skipping extraction: %s (%d bytes)",
                 cache_path, cache_path.stat().st_size)
        return cache_path

    stream = probe_audio_stream(media_path)
    layout = stream.get("channel_layout", "")
    if "LFE" not in layout.upper() and not any(
        layout.lower().startswith(p) for p in ("5.1", "6.1", "7.1")
    ):
        pytest.skip(
            f"audio layout {layout!r} has no LFE channel; set "
            "AUTO_BEQ_MEDIA_CHANNEL to bypass auto-detection"
        )

    if trim_start_s is not None or trim_end_s is not None:
        log.info(
            "APPLYING TRIM: start=%ss end=%ss - this is NOT the full film",
            trim_start_s, trim_end_s,
        )
    log.info("extracting LFE channel -> %s (fs=%d)", cache_path, target_fs)
    log.info("this may take 1-3 minutes for a feature-length movie...")
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
        "-af", "pan=mono|c0=LFE",
        "-ar", str(target_fs),
        "-ac", "1",
        str(cache_path),
    ]
    proc = subprocess.run(ff_args, capture_output=True, text=True)
    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("ffmpeg stderr:\n%s", proc.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
    size = cache_path.stat().st_size
    log.info("extracted %d bytes in %.1fs -> %s", size, elapsed, cache_path)
    return cache_path


def load_and_smooth(wav_path: Path, fs: int, freqs: np.ndarray) -> np.ndarray:
    """Load a WAV file, compute avg spectrum, interp to grid, smooth to 1/6-octave.

    Pipeline: WAV → Welch avg spectrum → interp to log grid → normalise
    to 80 Hz anchor → 1/6-octave smooth → re-anchor. Matches the
    ``test_real_media_roundtrip`` pipeline verbatim.
    """
    from model.auto_beq import smooth_fractional_octave
    from model.signal import Signal, read_wav_data

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
    anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
    measured_on_grid -= measured_on_grid[anchor_idx]
    measured_on_grid = smooth_fractional_octave(measured_on_grid, freqs, octaves=1.0 / 6.0)
    measured_on_grid -= measured_on_grid[anchor_idx]
    return measured_on_grid


# Backwards-compatible aliases (the existing test_auto_beq.py uses
# underscore-prefixed names).
_have_tool = have_tool
_probe_audio_stream = probe_audio_stream
_extract_lfe_wav = extract_lfe_wav
