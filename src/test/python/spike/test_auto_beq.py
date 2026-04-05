"""Spike integration test: can propose_filters match catalogue entries?

Parametrised over a list of (title, filter_count) fixtures from the
committed catalogue snapshot. For each entry:
  1. Compute the ground-truth magnitude curve G(f) from the entry's filters.
  2. Feed -G(f) to propose_filters as a simulated rolloff.
  3. Apply the proposed chain and assert it cancels the rolloff to tolerance.

Extending this test to dozens of titles means adding rows to FIXTURES.
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
from model.auto_beq import (
    DEFAULT_GRID,
    compute_match_metrics,
    evaluate_filter_chain,
    format_match_report,
    propose_filters,
)

log = logging.getLogger("auto_beq_spike")

# (title, filter_count, expected_verdict, notes)
# filter_count disambiguates when a title has multiple catalogue entries
# (different releases/rips). PASS cases use single-LowShelf entries because
# the spike optimizer only fits LowShelf + optional PEQ.
FIXTURES = [
    ("Battle: Los Angeles", 1, "PASS", "clean single-shelf, +4 dB @ 28 Hz"),
    ("Captain America: The Winter Soldier", 1, "PASS", "+3.8 dB @ 22 Hz"),
    ("Run Hide Fight", 1, "PASS", "+6 dB @ 17 Hz"),
]


def _ground_truth_curve(entry: dict, freqs_hz: np.ndarray, fs: int) -> np.ndarray:
    """Evaluate a catalogue entry's filter chain on our standard grid."""
    return evaluate_filter_chain(entry["filters"], freqs_hz, fs=fs)


@pytest.mark.parametrize("title,filter_count,expected,notes", FIXTURES)
def test_synthetic_roundtrip(catalogue_snapshot, title, filter_count, expected, notes):
    entry = catalogue_snapshot(title, filter_count=filter_count)
    fs = 1000
    freqs = DEFAULT_GRID

    ground_truth = _ground_truth_curve(entry, freqs, fs)
    target = -ground_truth  # simulate rolled-off input content

    proposed = propose_filters(target, freqs, fs=fs)
    metrics = compute_match_metrics(target, proposed, freqs, fs=fs)

    report = format_match_report(
        f"{title} ({notes})",
        entry["filters"],
        proposed,
        metrics,
        target_depth_db=float(target[0] - target[-1]),
    )
    print("\n" + report)

    assert proposed, "optimizer returned no filters for a rolloff curve"
    if expected == "PASS":
        assert metrics.verdict == "PASS", (
            f"{title}: expected PASS but got {metrics.verdict}\n{report}"
        )
    elif expected == "MARGINAL":
        assert metrics.verdict in ("PASS", "MARGINAL"), (
            f"{title}: expected PASS/MARGINAL but got FAIL\n{report}"
        )


def test_flat_input_produces_no_filters():
    """A flat target (no rolloff to correct) should yield an empty chain."""
    flat = np.zeros_like(DEFAULT_GRID)
    assert propose_filters(flat, DEFAULT_GRID) == []


def test_report_formatter_shape(catalogue_snapshot):
    entry = catalogue_snapshot("Battle: Los Angeles", filter_count=1)
    target = -_ground_truth_curve(entry, DEFAULT_GRID, 1000)
    proposed = propose_filters(target, DEFAULT_GRID, fs=1000)
    metrics = compute_match_metrics(target, proposed, DEFAULT_GRID, fs=1000)
    report = format_match_report("test", entry["filters"], proposed, metrics, 4.0)
    assert "Auto-BEQ spike" in report
    assert "Catalogue entry" in report
    assert "Proposed" in report
    assert "Match metrics" in report


# ---------------------------------------------------------------------------
# Real-media pass: optional, gated on an env var pointing to a media file.
# Set AUTO_BEQ_MEDIA_PATH and AUTO_BEQ_MEDIA_TITLE (matching a snapshot entry)
# to run. Extracts LFE via ffmpeg -> WAV -> Signal -> avg_spectrum and feeds
# the measured curve into propose_filters.
# ---------------------------------------------------------------------------


def _have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _probe_audio_stream(media_path: Path) -> dict:
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


def _extract_lfe_wav(media_path: Path, target_fs: int) -> Path:
    """Extract the LFE channel to a cached WAV next to the source file.

    Cache file: ``<source-stem>.lfe-<fs>hz.wav`` in the same directory.
    Returns the cache path. Skips ffmpeg if the cache already exists.
    """
    cache_path = media_path.with_suffix("")
    cache_path = cache_path.parent / f"{cache_path.name}.lfe-{target_fs}hz.wav"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        log.info("cached LFE WAV found, skipping extraction: %s (%d bytes)",
                 cache_path, cache_path.stat().st_size)
        return cache_path

    stream = _probe_audio_stream(media_path)
    layout = stream.get("channel_layout", "")
    if "LFE" not in layout.upper() and not any(
        layout.lower().startswith(p) for p in ("5.1", "6.1", "7.1")
    ):
        pytest.skip(
            f"audio layout {layout!r} has no LFE channel; set "
            "AUTO_BEQ_MEDIA_CHANNEL to bypass auto-detection"
        )

    log.info("extracting LFE channel -> %s (fs=%d)", cache_path, target_fs)
    log.info("this may take 1-3 minutes for a feature-length movie...")
    start = time.time()
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-i", str(media_path),
            "-af", "pan=mono|c0=LFE",
            "-ar", str(target_fs),
            "-ac", "1",
            str(cache_path),
        ],
        capture_output=True, text=True,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("ffmpeg stderr:\n%s", proc.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
    size = cache_path.stat().st_size
    log.info("extracted %d bytes in %.1fs -> %s", size, elapsed, cache_path)
    return cache_path


@pytest.mark.skipif(
    not os.environ.get("AUTO_BEQ_MEDIA_PATH"),
    reason="AUTO_BEQ_MEDIA_PATH not set",
)
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
def test_real_media_roundtrip(catalogue_snapshot, caplog):
    caplog.set_level(logging.INFO, logger="auto_beq_spike")

    media_path = Path(os.environ["AUTO_BEQ_MEDIA_PATH"])
    title = os.environ.get("AUTO_BEQ_MEDIA_TITLE", media_path.stem)
    filter_count_env = os.environ.get("AUTO_BEQ_MEDIA_FILTER_COUNT")
    filter_count = int(filter_count_env) if filter_count_env else None

    log.info("=== real-media roundtrip ===")
    log.info("media: %s", media_path)
    log.info("title: %r  filter_count: %s", title, filter_count)
    assert media_path.exists(), f"media file not found: {media_path}"
    log.info("media file size: %.1f MB", media_path.stat().st_size / 1e6)

    entry = catalogue_snapshot(title, filter_count=filter_count)
    log.info("catalogue entry matched: %d filters", len(entry["filters"]))

    fs = 1000
    freqs = DEFAULT_GRID

    wav_path = _extract_lfe_wav(media_path, fs)

    # Load via the app's signal pipeline.
    log.info("loading WAV into Signal pipeline")
    from model.signal import Signal, read_wav_data
    samples, read_fs, _ = read_wav_data(str(wav_path))
    assert read_fs == fs, f"expected fs={fs}, got {read_fs}"
    mono = samples[:, 0] if samples.ndim > 1 else samples
    duration_s = len(mono) / fs
    log.info("loaded %d samples (%.1f s = %.1f min)", len(mono), duration_s, duration_s / 60)
    sig = Signal(title, mono, fs=fs)

    log.info("computing average spectrum (Welch)")
    measured_freqs, measured_db = sig.avg_spectrum()
    log.info("raw spectrum: %d bins from %.1f to %.1f Hz",
             len(measured_freqs), measured_freqs[0], measured_freqs[-1])

    measured_on_grid = np.interp(freqs, measured_freqs, measured_db)
    anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
    measured_on_grid -= measured_on_grid[anchor_idx]
    log.info("measured curve on grid: 10Hz=%.1f 20Hz=%.1f 80Hz=%.1f 200Hz=%.1f dB",
             measured_on_grid[0],
             measured_on_grid[int(np.argmin(np.abs(freqs - 20.0)))],
             measured_on_grid[anchor_idx],
             measured_on_grid[-1])

    # Catalogue-implied correction curve (positive dB - shelves boost).
    ground_truth = _ground_truth_curve(entry, freqs, fs)
    log.info("catalogue curve on grid:  10Hz=%+.1f 20Hz=%+.1f 80Hz=%+.1f dB",
             ground_truth[0],
             ground_truth[int(np.argmin(np.abs(freqs - 20.0)))],
             ground_truth[anchor_idx])

    # Measured rolloff should approximate -ground_truth if the rip matches.
    band_mask = (freqs >= 20.0) & (freqs <= 80.0)
    divergence = float(
        np.mean(np.abs(measured_on_grid[band_mask] + ground_truth[band_mask]))
    )
    log.info("media-vs-catalogue divergence in 20-80 Hz: %.2f dB", divergence)

    log.info("running optimizer on measured curve")
    proposed = propose_filters(measured_on_grid, freqs, fs=fs)
    metrics = compute_match_metrics(measured_on_grid, proposed, freqs, fs=fs)
    report = format_match_report(
        f"{title} (real media)",
        entry["filters"],
        proposed,
        metrics,
        target_depth_db=float(measured_on_grid[0] - measured_on_grid[-1]),
    )
    print("\n" + report)

    # Hard-fail if the measured curve is nowhere near the catalogue entry:
    # that usually means the rip doesn't match the entry's release/master.
    assert divergence < 15.0, (
        f"measured curve diverges from catalogue by {divergence:.1f} dB - "
        "wrong release/rip?"
    )

    # Gate the test on the algorithm grade. By default we require PASS -
    # a FAIL grade means the optimizer genuinely couldn't match the
    # target and the algorithm needs improvement, not a softer test.
    # For known-hard cases (e.g. deep cascaded-shelf catalogue entries
    # the spike's shelf+PEQ scope can't match yet), set
    # AUTO_BEQ_MEDIA_EXPECTED_GRADE=FAIL or MARGINAL to record the
    # expectation explicitly.
    expected_grade = os.environ.get("AUTO_BEQ_MEDIA_EXPECTED_GRADE", "PASS")
    grade_rank = {"PASS": 0, "MARGINAL": 1, "FAIL": 2}
    assert grade_rank[metrics.verdict] <= grade_rank[expected_grade], (
        f"algorithm grade {metrics.verdict} worse than expected "
        f"{expected_grade} for {title}\n\n{report}"
    )
