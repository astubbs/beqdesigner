"""Spike integration test: can propose_filters match catalogue entries?

Parametrised over a list of (title, filter_count) fixtures from the
committed catalogue snapshot. For each entry:
  1. Compute the ground-truth magnitude curve G(f) from the entry's filters.
  2. Feed -G(f) to propose_filters as a simulated rolloff.
  3. Apply the proposed chain and assert it cancels the rolloff to tolerance.

Extending this test to dozens of titles means adding rows to FIXTURES.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
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


def _have_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True, timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(
    not os.environ.get("AUTO_BEQ_MEDIA_PATH"),
    reason="AUTO_BEQ_MEDIA_PATH not set",
)
@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_real_media_roundtrip(catalogue_snapshot):
    media_path = Path(os.environ["AUTO_BEQ_MEDIA_PATH"])
    title = os.environ.get("AUTO_BEQ_MEDIA_TITLE", media_path.stem)
    channel = int(os.environ.get("AUTO_BEQ_MEDIA_CHANNEL", "4"))  # 4 = LFE in 5.1
    filter_count_env = os.environ.get("AUTO_BEQ_MEDIA_FILTER_COUNT")
    filter_count = int(filter_count_env) if filter_count_env else None

    entry = catalogue_snapshot(title, filter_count=filter_count)
    fs = 1000
    freqs = DEFAULT_GRID

    # 1. Extract the requested channel to a 1 kHz mono WAV.
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "extracted.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(media_path),
                "-af", f"pan=mono|c0=c{channel - 1}",
                "-ar", str(fs),
                "-ac", "1",
                str(wav_path),
            ],
            check=True, capture_output=True,
        )
        assert wav_path.exists() and wav_path.stat().st_size > 0

        # 2. Load via the app's signal pipeline.
        from model.signal import Signal, read_wav_data
        samples, read_fs, _ = read_wav_data(str(wav_path))
        assert read_fs == fs
        sig = Signal(title, samples[:, 0] if samples.ndim > 1 else samples, fs=fs)

        # 3. Measured curve = average spectrum, interpolated to our grid and
        #    normalised so its 80 Hz value is 0 dB (matching evaluate_filter_chain).
        measured_freqs, measured_db = sig.avg_spectrum()
        measured_on_grid = np.interp(freqs, measured_freqs, measured_db)
        anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
        measured_on_grid -= measured_on_grid[anchor_idx]

        # 4. Compare against catalogue-implied curve (positive dB because the
        #    filters represent the correction). For comparison we negate:
        ground_truth = _ground_truth_curve(entry, freqs, fs)
        # Measured rolloff should approximate -ground_truth when the rip
        # matches the catalogue entry.
        band_mask = (freqs >= 20.0) & (freqs <= 80.0)
        divergence = float(
            np.mean(np.abs(measured_on_grid[band_mask] + ground_truth[band_mask]))
        )
        print(f"\nMedia-vs-catalogue divergence in 20-80 Hz: {divergence:.2f} dB")

        # 5. Run the optimizer on the measured curve.
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

        # Assertions are loose for real media - we report, don't fail hard,
        # unless the curve is completely off (indicates wrong file).
        assert divergence < 10.0, (
            f"measured curve diverges from catalogue by {divergence:.1f} dB - "
            "wrong release/rip?"
        )
