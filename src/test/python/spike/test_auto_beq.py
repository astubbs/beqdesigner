"""Spike integration tests for the auto-BEQ proposal pipeline.

The spike's central question (from the vision brief):

    Can we take the human out of the chain of making BEQ filters?
    Given only a measured LFE curve from a media file, can we
    automatically propose filter parameters that resemble what a
    human BEQ expert prescribed (i.e. the catalogue entry)?

Three tests, in order of how much of the production problem they
exercise:

1. test_synthetic_roundtrip - PURE CURVE FITTING, no measurement.
   Validates that scipy.optimize + an N-filter chain can reproduce
   known catalogue response curves when handed them directly. If
   this fails, scipy can't even fit known targets and nothing else
   is worth trying.

2. test_real_media_roundtrip - THE ACTUAL SPIKE QUESTION.
   Extracts LFE from a real media file, computes its spectrum,
   feeds the measured curve into propose_filters, and compares the
   resulting filter chain's response to the catalogue entry's
   response. Currently expected to FAIL: we don't yet have a way to
   turn "measured LFE" into "catalogue-shaped correction target".
   These failures are the core remaining work of the spike - each
   one tells us how wrong our current heuristic is per-title.

3. Supporting sanity tests (flat input, report formatter).

The grading thresholds (<2 dB mean / <5 dB max error in 5-80 Hz)
come from the vision brief. The PASS verdict means "this proposal
resembles expert output closely enough to serve as a starting point
for a magic-wand button".
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
    propose_filters_from_measured,
    smooth_fractional_octave,
)

log = logging.getLogger("auto_beq_spike")

# Synthetic-roundtrip fixtures. For each (title, filter_count):
#   - Compute the catalogue entry's response curve G(f).
#   - Hand G(f) directly to the fitter (via -G as the "cancel" target).
#   - Assert the fitter reproduces G(f) within the pass thresholds.
#
# filter_count disambiguates when a title has multiple catalogue entries
# (different releases). All of these are 1-filter entries - easy targets
# for the fitter. They are a floor gate: if synthetic fails, the fitter
# itself is broken and real-media can't possibly pass.
SYNTHETIC_FIXTURES = [
    ("Battle: Los Angeles", 1, "PASS", "clean single-shelf, +4 dB @ 28 Hz"),
    ("Captain America: The Winter Soldier", 1, "PASS", "+3.8 dB @ 22 Hz"),
    ("Run Hide Fight", 1, "PASS", "+6 dB @ 17 Hz"),
]


def _ground_truth_curve(entry: dict, freqs_hz: np.ndarray, fs: int) -> np.ndarray:
    """Evaluate a catalogue entry's filter chain on our standard grid."""
    return evaluate_filter_chain(entry["filters"], freqs_hz, fs=fs)


@pytest.mark.parametrize("title,filter_count,expected,notes", SYNTHETIC_FIXTURES)
def test_synthetic_roundtrip(catalogue_snapshot, title, filter_count, expected, notes):
    """FLOOR GATE: fitter can reproduce a known catalogue curve.

    No measurement involved. Computes the catalogue entry's response
    curve and asks the fitter to reproduce it. This tests the fitter
    in isolation from the measurement pipeline.

    Passing here does NOT prove the magic wand works. It only proves
    scipy.optimize + the N-filter chain can fit known bass-curve
    shapes. This is a prerequisite for the real test below.
    """
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
# Real-media pass: parametrised over a user-maintained media manifest.
#
# The manifest is a JSON list of entries, each with:
#   {
#     "path": "/absolute/path/to/file.mkv",
#     "title": "Catalogue title",
#     "filter_count": 5,
#     "expected_grade": "PASS" | "MARGINAL" | "FAIL",
#     "notes": "free-form"
#   }
# Defaults: expected_grade=PASS, filter_count=None (first match).
#
# Location (in order):
#   $AUTO_BEQ_MEDIA_MANIFEST (file path)
#   ~/.config/beqdesigner/auto_beq_media.json
#
# Entries whose path does not exist on disk are skipped silently so the
# manifest can be shared across machines with different media libraries.
# Each entry runs as its own parametrised test case; adding a new film is
# "add one more JSON object" in the manifest.
# ---------------------------------------------------------------------------


def _have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _load_media_manifest() -> list[dict]:
    manifest_env = os.environ.get("AUTO_BEQ_MEDIA_MANIFEST")
    if manifest_env:
        manifest_path = Path(manifest_env)
    else:
        manifest_path = Path.home() / ".config" / "beqdesigner" / "auto_beq_media.json"
    if not manifest_path.exists():
        return []
    with manifest_path.open() as f:
        entries = json.load(f)
    runnable = []
    for entry in entries:
        path = Path(entry["path"]).expanduser()
        if not path.exists():
            continue
        runnable.append({
            "path": str(path),
            "title": entry["title"],
            "filter_count": entry.get("filter_count"),
            "expected_grade": entry.get("expected_grade", "PASS"),
            "notes": entry.get("notes", ""),
        })
    return runnable


MEDIA_MANIFEST = _load_media_manifest()


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


@pytest.mark.skipif(not MEDIA_MANIFEST, reason="no media manifest / no files on disk")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "manifest_entry",
    MEDIA_MANIFEST,
    ids=[e["title"] for e in MEDIA_MANIFEST],
)
def test_real_media_roundtrip(catalogue_snapshot, caplog, manifest_entry):
    """THE CORE SPIKE QUESTION: end-to-end magic-wand simulation.

    Exercises every step of the production "magic wand" workflow:
      1. Extract LFE from a real media file (ffmpeg, cached).
      2. Load samples via the app's Signal pipeline.
      3. Compute the average spectrum (Welch), interp to log grid,
         normalise to 80 Hz anchor, smooth to 1/6-octave.
      4. Hand ONLY the measured curve to propose_filters. No
         catalogue access at inference time - production has none.
      5. Compare the proposed chain's response to the catalogue
         entry's response using the vision-brief thresholds
         (<2 dB mean / <5 dB max in 5-80 Hz).

    This is the test that determines whether the spike's core
    value-prop works: can we take the human out of BEQ-making?

    Currently expected to FAIL on deep catalogue entries because
    propose_filters has no heuristic to bridge measured-curve to
    catalogue-shaped target. Each failure is data: it tells us how
    aggressively each title's expert chose to extend versus what
    the measured content naturally shows. Fix by teaching the
    fitter how to construct a target-correction curve from the
    measured curve alone.
    """
    caplog.set_level(logging.INFO, logger="auto_beq_spike")

    media_path = Path(manifest_entry["path"])
    title = manifest_entry["title"]
    filter_count = manifest_entry["filter_count"]
    expected_grade = manifest_entry["expected_grade"]
    notes = manifest_entry["notes"]

    log.info("=== real-media roundtrip ===")
    log.info("media: %s", media_path)
    log.info("title: %r  filter_count: %s  expected_grade: %s",
             title, filter_count, expected_grade)
    if notes:
        log.info("notes: %s", notes)
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
    # Smooth to 1/6-octave - standard for BEQ-style analysis. Narrow
    # resonances in the raw spectrum are mastering artefacts, not
    # features an IIR filter should chase.
    measured_on_grid = smooth_fractional_octave(measured_on_grid, freqs, octaves=1.0 / 6.0)
    measured_on_grid -= measured_on_grid[anchor_idx]
    log.info("smoothed curve on grid: 10Hz=%.1f 20Hz=%.1f 80Hz=%.1f 200Hz=%.1f dB",
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

    # Pipeline sanity: the extracted LFE should contain real LFE content
    # (not silence, not a constant). 5 dB of dynamic range in-band is a
    # weak lower bound that catches "forgot to select the right channel"
    # or "file is silent".
    band_mask_5_80 = (freqs >= 5.0) & (freqs <= 80.0)
    in_band_range = float(
        measured_on_grid[band_mask_5_80].max() - measured_on_grid[band_mask_5_80].min()
    )
    log.info("measured in-band dynamic range: %.1f dB", in_band_range)
    assert in_band_range > 5.0, (
        "measured curve has <5 dB dynamic range in 5-80 Hz - extraction "
        "pipeline probably extracted silence or wrong channel"
    )

    # ===== THE CORE SPIKE QUESTION =====
    # Feed ONLY the measured curve into propose_filters (no catalogue
    # access at inference time - this matches production, where the
    # magic wand has no ground truth). Grade the proposed chain's
    # response against the catalogue entry's response.
    #
    # A PASS means "our proposed chain resembles expert output within
    # the tolerance thresholds". A FAIL is the central spike finding
    # that tells us we don't yet know how to map measured LFE curves
    # to expert-shaped correction targets.
    log.info("running propose_filters_from_measured (production scenario)")
    proposed = propose_filters_from_measured(measured_on_grid, freqs, fs=fs)
    # compute_match_metrics grades |target + evaluate(chain)|.
    # To grade |evaluate(proposed) - catalogue|, pass target=-catalogue.
    metrics = compute_match_metrics(-ground_truth, proposed, freqs, fs=fs)
    report = format_match_report(
        f"{title} (measured -> proposed filters vs catalogue)",
        entry["filters"],
        proposed,
        metrics,
        target_depth_db=float(ground_truth.max() - ground_truth.min()),
    )
    print("\n" + report)

    # Gate the test on the algorithm grade. Manifest entries default to
    # expecting PASS; known-hard cases (deep cascaded-shelf catalogue
    # entries the spike's shelf+PEQ scope can't match yet) can set
    # "expected_grade": "FAIL" or "MARGINAL" in the manifest to record
    # the expectation explicitly. A FAIL grade means the optimizer
    # genuinely couldn't match the target and the algorithm needs
    # improvement, not a softer test.
    grade_rank = {"PASS": 0, "MARGINAL": 1, "FAIL": 2}
    assert grade_rank[metrics.verdict] <= grade_rank[expected_grade], (
        f"algorithm grade {metrics.verdict} worse than expected "
        f"{expected_grade} for {title}\n\n{report}"
    )
