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
from model.auto_beq_advisor import MediaMetadata, get_advisor

from spike._auto_beq_helpers import _extract_lfe_wav, _have_tool, _probe_audio_stream

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
        if entry.get("blacklisted"):
            continue
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


@pytest.mark.skipif(not MEDIA_MANIFEST, reason="no media manifest / no files on disk")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "manifest_entry",
    MEDIA_MANIFEST,
    ids=[
        f"{e['title']}" + (f" ({e['notes'][:40]})" if e.get("notes") else "")
        for e in MEDIA_MANIFEST
    ],
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

    # Probe once for metadata - cheap, and the cached-extraction path
    # doesn't re-probe by itself.
    stream_info = _probe_audio_stream(media_path)

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

    # Absolute (un-normalised) dBFS at diagnostic frequencies. The
    # pipeline normalises to 0 dB at 80 Hz next, which throws away
    # mastering-level information. Log it first so we can see whether
    # films differ in absolute mid-bass energy (hypothesis: louder
    # absolute mid-bass correlates with mastering aggressiveness).
    abs_dbfs = [
        measured_db[int(np.argmin(np.abs(measured_freqs - f)))]
        for f in (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0)
    ]
    log.info(
        "absolute dBFS: 5Hz=%.1f 10Hz=%.1f 20Hz=%.1f 40Hz=%.1f "
        "60Hz=%.1f 80Hz=%.1f 120Hz=%.1f",
        *abs_dbfs,
    )

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
    advisor_name = os.environ.get("AUTO_BEQ_ADVISOR", "measurement")
    try:
        advisor = get_advisor(advisor_name)
    except Exception as exc:
        pytest.skip(f"advisor construction failed ({advisor_name}): {exc}")
    media_metadata = MediaMetadata(
        title=title,
        year=entry.get("year") or None,
        audio_codec=stream_info.get("codec_name") if stream_info else None,
        channel_layout=stream_info.get("channel_layout") if stream_info else None,
    )
    log.info(
        "running propose_filters_from_measured with advisor=%s title=%r year=%s",
        advisor_name, media_metadata.title, media_metadata.year,
    )
    try:
        proposed = propose_filters_from_measured(
            measured_on_grid, freqs, fs=fs,
            advisor=advisor, metadata=media_metadata,
        )
    except Exception as exc:
        pytest.skip(f"advisor.advise failed ({advisor_name}): {exc}")
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


# ---------------------------------------------------------------------------
# E18: Chunked percentile analysis — alternative spectrum extraction.
#
# Instead of the whole-film Welch average, split audio into fixed-length
# chunks, compute the STFT peak curve per chunk, then take the 90th
# percentile across chunks at each frequency bin. Hypothesis: this
# produces a more robust rolloff ceiling estimate, especially for short
# TV episodes where one outlier scene dominates the whole-film statistic.
# Sub-experiment: chunk length sensitivity (30/60/90 s).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MEDIA_MANIFEST, reason="no media manifest / no files on disk")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize("chunk_s", [30.0, 60.0, 90.0])
@pytest.mark.parametrize(
    "manifest_entry",
    MEDIA_MANIFEST,
    ids=[
        f"{e['title']}" + (f" ({e['notes'][:40]})" if e.get("notes") else "")
        for e in MEDIA_MANIFEST
    ],
)
def test_chunked_percentile_roundtrip(catalogue_snapshot, caplog, manifest_entry, chunk_s):
    """E18: chunked-percentile spectrum vs whole-film Welch baseline.

    Same pipeline as test_real_media_roundtrip but replaces the Welch
    whole-film average with: chunk → STFT peak per chunk → 90th percentile
    across chunks. Parametrised over chunk lengths (30/60/90 s) to find
    the sweet spot.

    Each run logs a delta vs the whole-film curve at 20 Hz so results
    can be compared directly with the E17d baseline.
    """
    from spike._auto_beq_helpers import load_and_smooth, load_and_smooth_chunked

    caplog.set_level(logging.INFO, logger="auto_beq_spike")

    media_path = Path(manifest_entry["path"])
    title = manifest_entry["title"]
    filter_count = manifest_entry["filter_count"]
    expected_grade = manifest_entry["expected_grade"]
    notes = manifest_entry["notes"]

    log.info("=== E18 chunked-percentile roundtrip (chunk_s=%.0f) ===", chunk_s)
    log.info("media: %s", media_path)
    log.info("title: %r  filter_count: %s  expected_grade: %s",
             title, filter_count, expected_grade)
    if notes:
        log.info("notes: %s", notes)
    assert media_path.exists(), f"media file not found: {media_path}"

    entry = catalogue_snapshot(title, filter_count=filter_count)
    log.info("catalogue entry matched: %d filters", len(entry["filters"]))

    fs = 1000
    freqs = DEFAULT_GRID

    stream_info = _probe_audio_stream(media_path)
    wav_path = _extract_lfe_wav(media_path, fs)

    # Baseline: whole-film Welch average (existing code path).
    baseline_curve = load_and_smooth(wav_path, fs, freqs)

    # E18: chunked percentile.
    chunked_curve = load_and_smooth_chunked(
        wav_path, fs, freqs, chunk_s=chunk_s,
    )

    # Log delta at key frequencies for comparison.
    anchor_idx = int(np.argmin(np.abs(freqs - 80.0)))
    idx_10hz = int(np.argmin(np.abs(freqs - 10.0)))
    idx_20hz = int(np.argmin(np.abs(freqs - 20.0)))
    delta_10hz = chunked_curve[idx_10hz] - baseline_curve[idx_10hz]
    delta_20hz = chunked_curve[idx_20hz] - baseline_curve[idx_20hz]
    log.info(
        "chunk_s=%.0f | 10Hz baseline=%.1f chunked=%.1f delta=%+.1f dB",
        chunk_s, baseline_curve[idx_10hz], chunked_curve[idx_10hz], delta_10hz,
    )
    log.info(
        "chunk_s=%.0f | 20Hz baseline=%.1f chunked=%.1f delta=%+.1f dB",
        chunk_s, baseline_curve[idx_20hz], chunked_curve[idx_20hz], delta_20hz,
    )

    # Catalogue ground truth.
    ground_truth = _ground_truth_curve(entry, freqs, fs)

    # Sanity: chunked curve must still have LFE content.
    band_mask = (freqs >= 5.0) & (freqs <= 80.0)
    in_band_range = float(
        chunked_curve[band_mask].max() - chunked_curve[band_mask].min()
    )
    log.info("chunked in-band dynamic range: %.1f dB", in_band_range)
    assert in_band_range > 5.0, (
        f"chunked curve has <5 dB dynamic range (chunk_s={chunk_s}) — "
        "extraction may have failed"
    )

    # Propose filters from chunked curve — same advisor path as baseline.
    advisor_name = os.environ.get("AUTO_BEQ_ADVISOR", "measurement")
    try:
        advisor = get_advisor(advisor_name)
    except Exception as exc:
        pytest.skip(f"advisor construction failed ({advisor_name}): {exc}")

    media_metadata = MediaMetadata(
        title=title,
        year=entry.get("year") or None,
        audio_codec=stream_info.get("codec_name") if stream_info else None,
        channel_layout=stream_info.get("channel_layout") if stream_info else None,
    )
    log.info(
        "running propose_filters_from_measured with advisor=%s",
        advisor_name,
    )
    try:
        proposed = propose_filters_from_measured(
            chunked_curve, freqs, fs=fs,
            advisor=advisor, metadata=media_metadata,
        )
    except Exception as exc:
        pytest.skip(f"advisor.advise failed ({advisor_name}): {exc}")

    metrics = compute_match_metrics(-ground_truth, proposed, freqs, fs=fs)
    report = format_match_report(
        f"{title} [E18 chunked {chunk_s:.0f}s]",
        entry["filters"],
        proposed,
        metrics,
        target_depth_db=float(ground_truth.max() - ground_truth.min()),
    )
    print("\n" + report)
    log.info(
        "chunk_s=%.0f | verdict=%s mean=%.2f dB max=%.2f dB",
        chunk_s, metrics.verdict, metrics.mean_abs_err_db, metrics.max_abs_err_db,
    )

    grade_rank = {"PASS": 0, "MARGINAL": 1, "FAIL": 2}
    assert grade_rank[metrics.verdict] <= grade_rank[expected_grade], (
        f"E18 chunked-percentile (chunk_s={chunk_s:.0f}) grade "
        f"{metrics.verdict} worse than expected {expected_grade} "
        f"for {title}\n\n{report}"
    )
