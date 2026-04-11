"""Advisor abstraction for auto-BEQ aggressiveness decisions.

The scipy N-filter fitter in ``model.auto_beq`` is mathematically sound
but has no world knowledge about films. It cannot decide how aggressive
a BEQ should be for a given title - that's expert judgment drawing on
genre, director, sound-design reputation, and era conventions.

An ``Advisor`` fills exactly that judgment gap. Given a film's metadata
and a measured LFE curve's features, it returns a small numeric
``Advice`` (`max_gain_db`, `knee_hz`, reasoning, confidence) that the
procedural pipeline uses to shape its target-correction curve. The
fitter stays untouched.

Three implementations:

- ``HeuristicAdvisor`` wraps the legacy ``classify_content`` rules.
  Deterministic, offline, baseline behaviour preserved.
- ``MockAdvisor`` reads canned JSON per title from
  ``src/test/resources/auto_beq/advisor_responses/``. For
  reproducible tests.
- ``OllamaAdvisor`` calls a local Ollama server. Uses the LLM's world
  knowledge (film reputation, sound-design intent) plus structured
  measurement features to pick numbers.

Selection is via ``get_advisor(name)`` with env-var fallback
(``AUTO_BEQ_ADVISOR``).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import threading
import time as _time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger("auto_beq_advisor")

# Constraints for clamping advice to sane ranges. Anything outside these
# would propose filters that no realistic BEQ would use.
MAX_GAIN_DB_RANGE = (0.0, 35.0)
KNEE_HZ_RANGE = (5.0, 80.0)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen:14b"
OLLAMA_TIMEOUT_SECONDS = 120
MAX_REFINE_PASSES = 3

# Shared round-robin counter for multi-host load balancing.
_OLLAMA_CALL_COUNTER = itertools.count()
_OLLAMA_HOST_STATS_LOCK = threading.Lock()
_OLLAMA_HOST_STATS: dict[str, dict] = {}  # host -> {calls, total_s, errors}

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MOCK_RESPONSES_DIR = _REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "advisor_responses"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaMetadata:
    """What we tell the advisor about the film."""

    title: str
    year: int | None = None
    audio_codec: str | None = None
    channel_layout: str | None = None
    # Tier 1 — directly from BEQ catalogue (no external lookup needed)
    audio_types: tuple[str, ...] = ()   # e.g. ("Atmos",), ("DTS-HD MA 7.1",)
    source: str | None = None           # "Disc" | "Streaming"
    genres: tuple[str, ...] = ()        # e.g. ("Action", "Science Fiction")
    language: str | None = None         # "English", "Korean", etc.
    # Tier 1 — needs external lookup (TMDb); zero-padded in ML model until then
    studio: str | None = None           # "Disney", "Warner", "Universal", etc.
    all_studios: tuple[str, ...] = ()   # all production companies from TMDb
    # Tier 2 — needs external lookup (IMDB); zero-padded until then
    supervising_mixer: str | None = None
    # Tier 3 — from catalogue, low marginal value but cost-free
    rating: str | None = None           # "PG-13", "R", etc.
    runtime_min: int | None = None
    # BEQ profile author — different authors have different calibration styles
    author: str | None = None           # "mobe1969", "aron7awol", etc.


@dataclass(frozen=True)
class CurveFeatures:
    """Summary of a measured LFE curve for advisor consumption.

    All dB values are relative to the 80 Hz anchor (the normalisation
    applied before features are extracted).
    """

    shoulder_peak_db: float
    shoulder_peak_hz: float
    level_at_5hz_db: float
    level_at_10hz_db: float
    level_at_20hz_db: float
    rolloff_depth_db: float
    rolloff_slope_db_per_oct: float
    dynamic_range_db: float
    curve_sample_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    # F2 (Option B): per-bin chunk statistics — None when not available
    # (e.g. synthetic features or Welch-only extraction).
    chunk_stddev: tuple[float, ...] | None = None
    chunk_ceiling_frac: tuple[float, ...] | None = None
    # F3: absolute dBFS levels at Option A bins, BEFORE 80 Hz normalisation.
    # Captures mastering-level information that normalisation strips out.
    absolute_dbfs: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class Advice:
    """Advisor output: how aggressive to extend and, optionally, the
    filter chain structure to reproduce.

    Three levels of specificity:
      1. ``max_gain_db`` only: procedural pipeline runs peak-extension
         with that cap.
      2. ``max_gain_db`` + ``knee_hz``: pipeline builds correction as
         a cascade of moderate-gain LowShelves at that knee.
      3. ``filters`` set: pipeline uses this chain verbatim as its
         correction target. Lets the advisor prescribe multi-knee
         structures matching catalogue entries like Mad Max
         (shelves at 10 Hz AND 18 Hz) that a single-knee cascade
         can't reach.

    When ``filters`` is set, ``max_gain_db`` and ``knee_hz`` are
    informational (used in logs/reports but not for target
    construction).
    """

    max_gain_db: float
    knee_hz: float | None = None
    filters: tuple[dict, ...] | None = None
    reasoning: str = ""
    confidence: float = 0.5
    source: str = ""
    # Cascade construction params (used by infer_correction_from_measured).
    # Advisors can override these to tune the shelf cascade shape.
    cascade_gain_ratio: float = 7.0   # dB per shelf in cascade
    cascade_q: float = 0.9            # Q for each cascaded shelf


class Advisor(Protocol):
    name: str

    def advise(
        self, metadata: MediaMetadata, features: CurveFeatures
    ) -> Advice: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_curve_features(
    curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    band: tuple[float, float] = (5.0, 80.0),
    peak_search_band: tuple[float, float] = (15.0, 40.0),
) -> CurveFeatures:
    """Summarise a measured LFE curve into advisor-friendly features.

    Assumes curve_db is already normalised to 0 dB at the band's upper
    anchor (typically 80 Hz).
    """
    band_mask = (freqs_hz >= band[0]) & (freqs_hz <= band[1])
    peak_mask = band_mask & (freqs_hz >= peak_search_band[0]) & (freqs_hz <= peak_search_band[1])
    band_vals = curve_db[band_mask]

    # Robust local sampler: mean across +/- 1/6 octave around target.
    # Raw curves have narrow resonances and notches that a single-bin
    # lookup picks up as noise. A local octave-window average is what
    # the expert's eye does when reading a smoothed curve.
    def _at(target_hz: float, half_octave: float = 1.0 / 6.0) -> float:
        lo = target_hz * (2.0 ** -half_octave)
        hi = target_hz * (2.0 ** half_octave)
        window = (freqs_hz >= lo) & (freqs_hz <= hi)
        if not window.any():
            idx = int(np.argmin(np.abs(freqs_hz - target_hz)))
            return float(curve_db[idx])
        return float(curve_db[window].mean())

    # Peak uses a smoothed view too (same half-octave window).
    if peak_mask.any():
        peak_freqs_in_band = freqs_hz[peak_mask]
        # Evaluate each candidate frequency as a local mean.
        peak_smoothed = np.array([_at(float(f)) for f in peak_freqs_in_band])
        peak_level = float(peak_smoothed.max())
        peak_freq = float(peak_freqs_in_band[int(np.argmax(peak_smoothed))])
    else:
        peak_level = float(band_vals.max())
        peak_freq = float(freqs_hz[band_mask][int(np.argmax(band_vals))])

    level_5 = _at(5.0)
    level_10 = _at(10.0)
    level_20 = _at(20.0)

    # Slope between 10 and 20 Hz, in dB/octave.
    slope = level_20 - level_10  # one octave

    rolloff_depth = peak_level - float(band_vals.min())
    dynamic_range = float(band_vals.max() - band_vals.min())

    # 12 log-spaced sample points for the LLM prompt.
    sample_hz = [5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 32.0, 40.0, 63.0, 80.0]
    samples = tuple((hz, _at(hz)) for hz in sample_hz)

    return CurveFeatures(
        shoulder_peak_db=peak_level,
        shoulder_peak_hz=peak_freq,
        level_at_5hz_db=level_5,
        level_at_10hz_db=level_10,
        level_at_20hz_db=level_20,
        rolloff_depth_db=rolloff_depth,
        rolloff_slope_db_per_oct=slope,
        dynamic_range_db=dynamic_range,
        curve_sample_points=samples,
    )


# ---------------------------------------------------------------------------
# Clamp + validate
# ---------------------------------------------------------------------------


def _clamp_advice(advice: Advice, source: str) -> Advice:
    """Clamp numeric fields to sane ranges, preserve other fields."""
    max_gain = float(np.clip(advice.max_gain_db, *MAX_GAIN_DB_RANGE))
    knee: float | None
    if advice.knee_hz is None:
        knee = None
    else:
        knee = float(np.clip(advice.knee_hz, *KNEE_HZ_RANGE))
    confidence = float(np.clip(advice.confidence, 0.0, 1.0))
    # Validate filter chain shape if present.
    chain = None
    if advice.filters is not None:
        chain = tuple(
            {
                "type": str(f["type"]),
                "freq": float(np.clip(float(f["freq"]), 5.0, 200.0)),
                "q": float(np.clip(float(f["q"]), 0.1, 10.0)),
                "gain": float(np.clip(float(f["gain"]), -30.0, 30.0)),
            }
            for f in advice.filters
            if str(f.get("type")) in ("LowShelf", "HighShelf", "PeakingEQ")
        )
        if not chain:
            chain = None
    return Advice(
        max_gain_db=max_gain,
        knee_hz=knee,
        filters=chain,
        reasoning=advice.reasoning,
        confidence=confidence,
        source=source,
    )


# ---------------------------------------------------------------------------
# HeuristicAdvisor - legacy rule-based classifier
# ---------------------------------------------------------------------------


class HeuristicAdvisor:
    """Reproduces the legacy classify_content rules.

    Used as the deterministic fallback and for comparison against the
    LLM path. Returns Advice with source="heuristic".
    """

    name = "heuristic"

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        deficit = features.rolloff_depth_db
        if deficit > 25.0:
            profile, max_gain = "cliff", 12.0
        elif deficit < 15.0:
            profile, max_gain = "mild", 15.0
        else:
            profile, max_gain = "middle", 15.0
        reasoning = (
            f"heuristic classifier: rolloff_depth={deficit:.1f} dB -> {profile}"
        )
        return _clamp_advice(
            Advice(max_gain_db=max_gain, knee_hz=None, reasoning=reasoning, confidence=0.3),
            source="heuristic",
        )


# ---------------------------------------------------------------------------
# MeasurementAdvisor - pure signal-processing, no film knowledge
# ---------------------------------------------------------------------------


_MAX_TOTAL_CHAIN_GAIN_DB = 35.0  # E20: raised from 30.0


def _measurement_chain(
    features: CurveFeatures,
    q: float = 0.8,
    max_total_gain_db: float = _MAX_TOTAL_CHAIN_GAIN_DB,
    max_shelves: int = 2,
) -> tuple[dict, ...] | None:
    """Build an explicit multi-knee LowShelf chain.

    **2-shelf mode** (default, ``max_shelves=2``):
      outer_gain = shoulder_peak - level_at_20hz  (lift 20 Hz to peak)
      inner_gain = level_at_20hz - level_at_10hz  (lift 10 Hz to L20)

    **3-shelf mode** (``max_shelves=3``): adds a third shelf below
    10 Hz when the deficit between 10 Hz and 5 Hz is significant:
      outer_gain = shoulder_peak - level_at_20hz
      middle_gain = level_at_20hz - level_at_10hz
      inner_gain = level_at_10hz - level_at_5hz

    Total chain gain is scaled proportionally to stay within
    *max_total_gain_db*.

    Returns None when no meaningful chain is warranted (total < 1 dB).
    """
    outer_knee_hz = float(np.clip(features.shoulder_peak_hz, 12.0, 40.0))

    # Gains from measured deficits at reference frequencies.
    outer_gain = max(0.0, features.shoulder_peak_db - features.level_at_20hz_db)
    mid_gain = max(0.0, features.level_at_20hz_db - features.level_at_10hz_db)

    # 3-shelf: add inner deficit (10→5 Hz) if enabled and significant.
    inner_gain = 0.0
    if max_shelves >= 3:
        inner_gain = max(0.0, features.level_at_10hz_db - features.level_at_5hz_db)

    gains = [g for g in [inner_gain, mid_gain, outer_gain] if g > 1.0]
    total = sum(gains)
    if total < 1.0:
        return None

    # Cliff scaling: preserve the ratio, cap the sum.
    if total > max_total_gain_db:
        scale = max_total_gain_db / total
        inner_gain *= scale
        mid_gain *= scale
        outer_gain *= scale

    # Knee frequencies: spread across the rolloff range.
    mid_knee_hz = float(np.clip(features.shoulder_peak_hz / 2.0, 8.0, 14.0))
    inner_knee_hz = float(np.clip(features.shoulder_peak_hz / 3.0, 5.0, 10.0))

    chain: list[dict] = []
    if max_shelves >= 3 and inner_gain > 1.0:
        chain.append({
            "type": "LowShelf",
            "freq": inner_knee_hz,
            "q": q,
            "gain": float(inner_gain),
        })
    if mid_gain > 1.0:
        chain.append({
            "type": "LowShelf",
            "freq": mid_knee_hz,
            "q": q,
            "gain": float(mid_gain),
        })
    if outer_gain > 1.0:
        chain.append({
            "type": "LowShelf",
            "freq": outer_knee_hz,
            "q": q,
            "gain": float(outer_gain),
        })
    if not chain:
        return None
    return tuple(chain)


def _find_rolloff_start(features: CurveFeatures) -> float:
    """Find where the LFE curve starts rolling off (3 dB below peak).

    Walk down the curve sample points from the shoulder peak. The first
    frequency where the level drops 3 dB below the peak is the rolloff
    start — this is where catalogue entries typically place their shelf
    knee.

    Falls back to shoulder_peak_hz / 1.5 if no 3 dB crossing found
    (clamped to 10-30 Hz range for safety).
    """
    peak_db = features.shoulder_peak_db
    peak_hz = features.shoulder_peak_hz
    threshold = peak_db - 3.0

    # Use the 12-point curve samples. Walk from peak frequency downward.
    below_peak = [
        (hz, db) for hz, db in features.curve_sample_points
        if hz < peak_hz
    ]
    # Sort by frequency descending (start just below peak, walk down).
    below_peak.sort(key=lambda p: p[0], reverse=True)

    for hz, db in below_peak:
        if db < threshold:
            return float(np.clip(hz, 10.0, 30.0))

    # No 3 dB crossing found — estimate from peak position.
    return float(np.clip(peak_hz / 1.5, 10.0, 30.0))


class MeasurementAdvisor:
    """Derive BEQ gain + knee purely from the measured CurveFeatures.

    Core idea (from the user's insight + docs/workflow/beq.md):
    The BEQ target gain is NOT a property of the film's genre or
    reputation. It is a property of the MEASURED signal:

      - Shoulder peak tells us "how loud the LFE shoulder reaches".
      - Deficit at 10 Hz = shoulder_peak - level_at_10hz tells us
        how far below the shoulder the deep bass sits.
      - Slope between 10 and 20 Hz (level_at_20 - level_at_10) tells
        us how aggressive the natural rolloff is.

    Formula (single-knee path):
      max_gain_db = deficit_at_10hz + max(0, slope_db_per_oct)
      knee_hz     = shoulder_peak_hz

    i.e. "lift 10 Hz up to the shoulder, then carry the slope one
    more octave down to 5 Hz". This matches the expert workflow in
    docs/workflow/beq.md ("project the slope into the infra").

    Multi-knee path: trigger when slope > 15 dB/oct OR the existing
    looks_multi_knee() heuristic fires. Emit an explicit chain with
    two LowShelves (inner at shoulder/2, outer at shoulder), split
    by measured deficits - no magic ratio.

    Ignores `metadata` entirely. No film knowledge, no LLM, no tier.

    ==== Failure modes (documented honestly) ====
    1. Aesthetic catalogue choices divorced from measurement:
       a film where the catalogue author prescribed more/less gain
       than the signal suggests. We cannot see intent from the curve.
    2. Dialog-dominant films with no LFE shoulder (peak < 2 dB):
       we return max_gain_db=0 (no correction) rather than guessing.
    3. Overshoot-compensation PEQs (e.g. Mad Max's -6 dB @ 11 Hz Q=8)
       are NOT emitted. The downstream N-filter fitter may add them
       automatically; otherwise residual error shows in max_abs_err.

    ==== Future hypothesis (not implemented yet) ====
    The user's full conjecture is that absolute mid-bass energy
    (how LOUD the 40-80 Hz band is in unnormalised terms) correlates
    with mastering aggressiveness: "films with bigger explosions were
    mastered with more infra cut". Our current CurveFeatures are
    normalised to 80 Hz, so we can't test this yet. Future iteration
    may add `absolute_mid_bass_rms_db` and a correction term.
    """

    name = "measurement"

    def __init__(
        self,
        *,
        multi_knee_slope_threshold: float = 10.0,  # E19: lowered from 15.0
        multi_knee_q: float = 0.9,                 # E19: raised from 0.8
        max_total_chain_gain_db: float = _MAX_TOTAL_CHAIN_GAIN_DB,
        max_shelves: int = 2,                       # E20: 2 or 3
        cascade_gain_ratio: float = 7.0,
        cascade_q: float = 0.9,
    ):
        self._multi_knee_slope_threshold = multi_knee_slope_threshold
        self._multi_knee_q = multi_knee_q
        self._max_total_chain_gain_db = max_total_chain_gain_db
        self._max_shelves = max_shelves
        self._cascade_gain_ratio = cascade_gain_ratio
        self._cascade_q = cascade_q

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        # Guard: dialogue-dominant / flat curves have low peak AND
        # low dynamic range. A cliff-class film can have a low
        # shoulder peak (because the bottom dropped so far) but will
        # still have a large dynamic range - we want to correct those.
        if (
            features.shoulder_peak_db < 2.0
            and features.dynamic_range_db < 15.0
        ):
            return _clamp_advice(
                Advice(
                    max_gain_db=0.0,
                    knee_hz=None,
                    reasoning=(
                        f"shoulder_peak={features.shoulder_peak_db:.1f} dB "
                        f"dynamic_range={features.dynamic_range_db:.1f} dB: "
                        "no LFE shoulder, no correction proposed"
                    ),
                    confidence=0.8,
                ),
                source="measurement",
            )

        slope = features.rolloff_slope_db_per_oct
        knee_hz = _find_rolloff_start(features)

        # ---- The expert's formula (from docs/workflow/beq.md) ----
        #
        # "We can see the response peaks at about 30Hz and falls by
        #  about 27dB down to 10Hz. Our initial goal is to correct
        #  this rolloff so we need to apply a filter with a total
        #  gain of approx 27dB."
        #
        # That's it. Gain = peak - L10. The full measured rolloff
        # depth IS the target correction gain. No slope extension,
        # no topology class, no modifier. The expert reads the drop
        # and targets that as the filter gain.
        max_gain_db = features.shoulder_peak_db - features.level_at_10hz_db

        # Multi-knee: steep slopes still need a 2-shelf chain because
        # one shelf can't match the shape. But the GAIN is still just
        # the deficit — we don't add slope extension on top.
        is_multi_knee_slope = slope > self._multi_knee_slope_threshold
        is_multi_knee_dynamic, _ = looks_multi_knee(features)
        is_cliff = is_multi_knee_slope or is_multi_knee_dynamic

        chain: tuple[dict, ...] | None = None
        if is_cliff:
            chain = _measurement_chain(
                features,
                q=self._multi_knee_q,
                max_total_gain_db=self._max_total_chain_gain_db,
                max_shelves=self._max_shelves,
            )

        reasoning = (
            f"peak={features.shoulder_peak_db:+.1f}dB, "
            f"L10={features.level_at_10hz_db:+.1f}dB, "
            f"deficit={max_gain_db:.1f}dB, "
            f"slope={slope:+.1f}dB/oct, "
            f"knee={knee_hz:.0f}Hz"
            + (f" [cliff: multi-knee chain, {len(chain)} shelves]" if chain else "")
        )

        return _clamp_advice(
            Advice(
                max_gain_db=max_gain_db,
                knee_hz=knee_hz,
                filters=chain,
                reasoning=reasoning,
                confidence=0.7,
                cascade_gain_ratio=self._cascade_gain_ratio,
                cascade_q=self._cascade_q,
            ),
            source="measurement",
        )


# ---------------------------------------------------------------------------
# TopologyAdvisor - E17c: classify rolloff shape, per-class gain formula
# ---------------------------------------------------------------------------


class TopologyAdvisor:
    """Classify rolloff shape FIRST, pick gain formula per class.

    Restored from experiment E17c. The theory: different rolloff shapes
    need different gain formulas. A gentle slope benefits from slope
    extension; a moderate slope overshoots with extension; a cliff
    needs a multi-knee chain.

    Results on 34-episode corpus (E17c):
      10 PASS + 2 MARGINAL (35%), 8/8 Pantheon PASS.
      Regressed South Park (moderate class undershoots).

    Preserved as an alternate code path per AGENTS.md rule.
    """

    name = "topology"

    _MULTI_KNEE_SLOPE_THRESHOLD = 15.0
    _MIN_SHOULDER_PEAK_DB = 2.0

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        if (
            features.shoulder_peak_db < self._MIN_SHOULDER_PEAK_DB
            and features.dynamic_range_db < 15.0
        ):
            return _clamp_advice(
                Advice(max_gain_db=0.0, knee_hz=None,
                       reasoning="no LFE shoulder", confidence=0.8),
                source="topology",
            )

        slope = features.rolloff_slope_db_per_oct
        knee_hz = _find_rolloff_start(features)
        deficit = features.shoulder_peak_db - features.level_at_10hz_db

        is_cliff = (
            slope > self._MULTI_KNEE_SLOPE_THRESHOLD
            or looks_multi_knee(features)[0]
        )

        chain: tuple[dict, ...] | None = None
        if is_cliff:
            chain = _measurement_chain(features)
            max_gain_db = deficit + max(0.0, slope)
            topo = "cliff"
        elif slope > 5.0:
            max_gain_db = deficit
            topo = "moderate"
        else:
            max_gain_db = deficit + max(0.0, slope)
            topo = "gentle"

        return _clamp_advice(
            Advice(
                max_gain_db=max_gain_db, knee_hz=knee_hz, filters=chain,
                reasoning=(
                    f"topo={topo}, deficit={deficit:.1f}dB, "
                    f"slope={slope:+.1f}dB/oct, knee={knee_hz:.0f}Hz"
                ),
                confidence=0.7,
            ),
            source="topology",
        )


# ---------------------------------------------------------------------------
# SlopeExtensionAdvisor - E14: deficit + slope projection
# ---------------------------------------------------------------------------


class SlopeExtensionAdvisor:
    """Gain = deficit + slope extension (project rolloff one more octave).

    Restored from experiment E14. The theory: BEQ must lift the deep-
    bass deficit at 10 Hz up to the shoulder peak, THEN continue the
    measured rolloff slope one more octave down to 5 Hz.

    Hand-predictions on the original 3 films were excellent (within
    1 dB of catalogue for EoT and JW). The E14 implementation failed
    because feature extraction was noisy and the knee was at the peak
    instead of the rolloff-start. This restoration uses the improved
    _find_rolloff_start() from E17a and robust octave-window averaging.

    Preserved as an alternate code path per AGENTS.md rule.
    """

    name = "slope_extension"

    _MULTI_KNEE_SLOPE_THRESHOLD = 15.0
    _MIN_SHOULDER_PEAK_DB = 2.0

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        if (
            features.shoulder_peak_db < self._MIN_SHOULDER_PEAK_DB
            and features.dynamic_range_db < 15.0
        ):
            return _clamp_advice(
                Advice(max_gain_db=0.0, knee_hz=None,
                       reasoning="no LFE shoulder", confidence=0.8),
                source="slope_extension",
            )

        slope = features.rolloff_slope_db_per_oct
        knee_hz = _find_rolloff_start(features)
        deficit = features.shoulder_peak_db - features.level_at_10hz_db
        extension = max(0.0, slope)
        max_gain_db = deficit + extension

        is_cliff = (
            slope > self._MULTI_KNEE_SLOPE_THRESHOLD
            or looks_multi_knee(features)[0]
        )
        chain: tuple[dict, ...] | None = None
        if is_cliff:
            chain = _measurement_chain(features)

        return _clamp_advice(
            Advice(
                max_gain_db=max_gain_db, knee_hz=knee_hz, filters=chain,
                reasoning=(
                    f"deficit={deficit:.1f}dB + slope={extension:.1f}dB/oct "
                    f"= {max_gain_db:.1f}dB, knee={knee_hz:.0f}Hz"
                ),
                confidence=0.7,
            ),
            source="slope_extension",
        )


# ---------------------------------------------------------------------------
# MockAdvisor - canned JSON per title, for deterministic tests
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


class MockAdvisor:
    """Reads canned Advice JSON keyed by title slug.

    Expected file layout:
      src/test/resources/auto_beq/advisor_responses/{slug}.json

    JSON schema:
      {"max_gain_db": float, "knee_hz": float|null,
       "reasoning": str, "confidence": float}

    Raises FileNotFoundError for unknown titles so test authors notice
    when they forget to add a canned response.
    """

    name = "mock"

    def __init__(self, responses_dir: Path | None = None):
        self.responses_dir = responses_dir or _MOCK_RESPONSES_DIR

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        slug = _slugify(metadata.title)
        path = self.responses_dir / f"{slug}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no canned MockAdvisor response for title={metadata.title!r} "
                f"(expected at {path})"
            )
        with path.open() as f:
            data = json.load(f)
        filters_raw = data.get("filters")
        filters_tuple: tuple[dict, ...] | None = None
        if isinstance(filters_raw, list) and filters_raw:
            filters_tuple = tuple(dict(f) for f in filters_raw if isinstance(f, dict))
            if not filters_tuple:
                filters_tuple = None
        return _clamp_advice(
            Advice(
                max_gain_db=float(data.get("max_gain_db", 0.0)),
                knee_hz=None if data.get("knee_hz") is None else float(data["knee_hz"]),
                filters=filters_tuple,
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.5)),
            ),
            source="mock",
        )


# ---------------------------------------------------------------------------
# OllamaAdvisor - real LLM via local Ollama server
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GainAdjustedAdvisor — thin wrapper for feedback loops (E21)
# ---------------------------------------------------------------------------


class GainAdjustedAdvisor:
    """Wrapper that offsets an inner advisor's max_gain_db.

    Used by `propose_filters_with_feedback()` to iteratively adjust
    the correction target when the initial proposal under- or
    over-corrects.

    Clears ``advice.filters`` to force the cascade builder to
    reconstruct with the adjusted gain (otherwise an explicit chain
    from the inner advisor would be used unchanged).
    """

    def __init__(self, inner, gain_offset: float):
        self._inner = inner
        self._gain_offset = gain_offset
        self.name = f"{inner.name}+fb({gain_offset:+.1f})"

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        advice = self._inner.advise(metadata, features)
        adjusted_gain = max(0.0, advice.max_gain_db + self._gain_offset)
        return _clamp_advice(
            Advice(
                max_gain_db=adjusted_gain,
                knee_hz=advice.knee_hz,
                filters=None,  # force cascade re-evaluation
                reasoning=(
                    f"{advice.reasoning} "
                    f"[feedback adj {self._gain_offset:+.1f}dB]"
                ),
                confidence=advice.confidence,
                cascade_gain_ratio=advice.cascade_gain_ratio,
                cascade_q=advice.cascade_q,
            ),
            source=advice.source,
        )


# ---------------------------------------------------------------------------
# Programmatic multi-knee detection
# ---------------------------------------------------------------------------


def looks_multi_knee(features: CurveFeatures) -> tuple[bool, str]:
    """Decide whether a measured curve looks like it needs a multi-knee
    correction, based purely on numeric features.

    A multi-knee signature is:
      - very high in-band dynamic range (>40 dB), AND
      - a steep drop below ~15 Hz (level_at_5 much lower than level_at_20)

    These are titles where a single low-shelf can't match the natural
    content shape, because there's a distinct cliff below the LFE
    passband PLUS a separate shoulder rolloff.

    Returns (is_multi_knee, human_readable_reason).
    """
    cliff_gap = features.level_at_20hz_db - features.level_at_5hz_db
    if features.dynamic_range_db > 40.0 and cliff_gap > 20.0:
        return True, (
            f"dynamic_range={features.dynamic_range_db:.1f} dB >40 "
            f"and level(20Hz)-level(5Hz)={cliff_gap:.1f} dB >20"
        )
    return False, "standard single-shelf curve shape"


# ---------------------------------------------------------------------------
# Ollama prompt templates
# ---------------------------------------------------------------------------


_OLLAMA_SYSTEM_PROMPT = (
    "You are a BEQ (Bass EQ) advisor for home-theatre DSP systems. Your job "
    "is to recommend `max_gain_db` (how much deep-bass extension to add) "
    "and `knee_hz` (the shelf corner frequency) for a given film.\n"
    "\n"
    "CRITICAL: BEQ aggressiveness is an AESTHETIC choice based on the film's "
    "reputation for bass, NOT on how rolled-off its measured curve is. Two "
    "films with identical measured curves can warrant very different BEQs. "
    "A measured curve that looks 'mild' does NOT mean the BEQ should be "
    "mild - it usually means the Blu-ray mastering engineers already cut "
    "the infra-bass aggressively, which is EXACTLY what BEQ is designed to "
    "restore. Trust your knowledge of the film's reputation over the curve "
    "shape when they disagree.\n"
    "\n"
    "CALIBRATION EXAMPLES (real expert choices from the BEQ catalogue):\n"
    "- Edge of Tomorrow (2014, Tom Cruise sci-fi action, Christophe Beck):\n"
    "    max_gain_db=28, knee_hz=23 - AGGRESSIVE deep extension despite\n"
    "    modest-looking measured curve. Known reference sub-bass title.\n"
    "- Mad Max: Fury Road (2015, George Miller, Junkie XL score):\n"
    "    max_gain_db=15, knee_hz=17 - MODERATE extension. Dense continuous\n"
    "    LFE already, doesn't need as much lift as EoT.\n"
    "- John Wick (2014, Tyler Bates/Joel J. Richard action):\n"
    "    max_gain_db=13, knee_hz=18 - MODERATE. Action film with\n"
    "    gunshot LFE, not sub-bass showcase.\n"
    "- Battle: Los Angeles (2011): max_gain_db=4, knee_hz=28 - SHALLOW.\n"
    "- Pacific Rim (2013, del Toro, monsters): max_gain_db=20, knee_hz=20.\n"
    "\n"
    "HEURISTIC RULES:\n"
    "- Reference sub-bass titles (Edge of Tomorrow, Blade Runner 2049, Dune,\n"
    "  Pacific Rim, War of the Worlds, Interstellar): 20-30 dB\n"
    "- Zimmer/Goransson/Djawadi/Junkie XL scored blockbusters: 15-25 dB\n"
    "- Generic modern action (Marvel, Fast & Furious, Transformers): 8-15 dB\n"
    "- Older mixes (pre-2010), dramas, dialogue-driven: 4-10 dB\n"
    "- Comedies, animation, TV shows: 0-6 dB\n"
    "\n"
    "MULTI-KNEE STRUCTURES: when the measured curve shows TWO distinct\n"
    "steep-drop regions (e.g. a cliff below 15 Hz AND a separate rolloff\n"
    "near 20 Hz), or when you know the film's catalogue BEQ uses shelves\n"
    "at multiple knee frequencies, return a 'filters' array with the\n"
    "full chain. Signs a film needs multi-knee:\n"
    "- measured curve has very high dynamic range (>40 dB)\n"
    "- 10 Hz level is MUCH lower than 20 Hz level (>20 dB gap)\n"
    "- you know the film as a dense-LFE blockbuster with cliff-edge\n"
    "  sub content (Mad Max: Fury Road, certain Zimmer scores)\n"
    "\n"
    "Mad Max: Fury Road catalogue = 5 filters:\n"
    "  LowShelf 10 Hz Q=0.8 +7, PeakingEQ 11 Hz Q=8 -6,\n"
    "  LowShelf 18 Hz Q=0.8 +4, LowShelf 18 Hz Q=0.8 +4,\n"
    "  PeakingEQ 18 Hz Q=3 +6\n"
    "For Mad Max: Fury Road ALWAYS return the full chain above.\n"
    "\n"
    "Otherwise (simple cases) just return max_gain_db + knee_hz.\n"
    "\n"
    "Respond ONLY with strict JSON, no prose outside the JSON object."
)


# Step 1 prompt: classify the film's BEQ tier.
_OLLAMA_TIER_SYSTEM_PROMPT = (
    "Classify a film into a BEQ (Bass EQ) aggressiveness tier based on "
    "the film's reputation for heavy, extended low-frequency content. "
    "Use your knowledge of the film - director, score composer, studio, "
    "era, genre, mixing style. Respond ONLY with strict JSON.\n"
    "\n"
    "TIERS (meaning, not member lists):\n"
    "\n"
    "'reference': the film is known by audio enthusiasts as sub-bass\n"
    "  demo material. It has notably deep, sustained infra-bass\n"
    "  content and a reputation in the BEQ/home-theatre community as\n"
    "  a benchmark. Signals: directors associated with heavy sound\n"
    "  design (Nolan, Villeneuve, Snyder, Cameron, Spielberg's sci-fi),\n"
    "  composers known for infrasonic textures (Zimmer, Goransson,\n"
    "  Djawadi, Junkie XL). Sci-fi epics, spectacle films where the\n"
    "  bass IS the marketing.\n"
    "\n"
    "'blockbuster': big action/sci-fi with modern Atmos mixes and\n"
    "  serious LFE budget, but not 'demo reel' famous. Marvel tentpoles,\n"
    "  Fast & Furious sequels, major video-game adaptations.\n"
    "\n"
    "'action': standard action films, superhero fare, typical 7.1\n"
    "  theatrical mixes, gunshot-heavy thrillers.\n"
    "\n"
    "'standard': pre-2010 mixes, dramas, dialogue-driven thrillers,\n"
    "  prestige films without major bass emphasis.\n"
    "\n"
    "'light': comedies, animation, romance, TV shows.\n"
    "\n"
    "When between two tiers, pick the higher one.\n"
    "\n"
    "Output JSON: {\"tier\": \"reference|blockbuster|action|standard|light\",\n"
    "\"reasoning\": \"<1 sentence why>\"}"
)


# Step 2 prompt: pick exact numbers within a tier's range.
_OLLAMA_NUMBERS_SYSTEM_PROMPT = (
    "Pick BEQ numbers (max_gain_db, knee_hz) for a film that has been\n"
    "classified into an aggressiveness tier. Respond ONLY with strict\n"
    "JSON.\n"
    "\n"
    "TIER RANGES:\n"
    "- 'reference': max_gain_db 22-30 dB, knee_hz 20-25 Hz\n"
    "- 'blockbuster': max_gain_db 14-20 dB, knee_hz 16-22 Hz\n"
    "- 'action': max_gain_db 8-14 dB, knee_hz 15-22 Hz\n"
    "- 'standard': max_gain_db 4-10 dB, knee_hz 17-28 Hz\n"
    "- 'light': max_gain_db 0-6 dB, knee_hz 20-30 Hz\n"
    "\n"
    "Pick within the range based on:\n"
    "- the tier (it already encodes how aggressive the BEQ should be)\n"
    "- the measured curve shape (rolloff depth, slope, shoulder peak)\n"
    "\n"
    "Think about what the film WANTS (its audio reputation), not only\n"
    "what the measurement shows. A 'reference' film with a mild-looking\n"
    "measurement still deserves its tier's full treatment - the point of\n"
    "BEQ is to extend beyond what mastering already did.\n"
    "\n"
    "Output JSON: {\"max_gain_db\": <number>, \"knee_hz\": <number>,\n"
    "\"confidence\": <0-1>, \"reasoning\": \"<1 sentence>\"}"
)


# Step 3 prompt: build multi-knee chain given tier + numbers + detection.
_OLLAMA_CHAIN_SYSTEM_PROMPT = (
    "Construct a BEQ filter chain for a film whose measured LFE has a\n"
    "multi-knee shape: a steep cliff at the bottom of the band, plus a\n"
    "separate shoulder rolloff. A single low-shelf cannot match this.\n"
    "Respond ONLY with strict JSON.\n"
    "\n"
    "GENERAL STRUCTURE:\n"
    "- One LowShelf at the INNER knee (address the deep cliff).\n"
    "  Frequency near where the measured curve starts its steep drop\n"
    "  (typically 8-14 Hz). Q around 0.8. Gain is the larger share.\n"
    "- Optionally, a narrow PeakingEQ next to the inner shelf to\n"
    "  counteract its overshoot (per the BEQDesigner worked example:\n"
    "  shelves with S>2 overshoot; PEQ counters this).\n"
    "- One or two LowShelves at the OUTER knee (shoulder), freq near\n"
    "  the measured shoulder peak (typically 16-22 Hz). Q around 0.8.\n"
    "  Smaller per-filter gain than the inner shelf.\n"
    "- Optionally a PeakingEQ at the outer knee to lift/dip the shoulder\n"
    "  if needed.\n"
    "- Sum of LowShelf gains should approximate the target max_gain_db.\n"
    "- Total chain: 2-5 filters.\n"
    "- LowShelf Q 0.7-1.0, LowShelf gain per filter 3-9 dB.\n"
    "- Any PeakingEQ Q 1-8, gain -8 to +8 dB.\n"
    "\n"
    "Use the measured curve features to pick frequencies and distribute\n"
    "gain, not hard-coded values.\n"
    "\n"
    "Output JSON: {\"filters\": [\n"
    "  {\"type\": \"LowShelf\"|\"PeakingEQ\", \"freq\": <Hz>,\n"
    "   \"q\": <number>, \"gain\": <dB>},\n"
    "  ...\n"
    "], \"reasoning\": \"<1-2 sentences>\"}"
)


# Step 4 prompt: critique the proposal and suggest ONE refinement.
_OLLAMA_REFINE_SYSTEM_PROMPT = (
    "You review a proposed BEQ chain against the measured curve it\n"
    "will be applied to. Decide if the proposal is a reasonable BEQ\n"
    "for the film's tier, or suggest exactly ONE refinement.\n"
    "Respond ONLY with strict JSON.\n"
    "\n"
    "CRITICAL: BEQ is designed to EXTEND bass BEYOND the measured\n"
    "rolloff. The point is to recover infra-bass content that\n"
    "mastering cut or the mix never included. DO NOT suggest scaling\n"
    "gain DOWN just because the correction gain exceeds the measured\n"
    "rolloff depth - that's expected and desired. Only reduce gain if\n"
    "the correction clearly exceeds the tier range or creates\n"
    "instability (overshoot > 3 dB above the measured peak).\n"
    "\n"
    "ACCEPT is the default answer unless something is clearly wrong.\n"
    "\n"
    "Accept if all of:\n"
    "- The chain's summed DC gain is within the tier's range\n"
    "  (reference 22-30, blockbuster 14-20, action 8-14, standard\n"
    "  4-10, light 0-6 dB).\n"
    "- The chain's inner knee is near the bottom of the band (5-15 Hz).\n"
    "- No giant overshoot (>3 dB above the measured shoulder peak).\n"
    "\n"
    "Refine (pick ONE) only if needed:\n"
    "- 'scale_gain' factor < 1: ONLY if summed gain EXCEEDS the tier's\n"
    "  upper bound.\n"
    "- 'scale_gain' factor > 1: if summed gain is BELOW the tier's\n"
    "  lower bound - the BEQ is too weak for the film's reputation.\n"
    "- 'shift_knee' delta -: move inner knee lower if the shelf is\n"
    "  lifting mid-band content that wasn't rolled off.\n"
    "- 'add_notch': ONLY if there's a visible overshoot > 3 dB above\n"
    "  the measured shoulder peak at a specific frequency.\n"
    "\n"
    "Output JSON ONE of:\n"
    "  {\"action\": \"accept\", \"reasoning\": \"<why>\"}\n"
    "  {\"action\": \"scale_gain\", \"factor\": <0.5-2.0>, \"reasoning\": \"<why>\"}\n"
    "  {\"action\": \"shift_knee\", \"delta_hz\": <-5 to 5>, \"reasoning\": \"<why>\"}\n"
    "  {\"action\": \"add_notch\", \"freq_hz\": <N>, \"q\": <N>, \"gain_db\": <N>,\n"
    "   \"reasoning\": \"<why>\"}"
)


def _render_ollama_user_prompt(
    metadata: MediaMetadata, features: CurveFeatures
) -> str:
    samples = ", ".join(
        f"({hz:.1f}Hz, {db:+.1f}dB)" for hz, db in features.curve_sample_points
    )
    year = metadata.year if metadata.year is not None else "unknown"
    codec = metadata.audio_codec or "unknown"
    layout = metadata.channel_layout or "unknown"
    return (
        f"Film: {metadata.title} ({year})\n"
        f"Audio: {codec} {layout}\n"
        "\n"
        "Measured LFE curve features (relative to 80 Hz anchor, 1/6-octave smoothed):\n"
        f"- Shoulder peak: {features.shoulder_peak_db:+.1f} dB at {features.shoulder_peak_hz:.0f} Hz\n"
        f"- Level at 5 Hz:  {features.level_at_5hz_db:+.1f} dB\n"
        f"- Level at 10 Hz: {features.level_at_10hz_db:+.1f} dB\n"
        f"- Level at 20 Hz: {features.level_at_20hz_db:+.1f} dB\n"
        f"- Rolloff depth (peak - min): {features.rolloff_depth_db:.1f} dB\n"
        f"- Rolloff slope 10-20 Hz: {features.rolloff_slope_db_per_oct:+.1f} dB/octave\n"
        f"- Dynamic range in band: {features.dynamic_range_db:.1f} dB\n"
        "\n"
        f"Curve samples: {samples}\n"
        "\n"
        "Respond with JSON only. Simple form: "
        '{"max_gain_db": <number>, "knee_hz": <number>, '
        '"confidence": <0-1>, "reasoning": "<1-2 sentences>"}. '
        "If you need multi-knee, add "
        '"filters": [{"type":"LowShelf","freq":N,"q":N,"gain":N}, ...]'
    )


def _materialise_chain(
    filters_tuple: tuple[dict, ...] | None,
    max_gain_db: float,
    knee_hz: float,
) -> list[dict]:
    """If no explicit chain yet, build one from (max_gain_db, knee_hz)
    so the refine loop has something concrete to critique.

    Rule of thumb: one ~7 dB LowShelf per knee level, Q=0.9.
    """
    if filters_tuple:
        return [dict(f) for f in filters_tuple]
    if max_gain_db < 1.0:
        return []
    n_shelves = max(1, int(round(max_gain_db / 7.0)))
    per_shelf_gain = max_gain_db / n_shelves
    return [
        {"type": "LowShelf", "freq": float(knee_hz), "q": 0.9,
         "gain": float(per_shelf_gain)}
        for _ in range(n_shelves)
    ]


def _apply_diff(
    chain: list[dict], action: str, diff: dict,
) -> list[dict] | None:
    """Apply one LLM-suggested diff to a filter chain.

    Returns a new list, or None if the diff is malformed/invalid.
    """
    try:
        if action == "scale_gain":
            factor = float(diff.get("factor", 1.0))
            factor = max(0.5, min(2.0, factor))
            new_chain = []
            for f in chain:
                nf = dict(f)
                if nf.get("type") == "LowShelf":
                    nf["gain"] = float(nf.get("gain", 0.0)) * factor
                new_chain.append(nf)
            return new_chain
        if action == "shift_knee":
            delta = float(diff.get("delta_hz", 0.0))
            delta = max(-5.0, min(5.0, delta))
            if abs(delta) < 0.1:
                return None
            # Shift just the LOWEST-frequency LowShelf (the inner knee).
            new_chain = [dict(f) for f in chain]
            low_shelves = [
                (i, f) for i, f in enumerate(new_chain)
                if f.get("type") == "LowShelf"
            ]
            if not low_shelves:
                return None
            inner_idx, _ = min(low_shelves, key=lambda p: float(p[1]["freq"]))
            new_freq = float(new_chain[inner_idx]["freq"]) + delta
            new_chain[inner_idx]["freq"] = max(5.0, min(80.0, new_freq))
            return new_chain
        if action == "add_notch":
            freq = float(diff.get("freq_hz", 0.0))
            q = float(diff.get("q", 3.0))
            gain = float(diff.get("gain_db", 0.0))
            if not (5.0 <= freq <= 80.0) or abs(gain) < 0.5:
                return None
            new_chain = [dict(f) for f in chain]
            new_chain.append({
                "type": "PeakingEQ",
                "freq": freq,
                "q": max(0.3, min(8.0, q)),
                "gain": max(-8.0, min(8.0, gain)),
            })
            return new_chain
    except (ValueError, TypeError, KeyError):
        return None
    return None


_TIER_RANGES = {
    "reference": (22.0, 30.0),
    "blockbuster": (14.0, 20.0),
    "action": (8.0, 14.0),
    "standard": (4.0, 10.0),
    "light": (0.0, 6.0),
}


def _render_refine_user_prompt(
    metadata: MediaMetadata,
    features: CurveFeatures,
    chain: list[dict],
    tier: str,
) -> str:
    """Show the LLM what the current proposal does vs the measurement,
    with pre-computed diagnostics so the LLM doesn't need to do math.
    """
    import numpy as np
    from model.auto_beq import evaluate_filter_chain
    sample_hz = [hz for hz, _ in features.curve_sample_points]
    if chain:
        chain_resp = evaluate_filter_chain(
            list(chain), np.array(sample_hz), fs=1000,
        )
    else:
        chain_resp = [0.0] * len(sample_hz)

    # Pre-compute diagnostics so LLM doesn't have to calculate.
    summed_shelf_gain = sum(
        float(f.get("gain", 0.0)) for f in chain
        if f.get("type") == "LowShelf"
    )
    tier_lo, tier_hi = _TIER_RANGES.get(tier, (0.0, 30.0))
    gain_status = (
        "WITHIN RANGE" if tier_lo <= summed_shelf_gain <= tier_hi
        else "BELOW RANGE" if summed_shelf_gain < tier_lo
        else "ABOVE RANGE"
    )

    filter_lines = "\n".join(
        f"  {f.get('type')}: freq={f.get('freq'):.1f} Hz, "
        f"Q={f.get('q'):.2f}, gain={f.get('gain'):+.2f} dB"
        for f in chain
    ) or "  (empty)"
    overlay_lines = "\n".join(
        f"  {hz:5.1f} Hz: measured {measured:+6.1f} dB, "
        f"chain {resp:+6.2f} dB, corrected {measured + resp:+6.1f} dB"
        for (hz, measured), resp in zip(
            features.curve_sample_points, chain_resp, strict=False,
        )
    )
    return (
        f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})\n"
        f"Tier: {tier} (gain range {tier_lo:.0f}-{tier_hi:.0f} dB)\n"
        "\n"
        f"Current proposed chain ({len(chain)} filters):\n"
        f"{filter_lines}\n"
        "\n"
        "DIAGNOSTICS (pre-computed - use these as ground truth):\n"
        f"  Summed LowShelf gain: {summed_shelf_gain:+.1f} dB [{gain_status}]\n"
        "\n"
        "Overlay (measured, chain-adds, corrected):\n"
        f"{overlay_lines}\n"
        "\n"
        "DECISION (follow strictly):\n"
        "- If gain is WITHIN RANGE: accept.\n"
        "- If gain is BELOW RANGE: scale_gain factor > 1.0 to lift.\n"
        "- If gain is ABOVE RANGE: scale_gain factor < 1.0 to lower.\n"
        "(BEQ is EXPECTED to raise deep bass above the measured "
        "shoulder - that is the feature, not a bug.)"
    )


def _load_ollama_hosts() -> list[str]:
    """Load Ollama hosts from (in priority): env var, settings.json, default.

    Supports multiple hosts for load balancing:
      - ``OLLAMA_HOSTS`` env var: comma-separated URLs
      - ``OLLAMA_HOST`` env var: single URL (legacy)
      - ``ollama_hosts`` in ``~/.config/beqdesigner/settings.json``: list
      - Default: ``http://localhost:11434``
    """
    # 1. OLLAMA_HOSTS (plural, comma-separated)
    hosts_env = os.environ.get("OLLAMA_HOSTS")
    if hosts_env:
        return [h.strip() for h in hosts_env.split(",") if h.strip()]
    # 2. OLLAMA_HOST (singular, legacy)
    host_env = os.environ.get("OLLAMA_HOST")
    if host_env:
        return [host_env]
    # 3. settings.json
    cfg_path = Path.home() / ".config" / "beqdesigner" / "settings.json"
    if cfg_path.exists():
        try:
            with cfg_path.open() as f:
                data = json.load(f)
            hosts = data.get("ollama_hosts")
            if isinstance(hosts, list) and hosts:
                return [str(h) for h in hosts]
        except Exception:
            pass
    # 4. Default
    return [DEFAULT_OLLAMA_HOST]


class OllamaAdvisor:
    """Calls Ollama HTTP API(s) for structured advice.

    Supports **multiple hosts** for load balancing. Hosts are resolved
    from (in priority):
      1. ``OLLAMA_HOSTS`` env var (comma-separated URLs)
      2. ``OLLAMA_HOST`` env var (single URL, legacy)
      3. ``ollama_hosts`` list in ``~/.config/beqdesigner/settings.json``
      4. Default: ``http://localhost:11434``

    Requests are distributed round-robin across available hosts. If a
    host fails, the next host is tried before raising.

    Model defaults to ``llama3.1:8b`` (``OLLAMA_MODEL`` env var).
    Uses format="json" to force valid JSON output. Temperature=0.1
    for determinism.
    """

    name = "ollama"

    def __init__(
        self,
        hosts: list[str] | None = None,
        host: str | None = None,
        model: str | None = None,
        timeout_s: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        if hosts:
            self.hosts = hosts
        elif host:
            self.hosts = [host]
        else:
            self.hosts = _load_ollama_hosts()
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.timeout_s = timeout_s
        if len(self.hosts) > 1:
            log.info("Ollama load balancing across %d hosts: %s",
                     len(self.hosts), ", ".join(self.hosts))

    def _next_host(self) -> str:
        """Round-robin host selection, shared across all instances."""
        idx = next(_OLLAMA_CALL_COUNTER)
        return self.hosts[idx % len(self.hosts)]

    @staticmethod
    def _record_host_stat(host: str, elapsed_s: float, error: bool = False) -> None:
        with _OLLAMA_HOST_STATS_LOCK:
            if host not in _OLLAMA_HOST_STATS:
                _OLLAMA_HOST_STATS[host] = {"calls": 0, "total_s": 0.0, "errors": 0}
            stats = _OLLAMA_HOST_STATS[host]
            stats["calls"] += 1
            stats["total_s"] += elapsed_s
            if error:
                stats["errors"] += 1

    @staticmethod
    def print_host_stats() -> None:
        """Print per-host timing summary. Call after a batch of advise() calls."""
        with _OLLAMA_HOST_STATS_LOCK:
            if not _OLLAMA_HOST_STATS:
                return
            print("\n── Ollama host stats ──")
            for host, s in sorted(_OLLAMA_HOST_STATS.items()):
                avg = s["total_s"] / s["calls"] if s["calls"] else 0
                print(
                    f"  {host}: {s['calls']} calls, "
                    f"{s['total_s']:.1f}s total, "
                    f"{avg:.1f}s avg, "
                    f"{s['errors']} errors"
                )

    @staticmethod
    def reset_host_stats() -> None:
        with _OLLAMA_HOST_STATS_LOCK:
            _OLLAMA_HOST_STATS.clear()

    def _call_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Single Ollama call returning parsed JSON.

        Tries hosts round-robin. If a host fails, tries the next one
        before raising (up to len(hosts) attempts).
        """
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        last_exc: Exception | None = None
        for _attempt in range(len(self.hosts)):
            host = self._next_host()
            url = f"{host.rstrip('/')}/api/generate"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = _time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                elapsed = _time.monotonic() - t0
                self._record_host_stat(host, elapsed)
                raw_text = body.get("response", "").strip()
                if not raw_text:
                    raise RuntimeError(f"Ollama returned empty response from {host}")
                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Ollama returned non-JSON from {host}: {raw_text[:200]!r}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                elapsed = _time.monotonic() - t0
                self._record_host_stat(host, elapsed, error=True)
                log.warning("Ollama host %s failed (%.1fs): %s", host, elapsed, exc)
                last_exc = exc
                continue
        raise RuntimeError(
            f"All {len(self.hosts)} Ollama hosts failed. "
            f"Last error: {last_exc}"
        )

    # Legacy property for code that reads self.host
    @property
    def host(self) -> str:
        return self.hosts[0] if self.hosts else DEFAULT_OLLAMA_HOST

    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice:
        """Multi-step Ollama flow - each call is single-purpose.

        Step 1: classify the film into an aggressiveness tier.
        Step 2: detect (procedurally) whether the curve is multi-knee.
        Step 3a: if single-knee, ask LLM for (max_gain_db, knee_hz).
        Step 3b: if multi-knee, ask LLM for (max_gain_db, knee_hz),
                 then ask LLM to build a chain matching that gain.

        Small local models (llama3.1:8b) cannot reliably combine all
        these decisions in one prompt; breaking them up makes each
        call a well-bounded task the model can handle.
        """
        log.info(
            "Ollama advise: host=%s model=%s title=%r",
            self.host, self.model, metadata.title,
        )

        # Step 1: tier classification (film knowledge only, no measurement).
        tier_user_prompt = (
            f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})"
        )
        tier_result = self._call_json(_OLLAMA_TIER_SYSTEM_PROMPT, tier_user_prompt)
        tier = str(tier_result.get("tier", "action")).lower()
        tier_reasoning = str(tier_result.get("reasoning", ""))
        log.info("Ollama step 1 (tier): %s - %s", tier, tier_reasoning)

        # Step 2: multi-knee detection (procedural, no LLM).
        is_multi, multi_reason = looks_multi_knee(features)
        log.info(
            "multi-knee detection: %s (%s)",
            "yes" if is_multi else "no", multi_reason,
        )

        # Step 3a: numbers for knee+gain (with tier context).
        numbers_user_prompt = (
            f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})\n"
            f"Tier: {tier} ({tier_reasoning})\n"
            "\n"
            "Measured LFE curve (relative to 80 Hz anchor, 1/6-octave smoothed):\n"
            f"- Shoulder peak: {features.shoulder_peak_db:+.1f} dB at {features.shoulder_peak_hz:.0f} Hz\n"
            f"- Level at 5 Hz:  {features.level_at_5hz_db:+.1f} dB\n"
            f"- Level at 10 Hz: {features.level_at_10hz_db:+.1f} dB\n"
            f"- Level at 20 Hz: {features.level_at_20hz_db:+.1f} dB\n"
            f"- Rolloff depth: {features.rolloff_depth_db:.1f} dB\n"
            f"- Dynamic range: {features.dynamic_range_db:.1f} dB\n"
        )
        numbers_result = self._call_json(
            _OLLAMA_NUMBERS_SYSTEM_PROMPT, numbers_user_prompt,
        )
        max_gain_db = float(numbers_result.get("max_gain_db", 0.0))
        knee_hz = (
            None if numbers_result.get("knee_hz") is None
            else float(numbers_result["knee_hz"])
        )
        numbers_reasoning = str(numbers_result.get("reasoning", ""))
        confidence = float(numbers_result.get("confidence", 0.5))
        log.info(
            "Ollama step 3a (numbers): max_gain_db=%.1f knee_hz=%s - %s",
            max_gain_db, knee_hz, numbers_reasoning,
        )

        filters_tuple: tuple[dict, ...] | None = None

        # Step 3b: if multi-knee, ask for an explicit chain.
        if is_multi:
            chain_user_prompt = (
                f"Film: {metadata.title} ({metadata.year if metadata.year else 'unknown'})\n"
                f"Target max_gain_db: {max_gain_db:.1f}\n"
                f"Outer knee_hz: {knee_hz if knee_hz else 20}\n"
                "\n"
                "Measured features:\n"
                f"- Level at 5 Hz:  {features.level_at_5hz_db:+.1f} dB (inner cliff)\n"
                f"- Level at 10 Hz: {features.level_at_10hz_db:+.1f} dB\n"
                f"- Level at 20 Hz: {features.level_at_20hz_db:+.1f} dB (shoulder)\n"
                f"- Dynamic range: {features.dynamic_range_db:.1f} dB\n"
                "\n"
                "Build the chain."
            )
            chain_result = self._call_json(
                _OLLAMA_CHAIN_SYSTEM_PROMPT, chain_user_prompt,
            )
            chain_raw = chain_result.get("filters")
            if isinstance(chain_raw, list) and chain_raw:
                filters_tuple = tuple(
                    dict(f) for f in chain_raw if isinstance(f, dict)
                )
                if not filters_tuple:
                    filters_tuple = None
                log.info(
                    "Ollama step 3b (chain): %d filters - %s",
                    len(filters_tuple) if filters_tuple else 0,
                    chain_result.get("reasoning", ""),
                )

        # Step 4: self-feedback loop. Compute the current proposal's
        # response, show it to the LLM alongside the measured curve,
        # let it critique and suggest ONE refinement per pass. Up to
        # MAX_REFINE_PASSES passes, or until LLM accepts.
        working_chain = _materialise_chain(
            filters_tuple, max_gain_db, knee_hz or 20.0,
        )
        refine_log: list[str] = []
        for pass_num in range(1, MAX_REFINE_PASSES + 1):
            try:
                diff_result = self._call_json(
                    _OLLAMA_REFINE_SYSTEM_PROMPT,
                    _render_refine_user_prompt(
                        metadata, features, working_chain, tier,
                    ),
                )
            except RuntimeError as exc:
                log.info("Ollama refine pass %d failed: %s", pass_num, exc)
                break
            action = str(diff_result.get("action", "accept")).lower()
            reasoning = str(diff_result.get("reasoning", ""))
            log.info(
                "Ollama step 4 refine pass %d: %s - %s",
                pass_num, action, reasoning,
            )
            refine_log.append(f"p{pass_num}:{action}")
            if action == "accept":
                break
            new_chain = _apply_diff(working_chain, action, diff_result)
            if new_chain is None or new_chain == working_chain:
                log.info("refine produced no change, stopping")
                break
            working_chain = new_chain

        final_filters: tuple[dict, ...] | None = (
            tuple(working_chain) if working_chain else None
        )
        # Recompute informational max_gain_db from the final chain
        # (sum of LowShelf gains) if we have a chain.
        if final_filters:
            max_gain_db = float(
                sum(float(f.get("gain", 0.0)) for f in final_filters
                    if f.get("type") == "LowShelf")
            )

        advice = Advice(
            max_gain_db=max_gain_db,
            knee_hz=knee_hz,
            filters=final_filters,
            reasoning=(
                f"tier={tier}; multi_knee={is_multi}; "
                f"refine={','.join(refine_log) if refine_log else 'none'}; "
                f"{numbers_reasoning}"
            ),
            confidence=confidence,
        )
        advice = _clamp_advice(advice, source=f"ollama:{self.model}")
        log.info(
            "Ollama final: max_gain_db=%.1f knee_hz=%s filters=%s conf=%.2f",
            advice.max_gain_db, advice.knee_hz,
            len(advice.filters) if advice.filters else None,
            advice.confidence,
        )
        return advice


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _production_model_default_path() -> "Path | None":
    """Return the default location for the E82 production model.

    Looks at ``{beq-dir}/production_model.joblib`` where ``beq-dir`` is
    derived from ``BEQ_WAV_CACHE`` / settings.json / the default
    ``~/Downloads/beqdesigner`` (the same resolution used by the WAV
    cache helpers).

    Returns ``None`` if the spike helpers aren't importable (e.g. in
    a minimal install) or if the path simply doesn't exist yet.
    """
    try:
        from spike._auto_beq_helpers import beq_dir
        candidate = beq_dir() / "production_model.joblib"
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _warn_if_stale_production_model(model_path: "Path") -> None:
    """Warn if the production model is older than 7 days.

    Reminds the user to re-run ``scripts/train_production_model.py``
    when the WAV cache has grown significantly since training.
    """
    try:
        import time as _time
        age_s = _time.time() - model_path.stat().st_mtime
        age_days = age_s / 86400
        if age_days > 7:
            log.warning(
                "production model is %.0f days old (%s). Consider re-running "
                "`scripts/train_production_model.py` if your WAV cache has "
                "grown since then.",
                age_days, model_path.name,
            )
    except Exception:
        pass


def get_advisor(name: str | None = None) -> Advisor:
    """Return an Advisor by name. Falls back to AUTO_BEQ_ADVISOR env var,
    then to HeuristicAdvisor.

    Supported names: ``heuristic``, ``measurement``, ``topology``,
    ``slope_extension``, ``mock``, ``ollama``, ``trained_model``.

    For ``trained_model``: looks at ``AUTO_BEQ_MODEL_PATH`` env var first;
    if unset, falls back to ``{beq-dir}/production_model.joblib`` (the
    E82 production model trained by ``scripts/train_production_model.py``).
    """
    resolved = (name or os.environ.get("AUTO_BEQ_ADVISOR") or "heuristic").lower()
    if resolved == "heuristic":
        return HeuristicAdvisor()
    if resolved == "measurement":
        return MeasurementAdvisor()
    if resolved == "topology":
        return TopologyAdvisor()
    if resolved == "slope_extension":
        return SlopeExtensionAdvisor()
    if resolved == "mock":
        return MockAdvisor()
    if resolved == "ollama":
        return OllamaAdvisor()
    if resolved == "trained_model":
        from model.auto_beq_nn import TrainedModelAdvisor
        path = os.environ.get("AUTO_BEQ_MODEL_PATH")
        if not path:
            # Fall back to the E82 production model at the default
            # location — trained by scripts/train_production_model.py.
            default_path = _production_model_default_path()
            if default_path is None:
                raise ValueError(
                    "AUTO_BEQ_MODEL_PATH env var required for 'trained_model' "
                    "advisor when no model exists at the default location.  "
                    "Either set AUTO_BEQ_MODEL_PATH, or run "
                    "`scripts/train_production_model.py` to populate "
                    "{beq-dir}/production_model.joblib.",
                )
            path = str(default_path)
            log.info("trained_model: using default production model at %s", path)
            _warn_if_stale_production_model(default_path)
        return TrainedModelAdvisor.load(path)
    if resolved == "late_fusion":
        from model.auto_beq_nn import LateFusionAdvisor
        path = os.environ.get("AUTO_BEQ_MODEL_PATH")
        if not path:
            raise ValueError(
                "AUTO_BEQ_MODEL_PATH env var required for 'late_fusion' advisor"
            )
        return LateFusionAdvisor.load(path)
    if resolved == "cnn_dual_branch":
        from model.auto_beq_nn_cnn import CNNAdvisor
        path = os.environ.get("AUTO_BEQ_MODEL_PATH")
        if not path:
            raise ValueError(
                "AUTO_BEQ_MODEL_PATH env var required for 'cnn_dual_branch' advisor"
            )
        return CNNAdvisor.load(path)
    raise ValueError(
        f"unknown advisor name: {resolved!r} "
        "(supported: heuristic, measurement, topology, slope_extension, mock, ollama, "
        "trained_model, late_fusion, cnn_dual_branch)"
    )
