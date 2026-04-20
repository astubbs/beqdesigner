"""Automated BEQ filter proposal.

Given a target magnitude curve (the rolloff an expert would correct with a BEQ),
proposes a small IIR filter chain whose response negates the curve across a
chosen band. Output filter dicts match the catalogue entry schema used by
``model.catalogue.CatalogueEntry`` and are directly consumable by
``model.iir.CompleteFilter``.

This is the spike optimizer: one low shelf plus one optional residual PEQ.
Multi-PEQ chains, topology preference, and multi-channel handling are out of
scope for this iteration (see docs/design/auto_beq.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from model.iir import HighShelf, LowShelf, PeakingEQ
from scipy import signal as sps
from scipy.optimize import minimize

if TYPE_CHECKING:
    from model.auto_beq_advisor import Advisor, MediaMetadata

log = logging.getLogger("auto_beq")

DEFAULT_FS = 1000
# BEQ filters extend infra-bass below 20 Hz, so the scoring band must reach
# down there - otherwise most of the correction happens outside the band.
DEFAULT_BAND = (5.0, 80.0)
DEFAULT_GRID = np.logspace(np.log10(5.0), np.log10(200.0), 240)


@dataclass(frozen=True)
class MatchMetrics:
    """Summary of how well a candidate filter chain matches a target curve."""

    mean_abs_err_db: float
    max_abs_err_db: float
    band_hz: tuple[float, float]
    verdict: str  # PASS | MARGINAL | FAIL

    def as_text(self) -> str:
        return (
            f"Match metrics ({self.band_hz[0]:.0f}-{self.band_hz[1]:.0f} Hz):\n"
            f"  Mean abs error: {self.mean_abs_err_db:5.2f} dB   (target <2.00)\n"
            f"  Max  abs error: {self.max_abs_err_db:5.2f} dB   (target <5.00)\n"
            f"  Verdict: {self.verdict}"
        )


def evaluate_filter_chain(
    filters: list[dict],
    freqs_hz: np.ndarray,
    fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Return magnitude-in-dB of the cascaded filter chain at ``freqs_hz``.

    Uses ``scipy.signal.freqz`` directly on each filter's biquad coefficients
    (the ``b`` and ``a`` attributes of ``iir.Biquad`` subclasses). Fast enough
    to call inside an optimizer inner loop.
    """
    if not filters:
        return np.zeros_like(freqs_hz)
    h_total = np.ones_like(freqs_hz, dtype=np.complex128)
    for f in filters:
        biquad = _construct_biquad(f, fs)
        _, h = sps.freqz(b=biquad.b, a=biquad.a, worN=freqs_hz, fs=fs)
        h_total *= h
    mag = np.abs(h_total)
    mag[mag == 0] = 1e-12
    return 20.0 * np.log10(mag)


def _construct_biquad(f: dict, fs: int):
    t = f["type"]
    if t == "LowShelf":
        return LowShelf(fs, f["freq"], f["q"], f["gain"])
    if t == "HighShelf":
        return HighShelf(fs, f["freq"], f["q"], f["gain"])
    if t == "PeakingEQ":
        return PeakingEQ(fs, f["freq"], f["q"], f["gain"])
    raise ValueError(f"Unsupported filter type: {t}")


def smooth_fractional_octave(
    curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    octaves: float = 1.0 / 6.0,
) -> np.ndarray:
    """Apply log-frequency Gaussian smoothing at ``octaves`` width.

    BEQ-style fitting works on smoothed curves - narrow resonances in the
    raw spectrum are mastering artefacts, not features a broad IIR
    filter should chase. 1/6-octave smoothing is the standard convention
    in speaker/room measurements.
    """
    log_freqs = np.log2(np.clip(freqs_hz, 1e-6, None))
    sigma_log = octaves / 2.355  # FWHM -> sigma
    smoothed = np.empty_like(curve_db, dtype=float)
    for i, lf in enumerate(log_freqs):
        weights = np.exp(-0.5 * ((log_freqs - lf) / sigma_log) ** 2)
        weights /= weights.sum()
        smoothed[i] = float(np.sum(weights * curve_db))
    return smoothed


def _band_mask(freqs_hz: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freqs_hz >= lo) & (freqs_hz <= hi)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def detect_rolloff_knee(
    target_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[float, float] | None:
    """Estimate (knee_freq_hz, rolloff_depth_db) from a target curve.

    The depth is measured as the difference between the curve at the band
    upper edge (taken as the ~0 dB anchor) and the band lower edge. Knee is
    the frequency where the curve first crosses -3 dB relative to the anchor.
    Returns ``None`` if the curve looks flat (no meaningful rolloff).
    """
    mask = _band_mask(freqs_hz, band)
    if not mask.any():
        return None
    band_freqs = freqs_hz[mask]
    band_curve = target_curve_db[mask]
    anchor_db = float(band_curve[-1])
    depth_db = anchor_db - float(band_curve[0])
    # target curve is a *rolloff* so its low-freq end sits below the high-freq
    # anchor; depth is positive when rolloff is present.
    if depth_db < 1.0:
        return None
    threshold = anchor_db - 3.0
    below = band_curve < threshold
    if not below.any():
        # no crossing in band - knee is below our lowest bin
        return float(band_freqs[0]), depth_db
    knee_idx = int(np.argmax(below))  # first True
    return float(band_freqs[knee_idx]), depth_db


def _fit_single_filter(
    target_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int,
    filter_type: str,
    seeds: tuple[float, float, float],
    bounds: list[tuple[float, float]],
    band: tuple[float, float],
) -> tuple[dict, float]:
    mask = _band_mask(freqs_hz, band)
    band_freqs = freqs_hz[mask]
    band_target = target_curve_db[mask]

    def objective(x):
        freq, q, gain = x
        candidate = [{"type": filter_type, "freq": freq, "q": q, "gain": gain}]
        resp = evaluate_filter_chain(candidate, band_freqs, fs=fs)
        # we want resp to CANCEL the rolloff: target + resp -> 0
        return _rms(band_target + resp)

    result = minimize(objective, x0=seeds, method="L-BFGS-B", bounds=bounds)
    freq, q, gain = result.x
    return (
        {"type": filter_type, "freq": float(freq), "q": float(q), "gain": float(gain)},
        float(result.fun),
    )


def _fit_peq_at_seed(
    err_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int,
    band: tuple[float, float],
    seed_freq: float,
    seed_gain: float,
) -> tuple[dict, float]:
    """Fit a single PEQ that minimises RMS of (err + response) in-band.

    Tries several initial Q values and returns the best fit. Q is capped
    at 4 so the optimizer cannot manufacture narrow ringing filters to
    chase single-bin artefacts.
    """
    mask = _band_mask(freqs_hz, band)
    band_freqs = freqs_hz[mask]
    band_err = err_curve_db[mask]

    def objective(x):
        freq, q, gain = x
        resp = evaluate_filter_chain(
            [{"type": "PeakingEQ", "freq": freq, "q": q, "gain": gain}],
            band_freqs, fs=fs,
        )
        return _rms(band_err + resp)

    # Q capped at 8 to allow matching narrow catalogue PEQ notches
    # (e.g. Mad Max's -6 dB @ 11 Hz Q=8). Below Q=8 the fitter tends
    # to avoid narrow features; above it, chasing single-bin spikes
    # becomes a real risk.
    bounds = [(band[0], band[1]), (0.3, 8.0), (-30.0, 30.0)]
    clamped_freq = float(np.clip(seed_freq, band[0], band[1]))
    clamped_gain = float(np.clip(seed_gain, -30.0, 30.0))
    best_fun = float("inf")
    best_x = None
    for q_seed in (0.7, 1.5, 3.0, 6.0):
        seeds = (clamped_freq, q_seed, clamped_gain)
        result = minimize(objective, x0=seeds, method="L-BFGS-B", bounds=bounds)
        if result.fun < best_fun:
            best_fun = float(result.fun)
            best_x = result.x
    freq, q, gain = best_x
    return (
        {"type": "PeakingEQ", "freq": float(freq), "q": float(q), "gain": float(gain)},
        best_fun,
    )


def _fit_low_shelf(
    err_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int,
    band: tuple[float, float],
) -> tuple[dict, float]:
    """Fit one LowShelf that minimises RMS of (err + response) in-band.

    Works whether err is positive or negative at low freqs - the shelf
    gain bound is bidirectional.
    """
    mask = _band_mask(freqs_hz, band)
    band_freqs = freqs_hz[mask]
    band_err = err_curve_db[mask]
    # Seed gain: negate the mean of err in the lowest octave of the band.
    low_octave = band_freqs <= band_freqs[0] * 2.0
    seed_gain = float(-np.mean(band_err[low_octave])) if low_octave.any() else 0.0

    def objective(x):
        freq, q, gain = x
        resp = evaluate_filter_chain(
            [{"type": "LowShelf", "freq": freq, "q": q, "gain": gain}],
            band_freqs, fs=fs,
        )
        return _rms(band_err + resp)

    seeds = (25.0, 0.7, float(np.clip(seed_gain, -30.0, 30.0)))
    bounds = [(5.0, 120.0), (0.3, 2.0), (-30.0, 30.0)]
    result = minimize(objective, x0=seeds, method="L-BFGS-B", bounds=bounds)
    freq, q, gain = result.x
    return (
        {"type": "LowShelf", "freq": float(freq), "q": float(q), "gain": float(gain)},
        float(result.fun),
    )


def classify_content(
    measured_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    band: tuple[float, float] = DEFAULT_BAND,
    peak_search_band: tuple[float, float] = (15.0, 40.0),
) -> tuple[str, float]:
    """Classify measured content into a tuning profile.

    Different films need different BEQ aggressiveness, and the
    measured curve gives us clues. This is a first-cut classifier
    with three profiles. Each returns (profile_name, max_gain_db).

    Profiles:
      - "cliff": content with a very deep rolloff (>25 dB deficit at
        5 Hz). These films have steep LFE rolloffs and the catalogue
        applies medium-gain corrections - we cap tightly to avoid
        over-extending. Example: Mad Max: Fury Road (45 dB range).
      - "mild": shallow rolloff (<15 dB deficit). Content is
        naturally fairly extended. Apply a generous cap. Example:
        John Wick (11 dB range).
      - "middle": moderate rolloff (15-25 dB deficit). Standard cap.
        Example: Edge of Tomorrow - BUT EoT's catalogue is
        expert-aggressive beyond what measured content suggests,
        which this heuristic cannot infer.
    """
    mask = _band_mask(freqs_hz, band)
    peak_mask = mask & (freqs_hz >= peak_search_band[0]) & (freqs_hz <= peak_search_band[1])
    if not peak_mask.any():
        return "flat", 0.0
    peak_level = float(measured_curve_db[peak_mask].max())
    band_vals = measured_curve_db[mask]
    deficit_at_low = peak_level - float(band_vals.min())

    if deficit_at_low > 25.0:
        return "cliff", 12.0
    if deficit_at_low < 15.0:
        return "mild", 15.0
    return "middle", 15.0


def infer_correction_from_measured(
    measured_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    band: tuple[float, float] = DEFAULT_BAND,
    peak_search_band: tuple[float, float] = (15.0, 40.0),
    max_gain_db: float | None = None,
    advisor: Advisor | None = None,
    metadata: MediaMetadata | None = None,
) -> np.ndarray | None:
    """Heuristic: build a target-correction curve from a measured LFE curve.

    BEQ philosophy is "extend bass where the content rolls off". The
    heuristic here is conservative peak-extension:

      1. Find the peak level in the LFE shoulder band (default 15-40 Hz).
         This approximates "what the content naturally reaches" in its
         strongest region.
      2. At every frequency below the peak, compute how far below the
         peak the measured curve sits. That gap IS the correction: lift
         the low end up to the peak level.
      3. Clamp tiny corrections to zero.

    This matches conservative BEQs (mild-to-medium). It will under-shoot
    aggressive catalogue entries (e.g. Edge of Tomorrow's +28 dB at
    10 Hz) because those embed the expert's decision to go well beyond
    what the measured content warrants. That mismatch is a known spike
    finding, not a bug.

    Returns None if no meaningful correction is needed (< 1 dB peak).
    """
    mask = _band_mask(freqs_hz, band)
    peak_mask = mask & (freqs_hz >= peak_search_band[0]) & (freqs_hz <= peak_search_band[1])
    if not peak_mask.any():
        return None
    peak_level = float(measured_curve_db[peak_mask].max())
    # Index of peak frequency within the search band.
    peak_freqs = freqs_hz[peak_mask]
    peak_vals = measured_curve_db[peak_mask]
    peak_freq = float(peak_freqs[int(np.argmax(peak_vals))])

    # Advisor path: delegate the aggressiveness + knee decisions to an
    # Advisor (heuristic/mock/ollama). The advisor sees film metadata
    # and a summarised view of the measured curve, and returns the
    # numeric knobs the procedural pipeline needs.
    #
    # When the advisor gives BOTH max_gain_db and knee_hz, its answer
    # is authoritative: we build the correction curve as the frequency
    # response of a single LowShelf(fs, knee_hz, 0.7, max_gain_db),
    # and the fitter reproduces it. This bypasses peak-extension
    # entirely so aggressive catalogue-style boosts (+28 dB) aren't
    # clamped to the content's measured shoulder level.
    profile = "unclassified"
    if advisor is not None and metadata is not None:
        from model.auto_beq_advisor import extract_curve_features
        features = extract_curve_features(
            measured_curve_db, freqs_hz, band=band, peak_search_band=peak_search_band
        )
        advice = advisor.advise(metadata, features)
        log.info(
            "advisor[%s]: max_gain_db=%.1f knee_hz=%s confidence=%.2f reasoning=%r",
            advice.source, advice.max_gain_db, advice.knee_hz,
            advice.confidence, advice.reasoning,
        )
        if advice.filters is not None:
            # Most specific advice: use the prescribed chain directly as
            # the correction target. Lets the advisor prescribe multi-
            # knee structures a single-cascade target can't reach.
            log.info(
                "advisor correction target: %d explicit filters from advice",
                len(advice.filters),
            )
            correction = evaluate_filter_chain(
                list(advice.filters), freqs_hz, fs=DEFAULT_FS,
            )
            if float(correction.max()) < 1.0:
                return None
            return correction
        if advice.max_gain_db < 1.0:
            return None
        if advice.knee_hz is not None:
            # Fully-specified advice: build correction as a CASCADE of
            # LowShelves matching typical catalogue construction.
            # Catalogue authors cascade moderate-gain shelves
            # (+4-7 dB each Q=0.8-1.0) to reach deep extension; a
            # single large shelf has a different knee shape.
            #
            # Rule of thumb: one shelf per ~N dB of requested gain.
            # For Mad Max (+15 dB) -> 2 shelves of +7.5 dB each.
            # For EoT (+28 dB) -> 4 shelves of +7 dB each.
            # For John Wick (+13 dB) -> 2 shelves of +6.5 dB each.
            gain_ratio = getattr(advice, "cascade_gain_ratio", 7.0)
            shelf_q = getattr(advice, "cascade_q", 0.9)
            n_shelves = max(1, int(round(advice.max_gain_db / gain_ratio)))
            per_shelf_gain = advice.max_gain_db / n_shelves
            shelf_chain = [
                {"type": "LowShelf", "freq": float(advice.knee_hz),
                 "q": shelf_q, "gain": float(per_shelf_gain)}
                for _ in range(n_shelves)
            ]
            log.info(
                "advisor correction target: %d cascaded LowShelf @ %.1f Hz Q=%.1f +%.2f dB each",
                n_shelves, advice.knee_hz, shelf_q, per_shelf_gain,
            )
            correction = evaluate_filter_chain(
                shelf_chain, freqs_hz, fs=DEFAULT_FS,
            )
            if float(correction.max()) < 1.0:
                return None
            return correction
        # Partial advice: cap-only, fall through to peak-extension.
        max_gain_db = advice.max_gain_db
        profile = "advisor"

    # Fall back to classifier cap if advisor didn't produce one and
    # caller didn't override.
    if max_gain_db is None:
        profile, max_gain_db = classify_content(
            measured_curve_db, freqs_hz, band=band, peak_search_band=peak_search_band
        )
    if max_gain_db <= 0.0:
        return None

    # For cliff-class content (steep rolloff), push the knee DOWN to
    # where the steep drop actually starts. Catalogue entries for
    # cliff content typically put their shelves at the drop-start
    # frequency, not at the content's shoulder peak.
    if profile == "cliff":
        # Walk from peak_freq downward, find the first frequency
        # where the curve falls 3 dB below peak_level.
        band_freqs = freqs_hz[mask]
        band_vals = measured_curve_db[mask]
        peak_idx_band = int(np.argmin(np.abs(band_freqs - peak_freq)))
        # scan downward
        drop_freq = peak_freq
        for i in range(peak_idx_band, -1, -1):
            if band_vals[i] < peak_level - 3.0:
                drop_freq = float(band_freqs[i])
                break
        peak_freq = drop_freq

    # Below the peak frequency, lift the curve up to peak_level, but
    # cap each bin at `max_gain_db`. Real BEQs top out around
    # +15-20 dB of DC gain; beyond that we'd be invoking expert
    # judgment we don't have. Above the peak frequency, correction=0
    # (we don't modify content that's already at or above the shoulder).
    correction = np.zeros_like(measured_curve_db)
    below_peak = freqs_hz < peak_freq
    correction[below_peak] = peak_level - measured_curve_db[below_peak]
    correction = np.clip(correction, 0.0, max_gain_db)
    # Smooth the knee transition so it's shelf-shaped rather than a
    # step function.
    correction = smooth_fractional_octave(correction, freqs_hz, octaves=1.0 / 3.0)

    if float(correction.max()) < 1.0:
        return None
    return correction


def propose_filters_from_measured(
    measured_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int = DEFAULT_FS,
    band: tuple[float, float] = DEFAULT_BAND,
    max_filters: int = 6,
    advisor: Advisor | None = None,
    metadata: MediaMetadata | None = None,
) -> list[dict]:
    """Production API: given a measured LFE curve, propose BEQ filters.

    Pipeline:
      1. (Optional) advisor consults film metadata + curve features
         and picks max_gain_db / knee_hz.
      2. Build a correction curve (conservative peak-extension).
      3. Fit an N-filter IIR chain that reproduces that correction.

    Returns an empty list if the measured curve doesn't need extension.
    """
    correction = infer_correction_from_measured(
        measured_curve_db, freqs_hz, band=band,
        advisor=advisor, metadata=metadata,
    )
    if correction is None:
        return []
    # The fitter treats its input as "cancel this curve". We want the
    # chain's response to EQUAL `correction`, so we flip the sign.
    return propose_filters(
        -correction, freqs_hz, fs=fs, band=band, max_filters=max_filters,
    )


def propose_filters_with_feedback(
    measured_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int = DEFAULT_FS,
    band: tuple[float, float] = DEFAULT_BAND,
    max_filters: int = 6,
    advisor: Advisor | None = None,
    metadata: MediaMetadata | None = None,
    max_iterations: int = 3,
    flatness_threshold_db: float = 2.0,
) -> list[dict]:
    """Iterative correction: propose filters, evaluate, adjust, repeat.

    Unlike ``propose_filters_from_measured`` (single-pass), this loops
    at the ADVISOR level. After each pass it evaluates how flat the
    corrected curve (measured + chain response) is in-band. If the
    residual deficit or overshoot exceeds *flatness_threshold_db*, it
    adjusts the advisor's gain and re-runs.

    This is different from the PEQ iteration inside ``propose_filters``
    which adjusts individual filter parameters to match a fixed target.
    Here we adjust the TARGET itself.

    Returns the best filter chain found across iterations.
    """
    from model.auto_beq_advisor import GainAdjustedAdvisor

    band_mask = (freqs_hz >= band[0]) & (freqs_hz <= band[1])
    current_advisor = advisor
    best_proposed = []
    best_flatness = float("inf")

    for iteration in range(max_iterations):
        proposed = propose_filters_from_measured(
            measured_curve_db, freqs_hz, fs=fs, band=band,
            max_filters=max_filters, advisor=current_advisor,
            metadata=metadata,
        )

        if not proposed:
            log.info("feedback iter %d: no filters proposed, stopping", iteration)
            break

        chain_response = evaluate_filter_chain(proposed, freqs_hz, fs=fs)
        corrected = measured_curve_db + chain_response
        band_corrected = corrected[band_mask]

        max_deficit = float(max(0.0, -np.min(band_corrected)))
        max_overshoot = float(max(0.0, np.max(band_corrected)))
        flatness = max(max_deficit, max_overshoot)

        log.info(
            "feedback iter %d: deficit=%.1f overshoot=%.1f flatness=%.1f "
            "(threshold=%.1f) advisor=%s",
            iteration, max_deficit, max_overshoot, flatness,
            flatness_threshold_db,
            current_advisor.name if current_advisor else "none",
        )

        if flatness < best_flatness:
            best_flatness = flatness
            best_proposed = proposed

        if flatness <= flatness_threshold_db:
            log.info("feedback converged at iter %d (flatness=%.1f)", iteration, flatness)
            break

        # Adjust: apply half the residual error as a gain offset.
        if max_deficit > max_overshoot:
            gain_adj = max_deficit * 0.5
        else:
            gain_adj = -max_overshoot * 0.5

        base_advisor = advisor if advisor is not None else current_advisor
        current_advisor = GainAdjustedAdvisor(base_advisor, gain_adj)

    return best_proposed


def propose_or_lookup(
    title: str,
    measured_curve_db: np.ndarray | None = None,
    freqs_hz: np.ndarray | None = None,
    year: int | None = None,
    audio_codec: str | None = None,
    catalogue: list[dict] | None = None,
    advisor: Advisor | None = None,
    metadata: MediaMetadata | None = None,
    fs: int = DEFAULT_FS,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[list[dict], str]:
    """Production API: catalogue first, auto-generate as fallback.

    1. Look up the BEQ catalogue for (title, year, audio_codec).
       If found, return the expert's filter chain directly.
    2. If no match AND measured_curve_db is provided, auto-generate
       via propose_filters_from_measured.
    3. If neither: return empty.

    Returns ``(filter_list, source)`` where source is
    ``'catalogue'`` or ``'auto:<advisor_name>'`` or ``'none'``.
    """
    from model.auto_beq_catalogue import lookup_catalogue

    entry = lookup_catalogue(
        title, year=year, audio_codec=audio_codec, catalogue=catalogue,
    )
    if entry is not None and entry.get("filters"):
        log.info("catalogue hit for %r: %d filters", title, len(entry["filters"]))
        return entry["filters"], "catalogue"

    if measured_curve_db is not None and freqs_hz is not None:
        log.info("no catalogue match for %r — falling back to auto-generation", title)
        filters = propose_filters_from_measured(
            measured_curve_db, freqs_hz, fs=fs, band=band,
            advisor=advisor, metadata=metadata,
        )
        source = f"auto:{advisor.name}" if advisor else "auto:heuristic"
        return filters, source

    return [], "none"


def propose_filters(
    target_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int = DEFAULT_FS,
    band: tuple[float, float] = DEFAULT_BAND,
    max_filters: int = 6,
    stop_max_err_db: float = 0.5,
) -> list[dict]:
    """Propose a filter chain whose response cancels ``target_curve_db``
    across ``band`` (i.e. target + chain_response ≈ 0).

    Iterative greedy fitter:
      1. Fit one LowShelf to the target (captures the low-end bulk).
      2. While max |residual| in-band exceeds ``stop_max_err_db`` and the
         chain has fewer than ``max_filters`` entries, add one PEQ seeded
         at the worst-residual frequency and optimize it.
      3. Stop.

    The shelf handles broad low-end lift or cut; subsequent PEQs chip
    away at humps, notches, and curvature the shelf can't match. This
    generalises to arbitrary in-band curves, not just simple rolloffs.
    """
    mask = _band_mask(freqs_hz, band)
    # Short-circuit: target is already small enough in-band.
    if np.max(np.abs(target_curve_db[mask])) < stop_max_err_db:
        return []

    chain: list[dict] = []

    # Stage 1: one LowShelf (bidirectional gain).
    shelf, _ = _fit_low_shelf(target_curve_db, freqs_hz, fs, band)
    chain.append(shelf)

    # Stage 2: iteratively add PEQs against the residual.
    while len(chain) < max_filters:
        chain_resp = evaluate_filter_chain(chain, freqs_hz, fs=fs)
        err = target_curve_db + chain_resp  # want -> 0
        band_err = err[mask]
        band_freqs = freqs_hz[mask]
        max_abs = float(np.max(np.abs(band_err)))
        if max_abs < stop_max_err_db:
            break
        worst_idx = int(np.argmax(np.abs(band_err)))
        seed_freq = float(band_freqs[worst_idx])
        seed_gain = float(-band_err[worst_idx])
        peq, new_rms = _fit_peq_at_seed(
            err, freqs_hz, fs, band, seed_freq, seed_gain
        )
        # Accept only if the fit actually reduces in-band RMS error by a
        # meaningful margin (otherwise we're just adding noise).
        prev_rms = _rms(band_err)
        if new_rms >= prev_rms - 0.05:
            break
        chain.append(peq)

    return chain


def compute_match_metrics(
    target_curve_db: np.ndarray,
    candidate_filters: list[dict],
    freqs_hz: np.ndarray,
    fs: int = DEFAULT_FS,
    band: tuple[float, float] = DEFAULT_BAND,
    mean_threshold_db: float = 2.0,
    max_threshold_db: float = 5.0,
) -> MatchMetrics:
    """Compare a candidate chain's response against a target correction curve.

    A perfect match means ``target + candidate_response == 0`` across the band.
    """
    candidate_resp = evaluate_filter_chain(candidate_filters, freqs_hz, fs=fs)
    err = target_curve_db + candidate_resp
    mask = _band_mask(freqs_hz, band)
    band_err = np.abs(err[mask])
    mean_err = float(np.mean(band_err))
    max_err = float(np.max(band_err))
    if mean_err < mean_threshold_db and max_err < max_threshold_db:
        verdict = "PASS"
    elif mean_err < mean_threshold_db * 1.5 and max_err < max_threshold_db * 1.5:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    return MatchMetrics(
        mean_abs_err_db=mean_err,
        max_abs_err_db=max_err,
        band_hz=band,
        verdict=verdict,
    )


def format_filter_list(filters: list[dict], label: str) -> str:
    """Human-readable multi-line listing of a filter chain."""
    lines = [f"{label}: {len(filters)} filter(s)"]
    for f in filters:
        lines.append(
            f"  {f['type']:<10s} freq={f['freq']:7.2f} Hz  "
            f"Q={f['q']:4.2f}  gain={f['gain']:+6.2f} dB"
        )
    return "\n".join(lines)


def format_match_report(
    title: str,
    ground_truth: list[dict],
    proposed: list[dict],
    metrics: MatchMetrics,
    target_depth_db: float,
) -> str:
    """Produce the full text report used by tests and the CLI playground."""
    parts = [
        f"=== Auto-BEQ spike: {title} ===",
        "",
        format_filter_list(ground_truth, "Catalogue entry"),
        "",
        format_filter_list(proposed, "Proposed"),
        "",
        f"Rolloff depth (|target| low end vs anchor): {abs(target_depth_db):.2f} dB",
        metrics.as_text(),
    ]
    return "\n".join(parts)
