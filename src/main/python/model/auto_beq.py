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

from dataclasses import dataclass

import numpy as np
from model.iir import HighShelf, LowShelf, PeakingEQ
from scipy import signal as sps
from scipy.optimize import minimize

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

    bounds = [(band[0], band[1]), (0.3, 4.0), (-30.0, 30.0)]
    clamped_freq = float(np.clip(seed_freq, band[0], band[1]))
    clamped_gain = float(np.clip(seed_gain, -30.0, 30.0))
    best_fun = float("inf")
    best_x = None
    for q_seed in (0.7, 1.5, 3.0):
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
