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
from scipy import signal as sps
from scipy.optimize import minimize

from model.iir import HighShelf, LowShelf, PeakingEQ

DEFAULT_FS = 1000
DEFAULT_BAND = (20.0, 80.0)
DEFAULT_GRID = np.logspace(np.log10(10.0), np.log10(200.0), 200)


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


def propose_filters(
    target_curve_db: np.ndarray,
    freqs_hz: np.ndarray,
    fs: int = DEFAULT_FS,
    band: tuple[float, float] = DEFAULT_BAND,
    residual_threshold_db: float = 1.0,
) -> list[dict]:
    """Propose a filter chain that cancels ``target_curve_db`` across ``band``.

    ``target_curve_db`` is the rolloff to correct - typically negative in the
    deep bass relative to the 80 Hz anchor. Returns a list of filter dicts in
    the catalogue-entry schema. Empty list if no meaningful rolloff detected.
    """
    knee = detect_rolloff_knee(target_curve_db, freqs_hz, band=band)
    if knee is None:
        return []
    knee_hz, depth_db = knee

    # Shelf seed: freq at knee, gain at rolloff depth, Q ~0.7 (Butterworth).
    shelf_seeds = (max(knee_hz, 20.0), 0.7, min(max(depth_db, 1.0), 18.0))
    shelf_bounds = [(15.0, 120.0), (0.3, 2.0), (0.0, 18.0)]
    shelf, shelf_err = _fit_single_filter(
        target_curve_db, freqs_hz, fs, "LowShelf", shelf_seeds, shelf_bounds, band
    )
    chain = [shelf]

    # Residual PEQ pass if the shelf alone leaves more than threshold dB.
    if shelf_err > residual_threshold_db:
        shelf_resp = evaluate_filter_chain(chain, freqs_hz, fs=fs)
        residual = target_curve_db + shelf_resp  # we want this to be zero
        # seed PEQ at the bin of max |residual| within the band
        mask = _band_mask(freqs_hz, band)
        band_freqs = freqs_hz[mask]
        band_residual = residual[mask]
        worst_idx = int(np.argmax(np.abs(band_residual)))
        peq_seeds = (
            float(band_freqs[worst_idx]),
            1.5,
            float(-band_residual[worst_idx]),  # oppose the residual
        )
        peq_bounds = [(20.0, 80.0), (0.5, 4.0), (-12.0, 12.0)]
        peq, _ = _fit_single_filter(
            target_curve_db, freqs_hz, fs, "PeakingEQ", peq_seeds, peq_bounds, band
        )
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
