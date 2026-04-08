"""Real-audio training spike for Experiment 18.

Trains on the full BEQ catalogue (synthetic features) with TMDb metadata,
then validates on titles where we have real extracted LFE WAV files. This
is the honest test: can the model trained on synthetic rolloff curves
generalise to predictions on noisy measured audio?

Skipped if no WAV files are available in the audio cache.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
from model.auto_beq import DEFAULT_GRID, compute_match_metrics, evaluate_filter_chain, smooth_fractional_octave
from model.auto_beq_advisor import MediaMetadata, extract_curve_features
from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
from model.auto_beq_nn import (
    N_FEATURES,
    build_feature_vector,
    catalogue_entry_to_labels,
    deduplicate_by_title,
    downstream_loss,
    labels_to_filters,
    save_model,
    train_xgboost,
)
from model.signal import read_wav_data

from spike._auto_beq_helpers import audio_cache_dir

log = logging.getLogger("auto_beq_nn_real")

_DEFAULT_FS = 1000
_TMDB_RE = re.compile(r"\[tmdb-(\d+)\]")


# ---------------------------------------------------------------------------
# Discovery: find WAV files and match to catalogue
# ---------------------------------------------------------------------------


def _discover_wav_catalogue_pairs() -> list[dict]:
    """Find all cached LFE WAVs that match a BEQ catalogue entry.

    Returns list of dicts with keys: wav_path, catalogue_entry, tmdb_id.
    """
    try:
        cache_root = audio_cache_dir()
    except RuntimeError:
        return []

    # Load full catalogue.
    from model.auto_beq_catalogue import _fetch_or_cache
    catalogue = _fetch_or_cache()

    # Index by TMDb ID.
    by_tmdb: dict[str, list[dict]] = {}
    for e in catalogue:
        tid = str(e.get("theMovieDB", "")).strip()
        if tid:
            by_tmdb.setdefault(tid, []).append(e)

    # Scan for WAV files.
    wav_files = sorted(cache_root.rglob("*.lfe-1000hz.wav"))
    pairs = []
    for wav in wav_files:
        m = _TMDB_RE.search(str(wav))
        if not m:
            continue
        tid = m.group(1)
        entries = by_tmdb.get(tid)
        if entries:
            pairs.append({
                "wav_path": wav,
                "catalogue_entry": entries[0],
                "tmdb_id": tid,
            })
    return pairs


def _extract_real_audio_features(wav_path: Path, freqs_hz: np.ndarray, fs: int):
    """Load WAV → spectrum → smooth → extract_curve_features.

    Same pipeline as test_auto_beq.py::test_real_media_roundtrip.
    """
    samples, read_fs, _ = read_wav_data(str(wav_path))
    assert read_fs == fs, f"expected fs={fs}, got {read_fs}"
    mono = samples[:, 0] if samples.ndim > 1 else samples

    from model.signal import Signal
    sig = Signal(str(wav_path.stem), mono, fs=fs)
    measured_freqs, measured_db = sig.avg_spectrum()

    # Interpolate onto standard log grid, normalise to 80 Hz, smooth.
    curve = np.interp(freqs_hz, measured_freqs, measured_db)
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    curve -= curve[anchor_idx]
    curve = smooth_fractional_octave(curve, freqs_hz, octaves=1.0 / 6.0)
    curve -= curve[anchor_idx]

    return extract_curve_features(curve, freqs_hz)


# ---------------------------------------------------------------------------
# Synthetic feature extraction (for non-WAV catalogue entries)
# ---------------------------------------------------------------------------


def _synthetic_features(entry: dict, freqs_hz: np.ndarray):
    correction = evaluate_filter_chain(entry["filters"], freqs_hz, fs=_DEFAULT_FS)
    rolloff = -correction
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    rolloff_norm = rolloff - rolloff[anchor_idx]
    return extract_curve_features(rolloff_norm, freqs_hz)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


_PAIRS = _discover_wav_catalogue_pairs()


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_train_full_catalogue_validate_real_audio(tmp_path):
    """Train on full catalogue (synthetic), validate on real-audio WAV files.

    This is the honest generalisation test: the model sees synthetic
    rolloff curves during training (derived from filter chains), then
    at validation time sees REAL measured LFE spectra from actual media
    files. The gap between synthetic and real performance tells us how
    much the "perfect inverse" assumption costs.
    """
    from model.auto_beq_catalogue import _fetch_or_cache

    log.info("=== E18 real-audio validation ===")

    # --- 1. Load and deduplicate full catalogue ---
    catalogue = _fetch_or_cache()
    log.info("full catalogue: %d entries", len(catalogue))

    deduped = deduplicate_by_title(catalogue)
    deduped = [e for e in deduped if e.get("filters")]
    log.info("after dedup + filter: %d unique titles with filters", len(deduped))

    # --- 2. Fetch TMDb metadata (cached, no artificial throttle) ---
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    # --- 3. Build full synthetic training dataset ---
    log.info("building synthetic training features for %d titles...", len(deduped))
    X_all, Y_all, entries_all = [], [], []
    for e in deduped:
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_all.append(build_feature_vector(features, metadata))
        Y_all.append(catalogue_entry_to_labels(e))
        entries_all.append(e)

    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)
    log.info("training dataset: X=%s Y=%s", X_all.shape, Y_all.shape)

    # --- 4. Build real-audio validation set ---
    real_tmdb_ids = {p["tmdb_id"] for p in _PAIRS}
    log.info("real-audio validation titles: %d", len(_PAIRS))
    for p in _PAIRS:
        log.info("  %s (tmdb=%s)", p["catalogue_entry"]["title"], p["tmdb_id"])

    # Remove real-audio titles from training set (honest held-out).
    train_mask = np.array([
        str(e.get("theMovieDB", "")).strip() not in real_tmdb_ids
        for e in entries_all
    ])
    X_train = X_all[train_mask]
    Y_train = Y_all[train_mask]
    log.info("training set (excluding real-audio titles): %d", len(X_train))

    # Build real-audio validation features.
    X_val, Y_val, val_entries = [], [], []
    for p in _PAIRS:
        entry = p["catalogue_entry"]
        wav_path = p["wav_path"]
        log.info("extracting real audio features: %s", wav_path.name[:80])
        features = _extract_real_audio_features(wav_path, DEFAULT_GRID, _DEFAULT_FS)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val.append(build_feature_vector(features, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)

    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)
    log.info("validation set (real audio): %d", len(X_val))

    # --- 5. Train ---
    log.info("training XGBoost on %d synthetic entries...", len(X_train))
    model = train_xgboost(X_train, Y_train, X_val, Y_val)

    # Save model for inspection.
    model_path = str(tmp_path / "e18_real_audio_model.joblib")
    save_model(model, model_path)

    # --- 6. Feature importances ---
    importances = model.feature_importances_
    audio_imp = importances[:9].sum()
    meta_imp = importances[9:].sum()
    log.info("feature importances: audio=%.3f metadata=%.3f", audio_imp, meta_imp)

    # Top features.
    from model.auto_beq_nn import _STUDIO_VOCAB, _MIXER_VOCAB
    feature_names = (
        [f"audio_{hz}Hz" for hz in [20, 25, 30, 35, 40, 50, 60, 70, 80]]
        + ["year", "fmt_atmos", "fmt_truehd", "fmt_dtshd", "fmt_ddatmos", "fmt_dd", "fmt_other"]
        + ["src_disc", "src_stream", "src_unk"]
        + [f"studio_{s}" for s in _STUDIO_VOCAB]
        + [f"mixer_{m}" for m in _MIXER_VOCAB]
        + [f"genre_{i}" for i in range(10)]
        + [f"country_{i}" for i in range(5)]
        + ["runtime", "rating"]
    )
    top_idx = np.argsort(importances)[::-1][:15]
    print("\n=== Top 15 features by importance ===")
    for i in top_idx:
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {name:20s} {importances[i]:.4f}")

    # --- 7. Evaluate on real-audio validation set ---
    Y_pred = model.predict(X_val)
    print(f"\n=== Real-audio validation ({len(val_entries)} titles) ===")
    print(f"{'Title':40s} {'Downstream':>10s} {'Verdict':>8s} {'Predicted filters'}")
    print("-" * 90)

    total_loss = 0.0
    for i, entry in enumerate(val_entries):
        pred_filters = labels_to_filters(Y_pred[i])
        target_filters = entry["filters"]
        loss = downstream_loss(pred_filters, target_filters, DEFAULT_GRID)
        total_loss += loss

        # Also compute match metrics for the standard grading.
        correction = evaluate_filter_chain(target_filters, DEFAULT_GRID, fs=_DEFAULT_FS)
        target_curve = -correction
        metrics = compute_match_metrics(target_curve, pred_filters, DEFAULT_GRID, fs=_DEFAULT_FS)

        pred_summary = ", ".join(
            f"{f['type']}({f['freq']:.0f}Hz,{f['gain']:+.1f}dB)"
            for f in pred_filters
        ) if pred_filters else "(empty)"

        title = entry["title"][:40]
        print(f"  {title:40s} {loss:10.2f} dB {metrics.verdict:>8s} {pred_summary}")

    mean_loss = total_loss / len(val_entries) if val_entries else 0
    print(f"\n  Mean downstream loss: {mean_loss:.2f} dB")
    print(f"  (< 2 dB = good, < 5 dB = marginal, > 5 dB = needs work)")

    # --- 8. Compare: same titles with synthetic features ---
    print(f"\n=== Same titles with SYNTHETIC features (sanity check) ===")
    X_synth_val = []
    for entry in val_entries:
        features = _synthetic_features(entry, DEFAULT_GRID)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_synth_val.append(build_feature_vector(features, metadata))
    X_synth_val = np.array(X_synth_val, dtype=np.float32)
    Y_synth_pred = model.predict(X_synth_val)

    synth_total_loss = 0.0
    for i, entry in enumerate(val_entries):
        pred_filters = labels_to_filters(Y_synth_pred[i])
        loss = downstream_loss(pred_filters, entry["filters"], DEFAULT_GRID)
        synth_total_loss += loss
    synth_mean = synth_total_loss / len(val_entries) if val_entries else 0
    print(f"  Mean downstream loss (synthetic): {synth_mean:.2f} dB")
    print(f"  Mean downstream loss (real audio): {mean_loss:.2f} dB")
    print(f"  Gap (real - synthetic):            {mean_loss - synth_mean:.2f} dB")


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_ablation_audio_vs_metadata(tmp_path):
    """E18e ablation: audio-only vs metadata-only vs full (audio+metadata).

    Trains three XGBoost models on the same catalogue data, each seeing a
    different subset of the 90-dim feature vector:
      - audio-only:    dims 0–8 (9 frequency bins), metadata zeroed
      - metadata-only: dims 9–89 (81 metadata features), audio zeroed
      - full:          all 90 dims (baseline, same as E18d)

    Evaluates all three on real-audio validation set. The delta between
    audio-only and full quantifies how much metadata contributes. If
    metadata-only outperforms audio-only, it means production context
    (studio, year, format) is a stronger signal than the measured curve
    for predicting BEQ filter parameters.
    """
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_nn import N_AUDIO_FEATURES

    log.info("=== E18e ablation: audio vs metadata ===")

    # --- Reuse data pipeline from main test ---
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    X_all, Y_all, entries_all = [], [], []
    for e in deduped:
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_all.append(build_feature_vector(features, metadata))
        Y_all.append(catalogue_entry_to_labels(e))
        entries_all.append(e)
    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    # Hold out real-audio titles.
    real_tmdb_ids = {p["tmdb_id"] for p in _PAIRS}
    train_mask = np.array([
        str(e.get("theMovieDB", "")).strip() not in real_tmdb_ids
        for e in entries_all
    ])
    X_train = X_all[train_mask]
    Y_train = Y_all[train_mask]

    # Real-audio validation features.
    X_val_real, Y_val, val_entries = [], [], []
    for p in _PAIRS:
        entry = p["catalogue_entry"]
        features = _extract_real_audio_features(p["wav_path"], DEFAULT_GRID, _DEFAULT_FS)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val_real.append(build_feature_vector(features, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)
    X_val_real = np.array(X_val_real, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    # Synthetic validation features (same titles).
    X_val_synth = []
    for entry in val_entries:
        features = _synthetic_features(entry, DEFAULT_GRID)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val_synth.append(build_feature_vector(features, metadata))
    X_val_synth = np.array(X_val_synth, dtype=np.float32)

    # --- Create feature masks ---
    n = N_AUDIO_FEATURES  # 9
    audio_mask = np.zeros(X_train.shape[1], dtype=bool)
    audio_mask[:n] = True
    meta_mask = ~audio_mask

    # --- Train 3 models ---
    def _mask_features(X, mask):
        X_masked = X.copy()
        X_masked[:, ~mask] = 0.0
        return X_masked

    def _eval_model(model, X_val, val_entries, label):
        Y_pred = model.predict(X_val)
        total = 0.0
        for i, entry in enumerate(val_entries):
            pred_filters = labels_to_filters(Y_pred[i])
            total += downstream_loss(pred_filters, entry["filters"], DEFAULT_GRID)
        return total / len(val_entries)

    variants = [
        ("audio-only", audio_mask),
        ("metadata-only", meta_mask),
        ("full (audio+meta)", np.ones(X_train.shape[1], dtype=bool)),
    ]

    print(f"\n{'='*70}")
    print(f"  E18e ABLATION: audio-only vs metadata-only vs full")
    print(f"  Training on {len(X_train)} synthetic entries")
    print(f"  Validating on {len(val_entries)} real-audio titles")
    print(f"{'='*70}\n")
    print(f"  {'Variant':25s} {'Real audio':>12s} {'Synthetic':>12s} {'Gap':>8s}")
    print(f"  {'-'*60}")

    for name, mask in variants:
        X_tr = _mask_features(X_train, mask)
        X_vr = _mask_features(X_val_real, mask)
        X_vs = _mask_features(X_val_synth, mask)

        model = train_xgboost(X_tr, Y_train)

        real_loss = _eval_model(model, X_vr, val_entries, name)
        synth_loss = _eval_model(model, X_vs, val_entries, name)
        gap = real_loss - synth_loss
        print(f"  {name:25s} {real_loss:10.2f} dB {synth_loss:10.2f} dB {gap:+6.2f} dB")

    print(f"\n  Interpretation:")
    print(f"  - If full < audio-only: metadata is helping")
    print(f"  - If metadata-only < audio-only: production context outweighs measured curve")
    print(f"  - Gap column: how much harder real audio is vs synthetic")
