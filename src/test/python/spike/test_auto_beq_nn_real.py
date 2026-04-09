"""Real-audio training spike for ML experiments (E25-E32).

Trains on the full BEQ catalogue (synthetic features) with TMDb metadata,
then validates on titles where we have real extracted LFE WAV files. This
is the honest test: can the model trained on synthetic rolloff curves
generalise to predictions on noisy measured audio?

Skipped if no WAV files are available in the audio cache.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
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
    train_late_fusion,
    train_xgboost,
)

from spike._auto_beq_helpers import (
    STRATEGY_BLENDED_07,
    STRATEGY_WELCH,
    discover_wav_catalogue_pairs,
    extract_features_with_strategy,
)

log = logging.getLogger("auto_beq_nn_real")

_DEFAULT_FS = 1000


def _extract_real_audio_features(wav_path: Path, freqs_hz: np.ndarray, fs: int):
    """Load WAV → Welch spectrum → smooth → extract_curve_features."""
    return extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=STRATEGY_WELCH)


def _extract_real_audio_features_chunked(wav_path: Path, freqs_hz: np.ndarray, fs: int):
    """Load WAV → blended (Welch + chunked P90) → extract_curve_features."""
    return extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=STRATEGY_BLENDED_07)


def _extract_one_wav(args: tuple) -> tuple:
    """Worker function for parallel feature extraction.

    Takes (wav_path, freqs_hz, fs, strategy) and returns (wav_path, features)
    or (wav_path, None) on failure. Runs in a separate process.
    """
    wav_path, freqs_hz, fs, strategy = args
    try:
        features = extract_features_with_strategy(Path(wav_path), freqs_hz, fs, strategy=strategy)
        return (str(wav_path), features)
    except Exception as exc:
        return (str(wav_path), None)


def _extract_features_parallel(
    pairs: list[dict],
    freqs_hz: np.ndarray,
    fs: int,
    strategy=None,
    max_workers: int | None = None,
) -> list[tuple]:
    """Extract audio features from multiple WAVs in parallel.

    Returns list of (pair, features) tuples. Failed extractions are skipped.
    Uses ProcessPoolExecutor since scipy Welch is single-threaded.
    """
    from spike._auto_beq_helpers import STRATEGY_WELCH
    if strategy is None:
        strategy = STRATEGY_WELCH

    work = [
        (str(p["wav_path"]), freqs_hz, fs, strategy)
        for p in pairs
    ]

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for wav_str, features in executor.map(_extract_one_wav, work):
            results[wav_str] = features

    elapsed = time.time() - t0
    ok_count = sum(1 for f in results.values() if f is not None)
    log.info("parallel feature extraction: %d/%d in %.1fs (%.1f WAVs/s)",
             ok_count, len(work), elapsed, ok_count / elapsed if elapsed > 0 else 0)

    out = []
    for p in pairs:
        features = results.get(str(p["wav_path"]))
        if features is not None:
            out.append((p, features))
    return out


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


_PAIRS = discover_wav_catalogue_pairs()


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
    from model.auto_beq_nn import _AUTHOR_VOCAB, _MIXER_VOCAB, _STUDIO_VOCAB
    feature_names = (
        [f"audio_{hz}Hz" for hz in [20, 25, 30, 35, 40, 50, 60, 70, 80]]
        + ["year", "fmt_atmos", "fmt_truehd", "fmt_dtshd", "fmt_ddatmos", "fmt_dd", "fmt_other"]
        + ["src_disc", "src_stream", "src_unk"]
        + [f"studio_{s}" for s in _STUDIO_VOCAB]
        + [f"mixer_{m}" for m in _MIXER_VOCAB]
        + [f"genre_{i}" for i in range(10)]
        + [f"country_{i}" for i in range(5)]
        + ["runtime", "rating"]
        + [f"author_{a}" for a in _AUTHOR_VOCAB]
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


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_late_fusion_vs_early(tmp_path):
    """E22: Late fusion (separate audio + metadata models, blended) vs E18 early fusion.

    Late fusion prevents the cross-feature overfitting observed in E18e by
    training independent sub-models on audio-only and metadata-only features,
    then blending their Y predictions.

    Permanent regression test for continuous assessment.
    """
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_nn import LateFusionAdvisor

    log.info("=== E22 late fusion vs E18 early fusion ===")

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

    # Build real-audio + synthetic validation features.
    X_val_real, X_val_synth, Y_val, val_entries = [], [], [], []
    for p in _PAIRS:
        entry = p["catalogue_entry"]
        real_feats = _extract_real_audio_features(p["wav_path"], DEFAULT_GRID, _DEFAULT_FS)
        synth_feats = _synthetic_features(entry, DEFAULT_GRID)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val_real.append(build_feature_vector(real_feats, metadata))
        X_val_synth.append(build_feature_vector(synth_feats, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)
    X_val_real = np.array(X_val_real, dtype=np.float32)
    X_val_synth = np.array(X_val_synth, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    def _mean_loss(model, X_val):
        Y_pred = model.predict(X_val)
        total = 0.0
        for i, entry in enumerate(val_entries):
            total += downstream_loss(labels_to_filters(Y_pred[i]), entry["filters"], DEFAULT_GRID)
        return total / len(val_entries)

    # Train E18 early fusion (baseline).
    log.info("training E18 early fusion...")
    model_early = train_xgboost(X_train, Y_train)

    # Train E22 late fusion at several alpha values.
    alphas = [0.3, 0.5, 0.7]
    late_models = {}
    for alpha in alphas:
        log.info("training E22 late fusion (alpha=%.1f)...", alpha)
        late_models[alpha] = train_late_fusion(X_train, Y_train, alpha=alpha)

    # Evaluate all.
    print(f"\n{'='*70}")
    print(f"  E22 LATE FUSION vs E18 EARLY FUSION")
    print(f"  Training: {len(X_train)} synthetic | Validation: {len(val_entries)} real-audio")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':30s} {'Real audio':>12s} {'Synthetic':>12s} {'Gap':>8s}")
    print(f"  {'-'*65}")

    early_real = _mean_loss(model_early, X_val_real)
    early_synth = _mean_loss(model_early, X_val_synth)
    print(f"  {'E18 early fusion':30s} {early_real:10.2f} dB {early_synth:10.2f} dB {early_real - early_synth:+6.2f} dB")

    for alpha in alphas:
        m = late_models[alpha]
        real = _mean_loss(m, X_val_real)
        synth = _mean_loss(m, X_val_synth)
        print(f"  {f'E22 late fusion (α={alpha:.1f})':30s} {real:10.2f} dB {synth:10.2f} dB {real - synth:+6.2f} dB")

    # Save best late fusion model for advisor integration test.
    best_alpha = min(alphas, key=lambda a: _mean_loss(late_models[a], X_val_real))
    best_model = late_models[best_alpha]
    model_path = str(tmp_path / "e22_late_fusion.joblib")
    save_model(best_model, model_path)

    # Verify LateFusionAdvisor works end-to-end.
    advisor = LateFusionAdvisor.load(model_path)
    entry = val_entries[0]
    feats = _synthetic_features(entry, DEFAULT_GRID)
    meta = enrich_media_metadata(entry, tmdb_cache)
    advice = advisor.advise(meta, feats)
    assert advice.source == "late_fusion"
    print(f"\n  Best alpha: {best_alpha}")
    print(f"  LateFusionAdvisor test: {advice.reasoning}")


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_cnn_dual_branch(tmp_path):
    """E23: CNN dual-branch vs E18 early fusion vs E22 late fusion.

    Trains a PyTorch CNN with separate audio (1D conv) and metadata (dense)
    branches. The architecture naturally provides late fusion via separate
    processing before merging at the penultimate layer.

    Permanent regression test for continuous assessment.
    """
    try:
        from model.auto_beq_nn_cnn import CNNAdvisor, CNNPredictor, train_cnn
    except ImportError:
        pytest.skip("PyTorch not available")

    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_nn import N_AUDIO_FEATURES

    log.info("=== E23 CNN dual-branch ===")

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

    # Build validation features.
    X_val_real, X_val_synth, Y_val, val_entries = [], [], [], []
    for p in _PAIRS:
        entry = p["catalogue_entry"]
        real_feats = _extract_real_audio_features(p["wav_path"], DEFAULT_GRID, _DEFAULT_FS)
        synth_feats = _synthetic_features(entry, DEFAULT_GRID)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val_real.append(build_feature_vector(real_feats, metadata))
        X_val_synth.append(build_feature_vector(synth_feats, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)
    X_val_real = np.array(X_val_real, dtype=np.float32)
    X_val_synth = np.array(X_val_synth, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    def _mean_loss(predictor, X_val):
        Y_pred = predictor.predict(X_val)
        total = 0.0
        for i, entry in enumerate(val_entries):
            total += downstream_loss(labels_to_filters(Y_pred[i]), entry["filters"], DEFAULT_GRID)
        return total / len(val_entries)

    # Train CNN only — don't mix torch and XGBoost in the same test
    # to avoid segfaults from library conflicts on macOS.
    log.info("training E23 CNN dual-branch...")
    cnn_model = train_cnn(
        X_train, Y_train, X_val_synth, Y_val,
        n_audio=N_AUDIO_FEATURES, epochs=200, patience=20,
    )
    cnn_predictor = CNNPredictor(cnn_model)

    cnn_real = _mean_loss(cnn_predictor, X_val_real)
    cnn_synth = _mean_loss(cnn_predictor, X_val_synth)

    # Print results with E18/E22 baselines from previous runs for reference.
    print(f"\n{'='*70}")
    print(f"  E23 CNN DUAL-BRANCH")
    print(f"  Training: {len(X_train)} synthetic | Validation: {len(val_entries)} real-audio")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':30s} {'Real audio':>12s} {'Synthetic':>12s} {'Gap':>8s}")
    print(f"  {'-'*65}")
    print(f"  {'E18 early fusion (ref)':30s} {'6.34':>10s} dB {'3.45':>10s} dB {'2.90':>6s} dB")
    print(f"  {'E22 late fusion α=0.7 (ref)':30s} {'4.03':>10s} dB {'3.68':>10s} dB {'0.35':>6s} dB")
    print(f"  {'E23 CNN dual-branch':30s} {cnn_real:10.2f} dB {cnn_synth:10.2f} dB {cnn_real - cnn_synth:+6.2f} dB")

    # Save + verify advisor round-trip.
    model_path = str(tmp_path / "e23_cnn.joblib")
    save_model(cnn_predictor, model_path)
    advisor = CNNAdvisor.load(model_path)
    entry = val_entries[0]
    feats = _synthetic_features(entry, DEFAULT_GRID)
    meta = enrich_media_metadata(entry, tmdb_cache)
    advice = advisor.advise(meta, feats)
    assert advice.source == "cnn_dual_branch"
    print(f"\n  CNNAdvisor test: {advice.reasoning}")


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_chunked_nn_training(tmp_path):
    """E32: Chunked audio features vs Welch-only for NN training.

    Compares the impact of blended extraction (Welch + chunked P90 at 60s)
    vs Welch-only on NN model performance. Same XGBoost model and synthetic
    training data — only the real-audio validation features differ.

    Uses parallel feature extraction for speed.
    """
    from model.auto_beq_catalogue import _fetch_or_cache

    log.info("=== E32 chunked NN training ===")

    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    # Build synthetic training set.
    X_train_list, Y_train_list, train_entries = [], [], []
    val_tmdb_ids = {p["tmdb_id"] for p in _PAIRS if p.get("tmdb_id")}
    for e in deduped:
        if str(e.get("theMovieDB", "")).strip() in val_tmdb_ids:
            continue
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_train_list.append(build_feature_vector(features, metadata))
        Y_train_list.append(catalogue_entry_to_labels(e))
    X_train = np.array(X_train_list, dtype=np.float32)
    Y_train = np.array(Y_train_list, dtype=np.float32)
    log.info("training set: %d synthetic entries", len(X_train))

    # Train model once (same for both strategies).
    model = train_xgboost(X_train, Y_train)

    # Extract features with both strategies in parallel.
    strategies = [
        ("Welch-only", STRATEGY_WELCH),
        ("Blended (α=0.7, 60s P90)", STRATEGY_BLENDED_07),
    ]

    print(f"\n{'='*70}")
    print(f"  E32 CHUNKED NN TRAINING")
    print(f"  Training: {len(X_train)} synthetic | Validation: {len(_PAIRS)} real-audio")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':35s} {'Real audio':>12s} {'Synthetic':>12s} {'Gap':>8s}")
    print(f"  {'-'*70}")

    for name, strategy in strategies:
        # Extract real-audio features.
        t0 = time.time()
        pairs_features = _extract_features_parallel(
            _PAIRS, DEFAULT_GRID, _DEFAULT_FS, strategy=strategy,
        )
        extract_time = time.time() - t0

        # Build validation arrays.
        X_val, Y_val, val_entries = [], [], []
        X_val_synth = []
        for p, features in pairs_features:
            entry = p["catalogue_entry"]
            if not entry.get("filters"):
                continue
            metadata = enrich_media_metadata(entry, tmdb_cache)
            X_val.append(build_feature_vector(features, metadata))
            Y_val.append(catalogue_entry_to_labels(entry))
            val_entries.append(entry)
            # Synthetic features for comparison.
            synth_feats = _synthetic_features(entry, DEFAULT_GRID)
            X_val_synth.append(build_feature_vector(synth_feats, metadata))

        if not val_entries:
            print(f"  {name:35s} (no valid entries)")
            continue

        X_val = np.array(X_val, dtype=np.float32)
        X_val_synth = np.array(X_val_synth, dtype=np.float32)
        Y_val = np.array(Y_val, dtype=np.float32)

        # Evaluate.
        Y_pred_real = model.predict(X_val)
        Y_pred_synth = model.predict(X_val_synth)

        real_loss = sum(
            downstream_loss(labels_to_filters(Y_pred_real[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(val_entries)
        ) / len(val_entries)
        synth_loss = sum(
            downstream_loss(labels_to_filters(Y_pred_synth[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(val_entries)
        ) / len(val_entries)

        print(f"  {name:35s} {real_loss:10.2f} dB {synth_loss:10.2f} dB {real_loss - synth_loss:+6.2f} dB  ({extract_time:.0f}s)")

    # Also test late fusion + blended (the combination of our two best approaches).
    log.info("extracting blended features for late fusion test...")
    blended_pairs = _extract_features_parallel(
        _PAIRS, DEFAULT_GRID, _DEFAULT_FS, strategy=STRATEGY_BLENDED_07,
    )
    X_val_blended, Y_val_b, val_entries_b = [], [], []
    for p, features in blended_pairs:
        entry = p["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val_blended.append(build_feature_vector(features, metadata))
        Y_val_b.append(catalogue_entry_to_labels(entry))
        val_entries_b.append(entry)
    if val_entries_b:
        X_val_blended = np.array(X_val_blended, dtype=np.float32)

        log.info("training late fusion (α=0.3) for blended comparison...")
        model_late = train_late_fusion(X_train, Y_train, alpha=0.3)

        Y_pred = model_late.predict(X_val_blended)
        late_blended_loss = sum(
            downstream_loss(labels_to_filters(Y_pred[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(val_entries_b)
        ) / len(val_entries_b)

        print(f"\n  {'Late fusion α=0.3 + Blended':35s} {late_blended_loss:10.2f} dB")
        print(f"  (compare: late fusion α=0.3 Welch-only was 3.27 dB on 171 titles)")


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_real_audio_training(tmp_path):
    """E33: Train on real audio features instead of synthetic.

    Compares three training approaches on the same held-out real test set:
    1. Synthetic-only (baseline): ~8k synthetic entries
    2. Real-only: train on real WAV features only
    3. Hybrid: real features where available + synthetic for the rest

    Plus late fusion (α=0.3) variants of each.
    """
    from model.auto_beq_catalogue import _fetch_or_cache
    from sklearn.model_selection import train_test_split

    log.info("=== E33 real audio training ===")

    # --- 1. Extract real audio features for ALL available WAVs ---
    t0 = time.time()
    all_real = _extract_features_parallel(
        _PAIRS, DEFAULT_GRID, _DEFAULT_FS, strategy=STRATEGY_BLENDED_07,
    )
    extract_time = time.time() - t0
    log.info("extracted %d real audio features in %.0fs", len(all_real), extract_time)

    # Filter to entries with filters.
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(
        [p["catalogue_entry"] for p, _ in all_real], cache=tmdb_cache,
    )

    real_X, real_Y, real_entries = [], [], []
    for p, features in all_real:
        entry = p["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        real_X.append(build_feature_vector(features, metadata))
        real_Y.append(catalogue_entry_to_labels(entry))
        real_entries.append(entry)

    real_X = np.array(real_X, dtype=np.float32)
    real_Y = np.array(real_Y, dtype=np.float32)
    log.info("real audio dataset: %d entries", len(real_X))

    if len(real_X) < 20:
        pytest.skip(f"need at least 20 real audio entries, have {len(real_X)}")

    # --- 2. Split real audio into train (80%) / test (20%) ---
    indices = np.arange(len(real_X))
    severity = [
        "heavy" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 20
        else "moderate" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 10
        else "gentle"
        for e in real_entries
    ]
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=severity if len(set(severity)) > 1 else None,
    )

    X_real_train = real_X[train_idx]
    Y_real_train = real_Y[train_idx]
    X_real_test = real_X[test_idx]
    Y_real_test = real_Y[test_idx]
    test_entries = [real_entries[i] for i in test_idx]
    train_tmdb_ids = {str(real_entries[i].get("theMovieDB", "")).strip() for i in train_idx}

    log.info("real train: %d, real test: %d", len(X_real_train), len(X_real_test))

    # --- 3. Build synthetic training set (excluding test titles) ---
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    test_tmdb_ids = {str(real_entries[i].get("theMovieDB", "")).strip() for i in test_idx}
    X_synth_list, Y_synth_list = [], []
    for e in deduped:
        if str(e.get("theMovieDB", "")).strip() in test_tmdb_ids:
            continue  # exclude test titles from synthetic training
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_synth_list.append(build_feature_vector(features, metadata))
        Y_synth_list.append(catalogue_entry_to_labels(e))
    X_synth = np.array(X_synth_list, dtype=np.float32)
    Y_synth = np.array(Y_synth_list, dtype=np.float32)
    log.info("synthetic train: %d (test titles excluded)", len(X_synth))

    # --- 4. Build hybrid training set ---
    # Real features for titles we have WAVs for (train split only),
    # synthetic for everything else.
    X_hybrid_list = list(X_real_train)
    Y_hybrid_list = list(Y_real_train)
    for i, e in enumerate(deduped):
        tid = str(e.get("theMovieDB", "")).strip()
        if tid in test_tmdb_ids or tid in train_tmdb_ids:
            continue  # already in real train or test
        X_hybrid_list.append(X_synth_list[i] if i < len(X_synth_list) else
                             build_feature_vector(_synthetic_features(e, DEFAULT_GRID),
                                                  enrich_media_metadata(e, tmdb_cache)))
        Y_hybrid_list.append(Y_synth_list[i] if i < len(Y_synth_list) else
                             catalogue_entry_to_labels(e))
    X_hybrid = np.array(X_hybrid_list, dtype=np.float32)
    Y_hybrid = np.array(Y_hybrid_list, dtype=np.float32)
    log.info("hybrid train: %d (%d real + %d synthetic)",
             len(X_hybrid), len(X_real_train), len(X_hybrid) - len(X_real_train))

    # --- 5. Train + evaluate ---
    def _mean_loss(model, X_test):
        Y_pred = model.predict(X_test)
        return sum(
            downstream_loss(labels_to_filters(Y_pred[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(test_entries)
        ) / len(test_entries)

    print(f"\n{'='*70}")
    print(f"  E33 REAL AUDIO TRAINING")
    print(f"  Real train: {len(X_real_train)} | Synthetic: {len(X_synth)}")
    print(f"  Hybrid: {len(X_hybrid)} | Test: {len(X_real_test)} (held-out real)")
    print(f"{'='*70}\n")
    print(f"  {'Training approach':35s} {'Early fusion':>14s} {'Late α=0.3':>14s}")
    print(f"  {'-'*65}")

    for name, X_tr, Y_tr in [
        (f"Synthetic-only ({len(X_synth)})", X_synth, Y_synth),
        (f"Real-only ({len(X_real_train)})", X_real_train, Y_real_train),
        (f"Hybrid ({len(X_hybrid)})", X_hybrid, Y_hybrid),
    ]:
        model_early = train_xgboost(X_tr, Y_tr)
        early_loss = _mean_loss(model_early, X_real_test)

        model_late = train_late_fusion(X_tr, Y_tr, alpha=0.3)
        late_loss = _mean_loss(model_late, X_real_test)

        print(f"  {name:35s} {early_loss:12.2f} dB {late_loss:12.2f} dB")
