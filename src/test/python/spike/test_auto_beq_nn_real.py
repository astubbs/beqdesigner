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
    train_xgboost_reweighted,
)

from spike._auto_beq_helpers import (
    STRATEGY_BLENDED_07,
    STRATEGY_WELCH,
    discover_wav_catalogue_pairs,
    extract_features_with_strategy,
)

log = logging.getLogger("auto_beq_nn_real")

_DEFAULT_FS = 1000


def _print_per_title_breakdown(Y_pred: np.ndarray, val_entries: list[dict], model_name: str):
    """Print per-title results sorted by loss, grouped by verdict."""
    from model.auto_beq import compute_match_metrics

    results = []
    for i, entry in enumerate(val_entries):
        pred_filters = labels_to_filters(Y_pred[i])
        target_filters = entry["filters"]
        loss = downstream_loss(pred_filters, target_filters, DEFAULT_GRID)

        correction = evaluate_filter_chain(target_filters, DEFAULT_GRID, fs=_DEFAULT_FS)
        target_curve = -correction
        metrics = compute_match_metrics(target_curve, pred_filters, DEFAULT_GRID, fs=_DEFAULT_FS)

        pred_types = [f["type"][0] for f in pred_filters]  # L/H/P
        target_types = [f["type"][0] for f in target_filters]

        results.append({
            "title": entry.get("title", "?")[:35],
            "year": str(entry.get("year", "?")),
            "author": entry.get("author", "?")[:12],
            "content_type": "TV" if entry.get("content_type", "") == "TV" else "film",
            "loss": loss,
            "verdict": metrics.verdict,
            "pred_types": "".join(pred_types),
            "target_types": "".join(target_types),
            "n_filters": len(target_filters),
        })

    results.sort(key=lambda r: r["loss"])

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    marginal_count = sum(1 for r in results if r["verdict"] == "MARGINAL")
    fail_count = sum(1 for r in results if r["verdict"] == "FAIL")
    mean_loss = sum(r["loss"] for r in results) / len(results) if results else 0

    print(f"\n=== {model_name}: per-title breakdown ({len(results)} titles) ===")
    print(f"  PASS: {pass_count} | MARGINAL: {marginal_count} | FAIL: {fail_count} | Mean: {mean_loss:.2f} dB")
    print(f"\n  {'Title':35s} {'Year':>4s} {'Author':>12s} {'Type':>4s} {'Loss':>7s} {'Grade':>8s} {'Pred→Tgt':>8s}")
    print(f"  {'-'*88}")

    for r in results:
        print(
            f"  {r['title']:35s} {r['year']:>4s} {r['author']:>12s} "
            f"{r['content_type']:>4s} {r['loss']:6.2f} dB {r['verdict']:>8s} "
            f"{r['pred_types']:>3s}→{r['target_types']:<3s}"
        )

    # Failure pattern analysis.
    from collections import defaultdict
    print(f"\n  --- Failure analysis ---")

    # By decade.
    decade_losses = defaultdict(list)
    for r in results:
        y = int(r["year"]) if r["year"].isdigit() else 0
        if y:
            decade_losses[f"{y // 10 * 10}s"].append(r["loss"])
    print(f"\n  {'Decade':>8s} {'n':>4s} {'Mean':>6s} {'>4dB':>5s}")
    for d in sorted(decade_losses):
        losses = decade_losses[d]
        mean = sum(losses) / len(losses)
        bad = sum(1 for l in losses if l >= 4)
        print(f"  {d:>8s} {len(losses):4d} {mean:5.1f}dB {bad:5d}")

    # By author.
    author_losses = defaultdict(list)
    for r in results:
        author_losses[r["author"]].append(r["loss"])
    print(f"\n  {'Author':>12s} {'n':>4s} {'Mean':>6s} {'<2dB':>5s} {'>4dB':>5s}")
    for a in sorted(author_losses, key=lambda a: -len(author_losses[a])):
        losses = author_losses[a]
        mean = sum(losses) / len(losses)
        good = sum(1 for l in losses if l < 2)
        bad = sum(1 for l in losses if l >= 4)
        print(f"  {a:>12s} {len(losses):4d} {mean:5.1f}dB {good:5d} {bad:5d}")

    # By content type.
    type_losses = defaultdict(list)
    for r in results:
        type_losses[r["content_type"]].append(r["loss"])
    print(f"\n  {'Type':>6s} {'n':>4s} {'Mean':>6s} {'<2dB':>5s} {'>4dB':>5s}")
    for t in sorted(type_losses):
        losses = type_losses[t]
        mean = sum(losses) / len(losses)
        good = sum(1 for l in losses if l < 2)
        bad = sum(1 for l in losses if l >= 4)
        print(f"  {t:>6s} {len(losses):4d} {mean:5.1f}dB {good:5d} {bad:5d}")

    # Filter count match.
    over = under = match = 0
    for r in results:
        pn = len(r["pred_types"])
        tn = len(r["target_types"])
        if pn > tn:
            over += 1
        elif pn < tn:
            under += 1
        else:
            match += 1
    print(f"\n  Filter count: {over} over, {under} under, {match} match")


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

    # --- 7. Evaluate on real-audio validation set (with per-title breakdown) ---
    Y_pred = model.predict(X_val)
    _print_per_title_breakdown(Y_pred, val_entries, "E25 early fusion")

    mean_loss = sum(
        downstream_loss(labels_to_filters(Y_pred[i]), e["filters"], DEFAULT_GRID)
        for i, e in enumerate(val_entries)
    ) / len(val_entries)

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

    # --- E36: Unknown author scenario ---
    # Zero out the author columns to simulate production inference
    # (where we're generating a BEQ, not reproducing a known author's work).
    from model.auto_beq_nn import N_AUTHOR
    author_start = N_FEATURES - N_AUTHOR  # author is the last N_AUTHOR dims
    X_val_no_author = X_val.copy()
    X_val_no_author[:, author_start:] = 0.0

    Y_pred_no_author = model.predict(X_val_no_author)
    no_author_loss = sum(
        downstream_loss(labels_to_filters(Y_pred_no_author[i]), e["filters"], DEFAULT_GRID)
        for i, e in enumerate(val_entries)
    ) / len(val_entries)
    print(f"\n=== E36: Unknown author (zeroed at inference) ===")
    print(f"  With author:    {mean_loss:.2f} dB")
    print(f"  Without author: {no_author_loss:.2f} dB")
    print(f"  Author impact:  {no_author_loss - mean_loss:+.2f} dB")


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

    # Per-title breakdown on best late fusion model.
    Y_pred_best = best_model.predict(X_val_real)
    _print_per_title_breakdown(Y_pred_best, val_entries, f"E27 late fusion α={best_alpha}")


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
    """E33/E77: Train on real audio features instead of synthetic.

    Compares three training approaches on the same held-out real test set:
    1. Synthetic-only (baseline): ~8k synthetic entries
    2. Real-only: train on real WAV features only
    3. Hybrid: real features where available + synthetic for the rest

    E33 (original run at 155 WAVs): real-only lost to synthetic by 1.14 dB
    and hybrid lost by 0.24 dB.  Not enough real data to beat synthetic.

    E77 (re-run at 932 WAVs): test whether 6× more real data is enough
    for real or hybrid training to beat synthetic-only.

    All three training approaches are tested under the current best
    config (F1 augmentation σ=0.5 + late fusion α=0.5), matching the
    production model (I1b-soft-blend).
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
    from model.auto_beq_nn import AugmentationConfig

    def _mean_loss(model, X_test):
        Y_pred = model.predict(X_test)
        return sum(
            downstream_loss(labels_to_filters(Y_pred[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(test_entries)
        ) / len(test_entries)

    # Current production config: F1 σ=0.5 + late fusion α=0.5.
    # Also run plain XGBoost and LF without augmentation for comparison
    # against the old E33 numbers (155-WAV regime).
    aug = AugmentationConfig(
        gaussian_sigma_db=0.5, per_bin_uniform_db=1.0, n_copies=3,
    )

    print(f"\n{'='*78}")
    print(f"  E77/E33 REAL AUDIO TRAINING (932-WAV cache re-run)")
    print(f"  Real train: {len(X_real_train)} | Synthetic: {len(X_synth)}")
    print(f"  Hybrid: {len(X_hybrid)} | Test: {len(X_real_test)} (held-out real)")
    print(f"{'='*78}\n")
    print(f"  {'Training approach':30s} "
          f"{'XGB plain':>12s} "
          f"{'LF α=0.5':>12s} "
          f"{'LF+aug':>12s}")
    print(f"  {'-' * 72}")
    results_table = []
    for name, X_tr, Y_tr in [
        (f"Synthetic-only ({len(X_synth)})", X_synth, Y_synth),
        (f"Real-only ({len(X_real_train)})", X_real_train, Y_real_train),
        (f"Hybrid ({len(X_hybrid)})", X_hybrid, Y_hybrid),
    ]:
        # 1. Plain XGBoost (matches old E33 numbers for comparison).
        model_plain = train_xgboost(X_tr, Y_tr)
        plain_loss = _mean_loss(model_plain, X_real_test)

        # 2. Late fusion α=0.5 (G2a single-alpha best).
        model_lf = train_late_fusion(X_tr, Y_tr, alpha=0.5)
        lf_loss = _mean_loss(model_lf, X_real_test)

        # 3. Late fusion α=0.5 + augmentation (F1 × G2a, production config).
        model_lf_aug = train_late_fusion(
            X_tr, Y_tr, alpha=0.5, augmentation=aug,
        )
        lf_aug_loss = _mean_loss(model_lf_aug, X_real_test)

        print(f"  {name:30s} "
              f"{plain_loss:10.2f} dB "
              f"{lf_loss:10.2f} dB "
              f"{lf_aug_loss:10.2f} dB")
        results_table.append((name, plain_loss, lf_loss, lf_aug_loss))

    # Print deltas vs synthetic-only baseline (LF+aug, production config).
    synth_lf_aug = results_table[0][3]
    print(f"\n  Delta vs synthetic-only (LF+aug column, production config):")
    for name, _plain, _lf, lf_aug in results_table[1:]:
        delta = lf_aug - synth_lf_aug
        sign = "+" if delta > 0 else ""
        if delta < -0.05:
            verdict = " ← WINS"
        elif delta > 0.05:
            verdict = " ← loses"
        else:
            verdict = " ← tied"
        print(f"    {name:30s} {sign}{delta:.2f} dB{verdict}")
    print()


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_reweighted_training(tmp_path):
    """E38: Reweighted training — focus on acoustically bad predictions.

    Two-stage training: Stage 1 trains normally with MSE, Stage 2 upweights
    samples where parameter-MSE produced bad acoustic (downstream) results.
    Compares standard, reweighted, late fusion, and reweighted + late fusion.

    Permanent regression test for continuous assessment.
    """
    from model.auto_beq_catalogue import _fetch_or_cache

    log.info("=== E38 reweighted training ===")

    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    # Build synthetic training set (excluding validation titles).
    val_tmdb_ids = {p["tmdb_id"] for p in _PAIRS if p.get("tmdb_id")}
    X_train_list, Y_train_list = [], []
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

    # Build real-audio validation features (parallel).
    t0 = time.time()
    pairs_features = _extract_features_parallel(
        _PAIRS, DEFAULT_GRID, _DEFAULT_FS, strategy=STRATEGY_WELCH,
    )
    log.info("extracted %d features in %.0fs", len(pairs_features), time.time() - t0)

    X_val, Y_val, val_entries = [], [], []
    for p, features in pairs_features:
        entry = p["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val.append(build_feature_vector(features, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    def _mean_loss(model):
        Y_pred = model.predict(X_val)
        return sum(
            downstream_loss(labels_to_filters(Y_pred[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(val_entries)
        ) / len(val_entries)

    # Train all strategies.
    log.info("training standard XGBoost...")
    model_std = train_xgboost(X_train, Y_train)

    log.info("training reweighted XGBoost (2 rounds)...")
    model_rw = train_xgboost_reweighted(X_train, Y_train, DEFAULT_GRID, n_rounds=2)

    log.info("training reweighted XGBoost (3 rounds)...")
    model_rw3 = train_xgboost_reweighted(X_train, Y_train, DEFAULT_GRID, n_rounds=3)

    log.info("training late fusion α=0.7...")
    model_late = train_late_fusion(X_train, Y_train, alpha=0.7)

    # Reweighted sub-models for late fusion: reweight the full-feature
    # model, then build late fusion from it. The reweighted weights come
    # from the early-fusion downstream loss — same weights apply to both
    # audio-only and metadata-only sub-models since the bad samples are
    # the same regardless of feature split.
    log.info("training reweighted late fusion α=0.7...")
    # Get weights from the reweighted model's round-1 evaluation.
    Y_pred_std = model_std.predict(X_train)
    rw_weights = np.ones(len(X_train), dtype=np.float32)
    for i in range(len(X_train)):
        pred_filters = labels_to_filters(Y_pred_std[i])
        target_filters = labels_to_filters(Y_train[i])
        loss = downstream_loss(pred_filters, target_filters, DEFAULT_GRID)
        rw_weights[i] = max(1.0, loss)

    n_audio = 9  # N_AUDIO_FEATURES
    X_audio = np.zeros_like(X_train)
    X_audio[:, :n_audio] = X_train[:, :n_audio]
    X_meta = np.zeros_like(X_train)
    X_meta[:, n_audio:] = X_train[:, n_audio:]

    from model.auto_beq_nn import LateFusionModel
    model_rw_audio = train_xgboost(X_audio, Y_train, sample_weight=rw_weights)
    model_rw_meta = train_xgboost(X_meta, Y_train, sample_weight=rw_weights)
    model_rw_late = LateFusionModel(model_rw_audio, model_rw_meta, alpha=0.7)

    print(f"\n{'='*70}")
    print(f"  E38 REWEIGHTED TRAINING")
    print(f"  Training: {len(X_train)} synthetic | Validation: {len(val_entries)} real-audio")
    print(f"{'='*70}\n")
    print(f"  {'Strategy':40s} {'Real audio':>12s}")
    print(f"  {'-'*55}")

    for name, model in [
        ("Standard XGBoost", model_std),
        ("Reweighted (2 rounds)", model_rw),
        ("Reweighted (3 rounds)", model_rw3),
        ("Late fusion α=0.7", model_late),
        ("Reweighted + Late fusion α=0.7", model_rw_late),
    ]:
        loss = _mean_loss(model)
        print(f"  {name:40s} {loss:10.2f} dB")


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_per_author_isolation(tmp_path):
    """E40: Per-author model performance — isolate each author's calibration.

    For each author with sufficient validation data:
    1. Multi-author model (all catalogue) validated on that author's WAVs
    2. Single-author model (only that author's catalogue) validated on their WAVs

    Shows whether the model has learned each author's style, and whether
    training on a single author improves predictions for their titles.

    Permanent regression test for continuous assessment.
    """
    from collections import defaultdict
    from model.auto_beq_catalogue import _fetch_or_cache

    log.info("=== E40 per-author isolation ===")

    # Extract real audio features for all WAVs (parallel).
    t0 = time.time()
    all_real = _extract_features_parallel(
        _PAIRS, DEFAULT_GRID, _DEFAULT_FS, strategy=STRATEGY_WELCH,
    )
    log.info("extracted %d features in %.0fs", len(all_real), time.time() - t0)

    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(
        [p["catalogue_entry"] for p, _ in all_real], cache=tmdb_cache,
    )

    # Build validation data grouped by author.
    by_author: dict[str, list[dict]] = defaultdict(list)
    for p, features in all_real:
        entry = p["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        author = entry.get("author", "unknown")
        by_author[author].append({
            "x": build_feature_vector(features, metadata),
            "y": catalogue_entry_to_labels(entry),
            "entry": entry,
        })

    # Build full synthetic training set.
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    all_val_tmdb = set()
    for items in by_author.values():
        for item in items:
            all_val_tmdb.add(str(item["entry"].get("theMovieDB", "")).strip())

    # Full training set (all authors, excluding validation titles).
    X_train_all, Y_train_all = [], []
    # Per-author training sets.
    train_by_author: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for e in deduped:
        tid = str(e.get("theMovieDB", "")).strip()
        if tid in all_val_tmdb:
            continue
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        x = build_feature_vector(features, metadata)
        y = catalogue_entry_to_labels(e)
        X_train_all.append(x)
        Y_train_all.append(y)
        author = e.get("author", "unknown")
        train_by_author[author][0].append(x)
        train_by_author[author][1].append(y)

    X_train_all = np.array(X_train_all, dtype=np.float32)
    Y_train_all = np.array(Y_train_all, dtype=np.float32)

    # Train multi-author model once.
    log.info("training multi-author model (%d entries)...", len(X_train_all))
    model_all = train_late_fusion(X_train_all, Y_train_all, alpha=0.3)

    # Results table.
    min_val_titles = 10
    authors_to_test = [a for a in sorted(by_author, key=lambda a: -len(by_author[a]))
                       if len(by_author[a]) >= min_val_titles]

    print(f"\n{'='*75}")
    print(f"  E40 PER-AUTHOR ISOLATION")
    print(f"  Multi-author training: {len(X_train_all)} synthetic")
    print(f"  Authors with ≥{min_val_titles} validation titles: {len(authors_to_test)}")
    print(f"{'='*75}\n")
    print(f"  {'Author':>12s} {'Val':>4s} {'Train':>6s} {'Multi-author':>14s} {'Single-author':>15s} {'Delta':>7s}")
    print(f"  {'-'*62}")

    for author in authors_to_test:
        items = by_author[author]
        X_val = np.array([item["x"] for item in items], dtype=np.float32)
        val_entries = [item["entry"] for item in items]

        # Multi-author model on this author's validation.
        Y_pred_multi = model_all.predict(X_val)
        multi_loss = sum(
            downstream_loss(labels_to_filters(Y_pred_multi[i]), e["filters"], DEFAULT_GRID)
            for i, e in enumerate(val_entries)
        ) / len(val_entries)

        # Single-author model.
        author_train = train_by_author[author]
        if len(author_train[0]) < 50:
            single_loss_str = "(too few train)"
            delta_str = ""
        else:
            X_tr = np.array(author_train[0], dtype=np.float32)
            Y_tr = np.array(author_train[1], dtype=np.float32)
            model_single = train_late_fusion(X_tr, Y_tr, alpha=0.3)
            Y_pred_single = model_single.predict(X_val)
            single_loss = sum(
                downstream_loss(labels_to_filters(Y_pred_single[i]), e["filters"], DEFAULT_GRID)
                for i, e in enumerate(val_entries)
            ) / len(val_entries)
            single_loss_str = f"{single_loss:13.2f} dB"
            delta = single_loss - multi_loss
            delta_str = f"{delta:+6.2f} dB"

        print(f"  {author:>12s} {len(items):4d} {len(author_train[0]):6d} {multi_loss:12.2f} dB {single_loss_str:>15s} {delta_str:>7s}")
