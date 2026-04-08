"""Experiment 19: NN training with chunked/blended audio features.

Same XGBoost pipeline as E18 (test_auto_beq_nn_real.py), but replaces
the Welch-only real-audio feature extraction with the chunked/blended
strategies from E18b. Runs all four strategies side-by-side so we can
compare downstream loss and verdict changes.

Hypothesis: the NN may benefit from chunked/blended extraction because
it captures transient bass events that Welch dilutes, giving the model
more accurate input features at validation time.

Skipped if no WAV files are available in the audio cache.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from model.auto_beq import DEFAULT_GRID, compute_match_metrics, evaluate_filter_chain
from model.auto_beq_metadata import enrich_media_metadata
from model.auto_beq_nn import (
    build_feature_vector,
    downstream_loss,
    labels_to_filters,
    save_model,
    train_xgboost,
)

from spike._auto_beq_helpers import (
    STRATEGY_BLENDED_03,
    STRATEGY_BLENDED_07,
    STRATEGY_CHUNKED_P90,
    STRATEGY_WELCH,
    build_training_dataset,
    discover_wav_catalogue_pairs,
    extract_features_with_strategy,
    synthetic_features,
)

log = logging.getLogger("auto_beq_nn_chunked")

_DEFAULT_FS = 1000

# Strategies to compare: Welch baseline + the three best from E18b.
_STRATEGIES = [
    STRATEGY_WELCH,
    STRATEGY_BLENDED_07,
    STRATEGY_BLENDED_03,
    STRATEGY_CHUNKED_P90,
]

_PAIRS = discover_wav_catalogue_pairs()


def _verdict_rank(verdict: str) -> int:
    """PASS=0, MARGINAL=1, FAIL=2 — lower is better."""
    return {"PASS": 0, "MARGINAL": 1, "FAIL": 2}.get(verdict, 3)


@pytest.mark.skipif(not _PAIRS, reason="no WAV files matched to catalogue entries")
def test_nn_chunked_strategy_comparison(tmp_path):
    """E19: Train on full catalogue, validate with multiple extraction strategies.

    Trains one XGBoost model on synthetic features (same as E18), then
    builds separate validation feature sets for each extraction strategy
    (Welch, blend-a0.7, blend-a0.3, chunked-P90). Compares downstream
    loss and verdict across strategies to see which extraction method
    gives the NN the best real-audio inputs.
    """
    log.info("=== E19: NN + chunked audio features ===")

    # --- 1. Build training dataset (catalogue + TMDb cache) ---
    X_all, Y_all, entries_all, tmdb_cache = build_training_dataset(DEFAULT_GRID)

    # --- 2. Held-out split: remove real-audio titles from training ---
    real_tmdb_ids = {p["tmdb_id"] for p in _PAIRS}
    log.info("real-audio validation titles: %d", len(_PAIRS))
    for p in _PAIRS:
        log.info("  %s (tmdb=%s)", p["catalogue_entry"]["title"], p["tmdb_id"])

    train_mask = np.array([
        str(e.get("theMovieDB", "")).strip() not in real_tmdb_ids
        for e in entries_all
    ])
    X_train = X_all[train_mask]
    Y_train = Y_all[train_mask]
    log.info("training set (excluding real-audio titles): %d", len(X_train))

    # --- 3. Train (once — same model for all strategies) ---
    log.info("training XGBoost on %d synthetic entries...", len(X_train))
    model = train_xgboost(X_train, Y_train)

    model_path = str(tmp_path / "e19_nn_chunked_model.joblib")
    save_model(model, model_path)

    # --- 4. Feature importances ---
    importances = model.feature_importances_
    audio_imp = importances[:9].sum()
    meta_imp = importances[9:].sum()
    log.info("feature importances: audio=%.3f metadata=%.3f", audio_imp, meta_imp)

    feature_names = (
        [f"audio_{hz}Hz" for hz in [20, 25, 30, 35, 40, 50, 60, 70, 80]]
        + ["year", "fmt_atmos", "fmt_truehd", "fmt_dtshd", "fmt_ddatmos", "fmt_dd", "fmt_other"]
        + ["src_disc", "src_stream", "src_unk"]
        + [f"studio_{i}" for i in range(16)]
        + [f"mixer_{i}" for i in range(8)]
        + [f"genre_{i}" for i in range(10)]
        + [f"country_{i}" for i in range(5)]
        + ["runtime", "rating"]
    )
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n=== Top 10 features by importance ===")
    for i in top_idx:
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {name:20s} {importances[i]:.4f}")

    # --- 5. Evaluate each strategy on real-audio validation ---
    strategy_results: dict[str, list[dict]] = {}
    val_entries: list[dict] = []  # populated on first strategy pass

    for strategy in _STRATEGIES:
        label = strategy.label
        log.info("--- Evaluating strategy: %s ---", label)

        X_val, Y_val, strat_entries = [], [], []
        for p in _PAIRS:
            entry = p["catalogue_entry"]
            wav_path = p["wav_path"]
            log.info("  extracting [%s]: %s", label, wav_path.name[:60])
            from model.auto_beq_nn import catalogue_entry_to_labels
            features = extract_features_with_strategy(
                wav_path, DEFAULT_GRID, _DEFAULT_FS, strategy,
            )
            metadata = enrich_media_metadata(entry, tmdb_cache)
            X_val.append(build_feature_vector(features, metadata))
            Y_val.append(catalogue_entry_to_labels(entry))
            strat_entries.append(entry)

        if not val_entries:
            val_entries = strat_entries

        X_val = np.array(X_val, dtype=np.float32)
        Y_pred = model.predict(X_val)

        results = []
        for i, entry in enumerate(strat_entries):
            pred_filters = labels_to_filters(Y_pred[i])
            target_filters = entry["filters"]
            loss = downstream_loss(pred_filters, target_filters, DEFAULT_GRID)
            correction = evaluate_filter_chain(target_filters, DEFAULT_GRID, fs=_DEFAULT_FS)
            target_curve = -correction
            metrics = compute_match_metrics(
                target_curve, pred_filters, DEFAULT_GRID, fs=_DEFAULT_FS,
            )
            results.append({
                "title": entry["title"],
                "loss": loss,
                "verdict": metrics.verdict,
                "mean_err": metrics.mean_abs_err_db,
                "max_err": metrics.max_abs_err_db,
                "filters": pred_filters,
            })
        strategy_results[label] = results

    # --- 6. Print comparison table ---
    print(f"\n{'=' * 100}")
    print("=== E19: Strategy comparison — NN validation on real audio ===")
    print(f"{'=' * 100}")

    titles = [r["title"] for r in strategy_results[_STRATEGIES[0].label]]
    header_labels = [s.label for s in _STRATEGIES]
    print(f"\n{'Title':40s}", end="")
    for lbl in header_labels:
        print(f" | {lbl:>20s}", end="")
    print()
    print("-" * (40 + 23 * len(header_labels)))

    for ti, title in enumerate(titles):
        short = title[:40]
        print(f"{short:40s}", end="")
        for lbl in header_labels:
            r = strategy_results[lbl][ti]
            print(f" | {r['loss']:6.2f} dB {r['verdict']:>8s}", end="")
        print()

    # Summary per strategy.
    print(f"\n{'Strategy':25s} {'Mean loss':>10s} {'PASS':>6s} {'MARGINAL':>9s} {'FAIL':>6s}")
    print("-" * 60)
    welch_results = strategy_results[STRATEGY_WELCH.label]
    for strategy in _STRATEGIES:
        label = strategy.label
        results = strategy_results[label]
        mean_loss = np.mean([r["loss"] for r in results])
        n_pass = sum(1 for r in results if r["verdict"] == "PASS")
        n_marg = sum(1 for r in results if r["verdict"] == "MARGINAL")
        n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
        print(f"  {label:23s} {mean_loss:10.2f} {n_pass:6d} {n_marg:9d} {n_fail:6d}")

    # Delta vs Welch baseline.
    print(f"\n{'Strategy':25s} {'Δ loss vs Welch':>15s} {'Grade improved':>15s} {'Grade degraded':>15s}")
    print("-" * 75)
    for strategy in _STRATEGIES[1:]:
        label = strategy.label
        results = strategy_results[label]
        delta_loss = np.mean([r["loss"] for r in results]) - np.mean([r["loss"] for r in welch_results])
        improved = sum(
            1 for r, w in zip(results, welch_results)
            if _verdict_rank(r["verdict"]) < _verdict_rank(w["verdict"])
        )
        degraded = sum(
            1 for r, w in zip(results, welch_results)
            if _verdict_rank(r["verdict"]) > _verdict_rank(w["verdict"])
        )
        print(f"  {label:23s} {delta_loss:+15.2f} {improved:15d} {degraded:15d}")

    # --- 7. Synthetic sanity check ---
    print("\n=== Synthetic features sanity check ===")
    X_synth_val = []
    for entry in val_entries:
        features = synthetic_features(entry, DEFAULT_GRID)
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

    welch_mean = np.mean([r["loss"] for r in welch_results])
    best_strat = min(_STRATEGIES[1:], key=lambda s: np.mean([r["loss"] for r in strategy_results[s.label]]))
    best_mean = np.mean([r["loss"] for r in strategy_results[best_strat.label]])

    print(f"  Synthetic mean loss:           {synth_mean:.2f} dB")
    print(f"  Welch real-audio mean loss:     {welch_mean:.2f} dB")
    print(f"  Best strategy ({best_strat.label}): {best_mean:.2f} dB")
    print(f"  Gap (Welch - synthetic):        {welch_mean - synth_mean:.2f} dB")
    print(f"  Gap (best - synthetic):         {best_mean - synth_mean:.2f} dB")
    if best_mean < welch_mean:
        print(f"  → Chunked extraction CLOSES the real-vs-synthetic gap by {welch_mean - best_mean:.2f} dB")
    else:
        print("  → Chunked extraction does NOT improve over Welch for NN validation")
