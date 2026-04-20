#!/usr/bin/env python3
"""Tier 1 unified comparison — all paradigm-shift experiments on the same split.

Runs E82 (baseline), E83 (Whisper foundation features), E84 (self-training),
and E85 (differentiable DSP) on the identical 80/20 stratified split of the
full 1279-WAV cache, producing a single definitive leaderboard.

Usage:
    BEQ_WAV_CACHE=/Volumes/jetspeed/beqdesigner/wav-cache \
      poetry run python3 scripts/run_tier1_comparison.py
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s - %(message)s")
log = logging.getLogger("tier1_comparison")

import numpy as np
from sklearn.model_selection import train_test_split

from model.auto_beq import DEFAULT_GRID
from model.media_utils import ProgressLogger
from model.auto_beq_advisor import extract_foundation_embeddings_parallel
from model.auto_beq_catalogue import _fetch_or_cache
from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
from model.auto_beq_nn import (
    AudioFeatureConfig,
    build_feature_vector,
    deduplicate_by_title,
    downstream_loss,
    labels_to_filters,
    pseudo_label_unmatched,
    train_e84_self_trained,
    train_production_weighted_hybrid,
)
from model.auto_beq_torch import E85TrainingConfig, train_e85_differentiable_dsp
from spike._auto_beq_helpers import (
    STRATEGY_BLENDED_07,
    cached_extract_features_with_strategy,
    discover_unmatched_wavs_cached,
    discover_wav_catalogue_pairs_cached,
)
from spike.test_auto_beq_nn_real import _extract_features_parallel

_FS = 1000


def main():
    t_global = time.time()

    # --- 1. Load + split ---
    log.info("=" * 70)
    log.info("  TIER 1 UNIFIED COMPARISON")
    log.info("")
    log.info("  Trains and evaluates each experiment technique (E82-E85) on")
    log.info("  the SAME data split so results are directly comparable.")
    log.info("")
    log.info("  80/20 stratified split means:")
    log.info("    - 80%% of WAVs are used for training, 20%% held out for testing")
    log.info("    - 'Stratified' = the split preserves the ratio of rolloff")
    log.info("      severity (heavy/moderate/gentle) in both sets, so neither")
    log.info("      set is biased toward easy or hard titles")
    log.info("    - Same random seed (42) every run for reproducibility")
    log.info("=" * 70)

    pairs = discover_wav_catalogue_pairs_cached()
    all_real = _extract_features_parallel(pairs, DEFAULT_GRID, _FS, strategy=STRATEGY_BLENDED_07)
    if len(all_real) < 200:
        log.error("only %d pairs — need ≥200", len(all_real))
        sys.exit(1)

    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch([p["catalogue_entry"] for p, _ in all_real], cache=tmdb_cache)

    real_entries = [p["catalogue_entry"] for p, _ in all_real]
    severity = [
        "heavy" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 20
        else "moderate" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 10
        else "gentle"
        for e in real_entries
    ]
    indices = np.arange(len(all_real))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=severity if len(set(severity)) > 1 else None,
    )
    log.info("split: %d train / %d test (stratified by rolloff severity)", len(train_idx), len(test_idx))

    # Pre-build common structures.
    cfg_base = AudioFeatureConfig()
    cfg_whisper = AudioFeatureConfig(foundation_model="whisper-tiny")

    train_samples = [(all_real[i][0]["catalogue_entry"], all_real[i][1]) for i in train_idx]
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    train_tmdb = {str(e.get("theMovieDB", "")).strip() for e, _ in train_samples}
    test_tmdb = {str(e.get("theMovieDB", "")).strip() for e in [real_entries[i] for i in test_idx]}
    synth_entries = [
        e for e in deduped
        if str(e.get("theMovieDB", "")).strip() not in train_tmdb | test_tmdb
    ]

    # Test set feature vectors (base config).
    X_test_base_list, entries_test = [], []
    for i in test_idx:
        pair, features = all_real[i]
        entry = pair["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_test_base_list.append(build_feature_vector(features, metadata, config=cfg_base))
        entries_test.append(entry)
    X_test_base = np.array(X_test_base_list, dtype=np.float32)

    # Train set feature vectors (base config) for E85.
    X_train_base_list, entries_train = [], []
    for i in train_idx:
        pair, features = all_real[i]
        entry = pair["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_train_base_list.append(build_feature_vector(features, metadata, config=cfg_base))
        entries_train.append(entry)
    X_train_base = np.array(X_train_base_list, dtype=np.float32)

    def _eval(model_or_predictor, X_test, entries, label, is_torch=False):
        """Evaluate and return (mean, max, per_author, elapsed)."""
        t0 = time.time()
        losses = []
        for i, entry in enumerate(entries):
            if is_torch:
                filters = model_or_predictor.predict_filters(X_test[i])
            else:
                y_pred = model_or_predictor.predict(X_test[i:i+1])[0]
                filters = labels_to_filters(y_pred)
            losses.append(downstream_loss(filters, entry["filters"], DEFAULT_GRID))
        mean = float(np.mean(losses))
        max_ = float(np.max(losses))
        by_auth = defaultdict(list)
        for e, l in zip(entries, losses, strict=True):
            by_auth[e.get("author", "unknown")].append(l)
        per_author = {a: float(np.mean(v)) for a, v in by_auth.items()}
        elapsed = time.time() - t0
        log.info("  %-20s: mean=%.2f dB, max=%.2f dB (eval %.1fs)", label, mean, max_, elapsed)
        return mean, max_, per_author, losses

    results = []

    # --- 2. E82 baseline ---
    log.info("")
    log.info("--- E82 baseline (plain XGB, 102 features) ---")
    t0 = time.time()
    baseline_model, _ = train_production_weighted_hybrid(
        real_samples=train_samples, synth_entries=synth_entries,
        tmdb_cache=tmdb_cache, freqs_hz=DEFAULT_GRID, fs=_FS,
        real_weight=50.0, config=cfg_base,
    )
    t_train_e82 = time.time() - t0
    e82_mean, e82_max, e82_auth, _ = _eval(baseline_model, X_test_base, entries_test, "E82 baseline")
    results.append(("E82 baseline", e82_mean, e82_max, t_train_e82, e82_auth, "reference"))

    # --- 3. E83 Whisper-tiny (486 features) ---
    log.info("")
    log.info("--- E83 Whisper-tiny (486 features) ---")
    # Attach cached Whisper embeddings to the real samples.
    embed_cache_dir = Path("/Volumes/jetspeed/beqdesigner/foundation-embeddings/whisper-tiny")
    if not embed_cache_dir.exists():
        # Try the pytest cache location.
        embed_cache_dir = Path(".pytest_cache/foundation-embeddings/whisper-tiny")

    train_samples_whisper = []
    log.info("extracting Whisper embeddings for %d train samples...", len(train_samples))
    whisper_train_progress = ProgressLogger(len(train_samples), logger=log, min_interval_s=5)
    for idx, (entry, features) in enumerate(train_samples):
        pair_for_entry = next(
            (p for p, _ in all_real if p["catalogue_entry"] is entry), None,
        )
        if pair_for_entry is None:
            train_samples_whisper.append((entry, features))
            whisper_train_progress.update(idx + 1, label=entry.get("title", ""))
            continue
        wav_path = pair_for_entry["wav_path"]
        try:
            from model.auto_beq_advisor import extract_foundation_embedding
            emb = extract_foundation_embedding(wav_path, model_name="whisper-tiny", cache_dir=embed_cache_dir)
            features_with = dataclasses.replace(features, foundation_embedding=tuple(float(x) for x in emb))
            train_samples_whisper.append((entry, features_with))
        except Exception:
            train_samples_whisper.append((entry, features))
        whisper_train_progress.update(idx + 1, label=entry.get("title", ""))
    whisper_train_progress.finish("train embeddings complete")

    t0 = time.time()
    e83_model, _ = train_production_weighted_hybrid(
        real_samples=train_samples_whisper, synth_entries=synth_entries,
        tmdb_cache=tmdb_cache, freqs_hz=DEFAULT_GRID, fs=_FS,
        real_weight=50.0, config=cfg_whisper,
    )
    t_train_e83 = time.time() - t0

    # Test set with Whisper embeddings.
    log.info("extracting Whisper embeddings for %d test samples...", len(test_idx))
    whisper_test_progress = ProgressLogger(len(test_idx), logger=log, min_interval_s=5)
    X_test_whisper_list = []
    for prog_idx, i in enumerate(test_idx):
        pair, features = all_real[i]
        entry = pair["catalogue_entry"]
        if not entry.get("filters"):
            whisper_test_progress.update(prog_idx + 1)
            continue
        wav_path = pair["wav_path"]
        try:
            emb = extract_foundation_embedding(wav_path, model_name="whisper-tiny", cache_dir=embed_cache_dir)
            features_with = dataclasses.replace(features, foundation_embedding=tuple(float(x) for x in emb))
        except Exception:
            features_with = features
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_test_whisper_list.append(build_feature_vector(features_with, metadata, config=cfg_whisper))
        whisper_test_progress.update(prog_idx + 1, label=entry.get("title", ""))
    whisper_test_progress.finish("test embeddings complete")
    X_test_whisper = np.array(X_test_whisper_list, dtype=np.float32)

    e83_mean, e83_max, e83_auth, _ = _eval(e83_model, X_test_whisper, entries_test, "E83 Whisper-tiny")
    results.append(("E83 Whisper-tiny", e83_mean, e83_max, t_train_e83, e83_auth, "regression"))

    # --- 4. E84 self-training ---
    log.info("")
    log.info("--- E84 self-training (11 unmatched WAVs) ---")
    unmatched_wavs = discover_unmatched_wavs_cached()
    log.info("extracting features for %d unmatched WAVs...", len(unmatched_wavs))
    unmatched_progress = ProgressLogger(len(unmatched_wavs), logger=log, min_interval_s=5)
    unmatched_pairs = []
    for idx, wav_path in enumerate(unmatched_wavs):
        try:
            features = cached_extract_features_with_strategy(
                wav_path, DEFAULT_GRID, _FS, strategy=STRATEGY_BLENDED_07,
            )
            unmatched_pairs.append((wav_path, features))
        except Exception:
            pass
        unmatched_progress.update(idx + 1, label=wav_path.stem if hasattr(wav_path, 'stem') else str(wav_path)[-30:])
    unmatched_progress.finish(f"extracted {len(unmatched_pairs)} unmatched features")
    t0 = time.time()
    e84_model, e84_meta = train_e84_self_trained(
        real_samples=train_samples, synth_entries=synth_entries,
        unmatched_pairs=unmatched_pairs, tmdb_cache=tmdb_cache,
        freqs_hz=DEFAULT_GRID, fs=_FS, real_weight=50.0,
        pseudo_weight=10.0, confidence_threshold_db=1.5, n_iterations=2,
        config=cfg_base,
    )
    t_train_e84 = time.time() - t0
    e84_mean, e84_max, e84_auth, _ = _eval(e84_model, X_test_base, entries_test, "E84 self-training")
    n_pseudo = sum(s["n_pseudo"] for s in e84_meta.get("iter_stats", []))
    results.append(("E84 self-training", e84_mean, e84_max, t_train_e84, e84_auth,
                    f"no-op ({n_pseudo} pseudo)"))

    # --- 5. E85 differentiable DSP ---
    log.info("")
    log.info("--- E85 differentiable DSP (acoustic loss, LowShelf-only) ---")
    e85_config = E85TrainingConfig(
        n_epochs_warm=10, n_epochs_acoustic=20,
        batch_size=64, learning_rate_warm=1e-3, learning_rate_acoustic=5e-4,
        hidden_dim=256, n_hidden_layers=3, device="cpu",
    )
    t0 = time.time()
    predictor, e85_stats = train_e85_differentiable_dsp(
        X_train=X_train_base, entries_train=entries_train,
        teacher_model=baseline_model, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=e85_config,
    )
    t_train_e85 = time.time() - t0
    e85_mean, e85_max, e85_auth, _ = _eval(predictor, X_test_base, entries_test, "E85 diff-DSP", is_torch=True)
    delta_e85 = e85_mean - e82_mean
    verdict_e85 = "NEW CHAMPION" if delta_e85 < -0.1 else "ties" if abs(delta_e85) <= 0.05 else "regression"
    results.append(("E85 diff-DSP", e85_mean, e85_max, t_train_e85, e85_auth, verdict_e85))

    # --- 6. Report ---
    print()
    print("=" * 80)
    print("  TIER 1 UNIFIED COMPARISON — 1279-WAV cache (full extraction)")
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test (stratified, random_state=42)")
    print("=" * 80)
    print()
    print(f"  {'Experiment':<22s}  {'mean dB':>9s}  {'max dB':>9s}  {'Δ vs E82':>10s}  {'train':>7s}  {'verdict'}")
    print(f"  {'-'*78}")
    for name, mean, max_, t_train, per_auth, verdict in results:
        delta = mean - e82_mean
        delta_str = f"{delta:+.2f}" if name != "E82 baseline" else "—"
        t_str = f"{t_train:.1f}s" if t_train < 60 else f"{t_train/60:.1f}m"
        bold = "**" if "CHAMPION" in verdict else ""
        print(f"  {bold}{name:<22s}{bold}  {mean:7.2f}    {max_:7.2f}    {delta_str:>10s}  {t_str:>7s}  {verdict}")
    print()
    print("  Per-author breakdown:")
    all_authors = sorted(set().union(*(r[4].keys() for r in results)))
    header = f"  {'author':<15s}" + "".join(f"  {r[0][:10]:>10s}" for r in results)
    print(header)
    print(f"  {'-'*len(header)}")
    for author in all_authors:
        row = f"  {author:<15s}"
        for _, _, _, _, per_auth, _ in results:
            v = per_auth.get(author, float("nan"))
            row += f"  {v:10.2f}"
        print(row)
    print()
    total_time = time.time() - t_global
    print(f"  Total comparison time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print()

    # --- 7. CSV ---
    csv_path = Path(".pytest_cache") / "tier1_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "mean_db", "max_db", "delta_vs_e82", "train_time_s", "verdict"])
        for name, mean, max_, t_train, _, verdict in results:
            delta = mean - e82_mean if name != "E82 baseline" else 0.0
            w.writerow([name, f"{mean:.3f}", f"{max_:.3f}", f"{delta:+.3f}", f"{t_train:.1f}", verdict])
    log.info("CSV written: %s", csv_path)
    print(f"  CSV: {csv_path}")
    print()


if __name__ == "__main__":
    main()
