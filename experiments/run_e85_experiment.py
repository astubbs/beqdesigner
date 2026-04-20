#!/usr/bin/env python3
"""E85 experiment runner — differentiable DSP vs E82 baseline.

Usage:
    BEQ_WAV_CACHE=/path/to/beqdesigner/wav-cache \
      poetry run python3 scripts/run_e85_experiment.py
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# Path setup for running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s - %(message)s")
log = logging.getLogger("e85_experiment")

import numpy as np
from model.auto_beq import DEFAULT_GRID
from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
from model.auto_beq_nn import (
    AudioFeatureConfig, build_feature_vector, deduplicate_by_title,
    downstream_loss, labels_to_filters, train_production_weighted_hybrid,
)
from model.auto_beq_torch import E85TrainingConfig, train_e85_differentiable_dsp
from spike._auto_beq_helpers import STRATEGY_BLENDED_07, discover_wav_catalogue_pairs_cached
from spike.test_auto_beq_nn_real import _extract_features_parallel
from sklearn.model_selection import train_test_split
from model.auto_beq_catalogue import _fetch_or_cache

_FS = 1000


def main():
    cfg = AudioFeatureConfig()

    # --- Load data ---
    pairs = discover_wav_catalogue_pairs_cached()
    all_real = _extract_features_parallel(pairs, DEFAULT_GRID, _FS, strategy=STRATEGY_BLENDED_07)
    if len(all_real) < 200:
        log.error("only %d pairs, need ≥200", len(all_real))
        return

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

    X_train_list, entries_train = [], []
    for i in train_idx:
        pair, features = all_real[i]
        entry = pair["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_train_list.append(build_feature_vector(features, metadata, config=cfg))
        entries_train.append(entry)
    X_train = np.array(X_train_list, dtype=np.float32)

    X_test_list, entries_test = [], []
    for i in test_idx:
        pair, features = all_real[i]
        entry = pair["catalogue_entry"]
        if not entry.get("filters"):
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_test_list.append(build_feature_vector(features, metadata, config=cfg))
        entries_test.append(entry)
    X_test = np.array(X_test_list, dtype=np.float32)

    log.info("split: %d train / %d test", len(X_train), len(X_test))

    # --- Baseline (E82 XGBoost) ---
    train_samples = [(all_real[i][0]["catalogue_entry"], all_real[i][1]) for i in train_idx]
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    train_tmdb = {str(e.get("theMovieDB", "")).strip() for e, _ in train_samples}
    test_tmdb = {str(e.get("theMovieDB", "")).strip() for e in entries_test}
    synth_entries = [
        e for e in deduped
        if str(e.get("theMovieDB", "")).strip() not in train_tmdb | test_tmdb
    ]

    baseline_model, _ = train_production_weighted_hybrid(
        real_samples=train_samples, synth_entries=synth_entries,
        tmdb_cache=tmdb_cache, freqs_hz=DEFAULT_GRID, fs=_FS,
        real_weight=50.0, config=cfg,
    )
    Y_pred_base = baseline_model.predict(X_test)
    base_losses = [
        downstream_loss(labels_to_filters(Y_pred_base[i]), e["filters"], DEFAULT_GRID)
        for i, e in enumerate(entries_test)
    ]
    base_mean = float(np.mean(base_losses))
    base_max = float(np.max(base_losses))
    log.info("E82 baseline: mean=%.2f dB, max=%.2f dB", base_mean, base_max)

    # --- E85 differentiable DSP ---
    e85_config = E85TrainingConfig(
        n_epochs_warm=10, n_epochs_acoustic=20,
        batch_size=64, learning_rate_warm=1e-3, learning_rate_acoustic=5e-4,
        hidden_dim=256, n_hidden_layers=3, device="cpu",
    )
    predictor, stats = train_e85_differentiable_dsp(
        X_train=X_train, entries_train=entries_train,
        teacher_model=baseline_model, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=e85_config,
    )

    e85_losses = []
    for i, entry in enumerate(entries_test):
        filters = predictor.predict_filters(X_test[i])
        loss = downstream_loss(filters, entry["filters"], DEFAULT_GRID)
        e85_losses.append(loss)
    e85_mean = float(np.mean(e85_losses))
    e85_max = float(np.max(e85_losses))
    log.info("E85 diff-DSP: mean=%.2f dB, max=%.2f dB", e85_mean, e85_max)

    # Per-author
    by_auth_base: dict[str, list[float]] = defaultdict(list)
    by_auth_e85: dict[str, list[float]] = defaultdict(list)
    for e, bl, el in zip(entries_test, base_losses, e85_losses, strict=True):
        a = e.get("author", "unknown")
        by_auth_base[a].append(bl)
        by_auth_e85[a].append(el)

    delta = e85_mean - base_mean
    sign = "+" if delta > 0 else ""
    verdict = " ← BEATS baseline" if delta < -0.05 else " ← ties" if abs(delta) <= 0.05 else " ← regression"

    print(f"\n{'='*72}")
    print(f"  E85 — DIFFERENTIABLE DSP (acoustic loss, fixed LowShelf topology)")
    print(f"  Split: {len(X_train)} train / {len(X_test)} test")
    print(f"  Warm: {e85_config.n_epochs_warm} ep, Acoustic: {e85_config.n_epochs_acoustic} ep")
    print(f"{'='*72}\n")
    print(f"  {'Model':<25s}  {'mean dB':>10s}  {'max dB':>10s}")
    print(f"  {'-'*48}")
    print(f"  {'baseline (E82)':<25s}  {base_mean:8.2f}    {base_max:8.2f}")
    print(f"  {'E85 diff-DSP':<25s}  {e85_mean:8.2f}    {e85_max:8.2f}")
    print(f"  {'Δ mean':<25s}  {sign}{delta:8.2f}{verdict}")
    print()
    print(f"  Per-author:")
    for a in sorted(set(by_auth_base) | set(by_auth_e85)):
        b = float(np.mean(by_auth_base[a])) if a in by_auth_base else float("nan")
        e = float(np.mean(by_auth_e85[a])) if a in by_auth_e85 else float("nan")
        d = e - b
        print(f"    {a:<15s}  {b:6.2f}  →  {e:6.2f}    {d:+5.2f}")
    print()
    print(f"  Training time: {stats['train_time_s']:.1f}s")

    # CSV
    csv_path = Path(".pytest_cache") / "e85_diff_dsp.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "n_train", "n_test", "base_mean", "e85_mean", "delta", "train_time_s"])
        w.writerow([int(time.time()), len(X_train), len(X_test),
                    f"{base_mean:.3f}", f"{e85_mean:.3f}", f"{delta:+.3f}", stats["train_time_s"]])
    log.info("CSV row appended: %s", csv_path)


if __name__ == "__main__":
    main()
