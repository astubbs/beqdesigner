#!/usr/bin/env python3
"""E85 experiment runner - differentiable DSP vs E82 baseline.

Usage:
    BEQ_WAV_CACHE=/path/to/beqdesigner/wav-cache \
      poetry run python3 scripts/run_e85_experiment.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from experiment_base import (
    evaluate_model,
    load_experiment_data,
    setup_experiment_paths,
    setup_logging,
)

setup_experiment_paths()

import numpy as np
from model.auto_beq import DEFAULT_GRID
from model.auto_beq_nn import downstream_loss, labels_to_filters
from model.auto_beq_torch import E85TrainingConfig, train_e85_differentiable_dsp

_FS = 1000


def main():
    log = setup_logging("e85_experiment")
    ed = load_experiment_data(log)

    # --- Baseline (E82 XGBoost) ---
    base_mean, base_max, by_auth_base = evaluate_model(
        ed.teacher_model, ed.entries_test, ed.X_test,
        "E82-baseline", is_torch=False, log=log,
    )

    # --- E85 differentiable DSP ---
    e85_config = E85TrainingConfig(
        n_epochs_warm=10, n_epochs_acoustic=20,
        batch_size=64, learning_rate_warm=1e-3, learning_rate_acoustic=5e-4,
        hidden_dim=256, n_hidden_layers=3, device="cpu",
    )
    predictor, stats = train_e85_differentiable_dsp(
        X_train=ed.X_train, entries_train=ed.entries_train,
        teacher_model=ed.teacher_model, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=e85_config,
    )

    e85_mean, e85_max, by_auth_e85 = evaluate_model(
        predictor, ed.entries_test, ed.X_test,
        "E85-diff-DSP", is_torch=True, log=log,
    )

    # --- Report ---
    delta = e85_mean - base_mean
    sign = "+" if delta > 0 else ""
    verdict = " <- BEATS baseline" if delta < -0.05 else " <- ties" if abs(delta) <= 0.05 else " <- regression"

    print(f"\n{'='*72}")
    print(f"  E85 - DIFFERENTIABLE DSP (acoustic loss, fixed LowShelf topology)")
    print(f"  Split: {len(ed.X_train)} train / {len(ed.X_test)} test")
    print(f"  Warm: {e85_config.n_epochs_warm} ep, Acoustic: {e85_config.n_epochs_acoustic} ep")
    print(f"{'='*72}\n")
    print(f"  {'Model':<25s}  {'mean dB':>10s}  {'max dB':>10s}")
    print(f"  {'-'*48}")
    print(f"  {'baseline (E82)':<25s}  {base_mean:8.2f}    {base_max:8.2f}")
    print(f"  {'E85 diff-DSP':<25s}  {e85_mean:8.2f}    {e85_max:8.2f}")
    print(f"  {'delta mean':<25s}  {sign}{delta:8.2f}{verdict}")
    print()
    print(f"  Per-author:")
    for a in sorted(set(by_auth_base) | set(by_auth_e85)):
        b = by_auth_base.get(a, float("nan"))
        e = by_auth_e85.get(a, float("nan"))
        d = e - b
        print(f"    {a:<15s}  {b:6.2f}  ->  {e:6.2f}    {d:+5.2f}")
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
        w.writerow([int(time.time()), len(ed.X_train), len(ed.X_test),
                    f"{base_mean:.3f}", f"{e85_mean:.3f}", f"{delta:+.3f}", stats["train_time_s"]])
    log.info("CSV row appended: %s", csv_path)


if __name__ == "__main__":
    main()
