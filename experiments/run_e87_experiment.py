#!/usr/bin/env python3
"""E87: more epochs + cosine annealing LR schedule.

Tests E85 (LowShelf-only) and E86 (multi-type) at 20 vs 100 acoustic
epochs, with cosine annealing on the 100-epoch variants.

Usage:
    poetry run python3 experiments/run_e87_experiment.py
"""
from __future__ import annotations

from experiment_base import (
    evaluate_model,
    load_experiment_data,
    setup_experiment_paths,
    setup_logging,
)

setup_experiment_paths()

from model.auto_beq import DEFAULT_GRID
from model.auto_beq_torch import E85TrainingConfig, train_e85_differentiable_dsp

_FS = 1000


def main():
    log = setup_logging("e87")
    ed = load_experiment_data(log)

    # Baseline E82.
    base_m, base_mx, base_pa = evaluate_model(
        ed.teacher_model, ed.entries_test, ed.X_test,
        "E82-baseline", is_torch=False, log=log,
    )

    # Experiment matrix: (label, multi_type, n_acoustic, cosine).
    configs = [
        ("E85 LS 20ep", False, 20, False),
        ("E85 LS 100ep+cosine", False, 100, True),
        ("E86 multi 20ep", True, 20, False),
        ("E86 multi 100ep+cosine", True, 100, True),
    ]

    results = [("E82 baseline", base_m, base_mx, base_pa)]
    for label, multi, n_ac, cosine in configs:
        log.info("")
        log.info("--- %s ---", label)
        p, _ = train_e85_differentiable_dsp(
            X_train=ed.X_train, entries_train=ed.entries_train,
            teacher_model=ed.teacher_model, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
            config=E85TrainingConfig(
                n_epochs_warm=10, n_epochs_acoustic=n_ac,
                multi_type=multi, cosine_annealing=cosine,
            ),
        )
        m, mx, pa = evaluate_model(
            p, ed.entries_test, ed.X_test,
            label, is_torch=True, log=log,
        )
        results.append((label, m, mx, pa))

    # Report.
    print(f"\n{'=' * 72}")
    print(f"  E87 - EPOCHS + LR SCHEDULE COMPARISON")
    print(f"  {len(ed.X_train)} train / {len(ed.X_test)} test")
    print(f"{'=' * 72}\n")
    print(f"  {'Model':<28s}  {'mean':>8s}  {'max':>8s}  {'delta mean':>10s}")
    print(f"  {'-' * 56}")
    for nm, m, mx, _ in results:
        d = m - base_m
        ds = f"{d:+.2f}" if nm != "E82 baseline" else "-"
        print(f"  {nm:<28s}  {m:6.2f}    {mx:6.2f}    {ds:>10s}")
    print(f"\n  Per-author:")
    all_authors = sorted(set().union(*(r[3].keys() for r in results)))
    for a in all_authors:
        vals = "  ".join(f"{r[0][:8]}={r[3].get(a, 0):.2f}" for r in results)
        print(f"    {a:<15s}  {vals}")
    print()


if __name__ == "__main__":
    main()
