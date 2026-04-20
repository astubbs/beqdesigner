#!/usr/bin/env python3
"""E86: multi-type topology experiment (LowShelf + HighShelf + PeakingEQ).

Usage:
    BEQ_WAV_CACHE=/path/to/beqdesigner/wav-cache \
      poetry run python3 scripts/run_e86_experiment.py
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
    log = setup_logging("e86")
    ed = load_experiment_data(log)

    # Baseline E82.
    base_m, base_mx, base_pa = evaluate_model(
        ed.teacher_model, ed.entries_test, ed.X_test,
        "E82-baseline", is_torch=False, log=log,
    )

    # E85 (v1 LowShelf-only).
    p85, _ = train_e85_differentiable_dsp(
        X_train=ed.X_train, entries_train=ed.entries_train,
        teacher_model=ed.teacher_model, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=E85TrainingConfig(n_epochs_warm=10, n_epochs_acoustic=20),
    )
    e85_m, e85_mx, e85_pa = evaluate_model(
        p85, ed.entries_test, ed.X_test,
        "E85-LowShelf-only", is_torch=True, log=log,
    )

    # E86 (v2 multi-type).
    p86, _ = train_e85_differentiable_dsp(
        X_train=ed.X_train, entries_train=ed.entries_train,
        teacher_model=ed.teacher_model, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=E85TrainingConfig(n_epochs_warm=10, n_epochs_acoustic=20, multi_type=True),
    )
    e86_m, e86_mx, e86_pa = evaluate_model(
        p86, ed.entries_test, ed.X_test,
        "E86-multi-type", is_torch=True, log=log,
    )

    # Report.
    print(f"\n{'=' * 72}")
    print(f"  E86 MULTI-TYPE TOPOLOGY COMPARISON")
    print(f"  {len(ed.X_train)} train / {len(ed.X_test)} test")
    print(f"{'=' * 72}\n")
    print(f"  {'Model':<22s}  {'mean':>8s}  {'max':>8s}  {'delta mean':>10s}")
    print(f"  {'-' * 50}")
    for nm, m, mx in [
        ("E82 baseline", base_m, base_mx),
        ("E85 LowShelf-only", e85_m, e85_mx),
        ("E86 multi-type", e86_m, e86_mx),
    ]:
        d = m - base_m
        ds = f"{d:+.2f}" if nm != "E82 baseline" else "-"
        print(f"  {nm:<22s}  {m:6.2f}    {mx:6.2f}    {ds:>10s}")
    print(f"\n  Per-author:")
    for a in sorted(set(base_pa) | set(e85_pa) | set(e86_pa)):
        print(
            f"    {a:<15s}  E82={base_pa.get(a, 0):.2f}  "
            f"E85={e85_pa.get(a, 0):.2f}  E86={e86_pa.get(a, 0):.2f}"
        )
    print()


if __name__ == "__main__":
    main()
