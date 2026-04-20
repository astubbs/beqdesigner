#!/usr/bin/env python3
"""Train the E85 differentiable-DSP production model.

Two-stage training: MSE warm-start from the E82 XGBoost teacher, then
acoustic-loss fine-tuning through a differentiable biquad layer. The
saved ``.pt`` file is compatible with ``TorchFilterAdvisor`` —  load
via ``AUTO_BEQ_ADVISOR=torch_differentiable`` or let ``get_advisor()``
discover it at ``{beq-dir}/e85_torch_filter.pt``.

Usage::

    # Train using the existing E82 production model as teacher:
    BEQ_WAV_CACHE=/Volumes/jetspeed/beqdesigner/wav-cache \\
      bin/beq-designer train-torch

    # Override hyperparams:
    bin/beq-designer train-torch \\
      --warm-epochs 15 --acoustic-epochs 30 --hidden-dim 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# NOTE: logging.basicConfig is called inside main(), not at module level.
# Module-level basicConfig adds handlers in ProcessPoolExecutor children.
log = logging.getLogger("train_torch_model")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train the E85 differentiable-DSP production model.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output path (default: {beq-dir}/e85_torch_filter.pt).",
    )
    parser.add_argument("--warm-epochs", type=int, default=10)
    parser.add_argument("--acoustic-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-warm", type=float, default=1e-3)
    parser.add_argument("--lr-acoustic", type=float, default=5e-4)
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
        )

    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_metadata import enrich_media_metadata
    from model.auto_beq_nn import (
        AudioFeatureConfig, build_feature_vector,
        train_production_weighted_hybrid,
    )
    from model.auto_beq_torch import (
        E85TrainingConfig, save_torch_predictor, train_e85_differentiable_dsp,
    )
    from spike._auto_beq_helpers import beq_shared_dir, prepare_training_data

    target_dir = beq_shared_dir()
    output_path = args.output or (target_dir / "e85_torch_filter.pt")
    meta_path = output_path.with_suffix(".meta.json")
    log.info("BEQ working directory: %s", target_dir)
    log.info("model output path:     %s", output_path)

    # --- Discover + extract + catalogue + metadata ---
    data = prepare_training_data(min_pairs=1)
    all_real = data["all_real"]
    deduped = data["deduped"]
    tmdb_cache = data["tmdb_cache"]
    real_samples = data["real_samples"]

    cfg = AudioFeatureConfig()

    # Build full training set (no hold-out - production uses ALL data).
    import numpy as np
    X_train_list, entries_train = [], []
    for entry, features in real_samples:
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_train_list.append(build_feature_vector(features, metadata, config=cfg))
        entries_train.append(entry)

    X_train = np.array(X_train_list, dtype=np.float32)
    log.info("training set: %d real samples, %d features", len(X_train), X_train.shape[1])

    # --- Train E82 teacher (needed for warm-start) ---
    log.info("training E82 XGBoost teacher for warm-start…")
    teacher_model, _ = train_production_weighted_hybrid(
        real_samples=real_samples,
        synth_entries=deduped,
        tmdb_cache=tmdb_cache,
        freqs_hz=DEFAULT_GRID,
        fs=1000,
        real_weight=50.0,
        config=cfg,
    )

    # --- Train E85 ---
    e85_config = E85TrainingConfig(
        n_epochs_warm=args.warm_epochs,
        n_epochs_acoustic=args.acoustic_epochs,
        batch_size=args.batch_size,
        learning_rate_warm=args.lr_warm,
        learning_rate_acoustic=args.lr_acoustic,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        device="cpu",
        cosine_annealing=True,  # E87: cosine LR schedule → 1.33 dB
    )
    predictor, stats = train_e85_differentiable_dsp(
        X_train=X_train,
        entries_train=entries_train,
        teacher_model=teacher_model,
        eval_freqs_hz=DEFAULT_GRID,
        fs=1000,
        config=e85_config,
    )

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_predictor(predictor, str(output_path))
    metadata = {
        **stats,
        "trained_at": int(time.time()),
        "n_real": len(X_train),
        "n_features": int(X_train.shape[1]),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log.info("metadata sidecar: %s", meta_path)

    # --- Summary ---
    print()
    print("=" * 70)
    print("  E85 TORCH MODEL TRAINED")
    print("=" * 70)
    print(f"  Model:              {output_path}")
    print(f"  Metadata:           {meta_path}")
    print(f"  Real samples:       {len(X_train)}")
    print(f"  Features:           {X_train.shape[1]}")
    print(f"  Warm epochs:        {e85_config.n_epochs_warm}")
    print(f"  Acoustic epochs:    {e85_config.n_epochs_acoustic}")
    print(f"  Training time:      {stats['train_time_s']} s")
    file_size_kb = output_path.stat().st_size / 1024
    print(f"  Model file size:    {file_size_kb:.1f} KB")
    print("=" * 70)
    print()
    print("To use this model in production:")
    print(f"  AUTO_BEQ_ADVISOR=torch_differentiable \\")
    print(f"  bin/beq-designer profile --media ...")
    print()


if __name__ == "__main__":
    main()
