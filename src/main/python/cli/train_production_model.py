#!/usr/bin/env python3
"""Train the production 50:1 weighted hybrid BEQ model (E82).

Builds the real-audio training set from the current WAV cache, pairs
each WAV with its catalogue entry, and combines with synthetic entries
for every catalogue title that doesn't have a matching real WAV.  Then
trains plain XGBoost with a ``sample_weight`` array that gives real
samples 50× the weight of synthetic samples.

On the 219-title test split (1091-WAV cache), this config matches
real-only plain XGB at **1.99 dB mean** — while retaining synthetic
coverage for the ~7k catalogue titles without real WAVs.  It's the
current production model per E82.

The saved model is compatible with ``TrainedModelAdvisor`` — load it
via ``AUTO_BEQ_ADVISOR=trained_model AUTO_BEQ_MODEL_PATH=<path>`` or
let ``get_advisor()`` discover it at the default location.

Usage::

    # Uses default WAV cache (from BEQ_WAV_CACHE env var or settings.json)
    bin/beq-designer train-production

    # Override WAV cache location (e.g. NAS mount)
    BEQ_WAV_CACHE=/Volumes/jetspeed/beqdesigner/wav-cache \\
      bin/beq-designer train-production

    # Experiment with a different real:synth weight ratio
    bin/beq-designer train-production --real-weight 20

Outputs to ``{beq-dir}/production_model.joblib`` plus a
``{beq-dir}/production_model.meta.json`` sidecar with provenance
(training time, n_real, n_synth, real_weight, xgb_params, WAV cache
mtime at training time).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# NOTE: logging.basicConfig is called inside main(), not at module level.
# Module-level basicConfig would add handlers in every ProcessPoolExecutor
# child that imports this module, causing log spam.
log = logging.getLogger("train_production_model")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train the production 50:1 weighted hybrid BEQ model (E82)",
    )
    parser.add_argument(
        "--real-weight", type=float, default=50.0,
        help="Per-sample weight for real samples vs 1.0 for synthetic "
             "(default: 50, matches E82 winning config).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output path for the model .joblib file "
             "(default: {beq-dir}/production_model.joblib).",
    )
    parser.add_argument(
        "--extraction-strategy", choices=["welch", "blend_07"], default="blend_07",
        help="Audio feature extraction strategy for real WAVs "
             "(default: blend_07 = blended Welch+chunked P90, matches E82).",
    )
    parser.add_argument(
        "--foundation-model", default=None,
        choices=[None, "whisper-tiny", "whisper-base", "whisper-small", "mock-16"],
        help="Optional audio foundation model for additional features "
             "(E83/T1.1). Default: None (E82 baseline). Values: whisper-tiny "
             "(384 dims), whisper-base (512), whisper-small (768), mock-16 "
             "(deterministic synthetic, for smoke tests).",
    )
    parser.add_argument(
        "--self-train", action="store_true",
        help="Enable E84/T1.3 semi-supervised self-training. Uses the "
             "unmatched real WAVs in the cache as unlabelled data; the "
             "trained teacher pseudo-labels them with a confidence filter "
             "and retrains for N iterations. Output goes to "
             "`production_model_e84.joblib` to keep E82 untouched.",
    )
    parser.add_argument(
        "--pseudo-weight", type=float, default=10.0,
        help="E84: per-sample weight for pseudo-labels (real=50, synth=1). "
             "Default: 10. Only used when --self-train is set.",
    )
    parser.add_argument(
        "--confidence-threshold-db", type=float, default=1.5,
        help="E84: maximum self-consistency mean abs error (dB) to accept "
             "a pseudo-label. Tighter = fewer, higher-quality pseudo-labels. "
             "Default: 1.5. Only used when --self-train is set.",
    )
    parser.add_argument(
        "--self-train-iterations", type=int, default=2,
        help="E84: number of self-training iterations after the baseline. "
             "Default: 2. Only used when --self-train is set.",
    )
    args = parser.parse_args(argv)

    # Configure logging only when called as main entry point, not when
    # imported as a module by ProcessPoolExecutor children.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
        )

    # Late imports so --help is fast.
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_advisor import extract_foundation_embedding
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import fetch_metadata_batch, load_cache
    from model.auto_beq_nn import (
        AudioFeatureConfig,
        deduplicate_by_title,
        save_model,
        train_e84_self_trained,
        train_production_weighted_hybrid,
    )
    from spike._auto_beq_helpers import (
        STRATEGY_BLENDED_07,
        STRATEGY_WELCH,
        beq_dir,
        cached_extract_features_with_strategy,
        discover_unmatched_wavs_cached,
        discover_wav_catalogue_pairs_cached,
    )
    from spike.test_auto_beq_nn_real import _extract_features_parallel

    strategy = STRATEGY_BLENDED_07 if args.extraction_strategy == "blend_07" else STRATEGY_WELCH
    feature_config = AudioFeatureConfig(foundation_model=args.foundation_model)
    if args.foundation_model:
        log.info(
            "E83: foundation model = %s (+%d dims → %d total features)",
            args.foundation_model, feature_config.foundation_dim, feature_config.n_features,
        )

    # --- Resolve paths ---
    target_dir = beq_dir()
    if args.self_train:
        # E84: keep the E82 production model untouched, write a parallel
        # file so both can be A/B compared.
        default_output = target_dir / "production_model_e84.joblib"
    else:
        default_output = target_dir / "production_model.joblib"
    output_path = args.output or default_output
    meta_path = output_path.with_suffix(".meta.json")
    log.info("BEQ working directory: %s", target_dir)
    log.info("model output path:     %s", output_path)
    if args.self_train:
        log.info(
            "E84 self-training enabled: pseudo_weight=%.1f, confidence≤%.1f dB, %d iterations",
            args.pseudo_weight, args.confidence_threshold_db, args.self_train_iterations,
        )

    # --- Discover WAV cache + catalogue ---
    pairs = discover_wav_catalogue_pairs_cached()
    if not pairs:
        log.error(
            "no catalogue-matched WAVs found in cache.  Run "
            "bin/beq-designer extract first to populate the WAV cache.",
        )
        sys.exit(1)
    log.info("discovered %d catalogue-matched WAVs", len(pairs))

    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    log.info("catalogue (deduped, with filters): %d entries", len(deduped))

    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    # --- Extract real features in parallel ---
    t_extract_start = time.time()
    all_real = _extract_features_parallel(
        pairs, DEFAULT_GRID, 1000, strategy=strategy,
    )
    t_extract = time.time() - t_extract_start
    log.info(
        "extracted %d real audio features in %.1fs (%.1f WAVs/s)",
        len(all_real), t_extract, len(all_real) / t_extract if t_extract > 0 else 0,
    )

    # Build real_samples list: (catalogue_entry, features) tuples.
    # E83: also attach a foundation model embedding to each CurveFeatures.
    real_samples = []
    t_embed_start = time.time()
    embed_cache_dir = target_dir / "foundation-embeddings" / (args.foundation_model or "none")
    n_embed_failed = 0
    n_all_real = len(all_real)
    next_pct = 10
    if args.foundation_model:
        log.info(
            "attaching %s embeddings to %d real samples…",
            args.foundation_model, n_all_real,
        )
    for i, (pair, features) in enumerate(all_real):
        entry = pair.get("catalogue_entry")
        if entry is None or not entry.get("filters"):
            continue
        if args.foundation_model:
            import dataclasses
            try:
                emb = extract_foundation_embedding(
                    pair["wav_path"],
                    model_name=args.foundation_model,
                    cache_dir=embed_cache_dir,
                )
                features = dataclasses.replace(
                    features, foundation_embedding=tuple(float(x) for x in emb),
                )
            except Exception as exc:
                log.warning(
                    "foundation embedding failed for %s: %s",
                    pair["wav_path"].name, exc,
                )
                n_embed_failed += 1
            pct = ((i + 1) * 100) // max(1, n_all_real)
            if pct >= next_pct and pct < 100:
                log.info("  %3d%%  %d/%d embeddings", pct, i + 1, n_all_real)
                while next_pct <= pct:
                    next_pct += 10
        real_samples.append((entry, features))
    if args.foundation_model:
        t_embed = time.time() - t_embed_start
        log.info(
            "foundation-model embedding: %d real samples in %.1fs "
            "(%.1f WAVs/s), %d failed",
            len(real_samples), t_embed,
            len(real_samples) / t_embed if t_embed > 0 else 0,
            n_embed_failed,
        )
    log.info("real samples (after filter): %d", len(real_samples))

    # --- E84: discover and extract features for unmatched WAVs (unlabelled). ---
    unmatched_pairs: list = []
    if args.self_train:
        unmatched_wavs = discover_unmatched_wavs_cached()
        log.info(
            "E84: %d unmatched WAVs available for pseudo-labelling",
            len(unmatched_wavs),
        )
        if unmatched_wavs:
            n_u = len(unmatched_wavs)
            log.info("extracting curve features from %d unmatched WAVs…", n_u)
            t_u0 = time.time()
            next_u_pct = 10
            for i, wav_path in enumerate(unmatched_wavs):
                try:
                    features = cached_extract_features_with_strategy(
                        wav_path, DEFAULT_GRID, 1000, strategy=strategy,
                    )
                    unmatched_pairs.append((wav_path, features))
                except Exception as exc:
                    log.debug("skipping %s: %s", wav_path.name, exc)
                pct = ((i + 1) * 100) // max(1, n_u)
                if pct >= next_u_pct and pct < 100:
                    log.info("  %3d%%  %d/%d unmatched WAVs", pct, i + 1, n_u)
                    while next_u_pct <= pct:
                        next_u_pct += 10
            log.info(
                "E84: extracted features for %d/%d unmatched WAVs in %.1fs",
                len(unmatched_pairs), n_u, time.time() - t_u0,
            )

    # --- Train ---
    t_train_start = time.time()
    if args.self_train:
        model, metadata = train_e84_self_trained(
            real_samples=real_samples,
            synth_entries=deduped,
            unmatched_pairs=unmatched_pairs,
            tmdb_cache=tmdb_cache,
            freqs_hz=DEFAULT_GRID,
            fs=1000,
            real_weight=args.real_weight,
            pseudo_weight=args.pseudo_weight,
            confidence_threshold_db=args.confidence_threshold_db,
            n_iterations=args.self_train_iterations,
            config=feature_config,
        )
        # Fill in summary fields the printer expects; the iter_stats key
        # already carries the per-iteration breakdown.
        metadata["n_real"] = len(real_samples)
        metadata["n_synth"] = len([e for e in deduped if e.get("filters")])
    else:
        model, metadata = train_production_weighted_hybrid(
            real_samples=real_samples,
            synth_entries=deduped,
            tmdb_cache=tmdb_cache,
            freqs_hz=DEFAULT_GRID,
            fs=1000,
            real_weight=args.real_weight,
            config=feature_config,
        )
    t_train = time.time() - t_train_start
    metadata["extract_time_s"] = round(t_extract, 1)
    metadata["train_time_s"] = round(t_train, 1)
    metadata["extraction_strategy"] = args.extraction_strategy

    # --- Compute WAV cache mtime for staleness detection ---
    try:
        from spike._auto_beq_helpers import _latest_wav_mtime, wav_cache_dir
        metadata["wav_cache_mtime"] = _latest_wav_mtime(wav_cache_dir())
    except Exception:
        metadata["wav_cache_mtime"] = 0.0

    # --- Save model + metadata sidecar ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(output_path))
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log.info("metadata sidecar: %s", meta_path)

    # --- Summary ---
    print()
    print("=" * 70)
    print("  PRODUCTION MODEL TRAINED")
    print("=" * 70)
    print(f"  Model:              {output_path}")
    print(f"  Metadata:           {meta_path}")
    print(f"  Real samples:       {metadata['n_real']}")
    print(f"  Synthetic samples:  {metadata['n_synth']}")
    print(f"  Real weight:        {metadata['real_weight']}:1")
    print(f"  Extraction time:    {metadata['extract_time_s']} s")
    print(f"  Training time:      {metadata['train_time_s']} s")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Model file size:    {file_size_mb:.1f} MB")
    print("=" * 70)
    print()
    print("To use this model in production:")
    print(f"  AUTO_BEQ_ADVISOR=trained_model \\")
    print(f"  AUTO_BEQ_MODEL_PATH={output_path} \\")
    print(f"  bin/beq-designer profile --media ...")
    print()


if __name__ == "__main__":
    main()
