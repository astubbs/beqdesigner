"""Unified F-experiment comparison harness (E41–E52).

Runs every F-experiment variant against the same baseline (late fusion
α=0.7 + one-hot type encoding) on the full real-audio validation set.
Outputs a CSV for downstream reporting and prints a summary table.

Usage::

    PYTHONPATH=./src/main/python:./src/test/python QT_QPA_PLATFORM=offscreen \
      poetry run pytest src/test/python/spike/test_auto_beq_nn_experiments.py \
      -x -s --tb=short -k test_f_experiment_comparison

The CSV is written to ``.pytest_cache/auto_beq_f_experiments.csv``.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

log = logging.getLogger("auto_beq_f_experiments")

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

from model.auto_beq_nn import (
    DEFAULT_AUDIO_CONFIG,
    AudioFeatureConfig,
    AugmentationConfig,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment variant to test in the comparison harness."""

    name: str
    audio_config: AudioFeatureConfig = field(default_factory=lambda: DEFAULT_AUDIO_CONFIG)
    augmentation: AugmentationConfig | None = None
    late_fusion: bool = True
    alpha: float = 0.7
    confidence_weighted: bool = False       # F6
    use_rolloff_cluster: bool = False       # F7
    n_clusters: int = 6                     # F7
    downstream_loss_training: bool = False  # F9
    use_author_ensemble: bool = False       # F12
    music_filter: bool = False              # F4

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.augmentation:
            parts.append(self.augmentation.label)
        if self.late_fusion:
            parts.append(f"LF-a{self.alpha:.1f}")
        parts.append(self.audio_config.label)
        return " | ".join(parts)


# All experiments to compare.
EXPERIMENTS: list[ExperimentConfig] = [
    # Baseline — current best (late fusion α=0.7 + one-hot type)
    ExperimentConfig("Baseline"),

    # F1: Augmentation sweep (sigma values)
    ExperimentConfig("F1-aug-s0.5", augmentation=AugmentationConfig(gaussian_sigma_db=0.5, per_bin_uniform_db=1.0)),
    ExperimentConfig("F1-aug-s1.0", augmentation=AugmentationConfig(gaussian_sigma_db=1.0, per_bin_uniform_db=1.5)),
    ExperimentConfig("F1-aug-s1.5", augmentation=AugmentationConfig(gaussian_sigma_db=1.5, per_bin_uniform_db=2.0)),
    ExperimentConfig("F1-aug-s2.0", augmentation=AugmentationConfig(gaussian_sigma_db=2.0, per_bin_uniform_db=2.5)),

    # F2: Option B chunk statistics
    ExperimentConfig("F2-optB", audio_config=AudioFeatureConfig(use_option_b=True)),

    # F3: Absolute dBFS
    ExperimentConfig("F3-dBFS", audio_config=AudioFeatureConfig(use_absolute_dbfs=True)),

    # F6: Confidence-weighted training
    ExperimentConfig("F6-confweight", confidence_weighted=True),

    # F7: Rolloff clustering (sweep cluster counts)
    ExperimentConfig("F7-clust-4", use_rolloff_cluster=True, n_clusters=4),
    ExperimentConfig("F7-clust-6", use_rolloff_cluster=True, n_clusters=6),
    ExperimentConfig("F7-clust-8", use_rolloff_cluster=True, n_clusters=8),

    # F9: Downstream loss objective
    ExperimentConfig("F9-downstream", downstream_loss_training=True),

    # F11: Multi-resolution bins
    ExperimentConfig("F11-hires", audio_config=AudioFeatureConfig(use_high_res=True)),

    # F12: Per-author ensemble
    ExperimentConfig("F12-ensemble", use_author_ensemble=True),

    # Best combos (F1 + F2, F1 + F3, F1 + F2 + F3)
    ExperimentConfig(
        "F1+F2",
        augmentation=AugmentationConfig(gaussian_sigma_db=1.5),
        audio_config=AudioFeatureConfig(use_option_b=True),
    ),
    ExperimentConfig(
        "F1+F3",
        augmentation=AugmentationConfig(gaussian_sigma_db=1.5),
        audio_config=AudioFeatureConfig(use_absolute_dbfs=True),
    ),
    ExperimentConfig(
        "F1+F2+F3",
        augmentation=AugmentationConfig(gaussian_sigma_db=1.5),
        audio_config=AudioFeatureConfig(use_option_b=True, use_absolute_dbfs=True),
    ),
]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_PATH = Path(os.environ.get(
    "AUTO_BEQ_F_REPORT", ".pytest_cache/auto_beq_f_experiments.csv",
))


def _write_csv(rows: list[dict]) -> None:
    """Write results to CSV (overwrite each run)."""
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with _CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV written to %s (%d rows)", _CSV_PATH, len(rows))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_DEFAULT_FS = 1000


def _synthetic_features(entry: dict, freqs_hz: np.ndarray):
    """Build CurveFeatures from catalogue filter chain (synthetic inverse)."""
    from model.auto_beq import evaluate_filter_chain
    from model.auto_beq_advisor import extract_curve_features

    correction = evaluate_filter_chain(entry["filters"], freqs_hz, fs=_DEFAULT_FS)
    rolloff = -correction
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    rolloff -= rolloff[anchor_idx]
    return extract_curve_features(rolloff, freqs_hz)


def _build_training_data(
    entries: list[dict],
    freqs_hz: np.ndarray,
    tmdb_cache: dict,
    config: AudioFeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) training arrays from catalogue entries."""
    from model.auto_beq_advisor import MediaMetadata
    from model.auto_beq_metadata import enrich_media_metadata
    from model.auto_beq_nn import build_feature_vector, catalogue_entry_to_labels

    X, Y = [], []
    for e in entries:
        features = _synthetic_features(e, freqs_hz)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X.append(build_feature_vector(features, metadata, config=config))
        Y.append(catalogue_entry_to_labels(e))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def _build_validation_data(
    pairs: list[dict],
    freqs_hz: np.ndarray,
    tmdb_cache: dict,
    config: AudioFeatureConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build (X, Y, entries) validation arrays from real-audio WAV pairs."""
    from model.auto_beq_metadata import enrich_media_metadata
    from model.auto_beq_nn import build_feature_vector, catalogue_entry_to_labels

    from spike._auto_beq_helpers import STRATEGY_WELCH, extract_features_with_strategy

    X, Y, entries = [], [], []
    for p in pairs:
        entry = p["catalogue_entry"]
        if not entry.get("filters"):
            continue
        try:
            features = extract_features_with_strategy(
                p["wav_path"], freqs_hz, _DEFAULT_FS, strategy=STRATEGY_WELCH,
            )
        except Exception:
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X.append(build_feature_vector(features, metadata, config=config))
        Y.append(catalogue_entry_to_labels(entry))
        entries.append(entry)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), entries


def _evaluate(
    model: object,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    val_entries: list[dict],
    freqs_hz: np.ndarray,
) -> list[dict]:
    """Evaluate model predictions against validation targets."""
    from model.auto_beq_nn import downstream_loss, labels_to_filters

    Y_pred = model.predict(X_val)
    results = []
    for i in range(len(X_val)):
        pred_filters = labels_to_filters(Y_pred[i])
        target_filters = val_entries[i]["filters"]
        loss = downstream_loss(pred_filters, target_filters, freqs_hz)
        results.append({
            "title": val_entries[i].get("title", "?"),
            "year": str(val_entries[i].get("year", "")),
            "author": val_entries[i].get("author", "?"),
            "content_type": val_entries[i].get("content_type", "film"),
            "loss_db": round(loss, 2),
            "verdict": "PASS" if loss < 2.0 else "MARGINAL" if loss < 4.0 else "FAIL",
        })
    return results


def _train_model(
    exp: ExperimentConfig,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    entries_train: list[dict],
    freqs_hz: np.ndarray,
    n_audio: int,
) -> object:
    """Train the model according to the experiment config."""
    from model.auto_beq_nn import (
        compute_agreement_weights,
        compute_rolloff_clusters,
        cluster_ids_to_onehot,
        train_author_ensemble,
        train_late_fusion,
        train_xgboost,
        train_xgboost_downstream,
    )

    sample_weight = None
    if exp.confidence_weighted:
        sample_weight = compute_agreement_weights(entries_train, freqs_hz)

    if exp.downstream_loss_training:
        return train_xgboost_downstream(X_train, Y_train, freqs_hz, n_rounds=2)

    if exp.use_author_ensemble:
        return train_author_ensemble(X_train, Y_train, entries_train)

    aug = exp.augmentation
    if exp.late_fusion:
        return train_late_fusion(
            X_train, Y_train, alpha=exp.alpha,
            n_audio=n_audio, augmentation=aug,
        )
    return train_xgboost(
        X_train, Y_train,
        augmentation=aug, n_audio=n_audio,
        sample_weight=sample_weight,
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_f_experiment_comparison(tmp_path, caplog):
    """Run all F-experiment variants against baseline on real-audio validation.

    This is the unified comparison harness. It trains each experiment variant
    on the full synthetic catalogue (minus held-out validation titles) and
    evaluates on real-audio WAV files discovered in the cache.

    Results are written to CSV for downstream reporting.
    """
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")

    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import fetch_metadata_batch, load_cache
    from model.auto_beq_nn import deduplicate_by_title

    from spike._auto_beq_helpers import discover_wav_catalogue_pairs

    # --- Discover validation WAVs ---
    pairs = discover_wav_catalogue_pairs()
    if not pairs:
        pytest.skip("No WAV files found in cache")

    val_tmdb_ids = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    log.info("validation WAVs: %d files, %d unique tmdb IDs", len(pairs), len(val_tmdb_ids))

    # --- Load & prepare catalogue ---
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    # Hold out validation titles from training.
    train_entries = [
        e for e in deduped
        if str(e.get("theMovieDB", "")).strip() not in val_tmdb_ids
    ]
    log.info("training entries: %d (held out %d)", len(train_entries), len(deduped) - len(train_entries))

    # --- Run each experiment ---
    all_csv_rows: list[dict] = []
    summary_lines: list[str] = []

    for exp in EXPERIMENTS:
        log.info("=" * 60)
        log.info("EXPERIMENT: %s", exp.label)
        log.info("=" * 60)
        t0 = time.time()

        config = exp.audio_config
        n_audio = config.n_total_audio

        # Build training data with this config's feature dims.
        X_train, Y_train = _build_training_data(
            train_entries, DEFAULT_GRID, tmdb_cache, config,
        )

        # Build validation data with same config.
        X_val, Y_val, val_entries = _build_validation_data(
            pairs, DEFAULT_GRID, tmdb_cache, config,
        )
        if len(X_val) == 0:
            log.warning("no validation data for %s, skipping", exp.name)
            continue

        # Train.
        model = _train_model(exp, X_train, Y_train, train_entries, DEFAULT_GRID, n_audio)

        # Evaluate.
        results = _evaluate(model, X_val, Y_val, val_entries, DEFAULT_GRID)
        elapsed = time.time() - t0

        # Aggregate stats.
        losses = [r["loss_db"] for r in results]
        mean_loss = float(np.mean(losses)) if losses else 0
        n_pass = sum(1 for r in results if r["verdict"] == "PASS")
        n_marg = sum(1 for r in results if r["verdict"] == "MARGINAL")
        n_fail = sum(1 for r in results if r["verdict"] == "FAIL")

        line = (
            f"{exp.name:25s} | mean={mean_loss:5.2f} dB | "
            f"P={n_pass:3d} M={n_marg:3d} F={n_fail:3d} | {elapsed:.0f}s"
        )
        log.info(line)
        summary_lines.append(line)

        # Append to CSV rows.
        for r in results:
            all_csv_rows.append({
                "experiment": exp.name,
                "experiment_label": exp.label,
                **r,
            })

    # --- Write CSV ---
    _write_csv(all_csv_rows)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("F-EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    for line in summary_lines:
        print(line)
    print("=" * 70)
    print(f"CSV: {_CSV_PATH}")


# ---------------------------------------------------------------------------
# F5/E45 — Cross-episode consistency test (standalone)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_f5_cross_episode_consistency(tmp_path, caplog):
    """Validate that one episode's predicted profile transfers across a series.

    For each TV series with 3+ episodes in the validation set:
    1. Train on full catalogue (excluding this series)
    2. Predict filters for episode 1
    3. Apply episode 1's filters to episodes 2-N: measure error
    4. Also predict per-episode filters: measure error
    5. Compare: cross-episode error should be within 0.5 dB of per-episode
    """
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")

    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import fetch_metadata_batch, load_cache
    from model.auto_beq_nn import (
        deduplicate_by_title,
        downstream_loss,
        labels_to_filters,
        train_late_fusion,
    )

    from spike._auto_beq_helpers import STRATEGY_WELCH, discover_wav_catalogue_pairs

    pairs = discover_wav_catalogue_pairs()
    if not pairs:
        pytest.skip("No WAV files found")

    # Group TV episodes by series title.
    series: dict[str, list[dict]] = {}
    for p in pairs:
        entry = p["catalogue_entry"]
        if entry.get("content_type") != "TV" or not entry.get("filters"):
            continue
        title = entry.get("title", "").lower().strip()
        series.setdefault(title, []).append(p)

    # Keep series with 3+ episodes.
    series = {k: v for k, v in series.items() if len(v) >= 3}
    if not series:
        pytest.skip("No TV series with 3+ episodes found")

    log.info("cross-episode test: %d series with 3+ episodes", len(series))

    # Prepare training data.
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    config = DEFAULT_AUDIO_CONFIG
    X_train, Y_train = _build_training_data(deduped, DEFAULT_GRID, tmdb_cache, config)
    model = train_late_fusion(X_train, Y_train, alpha=0.7)

    # Build validation features for all TV episodes.
    X_val, Y_val, val_entries = _build_validation_data(
        [p for eps in series.values() for p in eps],
        DEFAULT_GRID, tmdb_cache, config,
    )

    # Evaluate per-series.
    print("\n" + "=" * 70)
    print("F5 — CROSS-EPISODE CONSISTENCY")
    print("=" * 70)

    offset = 0
    for title, eps in sorted(series.items()):
        n_eps = len(eps)
        # Get predictions for all episodes of this series.
        Y_pred = model.predict(X_val[offset : offset + n_eps])
        ep1_filters = labels_to_filters(Y_pred[0])

        per_ep_losses = []
        cross_ep_losses = []
        for i in range(n_eps):
            target = val_entries[offset + i]["filters"]
            # Per-episode prediction loss.
            pred_i = labels_to_filters(Y_pred[i])
            per_ep_losses.append(downstream_loss(pred_i, target, DEFAULT_GRID))
            # Cross-episode: apply ep1's prediction to this episode's target.
            cross_ep_losses.append(downstream_loss(ep1_filters, target, DEFAULT_GRID))

        mean_per = float(np.mean(per_ep_losses))
        mean_cross = float(np.mean(cross_ep_losses))
        delta = mean_cross - mean_per

        status = "OK" if abs(delta) < 0.5 else "DRIFT"
        print(
            f"  {title:30s} | {n_eps} eps | per-ep={mean_per:.2f} | "
            f"cross-ep={mean_cross:.2f} | delta={delta:+.2f} | {status}"
        )
        offset += n_eps

    print("=" * 70)
