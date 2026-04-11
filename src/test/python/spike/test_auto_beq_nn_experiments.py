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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # G5: XGBoost hyperparameters
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    # G7: augmented ensemble
    use_augmented_ensemble: bool = False
    n_ensemble_seeds: int = 3
    # G8: per-author alpha selection
    use_per_author_alpha: bool = False
    # H1: response-space averaging dedup
    dedup_strategy: str = "format"   # format | mean | median | trusted
    # H2: author marginalization at inference
    marginalization: str | None = None  # None | uniform | frequency | quality
    # H3: training-data quality filtering
    drop_authors: tuple[str, ...] = ()
    # H4: response curve label encoding
    output_mode: str = "filter_params"  # filter_params | response_curve
    # I-series: predicted author selection at inference
    author_predictor: str | None = None  # None | hard | soft_blend | top3
    # I4 (E76): per-author dedicated late-fusion models with classifier routing
    use_author_ensemble_v2: bool = False

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
    # Note: cluster features are appended by the harness after build_feature_vector,
    # so we do NOT set use_rolloff_cluster on AudioFeatureConfig (which would cause
    # an assertion mismatch). Instead, the harness detects use_rolloff_cluster on
    # ExperimentConfig and handles cluster injection.
    ExperimentConfig("F7-clust-4", use_rolloff_cluster=True, n_clusters=4),
    ExperimentConfig("F7-clust-6", use_rolloff_cluster=True, n_clusters=6),
    ExperimentConfig("F7-clust-8", use_rolloff_cluster=True, n_clusters=8),

    # F9: Downstream loss objective
    ExperimentConfig("F9-downstream", downstream_loss_training=True),

    # F11: Multi-resolution bins
    ExperimentConfig("F11-hires", audio_config=AudioFeatureConfig(use_high_res=True)),

    # F12: Per-author ensemble
    ExperimentConfig("F12-ensemble", use_author_ensemble=True),

    # F-series combos (used σ=1.5, not optimal — kept for reference)
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

# Shorthand for the optimal augmentation config.
_AUG_05 = AugmentationConfig(gaussian_sigma_db=0.5, per_bin_uniform_db=1.0)

# G-series: combinations and tuning after F1 success.
G_EXPERIMENTS: list[ExperimentConfig] = [
    # Baseline + F1 reference (for comparison within G-series run)
    ExperimentConfig("Baseline"),
    ExperimentConfig("F1-s0.5", augmentation=_AUG_05),

    # --- G1: Combos with correct sigma (σ=0.5) ---
    ExperimentConfig("G1a-F1+F2", augmentation=_AUG_05,
                     audio_config=AudioFeatureConfig(use_option_b=True)),
    ExperimentConfig("G1b-F1+F3", augmentation=_AUG_05,
                     audio_config=AudioFeatureConfig(use_absolute_dbfs=True)),
    ExperimentConfig("G1c-F1+F7", augmentation=_AUG_05,
                     use_rolloff_cluster=True, n_clusters=8),
    ExperimentConfig("G1d-F1+F2+F3", augmentation=_AUG_05,
                     audio_config=AudioFeatureConfig(use_option_b=True, use_absolute_dbfs=True)),
    ExperimentConfig("G1e-F1+F2+F7", augmentation=_AUG_05,
                     audio_config=AudioFeatureConfig(use_option_b=True),
                     use_rolloff_cluster=True, n_clusters=8),
    ExperimentConfig("G1f-F1+F3+F7", augmentation=_AUG_05,
                     audio_config=AudioFeatureConfig(use_absolute_dbfs=True),
                     use_rolloff_cluster=True, n_clusters=8),
    ExperimentConfig("G1g-kitchen-sink", augmentation=_AUG_05,
                     audio_config=AudioFeatureConfig(use_option_b=True, use_absolute_dbfs=True),
                     use_rolloff_cluster=True, n_clusters=8),

    # --- G2: Alpha sweep with F1(σ=0.5) ---
    ExperimentConfig("G2a-a0.5", augmentation=_AUG_05, alpha=0.5),
    ExperimentConfig("G2b-a0.6", augmentation=_AUG_05, alpha=0.6),
    # α=0.7 is F1-s0.5 above
    ExperimentConfig("G2d-a0.8", augmentation=_AUG_05, alpha=0.8),
    ExperimentConfig("G2e-a0.9", augmentation=_AUG_05, alpha=0.9),

    # --- G3: Early fusion + augmentation ---
    ExperimentConfig("G3a-early", augmentation=_AUG_05, late_fusion=False),

    # --- G4: Fine-grained sigma sweep ---
    ExperimentConfig("G4a-s0.25", augmentation=AugmentationConfig(gaussian_sigma_db=0.25, per_bin_uniform_db=0.5)),
    ExperimentConfig("G4b-s0.3", augmentation=AugmentationConfig(gaussian_sigma_db=0.3, per_bin_uniform_db=0.6)),
    ExperimentConfig("G4c-s0.4", augmentation=AugmentationConfig(gaussian_sigma_db=0.4, per_bin_uniform_db=0.8)),
    # σ=0.5 is F1-s0.5 above
    ExperimentConfig("G4e-s0.6", augmentation=AugmentationConfig(gaussian_sigma_db=0.6, per_bin_uniform_db=1.2)),
    ExperimentConfig("G4f-s0.75", augmentation=AugmentationConfig(gaussian_sigma_db=0.75, per_bin_uniform_db=1.5)),

    # --- G5: XGBoost hyperparameter tuning (early fusion to isolate effect) ---
    ExperimentConfig("G5a-600t", augmentation=_AUG_05, late_fusion=False, n_estimators=600),
    ExperimentConfig("G5b-800t", augmentation=_AUG_05, late_fusion=False, n_estimators=800),
    ExperimentConfig("G5c-d8", augmentation=_AUG_05, late_fusion=False, max_depth=8),
    ExperimentConfig("G5d-lr03", augmentation=_AUG_05, late_fusion=False, learning_rate=0.03),
    ExperimentConfig("G5e-600t-d8-lr03", augmentation=_AUG_05, late_fusion=False,
                     n_estimators=600, max_depth=8, learning_rate=0.03),

    # --- G7: Augmented ensemble ---
    ExperimentConfig("G7a-ens3", augmentation=_AUG_05,
                     use_augmented_ensemble=True, n_ensemble_seeds=3),
    ExperimentConfig("G7b-ens5", augmentation=_AUG_05,
                     use_augmented_ensemble=True, n_ensemble_seeds=5),

    # --- G8: Per-author alpha selection ---
    # Single trained model, but at inference time pick the optimal alpha
    # for each title's author from PER_AUTHOR_ALPHA lookup table.  The
    # training-time alpha is irrelevant because predict_with_alphas()
    # bypasses self.alpha and computes Y_audio + Y_meta separately.
    ExperimentConfig("G8-perauth", augmentation=_AUG_05,
                     use_per_author_alpha=True),
]


# H-series: multi-author resolution.
H_EXPERIMENTS: list[ExperimentConfig] = [
    # Reference points
    ExperimentConfig("Baseline"),
    ExperimentConfig("F1-s0.5", augmentation=_AUG_05),               # F-best (α=0.7)
    ExperimentConfig("G2a-a0.5", augmentation=_AUG_05, alpha=0.5),  # G-best
    ExperimentConfig("G8-perauth", augmentation=_AUG_05,
                     use_per_author_alpha=True),                     # G-best alt

    # --- H1: Response-space averaging dedup ---
    ExperimentConfig("H1a-respavg-mean", augmentation=_AUG_05, alpha=0.5,
                     dedup_strategy="mean"),
    ExperimentConfig("H1b-respavg-median", augmentation=_AUG_05, alpha=0.5,
                     dedup_strategy="median"),
    ExperimentConfig("H1d-respavg-trusted", augmentation=_AUG_05, alpha=0.5,
                     dedup_strategy="trusted"),

    # --- H2: Author marginalization at inference ---
    ExperimentConfig("H2a-marg-uniform", augmentation=_AUG_05, alpha=0.5,
                     marginalization="uniform"),
    ExperimentConfig("H2b-marg-frequency", augmentation=_AUG_05, alpha=0.5,
                     marginalization="frequency"),
    ExperimentConfig("H2c-marg-quality", augmentation=_AUG_05, alpha=0.5,
                     marginalization="quality"),

    # --- H3: Quality filtering (drop noisy authors) ---
    ExperimentConfig("H3a-drop-remixmark", augmentation=_AUG_05, alpha=0.5,
                     drop_authors=("remixmark",)),

    # --- H4: Response curve prediction (label space change) ---
    ExperimentConfig("H4-resp-curve", augmentation=_AUG_05, alpha=0.5,
                     output_mode="response_curve"),

    # --- H5: Best combinations ---
    ExperimentConfig("H5a-H1+G8", augmentation=_AUG_05,
                     dedup_strategy="mean", use_per_author_alpha=True),
    ExperimentConfig("H5b-H1+H2c", augmentation=_AUG_05, alpha=0.5,
                     dedup_strategy="mean", marginalization="quality"),
    ExperimentConfig("H5c-H1+H3", augmentation=_AUG_05, alpha=0.5,
                     dedup_strategy="mean", drop_authors=("remixmark",)),
    ExperimentConfig("H5d-ultimate", augmentation=_AUG_05,
                     dedup_strategy="mean",
                     use_per_author_alpha=True,
                     drop_authors=("remixmark",)),
]


# I-series: automated author selection from metadata.
I_EXPERIMENTS: list[ExperimentConfig] = [
    # Reference points
    ExperimentConfig("Baseline"),
    ExperimentConfig("F1-s0.5", augmentation=_AUG_05),
    ExperimentConfig("G2a-a0.5", augmentation=_AUG_05, alpha=0.5),
    ExperimentConfig("G8-perauth-oracle", augmentation=_AUG_05,
                     use_per_author_alpha=True),

    # I1: Hard author prediction (argmax → alpha lookup)
    ExperimentConfig("I1a-hard", augmentation=_AUG_05,
                     author_predictor="hard"),

    # I1b: Soft blend (probability-weighted alpha)
    ExperimentConfig("I1b-soft-blend", augmentation=_AUG_05,
                     author_predictor="soft_blend"),

    # I1c: Top-3 weighted average
    ExperimentConfig("I1c-top3", augmentation=_AUG_05,
                     author_predictor="top3"),

    # I4 (E76): Per-author dedicated late-fusion models, routed by the
    # I1b metadata classifier.  Dedicated models trained on each of the
    # top-3 authors (mobe1969, aron7awol, kaelaria) + fallback shared
    # model for everyone else.
    ExperimentConfig("I4-dedicated-α0.5", augmentation=_AUG_05, alpha=0.5,
                     use_author_ensemble_v2=True),
    ExperimentConfig("I4-dedicated-α0.7", augmentation=_AUG_05, alpha=0.7,
                     use_author_ensemble_v2=True),
]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_PATH = Path(os.environ.get(
    "AUTO_BEQ_F_REPORT", ".pytest_cache/auto_beq_f_experiments.csv",
))


# _write_csv moved to _write_csv_to (takes path param) — defined in harness below.


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
    output_mode: str = "filter_params",
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) training arrays from catalogue entries.

    *output_mode*: ``"filter_params"`` (default, 24-dim Y) or
    ``"response_curve"`` (H4, 9-dim Y).
    """
    from model.auto_beq_advisor import MediaMetadata
    from model.auto_beq_metadata import enrich_media_metadata
    from model.auto_beq_nn import (
        build_feature_vector,
        catalogue_entry_to_labels,
        catalogue_entry_to_response_labels,
    )

    label_fn = (
        catalogue_entry_to_response_labels
        if output_mode == "response_curve"
        else catalogue_entry_to_labels
    )

    X, Y = [], []
    for e in entries:
        features = _synthetic_features(e, freqs_hz)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X.append(build_feature_vector(features, metadata, config=config))
        if output_mode == "response_curve":
            Y.append(label_fn(e, freqs_hz))
        else:
            Y.append(label_fn(e))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def _build_validation_data(
    pairs: list[dict],
    freqs_hz: np.ndarray,
    tmdb_cache: dict,
    config: AudioFeatureConfig,
    output_mode: str = "filter_params",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build (X, Y, entries) validation arrays from real-audio WAV pairs."""
    from model.auto_beq_metadata import enrich_media_metadata
    from model.auto_beq_nn import (
        build_feature_vector,
        catalogue_entry_to_labels,
        catalogue_entry_to_response_labels,
    )

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
        if output_mode == "response_curve":
            Y.append(catalogue_entry_to_response_labels(entry, freqs_hz))
        else:
            Y.append(catalogue_entry_to_labels(entry))
        entries.append(entry)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), entries


def _evaluate(
    model: object,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    val_entries: list[dict],
    freqs_hz: np.ndarray,
    per_author_alpha: bool = False,
    marginalization: str | None = None,
    n_audio: int = 9,
    output_mode: str = "filter_params",
    author_classifier: object | None = None,
    author_predictor: str | None = None,
) -> list[dict]:
    """Evaluate model predictions against validation targets.

    When *per_author_alpha* is True (G8), uses ``LateFusionModel.predict_with_alphas``
    with per-row alpha looked up from PER_AUTHOR_ALPHA.

    When *marginalization* is set (H2), uses ``LateFusionModel.predict_marginalized``
    with per-author weights ("uniform", "frequency", or "quality").

    When *output_mode* is "response_curve" (H4), the predicted Y is a 9-bin
    response curve, decoded via ``response_labels_to_filters``.
    """
    from model.auto_beq_nn import (
        AUTHOR_FREQUENCY_WEIGHTS,
        AUTHOR_QUALITY_WEIGHTS,
        DEFAULT_PER_AUTHOR_ALPHA,
        PER_AUTHOR_ALPHA,
        _AUTHOR_VOCAB,
        author_col_start,
        author_weights_array,
        downstream_loss,
        labels_to_filters,
        predict_alpha_from_metadata,
        response_labels_to_filters,
    )

    predicted_authors_log: list[str] | None = None

    if marginalization is not None and hasattr(model, "predict_marginalized"):
        weight_dict = {
            "uniform": None,
            "frequency": AUTHOR_FREQUENCY_WEIGHTS,
            "quality": AUTHOR_QUALITY_WEIGHTS,
        }.get(marginalization)
        weights = author_weights_array(weight_dict)
        Y_pred = model.predict_marginalized(
            X_val, author_col_start=author_col_start(n_audio), weights=weights,
        )
    elif (
        author_predictor is not None
        and author_classifier is not None
        and hasattr(model, "predict_with_alphas")
    ):
        # I-series: predict author from metadata, use that author's alpha.
        alphas, predicted_idx = predict_alpha_from_metadata(
            author_classifier, X_val, n_audio=n_audio, method=author_predictor,
        )
        Y_pred = model.predict_with_alphas(X_val, alphas)
        predicted_authors_log = [_AUTHOR_VOCAB[i] for i in predicted_idx]
    elif per_author_alpha and hasattr(model, "predict_with_alphas"):
        alphas = np.array([
            PER_AUTHOR_ALPHA.get(e.get("author", ""), DEFAULT_PER_AUTHOR_ALPHA)
            for e in val_entries
        ], dtype=np.float32)
        Y_pred = model.predict_with_alphas(X_val, alphas)
    else:
        Y_pred = model.predict(X_val)

    results = []
    for i in range(len(X_val)):
        if output_mode == "response_curve":
            pred_filters = response_labels_to_filters(Y_pred[i], freqs_hz)
        else:
            pred_filters = labels_to_filters(Y_pred[i])
        target_filters = val_entries[i]["filters"]
        loss = downstream_loss(pred_filters, target_filters, freqs_hz)
        row = {
            "title": val_entries[i].get("title", "?"),
            "year": str(val_entries[i].get("year", "")),
            "author": val_entries[i].get("author", "?"),
            "content_type": val_entries[i].get("content_type", "film"),
            "loss_db": round(loss, 2),
            "verdict": "PASS" if loss < 2.0 else "MARGINAL" if loss < 4.0 else "FAIL",
        }
        if predicted_authors_log:
            row["predicted_author"] = predicted_authors_log[i]
        results.append(row)
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

    # I4 (E76): per-author dedicated late-fusion models with classifier routing
    if exp.use_author_ensemble_v2:
        from model.auto_beq_nn import train_author_ensemble_v2
        return train_author_ensemble_v2(
            X_train, Y_train, entries_train,
            alpha=exp.alpha,
            n_audio=n_audio,
            augmentation=exp.augmentation,
        )

    # G7: augmented ensemble (multiple random seeds, averaged)
    if exp.use_augmented_ensemble and exp.augmentation:
        from model.auto_beq_nn import train_augmented_ensemble
        return train_augmented_ensemble(
            X_train, Y_train, exp.augmentation,
            n_seeds=exp.n_ensemble_seeds,
            n_audio=n_audio,
            late_fusion=exp.late_fusion,
            alpha=exp.alpha,
        )

    aug = exp.augmentation
    xgb_kwargs = {}
    if exp.n_estimators != 400:
        xgb_kwargs["n_estimators"] = exp.n_estimators
    if exp.max_depth != 6:
        xgb_kwargs["max_depth"] = exp.max_depth
    if exp.learning_rate != 0.05:
        xgb_kwargs["learning_rate"] = exp.learning_rate

    if exp.late_fusion:
        return train_late_fusion(
            X_train, Y_train, alpha=exp.alpha,
            n_audio=n_audio, augmentation=aug,
        )
    return train_xgboost(
        X_train, Y_train,
        augmentation=aug, n_audio=n_audio,
        sample_weight=sample_weight,
        **xgb_kwargs,
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def _run_experiment_batch(
    experiments: list[ExperimentConfig],
    batch_name: str = "EXPERIMENT",
    csv_path: Path = _CSV_PATH,
) -> None:
    """Shared harness: train + evaluate a batch of experiment configs."""
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import fetch_metadata_batch, load_cache
    from model.auto_beq_nn import deduplicate_by_title

    from spike._auto_beq_helpers import (
        discover_wav_catalogue_pairs,
        ensure_analysis_reports_current,
    )

    # Auto-refresh stale analysis reports (bias, author patterns,
    # acquisition) before running experiments.  Skipped if opted out via
    # env var (e.g. for tight iteration loops where the reports are
    # intentionally stale).
    if os.environ.get("AUTO_BEQ_SKIP_REPORT_REFRESH", "0") != "1":
        regenerated = ensure_analysis_reports_current()
        if regenerated:
            log.info(
                "refreshed %d stale analysis report(s): %s",
                len(regenerated), ", ".join(regenerated),
            )

    pairs = discover_wav_catalogue_pairs()
    if not pairs:
        pytest.skip("No WAV files found in cache")

    val_tmdb_ids = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    log.info("validation WAVs: %d files, %d unique tmdb IDs", len(pairs), len(val_tmdb_ids))

    catalogue = _fetch_or_cache()
    raw_with_filters = [e for e in catalogue if e.get("filters")]
    log.info("raw catalogue (with filters): %d entries", len(raw_with_filters))

    # Pre-warm TMDb cache for the format-deduped set (covers most entries
    # we'll need; H1 strategies will use cached entries from any author).
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    def _build_train_entries(
        dedup_strategy: str,
        drop_authors: tuple[str, ...],
    ) -> list[dict]:
        """Build the training entry list per experiment config."""
        from model.auto_beq_nn import (
            deduplicate_by_title,
            deduplicate_by_title_response_avg,
        )
        if dedup_strategy == "format":
            entries = deduplicate_by_title(raw_with_filters)
        else:
            entries = deduplicate_by_title_response_avg(
                raw_with_filters, DEFAULT_GRID, fs=_DEFAULT_FS,
                strategy=dedup_strategy,
            )
        # H3: drop noisy authors from training (validation untouched)
        if drop_authors:
            drop_set = {a.lower() for a in drop_authors}
            entries = [
                e for e in entries
                if str(e.get("author", "")).strip().lower() not in drop_set
            ]
        # Hold out validation titles
        return [
            e for e in entries
            if str(e.get("theMovieDB", "")).strip() not in val_tmdb_ids
        ]

    # Feature cache keyed by (audio_config, dedup_strategy, drop_authors,
    # output_mode).  Reuses identical matrices across experiments that
    # differ only in inference behaviour (alpha, marginalization).
    feature_cache: dict[tuple, tuple] = {}

    def _get_features(
        config: AudioFeatureConfig,
        dedup_strategy: str,
        drop_authors: tuple[str, ...],
        output_mode: str,
    ):
        key = (config, dedup_strategy, drop_authors, output_mode)
        if key not in feature_cache:
            log.info(
                "building features: config=%s dedup=%s drop=%s mode=%s",
                config.label, dedup_strategy, drop_authors, output_mode,
            )
            train_entries_local = _build_train_entries(dedup_strategy, drop_authors)
            Xt, Yt = _build_training_data(
                train_entries_local, DEFAULT_GRID, tmdb_cache, config,
                output_mode=output_mode,
            )
            Xv, Yv, ve = _build_validation_data(
                pairs, DEFAULT_GRID, tmdb_cache, config, output_mode=output_mode,
            )
            feature_cache[key] = (Xt, Yt, Xv, Yv, ve, train_entries_local)
        return feature_cache[key]

    # Pre-warm: enumerate all unique cache keys.
    unique_keys = {
        (e.audio_config, e.dedup_strategy, e.drop_authors, e.output_mode)
        for e in experiments
    }
    log.info("pre-building features for %d unique cache keys...", len(unique_keys))
    for cfg, dedup, drops, omode in unique_keys:
        _get_features(cfg, dedup, drops, omode)

    def _run_one(exp: ExperimentConfig) -> tuple[ExperimentConfig, list[dict], float]:
        t0 = time.time()
        config = exp.audio_config
        n_audio = config.n_total_audio

        X_train_base, Y_train, X_val_base, Y_val, val_entries, train_entries_local = (
            _get_features(config, exp.dedup_strategy, exp.drop_authors, exp.output_mode)
        )
        X_train = X_train_base.copy()
        X_val = X_val_base.copy()

        if len(X_val) == 0:
            return exp, [], time.time() - t0

        if exp.use_rolloff_cluster:
            from model.auto_beq_nn import cluster_ids_to_onehot, compute_rolloff_clusters

            kmeans, train_ids = compute_rolloff_clusters(
                X_train[:, :n_audio], n_clusters=exp.n_clusters,
            )
            X_train = np.hstack([X_train, cluster_ids_to_onehot(train_ids, exp.n_clusters)])
            val_ids = kmeans.predict(X_val[:, :n_audio])
            X_val = np.hstack([X_val, cluster_ids_to_onehot(val_ids, exp.n_clusters)])
            n_audio += exp.n_clusters

        model = _train_model(exp, X_train, Y_train, train_entries_local, DEFAULT_GRID, n_audio)

        # I-series: train author classifier on the same training entries.
        author_classifier = None
        if exp.author_predictor:
            from model.auto_beq_nn import train_author_classifier
            author_classifier = train_author_classifier(
                X_train, train_entries_local, n_audio=n_audio,
            )

        results = _evaluate(
            model, X_val, Y_val, val_entries, DEFAULT_GRID,
            per_author_alpha=exp.use_per_author_alpha,
            marginalization=exp.marginalization,
            n_audio=n_audio,
            output_mode=exp.output_mode,
            author_classifier=author_classifier,
            author_predictor=exp.author_predictor,
        )
        return exp, results, time.time() - t0

    all_csv_rows: list[dict] = []
    summary_lines: list[str] = []

    max_workers = int(os.environ.get("AUTO_BEQ_F_WORKERS", "2"))
    log.info("running %d experiments with max_workers=%d", len(experiments), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, exp): exp for exp in experiments}
        for future in as_completed(futures):
            exp, results, elapsed = future.result()

            if not results:
                log.warning("no results for %s, skipping", exp.name)
                continue

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

            for r in results:
                all_csv_rows.append({
                    "experiment": exp.name,
                    "experiment_label": exp.label,
                    **r,
                })

    _write_csv_to(all_csv_rows, csv_path)

    print("\n" + "=" * 70)
    print(f"{batch_name} COMPARISON SUMMARY")
    print("=" * 70)
    for line in sorted(summary_lines):
        print(line)
    print("=" * 70)
    print(f"CSV: {csv_path}")


def _write_csv_to(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Union of all keys across all rows (some experiments add extra columns
    # like predicted_author).
    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV written to %s (%d rows)", path, len(rows))


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_f_experiment_comparison(tmp_path, caplog):
    """Run all F-experiment variants against baseline."""
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")
    _run_experiment_batch(EXPERIMENTS, "F-EXPERIMENT", _CSV_PATH)


_G_CSV_PATH = Path(os.environ.get(
    "AUTO_BEQ_G_REPORT", ".pytest_cache/auto_beq_g_experiments.csv",
))

_H_CSV_PATH = Path(os.environ.get(
    "AUTO_BEQ_H_REPORT", ".pytest_cache/auto_beq_h_experiments.csv",
))

_I_CSV_PATH = Path(os.environ.get(
    "AUTO_BEQ_I_REPORT", ".pytest_cache/auto_beq_i_experiments.csv",
))


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_g_experiment_comparison(tmp_path, caplog):
    """Run G-series experiments: combinations and tuning after F1 success."""
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")
    _run_experiment_batch(G_EXPERIMENTS, "G-EXPERIMENT", _G_CSV_PATH)


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_h_experiment_comparison(tmp_path, caplog):
    """Run H-series experiments: multi-author resolution."""
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")
    _run_experiment_batch(H_EXPERIMENTS, "H-EXPERIMENT", _H_CSV_PATH)


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_i_experiment_comparison(tmp_path, caplog):
    """Run I-series experiments: automated author selection from metadata."""
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")
    _run_experiment_batch(I_EXPERIMENTS, "I-EXPERIMENT", _I_CSV_PATH)


# ---------------------------------------------------------------------------
# E75 — Baseline determinism regression test
# ---------------------------------------------------------------------------


def _verify_deterministic(
    exp: "ExperimentConfig",
    train_entries: list,
    X_train,
    Y_train,
    X_val,
    Y_val,
    val_entries: list,
    freqs_hz,
    n_runs: int = 3,
    epsilon_db: float = 0.005,
) -> list[float]:
    """Run *exp* N times back-to-back and verify mean loss is identical.

    Returns the list of per-run mean losses.  Asserts that (max - min)
    across runs is below *epsilon_db*.  Small epsilon (default 5e-3 dB)
    catches any non-determinism from XGBoost thread scheduling or
    concurrent seed contamination.

    Used to guard the E75 fix: XGBoost n_jobs=1 + deterministic seeds
    must produce bit-identical models across runs.
    """
    from model.auto_beq import DEFAULT_GRID  # noqa: F401 — used indirectly

    losses: list[float] = []
    for i in range(n_runs):
        model = _train_model(
            exp, X_train, Y_train, train_entries, freqs_hz,
            n_audio=exp.audio_config.n_total_audio,
        )
        results = _evaluate(
            model, X_val, Y_val, val_entries, freqs_hz,
            per_author_alpha=exp.use_per_author_alpha,
            marginalization=exp.marginalization,
            n_audio=exp.audio_config.n_total_audio,
            output_mode=exp.output_mode,
        )
        mean = float(np.mean([r["loss_db"] for r in results]))
        losses.append(mean)
        log.info("determinism run %d/%d: %s → mean=%.6f dB", i + 1, n_runs, exp.name, mean)

    spread = max(losses) - min(losses)
    assert spread < epsilon_db, (
        f"{exp.name} non-deterministic: losses={losses}, spread={spread:.4f} dB "
        f">= epsilon={epsilon_db:.4f} dB"
    )
    return losses


@pytest.mark.skipif(
    os.environ.get("AUTO_BEQ_SKIP_F_EXPERIMENTS", "0") == "1",
    reason="AUTO_BEQ_SKIP_F_EXPERIMENTS=1",
)
def test_baseline_determinism(tmp_path, caplog):
    """E75: Baseline config must be bit-identical across N consecutive runs.

    Regression guard for the n_jobs=1 fix.  If XGBoost picks up a
    default-thread configuration (or we reintroduce concurrent
    contention), the mean loss will drift across runs and this test
    fails immediately.

    Runs the cheap Baseline config (no augmentation, no classifier, no
    dedup changes) to keep the test time bounded.  Extend the
    ``_CONFIGS`` list below to verify determinism of additional configs
    when investigating related bugs.
    """
    caplog.set_level(logging.INFO, logger="auto_beq_f_experiments")

    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import fetch_metadata_batch, load_cache
    from model.auto_beq_nn import deduplicate_by_title

    from spike._auto_beq_helpers import discover_wav_catalogue_pairs

    pairs = discover_wav_catalogue_pairs()
    if not pairs:
        pytest.skip("No WAV files found in cache")

    val_tmdb = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)
    train_entries = [
        e for e in deduped
        if str(e.get("theMovieDB", "")).strip() not in val_tmdb
    ]

    config = AudioFeatureConfig()
    X_train, Y_train = _build_training_data(
        train_entries, DEFAULT_GRID, tmdb_cache, config,
    )
    X_val, Y_val, val_entries = _build_validation_data(
        pairs, DEFAULT_GRID, tmdb_cache, config,
    )

    # Configs to verify.  Keep this short — each config runs N times.
    _CONFIGS = [
        ExperimentConfig("Baseline"),
    ]

    for exp in _CONFIGS:
        losses = _verify_deterministic(
            exp, train_entries, X_train, Y_train, X_val, Y_val,
            val_entries, DEFAULT_GRID, n_runs=3, epsilon_db=0.005,
        )
        log.info("%s determinism OK: losses=%s", exp.name, losses)


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
