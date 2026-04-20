"""Shared fixtures and helpers for NN experiment tests.

Eliminates duplication across test_auto_beq_nn_real, test_auto_beq_nn_extract,
and test_auto_beq_nn_chunked by providing:

- extract_real_audio_features(): Welch-strategy wrapper
- compute_mean_loss(): Evaluate model predictions against target filters
- build_held_out_training_set(): Build training dataset with real-audio titles held out
- build_real_validation_features(): Build feature vectors from real WAV files
- build_synthetic_validation_features(): Build synthetic feature vectors for entries
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from model.auto_beq import DEFAULT_GRID
from model.auto_beq_metadata import enrich_media_metadata
from model.auto_beq_nn import (
    build_feature_vector,
    catalogue_entry_to_labels,
    downstream_loss,
    labels_to_filters,
)
from auto_beq._auto_beq_helpers import (
    STRATEGY_WELCH,
    extract_features_with_strategy,
    synthetic_features,
)

log = logging.getLogger(__name__)

_DEFAULT_FS = 1000


def extract_real_audio_features(wav_path: Path, freqs_hz: np.ndarray, fs: int):
    """Load WAV with Welch strategy for real-audio feature extraction."""
    return extract_features_with_strategy(wav_path, freqs_hz, fs, strategy=STRATEGY_WELCH)


def compute_mean_loss(
    model,
    X_val: np.ndarray,
    val_entries: list[dict],
    freqs_hz: np.ndarray | None = None,
) -> float:
    """Compute mean downstream loss for a model on a validation set.

    Parameters
    ----------
    model
        Trained model with a .predict() method (XGBoost-style).
    X_val
        Feature matrix for validation set.
    val_entries
        Catalogue entries (must have 'filters' key) for validation set.
    freqs_hz
        Frequency grid. Defaults to DEFAULT_GRID.

    Returns
    -------
    Mean downstream loss in dB.
    """
    if freqs_hz is None:
        freqs_hz = DEFAULT_GRID
    Y_pred = model.predict(X_val)
    total = 0.0
    for i, entry in enumerate(val_entries):
        total += downstream_loss(labels_to_filters(Y_pred[i]), entry["filters"], freqs_hz)
    return total / len(val_entries) if val_entries else 0.0


def build_held_out_training_set(
    tmdb_cache: dict,
    held_out_tmdb_ids: set[str],
    freqs_hz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build synthetic training dataset with specific TMDb IDs held out.

    Loads the full BEQ catalogue, deduplicates, builds feature vectors,
    and excludes entries whose TMDb ID is in ``held_out_tmdb_ids``.

    Parameters
    ----------
    tmdb_cache
        Pre-loaded TMDb metadata cache.
    held_out_tmdb_ids
        Set of TMDb ID strings to exclude from training.
    freqs_hz
        Frequency grid. Defaults to DEFAULT_GRID.

    Returns
    -------
    (X_train, Y_train, entries_train) - training arrays with held-out titles removed.
    """
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_nn import deduplicate_by_title

    if freqs_hz is None:
        freqs_hz = DEFAULT_GRID

    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    X_all, Y_all, entries_all = [], [], []
    for e in deduped:
        features = synthetic_features(e, freqs_hz)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_all.append(build_feature_vector(features, metadata))
        Y_all.append(catalogue_entry_to_labels(e))
        entries_all.append(e)

    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    # Remove held-out titles.
    train_mask = np.array([
        str(e.get("theMovieDB", "")).strip() not in held_out_tmdb_ids
        for e in entries_all
    ])
    X_train = X_all[train_mask]
    Y_train = Y_all[train_mask]
    entries_train = [e for e, m in zip(entries_all, train_mask) if m]

    log.info(
        "training set: %d entries (%d held out)",
        len(X_train), len(X_all) - len(X_train),
    )
    return X_train, Y_train, entries_train


def build_real_validation_features(
    pairs: list[dict],
    tmdb_cache: dict,
    freqs_hz: np.ndarray | None = None,
    fs: int = _DEFAULT_FS,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build real-audio feature vectors from WAV pairs.

    Parameters
    ----------
    pairs
        List of dicts with 'catalogue_entry' and 'wav_path' keys.
    tmdb_cache
        Pre-loaded TMDb metadata cache.
    freqs_hz
        Frequency grid. Defaults to DEFAULT_GRID.
    fs
        Sample rate for feature extraction.

    Returns
    -------
    (X_val, Y_val, val_entries) tuple.
    """
    if freqs_hz is None:
        freqs_hz = DEFAULT_GRID

    X_val, Y_val, val_entries = [], [], []
    for p in pairs:
        entry = p["catalogue_entry"]
        wav_path = p["wav_path"]
        log.info("extracting real audio features: %s", Path(wav_path).name[:80])
        features = extract_real_audio_features(wav_path, freqs_hz, fs)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val.append(build_feature_vector(features, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)

    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)
    log.info("validation set (real audio): %d", len(X_val))
    return X_val, Y_val, val_entries


def build_synthetic_validation_features(
    val_entries: list[dict],
    tmdb_cache: dict,
    freqs_hz: np.ndarray | None = None,
) -> np.ndarray:
    """Build synthetic feature vectors for a set of catalogue entries.

    Parameters
    ----------
    val_entries
        Catalogue entries to build features for.
    tmdb_cache
        Pre-loaded TMDb metadata cache.
    freqs_hz
        Frequency grid. Defaults to DEFAULT_GRID.

    Returns
    -------
    X_synth_val - float32 feature matrix.
    """
    if freqs_hz is None:
        freqs_hz = DEFAULT_GRID

    X_synth_val = []
    for entry in val_entries:
        features = synthetic_features(entry, freqs_hz)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_synth_val.append(build_feature_vector(features, metadata))
    return np.array(X_synth_val, dtype=np.float32)
