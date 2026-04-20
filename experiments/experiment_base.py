"""Shared experiment infrastructure for E85/E86/E87 runners.

Provides ``setup_experiment_paths()`` for sys.path manipulation, and
``load_experiment_data()`` which wraps ``prepare_training_data()`` to
produce the full data bundle every experiment runner needs: feature
vectors, train/test split, TMDb cache, baseline teacher model, and a
reusable evaluator function.

This eliminates ~91 lines of identical boilerplate that was copy-pasted
across run_e85_experiment.py, run_e86_experiment.py, and
run_e87_experiment.py.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[1]

_FS = 1000


def setup_experiment_paths() -> None:
    """Add src/main/python and src/test/python to sys.path.

    Safe to call multiple times - only inserts if not already present.
    """
    for p in (_REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def setup_logging(name: str) -> logging.Logger:
    """Configure root logging and return a named logger for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
    )
    return logging.getLogger(name)


class ExperimentData(NamedTuple):
    """Bundle of prepared data returned by ``load_experiment_data()``."""
    X_train: Any  # numpy float32 array
    X_test: Any   # numpy float32 array
    entries_train: list[dict]
    entries_test: list[dict]
    train_samples: list[tuple]
    synth_entries: list[dict]
    tmdb_cache: dict
    teacher_model: Any  # trained XGBoost baseline model


def load_experiment_data(
    log: logging.Logger,
    min_pairs: int = 200,
) -> ExperimentData:
    """Load data and train the E82 baseline teacher model.

    Uses ``prepare_training_data()`` from ``_auto_beq_helpers`` for the
    discovery-extract-split pipeline, then trains the production weighted
    hybrid XGBoost model as the baseline teacher.

    Returns an ``ExperimentData`` named tuple with everything an
    experiment runner needs.
    """
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_nn import (
        AudioFeatureConfig,
        train_production_weighted_hybrid,
    )
    from spike._auto_beq_helpers import prepare_training_data

    data = prepare_training_data(
        split=True,
        build_feature_vectors=True,
        min_pairs=min_pairs,
    )

    cfg = AudioFeatureConfig()

    log.info(
        "split: %d train / %d test",
        len(data["X_train"]),
        len(data["X_test"]),
    )

    # Train baseline E82 teacher model.
    teacher, _ = train_production_weighted_hybrid(
        real_samples=data["train_samples"],
        synth_entries=data["synth_entries"],
        tmdb_cache=data["tmdb_cache"],
        freqs_hz=DEFAULT_GRID,
        fs=_FS,
        real_weight=50.0,
        config=cfg,
    )

    return ExperimentData(
        X_train=data["X_train"],
        X_test=data["X_test"],
        entries_train=data["entries_train"],
        entries_test=data["entries_test"],
        train_samples=data["train_samples"],
        synth_entries=data["synth_entries"],
        tmdb_cache=data["tmdb_cache"],
        teacher_model=teacher,
    )


def evaluate_model(
    predictor,
    entries_test: list[dict],
    X_test,
    label: str,
    is_torch: bool = True,
    log: logging.Logger | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Evaluate a model on the test set, returning (mean, max, per_author).

    Parameters
    ----------
    predictor
        Model with ``.predict_filters(x)`` (torch) or ``.predict(X)``
        (XGBoost).
    entries_test
        Catalogue entries for the test set.
    X_test
        Feature matrix for the test set.
    label
        Display label for logging.
    is_torch
        If True, call ``predictor.predict_filters()``. Otherwise call
        ``predictor.predict()`` and convert via ``labels_to_filters()``.
    log
        Logger for progress output.

    Returns
    -------
    tuple of (mean_loss, max_loss, per_author_means)
    """
    import numpy as np
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_nn import downstream_loss, labels_to_filters

    losses: list[float] = []
    for i, entry in enumerate(entries_test):
        if is_torch:
            filters = predictor.predict_filters(X_test[i])
        else:
            y = predictor.predict(X_test[i:i + 1])[0]
            filters = labels_to_filters(y)
        losses.append(downstream_loss(filters, entry["filters"], DEFAULT_GRID))

    mean_loss = float(np.mean(losses))
    max_loss = float(np.max(losses))

    by_author: dict[str, list[float]] = defaultdict(list)
    for entry, loss in zip(entries_test, losses, strict=True):
        by_author[entry.get("author", "?")].append(loss)
    per_author = {a: float(np.mean(v)) for a, v in by_author.items()}

    if log is not None:
        log.info("%-30s mean=%.2f max=%.2f", label, mean_loss, max_loss)

    return mean_loss, max_loss, per_author
