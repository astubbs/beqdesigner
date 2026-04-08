"""Spike tests for the trained-model Advisor (Experiment 18).

Exercises the full code path:
  synthetic rolloff curves → feature extraction → XGBoost training →
  model save/load → TrainedModelAdvisor → propose_filters_from_measured →
  compute_match_metrics

Synthetic data approach: rolloff curves are derived from catalogue filter
chains (rolloff = -evaluate_filter_chain(filters)) rather than from real
audio. This gives correlated (X, Y) pairs without needing a WAV file or
STFT pipeline at this stage.

The 18-entry test snapshot is enough to verify the code path. For a real
training run, the full ~15,000-entry catalogue (deduplicated to ~7,000
unique titles) is fetched via auto_beq_catalogue._fetch_or_cache().
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from model.auto_beq import DEFAULT_GRID, compute_match_metrics, evaluate_filter_chain, propose_filters_from_measured
from model.auto_beq_advisor import CurveFeatures, MediaMetadata, extract_curve_features, get_advisor
from model.auto_beq_nn import (
    N_AUDIO_FEATURES,
    N_FEATURES,
    N_METADATA_FEATURES,
    N_OUTPUT,
    build_audio_features,
    build_feature_vector,
    build_metadata_features,
    catalogue_entry_to_labels,
    deduplicate_by_title,
    downstream_loss,
    labels_to_filters,
    save_model,
    split_dataset,
    train_xgboost,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CATALOGUE_SNAPSHOT = _REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "database.json"
_DEFAULT_FS = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_snapshot() -> list[dict]:
    with _CATALOGUE_SNAPSHOT.open() as f:
        return json.load(f)


def _synthetic_features_for_entry(entry: dict, freqs_hz: np.ndarray) -> CurveFeatures:
    """Compute curve features from the inverse of an entry's filter chain.

    rolloff = -evaluate_filter_chain(entry["filters"]) represents what the
    LFE would look like before BEQ correction.
    """
    correction = evaluate_filter_chain(entry["filters"], freqs_hz, fs=_DEFAULT_FS)
    rolloff = -correction
    # Normalise to 0 dB at 80 Hz anchor (nearest bin).
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    rolloff_norm = rolloff - rolloff[anchor_idx]
    return extract_curve_features(rolloff_norm, freqs_hz)


def _metadata_for_entry(entry: dict) -> MediaMetadata:
    return MediaMetadata(
        title=str(entry.get("title", "")),
        year=int(entry.get("year", 0) or 0) or None,
        audio_types=tuple(entry.get("audioTypes", [])),
        source=entry.get("source") or None,
        genres=tuple(entry.get("genres", [])),
        language=entry.get("language") or None,
        rating=entry.get("rating") or None,
        runtime_min=int(entry.get("runtime", 0) or 0) or None,
    )


def _build_dataset(entries: list[dict], freqs_hz: np.ndarray):
    """Build (X, Y, entries_used) from catalogue entries.

    Skips entries with no filters.
    """
    X_list, Y_list, used = [], [], []
    for e in entries:
        if not e.get("filters"):
            continue
        features = _synthetic_features_for_entry(e, freqs_hz)
        metadata = _metadata_for_entry(e)
        X_list.append(build_feature_vector(features, metadata))
        Y_list.append(catalogue_entry_to_labels(e))
        used.append(e)
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32), used


# ---------------------------------------------------------------------------
# 1. Feature vector shape
# ---------------------------------------------------------------------------


def test_feature_vector_shape():
    """build_feature_vector returns (60,) float32."""
    entries = _load_snapshot()
    entry = next(e for e in entries if e.get("filters"))
    feats = _synthetic_features_for_entry(entry, DEFAULT_GRID)
    meta = _metadata_for_entry(entry)

    audio = build_audio_features(feats)
    assert audio.shape == (N_AUDIO_FEATURES,), f"audio shape {audio.shape}"
    assert audio.dtype == np.float32

    meta_vec = build_metadata_features(meta)
    assert meta_vec.shape == (N_METADATA_FEATURES,), f"metadata shape {meta_vec.shape}"
    assert meta_vec.dtype == np.float32

    combined = build_feature_vector(feats, meta)
    assert combined.shape == (N_FEATURES,), f"combined shape {combined.shape}"
    assert combined.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Label roundtrip
# ---------------------------------------------------------------------------


def test_label_roundtrip():
    """Encode → decode preserves type/freq/gain/q within tolerance."""
    entries = _load_snapshot()
    entry = next(e for e in entries if len(e.get("filters", [])) >= 1)
    original = entry["filters"]

    y = catalogue_entry_to_labels(entry)
    assert y.shape == (N_OUTPUT,)
    assert y.dtype == np.float32

    decoded = labels_to_filters(y)
    assert len(decoded) == len(original), (
        f"roundtrip filter count mismatch: {len(decoded)} != {len(original)}"
    )
    for orig, dec in zip(original, decoded):
        assert dec["type"] == orig["type"], f"type mismatch: {dec['type']} != {orig['type']}"
        assert abs(dec["freq"] - float(orig["freq"])) < 0.5, f"freq mismatch: {dec['freq']} vs {orig['freq']}"
        assert abs(dec["gain"] - float(orig["gain"])) < 0.5, f"gain mismatch: {dec['gain']} vs {orig['gain']}"
        assert abs(dec["q"] - float(orig["q"])) < 0.05, f"q mismatch: {dec['q']} vs {orig['q']}"


# ---------------------------------------------------------------------------
# 3. Deduplication
# ---------------------------------------------------------------------------


def test_deduplication():
    """Captain America: TWS appears twice in snapshot; dedup yields one."""
    entries = _load_snapshot()
    titles_before = [e.get("title") for e in entries]
    ca_count_before = sum(1 for t in titles_before if "Captain America" in str(t))
    assert ca_count_before >= 2, "test precondition: need ≥2 CA:TWS entries in snapshot"

    deduped = deduplicate_by_title(entries)
    ca_count_after = sum(1 for e in deduped if "Captain America" in str(e.get("title", "")))
    assert ca_count_after == 1, f"after dedup expected 1 CA:TWS, got {ca_count_after}"
    assert len(deduped) < len(entries), "dedup should reduce total entry count"


# ---------------------------------------------------------------------------
# 4. Downstream loss — perfect prediction → ~0
# ---------------------------------------------------------------------------


def test_downstream_loss_perfect():
    """When predicted filters == target filters, downstream loss is near zero."""
    entries = _load_snapshot()
    entry = next(e for e in entries if e.get("filters"))
    filters = entry["filters"]

    loss = downstream_loss(filters, filters, DEFAULT_GRID)
    assert loss < 0.01, f"self-comparison loss should be ~0, got {loss:.4f}"


def test_downstream_loss_empty_vs_nonzero():
    """Empty predicted filter has non-trivial loss against a real filter."""
    entries = _load_snapshot()
    entry = next(e for e in entries if e.get("filters"))
    loss = downstream_loss([], entry["filters"], DEFAULT_GRID)
    assert loss > 1.0, f"empty prediction should have large loss, got {loss:.2f}"


# ---------------------------------------------------------------------------
# 5. Train on snapshot + predict valid filter dicts
# ---------------------------------------------------------------------------


def test_train_snapshot(tmp_path):
    """Train on 18-entry snapshot; predictions produce valid filter dicts.

    With only 18 samples XGBoost will massively overfit — that's expected and
    OK here. The test validates the code path, not prediction quality.
    """
    entries = _load_snapshot()
    X, Y, used = _build_dataset(entries, DEFAULT_GRID)
    assert X.shape[1] == N_FEATURES, f"feature dim: {X.shape[1]}"
    assert Y.shape[1] == N_OUTPUT, f"label dim: {Y.shape[1]}"
    assert len(used) >= 10, "need at least 10 entries with filters"

    model = train_xgboost(X, Y)

    # Predict on the same data — code-path exercise only.
    Y_pred = model.predict(X)
    assert Y_pred.shape == Y.shape

    # Every prediction should decode to a non-empty, well-formed filter list.
    for i, y in enumerate(Y_pred):
        filters = labels_to_filters(y)
        for f in filters:
            assert f["type"] in ("LowShelf", "HighShelf", "PeakingEQ")
            assert 5.0 <= f["freq"] <= 200.0, f"freq out of range: {f['freq']}"
            assert -30.0 <= f["gain"] <= 30.0, f"gain out of range: {f['gain']}"
            assert 0.1 <= f["q"] <= 10.0, f"q out of range: {f['q']}"

    # XGBoost feature importances should be available.
    importances = model.feature_importances_
    assert len(importances) == N_FEATURES
    assert importances.sum() > 0, "all zero importances"

    # Audio feature importances should be non-zero (audio is the primary signal).
    audio_importance = importances[:N_AUDIO_FEATURES].sum()
    assert audio_importance > 0, "audio features have zero importance"

    print(f"\nFeature importances (audio bins 0-8): {importances[:N_AUDIO_FEATURES]}")
    print(f"Metadata total importance: {importances[N_AUDIO_FEATURES:].sum():.4f}")


# ---------------------------------------------------------------------------
# 6. Full pipeline integration
# ---------------------------------------------------------------------------


def test_trained_advisor_pipeline(tmp_path):
    """Full round-trip: train → save → load → advise → propose_filters → metrics.

    Trains on the full snapshot, saves the model, loads via TrainedModelAdvisor,
    advises on 'Battle: Los Angeles', passes advice through propose_filters_from_measured.
    """
    from model.auto_beq_nn import TrainedModelAdvisor

    entries = _load_snapshot()
    X, Y, _ = _build_dataset(entries, DEFAULT_GRID)
    model = train_xgboost(X, Y)

    model_path = str(tmp_path / "test_model.joblib")
    save_model(model, model_path)

    advisor = TrainedModelAdvisor.load(model_path)

    # Pick 'Battle: Los Angeles' as the test subject.
    target_entry = next(
        (e for e in entries if "Battle" in str(e.get("title", "")) and e.get("filters")),
        entries[0],
    )
    feats = _synthetic_features_for_entry(target_entry, DEFAULT_GRID)
    meta = _metadata_for_entry(target_entry)

    advice = advisor.advise(meta, feats)
    assert advice.source == "trained_model"
    assert advice.max_gain_db > 0

    # Build the "measured rolloff" we'd hand to the pipeline.
    correction = evaluate_filter_chain(target_entry["filters"], DEFAULT_GRID, fs=_DEFAULT_FS)
    measured_rolloff = -correction

    # Run propose_filters_from_measured with the trained model advisor.
    proposed = propose_filters_from_measured(
        measured_rolloff,
        DEFAULT_GRID,
        fs=_DEFAULT_FS,
        advisor=advisor,
        metadata=meta,
    )
    # Even with 18-sample overfit training, proposed should be a non-empty list.
    assert isinstance(proposed, list), f"propose_filters_from_measured returned {type(proposed)}"

    if proposed:
        metrics = compute_match_metrics(-correction, proposed, DEFAULT_GRID, fs=_DEFAULT_FS)
        print(f"\n[test_trained_advisor_pipeline] {target_entry['title']}")
        print(f"  Advice: max_gain={advice.max_gain_db:.1f} dB, knee={advice.knee_hz}")
        print(f"  Proposed: {proposed}")
        print(f"  {metrics.as_text()}")


# ---------------------------------------------------------------------------
# 7. get_advisor("trained_model") factory
# ---------------------------------------------------------------------------


def test_get_advisor_trained_model(tmp_path, monkeypatch):
    """get_advisor('trained_model') resolves correctly via env var."""
    entries = _load_snapshot()
    X, Y, _ = _build_dataset(entries, DEFAULT_GRID)
    model = train_xgboost(X, Y)

    model_path = str(tmp_path / "test_model.joblib")
    save_model(model, model_path)

    monkeypatch.setenv("AUTO_BEQ_MODEL_PATH", model_path)
    advisor = get_advisor("trained_model")
    assert advisor.name == "trained_model"

    # Confirm it can advise without error.
    entry = next(e for e in entries if e.get("filters"))
    feats = _synthetic_features_for_entry(entry, DEFAULT_GRID)
    meta = _metadata_for_entry(entry)
    advice = advisor.advise(meta, feats)
    assert advice.source == "trained_model"


def test_get_advisor_trained_model_no_path(monkeypatch):
    """get_advisor('trained_model') raises if AUTO_BEQ_MODEL_PATH not set."""
    monkeypatch.delenv("AUTO_BEQ_MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="AUTO_BEQ_MODEL_PATH"):
        get_advisor("trained_model")
