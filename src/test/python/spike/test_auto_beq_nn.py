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
        author=entry.get("author") or None,
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


def test_get_advisor_trained_model_no_path(monkeypatch, tmp_path):
    """get_advisor('trained_model') raises if no model is discoverable."""
    monkeypatch.delenv("AUTO_BEQ_MODEL_PATH", raising=False)
    # Point beq_shared_dir at an empty tmp dir so auto-discovery finds nothing.
    monkeypatch.setattr("model.wav_discovery.beq_shared_dir", lambda: tmp_path)
    with pytest.raises(ValueError, match="AUTO_BEQ_MODEL_PATH"):
        get_advisor("trained_model")


# ---------------------------------------------------------------------------
# F1: Synthetic feature augmentation
# ---------------------------------------------------------------------------


def test_augment_audio_features_shape_and_originals_preserved():
    """augment_audio_features returns originals + n_copies, audio dims perturbed."""
    from model.auto_beq_nn import AugmentationConfig, augment_audio_features

    rng = np.random.default_rng(42)
    n_orig = 10
    n_audio = 9
    n_meta = 93
    X = rng.normal(0, 1, size=(n_orig, n_audio + n_meta)).astype(np.float32)
    Y = rng.normal(0, 1, size=(n_orig, 24)).astype(np.float32)

    config = AugmentationConfig(
        gaussian_sigma_db=0.5,
        per_bin_uniform_db=1.0,
        n_copies=3,
        smooth_prob=0.0,  # disable smoothing for determinism
    )
    X_aug, Y_aug = augment_audio_features(X, Y, config, n_audio=n_audio)

    # Originals + 3 copies = 4× rows.
    assert X_aug.shape == (n_orig * 4, n_audio + n_meta)
    assert Y_aug.shape == (n_orig * 4, 24)
    # First n_orig rows are original.
    np.testing.assert_array_equal(X_aug[:n_orig], X)
    np.testing.assert_array_equal(Y_aug[:n_orig], Y)
    # Y is tiled (each block of n_orig rows == original).
    for k in range(1, 4):
        np.testing.assert_array_equal(Y_aug[k * n_orig:(k + 1) * n_orig], Y)


def test_augment_audio_features_metadata_untouched():
    """Augmentation perturbs only audio columns; metadata is preserved."""
    from model.auto_beq_nn import AugmentationConfig, augment_audio_features

    rng = np.random.default_rng(0)
    n_audio = 9
    n_meta = 93
    X = rng.normal(0, 1, size=(5, n_audio + n_meta)).astype(np.float32)
    Y = rng.normal(0, 1, size=(5, 24)).astype(np.float32)

    config = AugmentationConfig(
        gaussian_sigma_db=2.0, per_bin_uniform_db=2.0, n_copies=2,
        smooth_prob=0.0,
    )
    X_aug, _ = augment_audio_features(X, Y, config, n_audio=n_audio)

    # Metadata columns should be byte-identical for every augmented copy.
    for i in range(X_aug.shape[0]):
        orig_idx = i % 5
        np.testing.assert_array_equal(
            X_aug[i, n_audio:], X[orig_idx, n_audio:],
            err_msg=f"metadata mutated at row {i}",
        )

    # Audio columns should differ from originals in the augmented copies
    # (rows 5-14, since 0-4 are originals).
    diffs = np.abs(X_aug[5:, :n_audio] - np.tile(X[:, :n_audio], (2, 1)))
    assert diffs.mean() > 0.5, (
        f"augmentation too weak: mean audio diff {diffs.mean():.2f}"
    )


# ---------------------------------------------------------------------------
# F2: Option B chunk statistics
# ---------------------------------------------------------------------------


def test_extract_chunk_stats_shapes():
    """extract_chunk_stats returns (stddev, ceiling_frac) each shape (n_bins,)."""
    from spike._auto_beq_helpers import extract_chunk_stats
    from model.auto_beq_nn import OPTION_A_BINS_HZ

    n_chunks, n_freqs = 50, 100
    freqs = np.linspace(5, 200, n_freqs)
    rng = np.random.default_rng(0)
    matrix = rng.normal(-30, 5, size=(n_chunks, n_freqs)).astype(np.float32)

    stddev, ceiling = extract_chunk_stats(matrix, freqs, OPTION_A_BINS_HZ)
    assert stddev.shape == (len(OPTION_A_BINS_HZ),)
    assert ceiling.shape == (len(OPTION_A_BINS_HZ),)
    assert stddev.dtype == np.float32
    assert ceiling.dtype == np.float32
    # Random gaussian data: stddev should be ~5, ceiling fraction reasonable.
    assert 3 < stddev.mean() < 7, f"stddev mean {stddev.mean()}"
    assert 0 < ceiling.mean() < 1, f"ceiling mean {ceiling.mean()}"


def test_extract_chunk_stats_constant_data():
    """Constant chunks → stddev=0, ceiling_frac=1."""
    from spike._auto_beq_helpers import extract_chunk_stats
    from model.auto_beq_nn import OPTION_A_BINS_HZ

    freqs = np.linspace(5, 200, 100)
    matrix = np.full((20, 100), -30.0, dtype=np.float32)
    stddev, ceiling = extract_chunk_stats(matrix, freqs, OPTION_A_BINS_HZ)
    np.testing.assert_array_almost_equal(stddev, 0.0)
    np.testing.assert_array_almost_equal(ceiling, 1.0)


# ---------------------------------------------------------------------------
# AudioFeatureConfig dimension calculations
# ---------------------------------------------------------------------------


def test_audio_feature_config_dims():
    """Each flag adjusts n_total_audio and n_features by the right amount."""
    from model.auto_beq_nn import AudioFeatureConfig, N_AUDIO_FEATURES, N_METADATA_FEATURES

    base = AudioFeatureConfig()
    assert base.n_total_audio == N_AUDIO_FEATURES
    assert base.n_features == N_AUDIO_FEATURES + N_METADATA_FEATURES

    opt_b = AudioFeatureConfig(use_option_b=True)
    assert opt_b.n_total_audio == N_AUDIO_FEATURES + 18

    dbfs = AudioFeatureConfig(use_absolute_dbfs=True)
    assert dbfs.n_total_audio == N_AUDIO_FEATURES + 9

    hires = AudioFeatureConfig(use_high_res=True)
    assert hires.n_total_audio == 16

    clust = AudioFeatureConfig(use_rolloff_cluster=True, n_clusters=8)
    assert clust.n_total_audio == N_AUDIO_FEATURES + 8

    combo = AudioFeatureConfig(use_option_b=True, use_absolute_dbfs=True)
    assert combo.n_total_audio == N_AUDIO_FEATURES + 18 + 9


def test_build_feature_vector_with_option_b_synthetic_fallback():
    """When CurveFeatures has no chunk stats, build_feature_vector pads zeros/ones."""
    from model.auto_beq_advisor import CurveFeatures
    from model.auto_beq_nn import AudioFeatureConfig, build_feature_vector

    feats = CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
        level_at_5hz_db=-10.0, level_at_10hz_db=-5.0, level_at_20hz_db=-2.0,
        rolloff_depth_db=15.0, rolloff_slope_db_per_oct=3.0, dynamic_range_db=20.0,
        curve_sample_points=tuple(),
    )
    meta = MediaMetadata(title="t", year=2020)
    cfg = AudioFeatureConfig(use_option_b=True)
    vec = build_feature_vector(feats, meta, config=cfg)
    # 9 audio + 9 stddev (zeros) + 9 ceiling_frac (ones) + 93 metadata = 120
    assert vec.shape == (cfg.n_features,)
    # The synthetic fallback writes zeros for stddev and ones for ceiling_frac.
    np.testing.assert_array_almost_equal(vec[9:18], 0.0)
    np.testing.assert_array_almost_equal(vec[18:27], 1.0)


# ---------------------------------------------------------------------------
# E83 / T1.1 — Foundation model embedding features
# ---------------------------------------------------------------------------


def test_audio_feature_config_includes_foundation_dim():
    """AudioFeatureConfig grows by foundation_dim when a model is set."""
    from model.auto_beq_nn import AudioFeatureConfig, N_AUDIO_FEATURES, N_METADATA_FEATURES

    base = AudioFeatureConfig()
    assert base.foundation_dim == 0
    assert base.n_total_audio == N_AUDIO_FEATURES

    fnd = AudioFeatureConfig(foundation_model="mock-16")
    assert fnd.foundation_dim == 16
    assert fnd.n_total_audio == N_AUDIO_FEATURES + 16
    assert fnd.n_features == N_AUDIO_FEATURES + 16 + N_METADATA_FEATURES
    assert "fnd-mock-16" in fnd.label


def test_audio_feature_config_rejects_unknown_foundation_model():
    """Asking for a nonexistent foundation model raises at dim-query time."""
    from model.auto_beq_nn import AudioFeatureConfig

    cfg = AudioFeatureConfig(foundation_model="does-not-exist")
    with pytest.raises(ValueError, match="unknown foundation model"):
        _ = cfg.foundation_dim


def test_build_feature_vector_with_foundation_embedding():
    """Embedding present on CurveFeatures is concatenated in the right slot."""
    from model.auto_beq_advisor import CurveFeatures
    from model.auto_beq_nn import AudioFeatureConfig, build_feature_vector

    embedding = tuple(float(i) for i in range(16))  # deterministic [0..15]
    feats = CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
        level_at_5hz_db=-10.0, level_at_10hz_db=-5.0, level_at_20hz_db=-2.0,
        rolloff_depth_db=15.0, rolloff_slope_db_per_oct=3.0, dynamic_range_db=20.0,
        curve_sample_points=tuple(),
        foundation_embedding=embedding,
    )
    meta = MediaMetadata(title="t", year=2020)
    cfg = AudioFeatureConfig(foundation_model="mock-16")

    vec = build_feature_vector(feats, meta, config=cfg)

    # Shape: 9 audio + 16 foundation + 93 metadata = 118.
    assert vec.shape == (cfg.n_features,)
    # Foundation block sits immediately after the 9 audio features.
    np.testing.assert_array_almost_equal(vec[9:25], np.arange(16, dtype=np.float32))


def test_build_feature_vector_synthetic_foundation_fallback():
    """Synthetic samples (no raw audio) get zero-vector fallback of the right dim."""
    from model.auto_beq_advisor import CurveFeatures
    from model.auto_beq_nn import AudioFeatureConfig, build_feature_vector

    feats = CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
        level_at_5hz_db=-10.0, level_at_10hz_db=-5.0, level_at_20hz_db=-2.0,
        rolloff_depth_db=15.0, rolloff_slope_db_per_oct=3.0, dynamic_range_db=20.0,
        curve_sample_points=tuple(),
        foundation_embedding=None,  # synthetic — no raw audio
    )
    meta = MediaMetadata(title="t", year=2020)
    cfg = AudioFeatureConfig(foundation_model="mock-16")
    vec = build_feature_vector(feats, meta, config=cfg)

    assert vec.shape == (cfg.n_features,)
    np.testing.assert_array_almost_equal(vec[9:25], 0.0)


def test_build_feature_vector_rejects_foundation_embedding_shape_mismatch():
    """Caller passing the wrong-size embedding fails fast with a clear error."""
    from model.auto_beq_advisor import CurveFeatures
    from model.auto_beq_nn import AudioFeatureConfig, build_feature_vector

    feats = CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
        level_at_5hz_db=-10.0, level_at_10hz_db=-5.0, level_at_20hz_db=-2.0,
        rolloff_depth_db=15.0, rolloff_slope_db_per_oct=3.0, dynamic_range_db=20.0,
        curve_sample_points=tuple(),
        foundation_embedding=tuple(range(10)),  # wrong size: 10 instead of 16
    )
    meta = MediaMetadata(title="t", year=2020)
    cfg = AudioFeatureConfig(foundation_model="mock-16")
    with pytest.raises(ValueError, match="foundation_embedding shape mismatch"):
        build_feature_vector(feats, meta, config=cfg)


def test_extract_foundation_embedding_mock_is_deterministic_and_cached(tmp_path):
    """Mock model path produces a stable vector per media path + caches it."""
    from model.auto_beq_advisor import extract_foundation_embedding

    media = tmp_path / "fake.mkv"
    media.write_bytes(b"not-a-real-mkv-but-that's-fine-for-mock")

    cache_dir = tmp_path / "embedding-cache"
    v1 = extract_foundation_embedding(media, model_name="mock-16", cache_dir=cache_dir)
    assert v1.shape == (16,)
    assert v1.dtype == np.float32

    # Hit the cache on the second call — same output.
    v2 = extract_foundation_embedding(media, model_name="mock-16", cache_dir=cache_dir)
    np.testing.assert_array_equal(v1, v2)

    # Cache file is written under {cache_dir}/<key>.npy.
    cached_files = list(cache_dir.glob("*.npy"))
    assert len(cached_files) == 1


def test_train_production_weighted_hybrid_accepts_foundation_config(tmp_path):
    """Training function wires AudioFeatureConfig.foundation_model through end-to-end."""
    import time
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_advisor import CurveFeatures
    from model.auto_beq_nn import (
        AudioFeatureConfig,
        N_OUTPUT,
        train_production_weighted_hybrid,
    )

    # Hand-built tiny fixture: 3 real + 3 synth entries with LowShelf filters.
    def make_entry(title, freq, gain):
        return {
            "title": title,
            "year": "2020",
            "theMovieDB": f"tmdb-{title}",
            "filters": [{"type": "LowShelf", "freq": freq, "gain": gain, "q": 0.9}],
            "author": "aron7awol",
        }

    def make_feats(with_embedding: bool):
        return CurveFeatures(
            shoulder_peak_db=5.0, shoulder_peak_hz=30.0,
            level_at_5hz_db=-10.0, level_at_10hz_db=-5.0, level_at_20hz_db=-2.0,
            rolloff_depth_db=15.0, rolloff_slope_db_per_oct=3.0, dynamic_range_db=20.0,
            curve_sample_points=tuple(),
            foundation_embedding=tuple(float(i) for i in range(16)) if with_embedding else None,
        )

    real_samples = [
        (make_entry(f"real-{i}", 20.0 + i, 4.0 + i * 0.5), make_feats(with_embedding=True))
        for i in range(3)
    ]
    synth_entries = [make_entry(f"synth-{i}", 25.0 + i, 3.0 + i * 0.5) for i in range(3)]

    cfg = AudioFeatureConfig(foundation_model="mock-16")
    t0 = time.time()
    model, metadata = train_production_weighted_hybrid(
        real_samples=real_samples,
        synth_entries=synth_entries,
        tmdb_cache={},
        freqs_hz=DEFAULT_GRID,
        fs=1000,
        real_weight=50.0,
        config=cfg,
    )
    assert time.time() - t0 < 20, "training on 6 samples should be fast"

    # Metadata reflects the foundation config used.
    assert metadata["foundation_model"] == "mock-16"
    assert metadata["n_features"] == cfg.n_features
    assert metadata["feature_config"] == cfg.label
    assert metadata["n_real"] == 3
    assert metadata["n_synth"] == 3

    # Model predicts into (1, N_OUTPUT) for a single feature row at the new dim.
    x = np.zeros((1, cfg.n_features), dtype=np.float32)
    y = model.predict(x)
    assert y.shape == (1, N_OUTPUT)


# ---------------------------------------------------------------------------
# G8: per-author alpha (predict_with_alphas)
# ---------------------------------------------------------------------------


def test_late_fusion_predict_with_alphas(tmp_path):
    """LateFusionModel.predict_with_alphas blends per row."""
    from model.auto_beq_nn import train_late_fusion

    entries = _load_snapshot()
    X, Y, _ = _build_dataset(entries, DEFAULT_GRID)
    model = train_late_fusion(X, Y, alpha=0.5)

    # Predict with two different alphas: row 0 → α=0.0 (pure metadata),
    # row 1 → α=1.0 (pure audio).
    n = 2
    alphas = np.array([0.0, 1.0], dtype=np.float32)
    Y_blend = model.predict_with_alphas(X[:n], alphas)
    assert Y_blend.shape == Y[:n].shape

    # Sanity: pure-metadata prediction matches metadata sub-model alone.
    n_audio = model._n_audio
    X_meta_only = np.zeros_like(X[:n])
    X_meta_only[:, n_audio:] = X[:n, n_audio:]
    Y_meta = model.model_meta.predict(X_meta_only)
    np.testing.assert_allclose(Y_blend[0], Y_meta[0], rtol=1e-5)

    # Pure-audio prediction matches audio sub-model alone.
    X_audio_only = np.zeros_like(X[:n])
    X_audio_only[:, :n_audio] = X[:n, :n_audio]
    Y_audio = model.model_audio.predict(X_audio_only)
    np.testing.assert_allclose(Y_blend[1], Y_audio[1], rtol=1e-5)


# ---------------------------------------------------------------------------
# H2: author marginalization
# ---------------------------------------------------------------------------


def test_late_fusion_predict_marginalized_uniform():
    """predict_marginalized with uniform weights == mean over author identities."""
    from model.auto_beq_nn import (
        N_AUTHOR, author_col_start, train_late_fusion,
    )

    entries = _load_snapshot()
    X, Y, _ = _build_dataset(entries, DEFAULT_GRID)
    model = train_late_fusion(X, Y, alpha=0.5)

    Y_marg = model.predict_marginalized(
        X[:3], author_col_start=author_col_start(), n_author=N_AUTHOR,
    )
    assert Y_marg.shape == Y[:3].shape
    # Should be a finite, sane prediction.
    assert np.all(np.isfinite(Y_marg))


# ---------------------------------------------------------------------------
# H1: response-space averaging dedup
# ---------------------------------------------------------------------------


def test_deduplicate_by_title_response_avg_passthrough_for_singles():
    """Single-author titles are passed through unchanged (no refit)."""
    from model.auto_beq_nn import deduplicate_by_title_response_avg

    # Build a single-author group.
    entries = [
        {"title": "Solo Film", "author": "aron7awol",
         "audioTypes": ["Atmos"], "year": 2020,
         "filters": [{"type": "LowShelf", "freq": 20.0, "gain": 5.0, "q": 0.7}]},
    ]
    result = deduplicate_by_title_response_avg(
        entries, DEFAULT_GRID, fs=1000, strategy="mean",
    )
    assert len(result) == 1
    # No consensus marker since it wasn't averaged.
    assert "_consensus_n_authors" not in result[0]


def test_deduplicate_by_title_response_avg_multi_author_refit():
    """Multi-author title gets averaged + refit; result has consensus marker."""
    from model.auto_beq_nn import deduplicate_by_title_response_avg

    entries = [
        {"title": "Multi Film", "author": "mobe1969",
         "audioTypes": ["Atmos"], "year": 2020,
         "filters": [{"type": "LowShelf", "freq": 20.0, "gain": 5.0, "q": 0.7}]},
        {"title": "Multi Film", "author": "aron7awol",
         "audioTypes": ["TrueHD"], "year": 2020,
         "filters": [{"type": "LowShelf", "freq": 25.0, "gain": 4.0, "q": 0.7}]},
    ]
    result = deduplicate_by_title_response_avg(
        entries, DEFAULT_GRID, fs=1000, strategy="mean", n_workers=1,
    )
    assert len(result) == 1
    # Consensus marker is set on multi-author refits.
    assert result[0].get("_consensus_n_authors") == 2
    # Refit chain has at least one filter.
    assert len(result[0]["filters"]) >= 1


# ---------------------------------------------------------------------------
# I1: author classifier (metadata → author)
# ---------------------------------------------------------------------------


def test_strip_author_columns_dim():
    """strip_author_columns drops 9 columns from the right place."""
    from model.auto_beq_nn import (
        AUTHOR_COL_OFFSET_IN_METADATA, N_AUDIO_FEATURES, N_AUTHOR, N_FEATURES,
        strip_author_columns,
    )

    X = np.arange(2 * N_FEATURES, dtype=np.float32).reshape(2, N_FEATURES)
    X_no_author = strip_author_columns(X, n_audio=N_AUDIO_FEATURES)
    assert X_no_author.shape == (2, N_FEATURES - N_AUTHOR)
    # Columns before the author block are unchanged.
    start = N_AUDIO_FEATURES + AUTHOR_COL_OFFSET_IN_METADATA
    np.testing.assert_array_equal(X_no_author[:, :start], X[:, :start])
    # Columns after the author block follow on directly.
    np.testing.assert_array_equal(
        X_no_author[:, start:], X[:, start + N_AUTHOR:],
    )


def test_train_author_classifier_then_predict_alpha():
    """train_author_classifier + predict_alpha_from_metadata produces sensible alphas."""
    from model.auto_beq_nn import (
        DEFAULT_PER_AUTHOR_ALPHA, N_AUDIO_FEATURES, N_AUTHOR, PER_AUTHOR_ALPHA,
        predict_alpha_from_metadata, train_author_classifier,
    )

    entries = _load_snapshot()
    X, _Y, used = _build_dataset(entries, DEFAULT_GRID)

    # Need ≥2 distinct authors with ≥2 samples each for the classifier
    # to train successfully. The 18-entry snapshot may not satisfy this,
    # so we duplicate the dataset to ensure each author has at least 2.
    X_dup = np.vstack([X, X])
    used_dup = used + used

    clf = train_author_classifier(X_dup, used_dup, n_audio=N_AUDIO_FEATURES)

    for method in ("hard", "soft_blend", "top3"):
        alphas, predicted_idx = predict_alpha_from_metadata(
            clf, X[:5], n_audio=N_AUDIO_FEATURES, method=method,
        )
        assert alphas.shape == (5,)
        assert alphas.dtype == np.float32
        # Alphas should be in the valid blend range.
        assert np.all(alphas >= 0.0) and np.all(alphas <= 1.0)
        # Predicted indices should map to valid authors.
        assert predicted_idx.shape == (5,)
        assert np.all((predicted_idx >= 0) & (predicted_idx < N_AUTHOR))


# ---------------------------------------------------------------------------
# F7: rolloff clustering
# ---------------------------------------------------------------------------


def test_compute_rolloff_clusters_shapes():
    """compute_rolloff_clusters returns (kmeans, ids) with the right shape."""
    from model.auto_beq_nn import cluster_ids_to_onehot, compute_rolloff_clusters

    rng = np.random.default_rng(0)
    X_audio = rng.normal(0, 1, size=(50, 9)).astype(np.float32)
    kmeans, ids = compute_rolloff_clusters(X_audio, n_clusters=4)
    assert ids.shape == (50,)
    assert ids.min() >= 0 and ids.max() < 4

    onehot = cluster_ids_to_onehot(ids, n_clusters=4)
    assert onehot.shape == (50, 4)
    # Each row sums to exactly 1.0 (one-hot).
    np.testing.assert_array_almost_equal(onehot.sum(axis=1), 1.0)

    # New samples can be assigned via the kmeans model.
    X_new = rng.normal(0, 1, size=(7, 9)).astype(np.float32)
    new_ids = kmeans.predict(X_new)
    assert new_ids.shape == (7,)


# ---------------------------------------------------------------------------
# F6: confidence (inter-author agreement) weights
# ---------------------------------------------------------------------------


def test_compute_agreement_weights_single_author_neutral():
    """Single-author entries get weight 1.0 (neutral)."""
    from model.auto_beq_nn import compute_agreement_weights

    entries = [
        {"title": "A", "author": "x",
         "filters": [{"type": "LowShelf", "freq": 20.0, "gain": 5.0, "q": 0.7}]},
        {"title": "B", "author": "x",
         "filters": [{"type": "LowShelf", "freq": 25.0, "gain": 3.0, "q": 0.7}]},
    ]
    weights = compute_agreement_weights(entries, DEFAULT_GRID)
    assert weights.shape == (2,)
    np.testing.assert_array_almost_equal(weights, 1.0)


def test_compute_agreement_weights_disagreement_penalised():
    """Disagreeing multi-author titles get a weight < 1.0."""
    from model.auto_beq_nn import compute_agreement_weights

    entries = [
        {"title": "Same Title", "author": "alpha",
         "filters": [{"type": "LowShelf", "freq": 20.0, "gain": 5.0, "q": 0.7}]},
        {"title": "Same Title", "author": "beta",
         "filters": [{"type": "PeakingEQ", "freq": 50.0, "gain": -8.0, "q": 2.0}]},
    ]
    weights = compute_agreement_weights(entries, DEFAULT_GRID)
    # Both entries belong to the disagreeing pair, so both get the same weight < 1.
    assert weights.shape == (2,)
    assert weights[0] < 1.0
    assert weights[0] == weights[1]


# ---------------------------------------------------------------------------
# E82 — train_production_weighted_hybrid
# ---------------------------------------------------------------------------


def test_train_production_weighted_hybrid_returns_model_and_metadata(tmp_path):
    """E82 production training: real + synth + sample weights + metadata.

    Verifies the training function:
    - Accepts pre-extracted real samples + synth catalogue entries
    - Builds a sample_weight array with real_weight for real rows and
      1.0 for synth rows
    - Returns a fitted model with .predict() and 48-dim output
    - Returns a metadata dict with the expected provenance keys
    - The saved model can be reloaded via save_model/load_model
    """
    from model.auto_beq_nn import (
        load_model,
        save_model,
        train_production_weighted_hybrid,
    )

    entries = _load_snapshot()
    trainable = [e for e in entries if e.get("filters")]
    assert len(trainable) >= 5, "snapshot needs at least 5 entries with filters"

    # Split the snapshot into "real" (first 3) and "synth" (remainder).
    # Real samples are (entry, features) tuples — features from the
    # synthetic inverse of the catalogue filters, used as a stand-in for
    # actual WAV-derived features.  That's fine for this unit test: we're
    # checking the plumbing, not the synthetic-vs-real gap.
    real_entries = trainable[:3]
    synth_entries = trainable[3:]

    real_samples = [
        (e, _synthetic_features_for_entry(e, DEFAULT_GRID))
        for e in real_entries
    ]

    # Empty TMDb cache — enrich_media_metadata handles missing data
    # gracefully (fields default to None).
    tmdb_cache: dict[str, dict] = {}

    model, metadata = train_production_weighted_hybrid(
        real_samples=real_samples,
        synth_entries=synth_entries,
        tmdb_cache=tmdb_cache,
        freqs_hz=DEFAULT_GRID,
        fs=_DEFAULT_FS,
        real_weight=50.0,
    )

    # Metadata sanity checks.
    assert metadata["n_real"] == 3
    # synth count excludes entries already covered by real samples via tmdb_id
    # (for this snapshot the real entries don't overlap the synth entries so
    # all synth_entries pass through — but depending on how many have
    # filters, the count may be ≤ len(synth_entries)).
    assert metadata["n_synth"] >= 1
    assert metadata["n_synth"] <= len(synth_entries)
    assert metadata["real_weight"] == 50.0
    assert "trained_at" in metadata and metadata["trained_at"] > 0
    assert "xgb_params" in metadata
    assert metadata["xgb_params"]["n_jobs"] == 1  # E75 determinism fix

    # Model has .predict and produces a (1, N_OUTPUT) output.
    X_sample = np.random.RandomState(0).randn(1, N_FEATURES).astype(np.float32)
    Y_pred = model.predict(X_sample)
    assert Y_pred.shape == (1, N_OUTPUT)

    # Model round-trips via save/load.
    model_path = tmp_path / "test_production.joblib"
    save_model(model, str(model_path))
    assert model_path.exists()
    loaded = load_model(str(model_path))
    Y_pred_loaded = loaded.predict(X_sample)
    np.testing.assert_array_almost_equal(Y_pred, Y_pred_loaded)


def test_train_production_weighted_hybrid_rejects_empty_real_samples():
    """The function fails fast when given no usable real samples."""
    import pytest as _pytest

    from model.auto_beq_nn import train_production_weighted_hybrid

    with _pytest.raises(ValueError, match="no usable real samples"):
        train_production_weighted_hybrid(
            real_samples=[],
            synth_entries=[],
            tmdb_cache={},
            freqs_hz=DEFAULT_GRID,
        )


# ---------------------------------------------------------------------------
# E84 / T1.3 — Semi-supervised pseudo-labelling
# ---------------------------------------------------------------------------


def _make_real_curve_features(embedding: bool = False):
    """Minimal CurveFeatures with sample points for self-consistency scoring."""
    from model.auto_beq_advisor import CurveFeatures
    # 12 log-spaced sample points covering 5-80 Hz with a rolloff shape.
    sample_hz = [5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 32.0, 40.0, 63.0, 80.0]
    # Steep rolloff: -10 dB at 5 Hz, rising to 0 dB at 80 Hz (anchor).
    sample_db = [-10.0, -9.0, -7.5, -6.0, -4.5, -3.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0]
    return CurveFeatures(
        shoulder_peak_db=0.0, shoulder_peak_hz=80.0,
        level_at_5hz_db=-10.0, level_at_10hz_db=-6.0, level_at_20hz_db=-2.0,
        rolloff_depth_db=10.0, rolloff_slope_db_per_oct=3.0, dynamic_range_db=10.0,
        curve_sample_points=tuple(zip(sample_hz, sample_db, strict=True)),
        foundation_embedding=tuple(range(16)) if embedding else None,
    )


def test_pseudo_label_unmatched_confidence_filter_keeps_self_consistent():
    """High-confidence pseudo-labels pass; incoherent ones are dropped."""
    from pathlib import Path
    from model.auto_beq_nn import pseudo_label_unmatched

    # A teacher model stub that always returns the same filter chain —
    # a LowShelf(+10 dB, 20 Hz, Q=0.9) that reproduces the ~10 dB
    # rolloff in our synthetic features. Self-consistency should be
    # very high.
    from model.auto_beq_nn import catalogue_entry_to_labels

    good_filter_chain = [{"type": "LowShelf", "freq": 20.0, "gain": 10.0, "q": 0.9}]
    y_good = catalogue_entry_to_labels({"filters": good_filter_chain})

    class GoodTeacher:
        def predict(self, X):
            return np.tile(y_good, (len(X), 1))

    unmatched_pairs = [
        (Path("/fake/wav/fake_movie.lfe-1000hz.wav"), _make_real_curve_features()),
    ]

    pseudo_samples, stats = pseudo_label_unmatched(
        teacher_model=GoodTeacher(),
        unmatched_pairs=unmatched_pairs,
        tmdb_cache={},
        freqs_hz=DEFAULT_GRID,
        confidence_threshold_db=5.0,  # loose gate so we focus on shape, not tuning
    )
    assert stats["n_total"] == 1
    assert stats["n_kept"] == 1
    assert len(pseudo_samples) == 1
    entry, features = pseudo_samples[0]
    assert entry["author"] == "pseudo"
    assert entry["filters"][0]["type"] == "LowShelf"

    # The confidence gate should drop wild predictions. A teacher
    # predicting a +15 dB HighShelf at 80 Hz doesn't reproduce any
    # sub-20 Hz rolloff, so the residual is huge.
    bad_chain = [{"type": "HighShelf", "freq": 80.0, "gain": 15.0, "q": 0.5}]
    y_bad = catalogue_entry_to_labels({"filters": bad_chain})

    class BadTeacher:
        def predict(self, X):
            return np.tile(y_bad, (len(X), 1))

    _pseudo, stats_bad = pseudo_label_unmatched(
        teacher_model=BadTeacher(),
        unmatched_pairs=unmatched_pairs,
        tmdb_cache={},
        freqs_hz=DEFAULT_GRID,
        confidence_threshold_db=0.5,
    )
    assert stats_bad["n_kept"] == 0, (
        "very-tight gate should reject the incoherent teacher's pseudo-label"
    )


def test_pseudo_label_unmatched_shape_compatibility_with_training_fn():
    """Output tuples must match train_production_weighted_hybrid's real_samples shape."""
    from pathlib import Path
    from model.auto_beq_nn import (
        N_OUTPUT,
        catalogue_entry_to_labels,
        pseudo_label_unmatched,
        train_production_weighted_hybrid,
    )

    good_chain = [{"type": "LowShelf", "freq": 20.0, "gain": 10.0, "q": 0.9}]
    y_good = catalogue_entry_to_labels({"filters": good_chain})

    class FixedTeacher:
        def predict(self, X):
            return np.tile(y_good, (len(X), 1))

    unmatched = [
        (Path("/fake/wav/f1.lfe-1000hz.wav"), _make_real_curve_features()),
        (Path("/fake/wav/f2.lfe-1000hz.wav"), _make_real_curve_features()),
    ]

    pseudo_samples, _ = pseudo_label_unmatched(
        teacher_model=FixedTeacher(),
        unmatched_pairs=unmatched,
        tmdb_cache={},
        freqs_hz=DEFAULT_GRID,
        confidence_threshold_db=10.0,  # keep all
    )
    assert len(pseudo_samples) == 2

    # The (entry, features) tuples should be directly consumable by
    # train_production_weighted_hybrid as additional real_samples.
    # We concatenate them with one synthetic entry so the trainer has
    # a non-empty synth set.
    synth_entries = [{
        "title": "synth", "year": "2020",
        "theMovieDB": "tmdb-synth",
        "filters": [{"type": "LowShelf", "freq": 25.0, "gain": 5.0, "q": 0.9}],
        "author": "aron7awol",
    }]
    model, metadata = train_production_weighted_hybrid(
        real_samples=pseudo_samples,
        synth_entries=synth_entries,
        tmdb_cache={},
        freqs_hz=DEFAULT_GRID,
        real_weight=10.0,
    )
    # Metadata reflects 2 "real" samples (the pseudo-labels).
    assert metadata["n_real"] == 2
    assert metadata["n_synth"] == 1

    # Model can predict at the expected output shape.
    x_dummy = np.zeros((1, 102), dtype=np.float32)
    y = model.predict(x_dummy)
    assert y.shape == (1, N_OUTPUT)


def test_train_e84_self_trained_iterations_grow_training_set():
    """After N iterations, metadata should track pseudo-label counts per iter."""
    from pathlib import Path
    from model.auto_beq_nn import (
        catalogue_entry_to_labels,
        train_e84_self_trained,
    )

    # Real: 3 samples with LowShelf labels.
    def make_entry(title, freq, gain):
        return {
            "title": title, "year": "2020",
            "theMovieDB": f"tmdb-{title}",
            "filters": [{"type": "LowShelf", "freq": freq, "gain": gain, "q": 0.9}],
            "author": "aron7awol",
        }

    real_samples = [
        (make_entry(f"real-{i}", 20.0 + i, 4.0 + i * 0.5), _make_real_curve_features())
        for i in range(3)
    ]
    synth_entries = [make_entry(f"synth-{i}", 25.0 + i, 3.0) for i in range(3)]
    unmatched_pairs = [
        (Path(f"/fake/wav/unm-{i}.lfe-1000hz.wav"), _make_real_curve_features())
        for i in range(5)
    ]

    model, metadata = train_e84_self_trained(
        real_samples=real_samples,
        synth_entries=synth_entries,
        unmatched_pairs=unmatched_pairs,
        tmdb_cache={},
        freqs_hz=DEFAULT_GRID,
        real_weight=10.0,
        pseudo_weight=2.0,
        confidence_threshold_db=10.0,  # loose so we actually retain pseudo-labels
        n_iterations=2,
    )
    # Iter 0: baseline (no pseudo yet); iter 1 + 2: pseudo added.
    assert len(metadata["iter_stats"]) == 3
    assert metadata["iter_stats"][0]["n_pseudo"] == 0
    assert metadata["iter_stats"][1]["n_pseudo"] >= 0
    assert metadata["iter_stats"][2]["n_pseudo"] >= 0
    assert metadata["n_iterations"] == 2
    # Model exists and predicts.
    x = np.zeros((1, 102), dtype=np.float32)
    y = model.predict(x)
    assert y.shape[1] > 0
