"""Training data preparation for BEQ model training.

Handles discovering WAV-catalogue pairs, extracting audio features,
loading metadata, and building training/test splits. Used by training
scripts (train_production_model, train_torch_model), evaluation, and
experiment runners. All functions are safe for CLI and Docker use - no
PyQt6 or Qt imports.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def build_training_dataset(freqs_hz: np.ndarray):
    """Load full catalogue, dedup, enrich with TMDb cache, build (X, Y, entries).

    Uses the local TMDb cache (expected to be fully populated). Does NOT
    call fetch_metadata_batch - if a TMDb ID is missing from the cache,
    its metadata fields will simply be empty/zero.

    Returns (X, Y, entries, tmdb_cache) where X is float32 [N, 60],
    Y is float32 [N, 16], entries is the list of catalogue dicts used.
    """
    from model.audio_extraction import synthetic_features
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import enrich_media_metadata, load_cache
    from model.auto_beq_nn import build_feature_vector, catalogue_entry_to_labels, deduplicate_by_title

    catalogue = _fetch_or_cache()
    log.info("full catalogue: %d entries", len(catalogue))

    deduped = deduplicate_by_title(catalogue)
    deduped = [e for e in deduped if e.get("filters")]
    log.info("after dedup + filter: %d unique titles with filters", len(deduped))

    tmdb_cache = load_cache()
    log.info("TMDb cache: %d entries", len(tmdb_cache))

    X_list, Y_list, used = [], [], []
    for e in deduped:
        features = synthetic_features(e, freqs_hz)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_list.append(build_feature_vector(features, metadata))
        Y_list.append(catalogue_entry_to_labels(e))
        used.append(e)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    log.info("training dataset: X=%s Y=%s", X.shape, Y.shape)
    return X, Y, used, tmdb_cache


def prepare_training_data(
    strategy=None,
    split: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    build_feature_vectors: bool = False,
    fetch_catalogue: bool = True,
    min_pairs: int = 0,
) -> dict:
    """Prepare training data from the WAV cache and BEQ catalogue.

    This is the shared data-preparation preamble used by training scripts
    (train_production_model, train_torch_model), evaluation (evaluate),
    and experiment runners (run_tier1_comparison). Extracting it here
    eliminates duplication of the discover-extract-catalogue-metadata
    sequence.

    Parameters
    ----------
    strategy : ExtractionStrategy, optional
        Audio extraction strategy. Defaults to STRATEGY_BLENDED_07.
    split : bool
        If True, split into train/test sets using stratified sampling.
    test_size : float
        Fraction held out for testing (only used when split=True).
    random_state : int
        Random seed for reproducible splits (only used when split=True).
    build_feature_vectors : bool
        If True, build numpy feature vector arrays (X_train, X_test)
        using AudioFeatureConfig(). Requires split=True.
    fetch_catalogue : bool
        If True, fetch the full catalogue and deduplicate to produce
        synth_entries. Set False if the caller only needs real samples
        (e.g. evaluate.py).
    min_pairs : int
        Minimum number of WAV-catalogue pairs required. Raises
        RuntimeError if fewer are found.

    Returns
    -------
    dict with keys:
        - pairs: raw WAV-catalogue pair dicts
        - all_real: list of (pair, CurveFeatures) tuples
        - tmdb_cache: dict of TMDb metadata
        - real_samples: list of (catalogue_entry, CurveFeatures) tuples
          (filtered to entries with filters)

    When fetch_catalogue=True (default), also includes:
        - catalogue: full catalogue list
        - deduped: deduplicated catalogue entries with filters
        - synth_entries: deduped entries excluding train/test TMDb IDs
          (only when split=True; otherwise same as deduped)

    When split=True, also includes:
        - train_idx, test_idx: numpy index arrays
        - train_samples: list of (entry, features) for train set
        - entries_test: list of catalogue entries in test set
        - severity: list of severity labels used for stratification

    When build_feature_vectors=True (requires split=True), also includes:
        - X_train: numpy float32 array of train feature vectors
        - X_test: numpy float32 array of test feature vectors
        - entries_train: list of catalogue entries matching X_train rows
    """
    import time

    import numpy as np
    from model.audio_extraction import STRATEGY_BLENDED_07
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_metadata import fetch_metadata_batch, load_cache
    from model.wav_discovery import discover_wav_catalogue_pairs_cached
    from spike.test_auto_beq_nn_real import _extract_features_parallel

    if strategy is None:
        strategy = STRATEGY_BLENDED_07

    _FS = 1000

    # --- 1. Discover WAV-catalogue pairs ---
    pairs = discover_wav_catalogue_pairs_cached()
    if not pairs:
        raise RuntimeError(
            "No catalogue-matched WAVs found in cache. "
            "Run bin/beq-designer extract first to populate the WAV cache."
        )
    if len(pairs) < min_pairs:
        raise RuntimeError(
            f"Only {len(pairs)} WAV-catalogue pairs found, need >= {min_pairs}."
        )
    log.info("discovered %d catalogue-matched WAVs", len(pairs))

    # --- 2. Extract audio features in parallel ---
    t0 = time.time()
    all_real = _extract_features_parallel(
        pairs, DEFAULT_GRID, _FS, strategy=strategy,
    )
    t_extract = time.time() - t0
    log.info(
        "extracted %d real audio features in %.1fs (%.1f WAVs/s)",
        len(all_real), t_extract,
        len(all_real) / t_extract if t_extract > 0 else 0,
    )

    # --- 3. Load TMDb metadata ---
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(
        [p["catalogue_entry"] for p, _ in all_real], cache=tmdb_cache,
    )

    # --- 4. Build real_samples (entry, features) filtered to entries with filters ---
    real_samples = []
    for pair, features in all_real:
        entry = pair.get("catalogue_entry")
        if entry is not None and entry.get("filters"):
            real_samples.append((entry, features))
    log.info("real samples (with filters): %d", len(real_samples))

    result = {
        "pairs": pairs,
        "all_real": all_real,
        "tmdb_cache": tmdb_cache,
        "real_samples": real_samples,
        "extract_time_s": round(t_extract, 1),
    }

    # --- 5. Optionally fetch and deduplicate the full catalogue ---
    if fetch_catalogue:
        from model.auto_beq_catalogue import _fetch_or_cache
        from model.auto_beq_nn import deduplicate_by_title

        catalogue = _fetch_or_cache()
        deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
        log.info("catalogue (deduped, with filters): %d entries", len(deduped))
        result["catalogue"] = catalogue
        result["deduped"] = deduped

    # --- 6. Optionally split into train/test ---
    if split:
        from sklearn.model_selection import train_test_split

        real_entries = [p["catalogue_entry"] for p, _ in all_real]
        severity = [
            "heavy" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 20
            else "moderate" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 10
            else "gentle"
            for e in real_entries
        ]
        indices = np.arange(len(all_real))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=severity if len(set(severity)) > 1 else None,
        )
        log.info(
            "split: %d train / %d test (stratified by rolloff severity)",
            len(train_idx), len(test_idx),
        )

        train_samples = [
            (all_real[i][0]["catalogue_entry"], all_real[i][1])
            for i in train_idx
        ]

        entries_test = []
        for i in test_idx:
            entry = all_real[i][0]["catalogue_entry"]
            if entry.get("filters"):
                entries_test.append(entry)

        result["train_idx"] = train_idx
        result["test_idx"] = test_idx
        result["train_samples"] = train_samples
        result["entries_test"] = entries_test
        result["severity"] = severity

        # Filter synth_entries to exclude titles in the train/test sets.
        if fetch_catalogue:
            train_tmdb = {
                str(e.get("theMovieDB", "")).strip()
                for e, _ in train_samples
            }
            test_tmdb = {
                str(real_entries[i].get("theMovieDB", "")).strip()
                for i in test_idx
            }
            synth_entries = [
                e for e in deduped
                if str(e.get("theMovieDB", "")).strip() not in train_tmdb | test_tmdb
            ]
            result["synth_entries"] = synth_entries
        else:
            result["synth_entries"] = []
    elif fetch_catalogue:
        # No split - synth_entries is the full deduped catalogue.
        result["synth_entries"] = deduped

    # --- 7. Optionally build feature vectors ---
    if build_feature_vectors:
        if not split:
            raise ValueError(
                "build_feature_vectors=True requires split=True"
            )
        from model.auto_beq_metadata import enrich_media_metadata
        from model.auto_beq_nn import AudioFeatureConfig, build_feature_vector

        cfg = AudioFeatureConfig()

        X_test_list = []
        for i in test_idx:
            pair, features = all_real[i]
            entry = pair["catalogue_entry"]
            if not entry.get("filters"):
                continue
            metadata = enrich_media_metadata(entry, tmdb_cache)
            X_test_list.append(build_feature_vector(features, metadata, config=cfg))
        X_test = np.array(X_test_list, dtype=np.float32)

        X_train_list, entries_train = [], []
        for i in train_idx:
            pair, features = all_real[i]
            entry = pair["catalogue_entry"]
            if not entry.get("filters"):
                continue
            metadata = enrich_media_metadata(entry, tmdb_cache)
            X_train_list.append(build_feature_vector(features, metadata, config=cfg))
            entries_train.append(entry)
        X_train = np.array(X_train_list, dtype=np.float32)

        result["X_train"] = X_train
        result["X_test"] = X_test
        result["entries_train"] = entries_train

    return result
