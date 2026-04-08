"""E31: Extract LFE WAVs from library media and validate NN models.

Discovers all catalogue-matched media across library roots, extracts
LFE WAVs for the smallest 50 titles (fastest extraction over NAS), and
runs the full validation suite.

This test creates real audio training/validation data by running ffmpeg
on actual media files. Skipped if library roots are not configured or
no matches found.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from model.auto_beq import DEFAULT_GRID, evaluate_filter_chain, smooth_fractional_octave
from model.auto_beq_advisor import extract_curve_features
from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
from model.auto_beq_nn import (
    N_AUDIO_FEATURES,
    build_feature_vector,
    catalogue_entry_to_labels,
    deduplicate_by_title,
    downstream_loss,
    labels_to_filters,
    train_late_fusion,
    train_xgboost,
)

from spike._auto_beq_helpers import audio_cache_dir, extract_lfe_wav
from spike.sweep_discover import inventory_root, match_media_files

log = logging.getLogger("auto_beq_nn_extract")

_DEFAULT_FS = 1000
_MAX_TITLES = 50


def _load_settings() -> dict:
    cfg = Path.home() / ".config" / "beqdesigner" / "settings.json"
    if cfg.exists():
        return json.loads(cfg.read_text())
    return {}


def _discover_and_sort() -> list[dict]:
    """Discover catalogue matches across all library roots, sorted by file size.

    Returns list of dicts with keys: path, title, year, catalogue_entry, size_bytes.
    Deduplicated by title (first/smallest file per title kept).
    """
    settings = _load_settings()
    roots = settings.get("library_roots", [])
    if not roots:
        return []

    from model.auto_beq_catalogue import _fetch_or_cache
    catalogue = _fetch_or_cache()

    # Inventory all roots.
    all_files = []
    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            log.info("skipping unmounted root: %s", root)
            continue
        files = inventory_root(root)
        all_files.extend((f, root) for f in files)

    if not all_files:
        return []

    log.info("total media files across all roots: %d", len(all_files))

    # Match against catalogue.
    result = match_media_files(all_files, catalogue)
    log.info("catalogue matches: %d", len(result.matches))

    # Sort by file size (smallest first) for fastest extraction.
    result.matches.sort(key=lambda m: m.size_bytes)

    # Deduplicate by title (keep smallest file per title).
    seen = set()
    unique = []
    for m in result.matches:
        if m.title not in seen:
            seen.add(m.title)
            unique.append({
                "path": m.path,
                "title": m.title,
                "year": m.year,
                "catalogue_entry": m.catalogue_entry,
                "size_bytes": m.size_bytes,
                "size_mb": m.size_mb,
                "content_type": m.catalogue_entry.get("content_type", "film"),
            })
    return unique


def _extract_real_audio_features(wav_path: Path, freqs_hz: np.ndarray, fs: int):
    from model.signal import Signal, read_wav_data

    samples, read_fs, _ = read_wav_data(str(wav_path))
    assert read_fs == fs, f"expected fs={fs}, got {read_fs}"
    mono = samples[:, 0] if samples.ndim > 1 else samples

    sig = Signal(str(wav_path.stem), mono, fs=fs)
    measured_freqs, measured_db = sig.avg_spectrum()

    curve = np.interp(freqs_hz, measured_freqs, measured_db)
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    curve -= curve[anchor_idx]
    curve = smooth_fractional_octave(curve, freqs_hz, octaves=1.0 / 6.0)
    curve -= curve[anchor_idx]

    return extract_curve_features(curve, freqs_hz)


def _synthetic_features(entry: dict, freqs_hz: np.ndarray):
    correction = evaluate_filter_chain(entry["filters"], freqs_hz, fs=_DEFAULT_FS)
    rolloff = -correction
    anchor_idx = int(np.argmin(np.abs(freqs_hz - 80.0)))
    rolloff_norm = rolloff - rolloff[anchor_idx]
    return extract_curve_features(rolloff_norm, freqs_hz)


# Discovery at module load (fast — just filesystem walk + catalogue match).
_CANDIDATES = _discover_and_sort()


@pytest.mark.skipif(not _CANDIDATES, reason="no library roots configured or no catalogue matches")
def test_extract_and_validate(tmp_path):
    """E31: Extract LFE from smallest 50 catalogue-matched titles, validate all models.

    1. Extract LFE WAVs (cached — only extracts if not already cached)
    2. Build real-audio features for each extracted title
    3. Train on full synthetic catalogue (excluding validation titles)
    4. Validate: E25 early fusion, E27 late fusion, ablation
    """
    from model.auto_beq_catalogue import _fetch_or_cache
    from spike._auto_beq_helpers import have_tool

    if not have_tool("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    titles = _CANDIDATES[:_MAX_TITLES]
    log.info("=== E31: extracting LFE for %d titles (sorted by size) ===", len(titles))

    # --- 1. Extract LFE WAVs ---
    extracted = []
    for i, t in enumerate(titles):
        media_path = Path(t["path"])
        log.info("[%d/%d] %s (%.0f MB)", i + 1, len(titles), t["title"], t["size_mb"])
        try:
            wav_path = extract_lfe_wav(media_path, _DEFAULT_FS)
            extracted.append({**t, "wav_path": wav_path})
        except Exception as exc:
            log.warning("extraction failed for %s: %s", t["title"], exc)

    log.info("extracted %d / %d titles", len(extracted), len(titles))
    if not extracted:
        pytest.skip("no WAVs extracted successfully")

    # --- 2. Build real-audio features ---
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(
        [t["catalogue_entry"] for t in extracted], cache=tmdb_cache
    )

    X_val_real, X_val_synth, Y_val, val_entries = [], [], [], []
    for t in extracted:
        entry = t["catalogue_entry"]
        if not entry.get("filters"):
            continue
        try:
            real_feats = _extract_real_audio_features(t["wav_path"], DEFAULT_GRID, _DEFAULT_FS)
        except Exception as exc:
            log.warning("feature extraction failed for %s: %s", t["title"], exc)
            continue
        synth_feats = _synthetic_features(entry, DEFAULT_GRID)
        metadata = enrich_media_metadata(entry, tmdb_cache)
        X_val_real.append(build_feature_vector(real_feats, metadata))
        X_val_synth.append(build_feature_vector(synth_feats, metadata))
        Y_val.append(catalogue_entry_to_labels(entry))
        val_entries.append(entry)

    if not val_entries:
        pytest.skip("no titles with filters after feature extraction")

    X_val_real = np.array(X_val_real, dtype=np.float32)
    X_val_synth = np.array(X_val_synth, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)
    log.info("validation set: %d titles with real audio features", len(val_entries))

    # --- 3. Train on full synthetic catalogue (excluding validation titles) ---
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]

    val_tmdb_ids = {
        str(e.get("theMovieDB", "")).strip() for e in val_entries
    }
    X_train_list, Y_train_list = [], []
    for e in deduped:
        if str(e.get("theMovieDB", "")).strip() in val_tmdb_ids:
            continue
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_train_list.append(build_feature_vector(features, metadata))
        Y_train_list.append(catalogue_entry_to_labels(e))
    X_train = np.array(X_train_list, dtype=np.float32)
    Y_train = np.array(Y_train_list, dtype=np.float32)
    log.info("training set: %d synthetic entries", len(X_train))

    # --- 4. Train + evaluate ---
    def _mean_loss(model, X_val):
        Y_pred = model.predict(X_val)
        total = 0.0
        for i, entry in enumerate(val_entries):
            total += downstream_loss(labels_to_filters(Y_pred[i]), entry["filters"], DEFAULT_GRID)
        return total / len(val_entries)

    log.info("training E25 early fusion...")
    model_early = train_xgboost(X_train, Y_train)

    log.info("training E27 late fusion (α=0.3)...")
    model_late = train_late_fusion(X_train, Y_train, alpha=0.3)

    # --- 5. Results ---
    n_film = sum(1 for t in extracted if t["content_type"] != "TV")
    n_tv = sum(1 for t in extracted if t["content_type"] == "TV")

    print(f"\n{'='*75}")
    print(f"  E31: LIBRARY EXTRACTION VALIDATION")
    print(f"  {len(val_entries)} titles validated ({n_film} film, {n_tv} TV)")
    print(f"  Trained on {len(X_train)} synthetic entries")
    print(f"{'='*75}\n")

    # Per-title results (early fusion).
    Y_pred = model_early.predict(X_val_real)
    print(f"  {'Title':40s} {'Size MB':>8s} {'Loss':>8s} {'Type':>5s}")
    print(f"  {'-'*65}")
    for i, entry in enumerate(val_entries):
        pred_filters = labels_to_filters(Y_pred[i])
        loss = downstream_loss(pred_filters, entry["filters"], DEFAULT_GRID)
        title = entry["title"][:40]
        size = extracted[i]["size_mb"]
        ctype = extracted[i]["content_type"][:4]
        print(f"  {title:40s} {size:8.0f} {loss:7.2f} dB {ctype}")

    # Summary table.
    print(f"\n  {'Strategy':30s} {'Real audio':>12s} {'Synthetic':>12s} {'Gap':>8s}")
    print(f"  {'-'*65}")

    for name, model in [
        ("E25 early fusion", model_early),
        ("E27 late fusion (α=0.3)", model_late),
    ]:
        real = _mean_loss(model, X_val_real)
        synth = _mean_loss(model, X_val_synth)
        print(f"  {name:30s} {real:10.2f} dB {synth:10.2f} dB {real - synth:+6.2f} dB")

    # Ablation.
    n_audio = N_AUDIO_FEATURES
    audio_mask = np.zeros(X_train.shape[1], dtype=bool)
    audio_mask[:n_audio] = True

    def _mask(X, mask):
        out = X.copy()
        out[:, ~mask] = 0.0
        return out

    print(f"\n  {'Ablation':30s} {'Real audio':>12s} {'Synthetic':>12s} {'Gap':>8s}")
    print(f"  {'-'*65}")
    for name, mask in [
        ("audio-only", audio_mask),
        ("metadata-only", ~audio_mask),
        ("full", np.ones(X_train.shape[1], dtype=bool)),
    ]:
        model = train_xgboost(_mask(X_train, mask), Y_train)
        real = _mean_loss(model, _mask(X_val_real, mask))
        synth = _mean_loss(model, _mask(X_val_synth, mask))
        print(f"  {name:30s} {real:10.2f} dB {synth:10.2f} dB {real - synth:+6.2f} dB")
