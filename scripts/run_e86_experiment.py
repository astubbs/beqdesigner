#!/usr/bin/env python3
"""E86: multi-type topology experiment (LowShelf + HighShelf + PeakingEQ).

Usage:
    BEQ_WAV_CACHE=/Volumes/jetspeed/beqdesigner/wav-cache \
      poetry run python3 scripts/run_e86_experiment.py
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s - %(message)s")
log = logging.getLogger("e86")


def main():
    import numpy as np
    from sklearn.model_selection import train_test_split

    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_catalogue import _fetch_or_cache
    from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
    from model.auto_beq_nn import (
        AudioFeatureConfig, build_feature_vector, deduplicate_by_title,
        downstream_loss, labels_to_filters, train_production_weighted_hybrid,
    )
    from model.auto_beq_torch import E85TrainingConfig, train_e85_differentiable_dsp
    from spike._auto_beq_helpers import STRATEGY_BLENDED_07, discover_wav_catalogue_pairs_cached
    from spike.test_auto_beq_nn_real import _extract_features_parallel

    cfg = AudioFeatureConfig()
    _FS = 1000

    pairs = discover_wav_catalogue_pairs_cached()
    all_real = _extract_features_parallel(pairs, DEFAULT_GRID, _FS, strategy=STRATEGY_BLENDED_07)
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch([p["catalogue_entry"] for p, _ in all_real], cache=tmdb_cache)

    real_entries = [p["catalogue_entry"] for p, _ in all_real]
    severity = [
        "heavy" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 20
        else "moderate" if sum(abs(float(f.get("gain", 0))) for f in e.get("filters", [])) >= 10
        else "gentle"
        for e in real_entries
    ]
    indices = np.arange(len(all_real))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=severity if len(set(severity)) > 1 else None,
    )

    X_train_list, entries_train = [], []
    for i in train_idx:
        p, f = all_real[i]
        e = p["catalogue_entry"]
        if not e.get("filters"):
            continue
        X_train_list.append(build_feature_vector(f, enrich_media_metadata(e, tmdb_cache), config=cfg))
        entries_train.append(e)
    X_train = np.array(X_train_list, dtype=np.float32)

    X_test_list, entries_test = [], []
    for i in test_idx:
        p, f = all_real[i]
        e = p["catalogue_entry"]
        if not e.get("filters"):
            continue
        X_test_list.append(build_feature_vector(f, enrich_media_metadata(e, tmdb_cache), config=cfg))
        entries_test.append(e)
    X_test = np.array(X_test_list, dtype=np.float32)

    log.info("split: %d train / %d test", len(X_train), len(X_test))

    # Baseline E82.
    train_samples = [(all_real[i][0]["catalogue_entry"], all_real[i][1]) for i in train_idx]
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    exclude = (
        {str(e.get("theMovieDB", "")).strip() for e, _ in train_samples}
        | {str(e.get("theMovieDB", "")).strip() for e in entries_test}
    )
    synth = [e for e in deduped if str(e.get("theMovieDB", "")).strip() not in exclude]
    teacher, _ = train_production_weighted_hybrid(
        real_samples=train_samples, synth_entries=synth,
        tmdb_cache=tmdb_cache, freqs_hz=DEFAULT_GRID, fs=_FS,
        real_weight=50.0, config=cfg,
    )

    def _eval(predictor, label, is_torch=True):
        losses = []
        for i, e in enumerate(entries_test):
            if is_torch:
                filters = predictor.predict_filters(X_test[i])
            else:
                y = predictor.predict(X_test[i:i + 1])[0]
                filters = labels_to_filters(y)
            losses.append(downstream_loss(filters, e["filters"], DEFAULT_GRID))
        m, mx = float(np.mean(losses)), float(np.max(losses))
        by_a: dict[str, list[float]] = defaultdict(list)
        for e, l in zip(entries_test, losses, strict=True):
            by_a[e.get("author", "?")].append(l)
        pa = {a: float(np.mean(v)) for a, v in by_a.items()}
        log.info("%-20s mean=%.2f max=%.2f", label, m, mx)
        return m, mx, pa

    base_m, base_mx, base_pa = _eval(teacher, "E82-baseline", is_torch=False)

    # E85 (v1 LowShelf-only).
    p85, _ = train_e85_differentiable_dsp(
        X_train=X_train, entries_train=entries_train,
        teacher_model=teacher, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=E85TrainingConfig(n_epochs_warm=10, n_epochs_acoustic=20),
    )
    e85_m, e85_mx, e85_pa = _eval(p85, "E85-LowShelf-only")

    # E86 (v2 multi-type).
    p86, _ = train_e85_differentiable_dsp(
        X_train=X_train, entries_train=entries_train,
        teacher_model=teacher, eval_freqs_hz=DEFAULT_GRID, fs=_FS,
        config=E85TrainingConfig(n_epochs_warm=10, n_epochs_acoustic=20, multi_type=True),
    )
    e86_m, e86_mx, e86_pa = _eval(p86, "E86-multi-type")

    # Report.
    print(f"\n{'=' * 72}")
    print(f"  E86 MULTI-TYPE TOPOLOGY COMPARISON")
    print(f"  {len(X_train)} train / {len(X_test)} test")
    print(f"{'=' * 72}\n")
    print(f"  {'Model':<22s}  {'mean':>8s}  {'max':>8s}  {'Δ mean':>8s}")
    print(f"  {'-' * 50}")
    for nm, m, mx in [
        ("E82 baseline", base_m, base_mx),
        ("E85 LowShelf-only", e85_m, e85_mx),
        ("E86 multi-type", e86_m, e86_mx),
    ]:
        d = m - base_m
        ds = f"{d:+.2f}" if nm != "E82 baseline" else "—"
        print(f"  {nm:<22s}  {m:6.2f}    {mx:6.2f}    {ds:>8s}")
    print(f"\n  Per-author:")
    for a in sorted(set(base_pa) | set(e85_pa) | set(e86_pa)):
        print(
            f"    {a:<15s}  E82={base_pa.get(a, 0):.2f}  "
            f"E85={e85_pa.get(a, 0):.2f}  E86={e86_pa.get(a, 0):.2f}"
        )
    print()


if __name__ == "__main__":
    main()
