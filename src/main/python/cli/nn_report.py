#!/usr/bin/env python3
"""Generate a comparison report: NN-predicted vs hand-coded BEQ filters.

For every title in the WAV cache that has a catalogue entry, trains the
best model (late fusion XGBoost), predicts filter parameters, and compares
against the catalogue's hand-coded filters.

Outputs a markdown report showing per-title predicted vs catalogue filters,
grouped by quality tier (< 2 dB, 2-4 dB, > 5 dB).

Usage:
    bin/beq-designer nn-report
    BEQ_WAV_CACHE=/path/to/wav-cache bin/beq-designer nn-report
    bin/beq-designer nn-report --output report.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src" / "main" / "python", _REPO_ROOT / "src" / "test" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

from model.auto_beq import DEFAULT_GRID, evaluate_filter_chain
from model.auto_beq_advisor import extract_curve_features
from model.auto_beq_catalogue import _fetch_or_cache
from model.auto_beq_metadata import enrich_media_metadata, fetch_metadata_batch, load_cache
from model.auto_beq_nn import (
    build_feature_vector,
    catalogue_entry_to_labels,
    deduplicate_by_title,
    downstream_loss,
    labels_to_filters,
    train_late_fusion,
)
from spike._auto_beq_helpers import (
    STRATEGY_WELCH,
    cached_extract_features_with_strategy,
    discover_wav_catalogue_pairs_cached,
)

log = logging.getLogger("nn_comparison_report")


def _synthetic_features(entry: dict, freqs: np.ndarray):
    correction = evaluate_filter_chain(entry["filters"], freqs, fs=1000)
    rolloff = -correction
    anchor = int(np.argmin(np.abs(freqs - 80.0)))
    rolloff -= rolloff[anchor]
    return extract_curve_features(rolloff, freqs)


def _fmt_filters(filters: list[dict]) -> str:
    if not filters:
        return "(none)"
    return ", ".join(
        f"{f['type']}({f['freq']:.0f}Hz, {f['gain']:+.1f}dB, Q={f['q']:.1f})"
        for f in filters
    )


def generate_report(output=None):
    out = output or sys.stdout

    def pr(s=""):
        print(s, file=out)

    t0 = time.time()

    # Discover WAVs.
    pairs = discover_wav_catalogue_pairs_cached()
    if not pairs:
        pr("No WAV files found in cache. Run extract_lfe.py first.")
        return

    # Load catalogue + metadata.
    catalogue = _fetch_or_cache()
    deduped = [e for e in deduplicate_by_title(catalogue) if e.get("filters")]
    tmdb_cache = load_cache()
    tmdb_cache = fetch_metadata_batch(deduped, cache=tmdb_cache)

    # Build synthetic training set (excluding validation titles).
    val_tmdb = {p["tmdb_id"] for p in pairs if p.get("tmdb_id")}
    X_train, Y_train = [], []
    for e in deduped:
        if str(e.get("theMovieDB", "")).strip() in val_tmdb:
            continue
        features = _synthetic_features(e, DEFAULT_GRID)
        metadata = enrich_media_metadata(e, tmdb_cache)
        X_train.append(build_feature_vector(features, metadata))
        Y_train.append(catalogue_entry_to_labels(e))
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)

    # Train best model.
    model = train_late_fusion(X_train, Y_train, alpha=0.3)

    # Predict for each title.
    results = []
    seen = set()
    for p in pairs:
        entry = p["catalogue_entry"]
        if not entry.get("filters"):
            continue
        title = entry.get("title", "?")
        if title in seen:
            continue
        seen.add(title)
        try:
            features = cached_extract_features_with_strategy(
                p["wav_path"], DEFAULT_GRID, 1000, strategy=STRATEGY_WELCH,
            )
        except Exception:
            continue
        metadata = enrich_media_metadata(entry, tmdb_cache)
        x = build_feature_vector(features, metadata)
        y_pred = model.predict(x.reshape(1, -1))[0]
        pred_filters = labels_to_filters(y_pred)
        target_filters = entry["filters"]
        loss = downstream_loss(pred_filters, target_filters, DEFAULT_GRID)

        results.append({
            "title": title,
            "year": str(entry.get("year", "")),
            "author": entry.get("author", "?"),
            "content_type": "TV" if entry.get("content_type") == "TV" else "film",
            "loss": loss,
            "predicted": _fmt_filters(pred_filters),
            "catalogue": _fmt_filters(target_filters),
            "n_pred": len(pred_filters),
            "n_cat": len(target_filters),
        })

    results.sort(key=lambda r: r["loss"])
    elapsed = time.time() - t0

    # Statistics.
    under2 = sum(1 for r in results if r["loss"] < 2)
    under3 = sum(1 for r in results if r["loss"] < 3)
    under5 = sum(1 for r in results if r["loss"] < 5)
    mean = sum(r["loss"] for r in results) / len(results) if results else 0

    by_author = {}
    for r in results:
        a = r["author"]
        by_author.setdefault(a, []).append(r["loss"])

    # Report.
    pr("# BEQ Filter Prediction — Comparison Report")
    pr()
    pr(f"Generated from {len(results)} unique titles with real extracted LFE audio,")
    pr(f"compared against hand-coded BEQ catalogue entries.")
    pr()
    pr("## Summary")
    pr()
    pr(f"| Metric | Value |")
    pr(f"|---|---|")
    pr(f"| Model | Late fusion XGBoost (α=0.3), one-hot filter types |")
    pr(f"| Training data | {len(X_train)} synthetic catalogue entries (no real audio needed) |")
    pr(f"| Validation titles | {len(results)} unique titles with real LFE WAVs |")
    pr(f"| Mean downstream error | **{mean:.2f} dB** (20-80 Hz band) |")
    pr(f"| Under 2 dB (expert quality) | {under2}/{len(results)} ({under2 * 100 // len(results)}%) |")
    pr(f"| Under 3 dB (good starting point) | {under3}/{len(results)} ({under3 * 100 // len(results)}%) |")
    pr(f"| Under 5 dB (usable) | {under5}/{len(results)} ({under5 * 100 // len(results)}%) |")
    pr(f"| Report generation time | {elapsed:.0f}s |")
    pr()

    pr("## Per-author performance")
    pr()
    pr(f"| Author | Titles | Mean | <2 dB | >4 dB |")
    pr(f"|---|---|---|---|---|")
    for a in sorted(by_author, key=lambda a: -len(by_author[a])):
        losses = by_author[a]
        m = sum(losses) / len(losses)
        good = sum(1 for l in losses if l < 2)
        bad = sum(1 for l in losses if l >= 4)
        pr(f"| {a} | {len(losses)} | {m:.2f} dB | {good} | {bad} |")
    pr()

    # Best predictions.
    best = [r for r in results if r["loss"] < 2]
    if best:
        pr(f"## Best predictions — under 2 dB ({len(best)} titles)")
        pr()
        pr("These match expert quality. The predicted filter chain produces a")
        pr("frequency response within 2 dB of the hand-coded catalogue entry")
        pr("across the 20-80 Hz bass extension band.")
        pr()
        for r in best:
            pr(f"### {r['title']} ({r['year']}) — {r['loss']:.2f} dB — by {r['author']}")
            pr(f"- **Predicted ({r['n_pred']} filters)**: {r['predicted']}")
            pr(f"- **Catalogue ({r['n_cat']} filters)**: {r['catalogue']}")
            pr()

    # Mid-range.
    mid = [r for r in results if 2 <= r["loss"] < 4]
    if mid:
        pr(f"## Good predictions — 2 to 4 dB ({len(mid)} titles)")
        pr()
        pr("Usable as a starting point. The predicted filters are in the right")
        pr("ballpark but may need manual tweaking of gain or frequency.")
        pr()
        for r in mid[:15]:
            pr(f"### {r['title']} ({r['year']}) — {r['loss']:.2f} dB — by {r['author']}")
            pr(f"- **Predicted ({r['n_pred']} filters)**: {r['predicted']}")
            pr(f"- **Catalogue ({r['n_cat']} filters)**: {r['catalogue']}")
            pr()
        if len(mid) > 15:
            pr(f"*... and {len(mid) - 15} more titles in this range.*")
            pr()

    # Worst.
    worst = [r for r in results if r["loss"] >= 5]
    if worst:
        pr(f"## Needs work — over 5 dB ({len(worst)} titles)")
        pr()
        pr("These predictions are too far from the catalogue to be directly usable.")
        pr("Typically older films (pre-1990) or titles with unusual rolloff patterns.")
        pr()
        for r in worst[:10]:
            pr(f"### {r['title']} ({r['year']}) — {r['loss']:.2f} dB — by {r['author']}")
            pr(f"- **Predicted ({r['n_pred']} filters)**: {r['predicted']}")
            pr(f"- **Catalogue ({r['n_cat']} filters)**: {r['catalogue']}")
            pr()
        if len(worst) > 10:
            pr(f"*... and {len(worst) - 10} more titles in this range.*")
            pr()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate comparison report: NN-predicted vs hand-coded BEQ filters.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output file (markdown). Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    if args.output:
        with args.output.open("w") as f:
            generate_report(output=f)
        print(f"Report written to {args.output}")
    else:
        generate_report()


if __name__ == "__main__":
    main()
