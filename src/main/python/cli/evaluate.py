#!/usr/bin/env python3
"""Full-library evaluation pipeline for the production model.

Loads the saved production model (not retrained), runs it on ALL
catalogue-matched titles, compares predictions against BEQ catalogue
ground truth using downstream_loss(), and produces a comprehensive
report (console + CSV + JSON).

Key difference from reassess (tier1 comparison):
  - Reassess: trains from scratch, evaluates on held-out split only.
    Measures generalisation - "which approach learns best?" (research)
  - Evaluate: loads saved production model, evaluates on ALL matched
    titles. Measures real-world performance - "how does the deployed
    model perform on my library?" (QA)

Usage:
    bin/beq-designer dev evaluate
    poetry run python -m cli.evaluate
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("evaluate")

_FS = 1000  # 1 kHz sample rate for curves


def _load_production_model(shared_dir: Path, auto_yes: bool = False):
    """Load the production XGBoost model from disk.

    Returns (model, metadata, model_path) or raises RuntimeError if
    no model is found.
    """
    import joblib
    from model.model_metadata import (
        check_training_staleness,
        format_model_banner,
        format_staleness_warning,
        load_model_metadata,
    )

    model_path = shared_dir / "production_model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"No production model found at {model_path}.\n"
            "Train one first: bin/beq-designer dev train"
        )

    metadata = load_model_metadata(model_path)
    banner = format_model_banner(metadata, model_path)
    log.info("Production model:\n%s", banner)

    # Check staleness and prompt if needed.
    try:
        from model.wav_discovery import _latest_wav_mtime, wav_cache_dir
        current_mtime = _latest_wav_mtime(wav_cache_dir())
    except Exception:
        current_mtime = 0.0

    staleness = check_training_staleness(metadata, current_mtime)
    warning = format_staleness_warning(staleness, metadata)
    if warning:
        log.warning(warning)

    if staleness["stale"] and not auto_yes:
        if sys.stdin.isatty():
            answer = input("Training data has changed. Retrain first? [y/N] ").strip().lower()
            if answer == "y":
                log.info("Retraining requested - run: bin/beq-designer dev train")
                sys.exit(0)

    try:
        model = joblib.load(str(model_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {model_path}: {exc}\n"
            "The model file may be corrupt. Retrain: bin/beq-designer dev train"
        ) from exc

    return model, metadata, model_path


def _load_e85_predictor(shared_dir: Path):
    """Try to load the E85 torch model. Returns predictor or None."""
    e85_path = shared_dir / "e85_torch_filter.pt"
    if not e85_path.exists():
        log.info("E85 torch model not found at %s - skipping E85 column", e85_path)
        return None
    try:
        from model.auto_beq_torch import load_torch_predictor
        predictor = load_torch_predictor(str(e85_path))
        return predictor
    except Exception as exc:
        log.warning("Failed to load E85 model: %s", exc)
        return None


def _reconstruct_split(all_real: list[tuple]) -> tuple[set[int], set[int]]:
    """Reconstruct the train/held-out split using the same logic as tier1.

    Returns (train_indices, test_indices) as sets for O(1) lookup.
    Uses the same severity classification and random_state=42 as
    run_tier1_comparison.py so membership labels are deterministic.
    """
    import numpy as np
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
        indices, test_size=0.2, random_state=42,
        stratify=severity if len(set(severity)) > 1 else None,
    )
    return set(int(i) for i in train_idx), set(int(i) for i in test_idx)


def _evaluate_pipeline(
    shared_dir: Path,
    auto_yes: bool = False,
) -> dict:
    """Run the full evaluation pipeline. Returns structured results dict."""
    import numpy as np
    from model.auto_beq import DEFAULT_GRID
    from model.auto_beq_metadata import enrich_media_metadata
    from model.auto_beq_nn import (
        AudioFeatureConfig,
        build_feature_vector,
        downstream_loss,
        labels_to_filters,
    )
    from model.media_utils import ProgressLogger
    from model.training_data import prepare_training_data

    t_global = time.time()

    # --- Stage 1: Load model ---
    log.info("=" * 70)
    log.info("  PRODUCTION MODEL EVALUATION")
    log.info("")
    log.info("  Evaluates the production model on ALL catalogue-matched titles.")
    log.info("  Titles marked 'train' were used during model training;")
    log.info("  'held-out' titles were not. A large gap between train and")
    log.info("  held-out performance indicates overfitting.")
    log.info("=" * 70)

    model, metadata, model_path = _load_production_model(shared_dir, auto_yes)

    e85_predictor = _load_e85_predictor(shared_dir)
    has_e85 = e85_predictor is not None

    # --- Stage 2+3: Discover pairs + extract features + load metadata ---
    data = prepare_training_data(fetch_catalogue=False, min_pairs=1)
    all_real = data["all_real"]
    tmdb_cache = data["tmdb_cache"]

    if not all_real:
        raise RuntimeError("Feature extraction produced zero results")

    # --- Stage 5 (before scoring): Reconstruct split ---
    train_indices, test_indices = _reconstruct_split(all_real)
    log.info(
        "Reconstructed split: %d train, %d held-out",
        len(train_indices), len(test_indices),
    )

    # --- Stage 4: Predict + Score ---
    cfg = AudioFeatureConfig()
    results_list = []
    progress = ProgressLogger(len(all_real), logger=log, min_interval_s=5)

    for idx, (pair, features) in enumerate(all_real):
        entry = pair["catalogue_entry"]
        if not entry.get("filters"):
            progress.update(idx + 1)
            continue

        title = entry.get("title", "?")
        author = entry.get("author", "?")
        tmdb_id = str(entry.get("theMovieDB", ""))

        metadata_enriched = enrich_media_metadata(entry, tmdb_cache)
        x = build_feature_vector(features, metadata_enriched, config=cfg)

        # E82 prediction.
        y_pred = model.predict(x.reshape(1, -1))[0]
        pred_filters = labels_to_filters(y_pred)
        e82_loss = downstream_loss(pred_filters, entry["filters"], DEFAULT_GRID)

        # E85 prediction (if available).
        e85_loss = None
        if has_e85:
            try:
                e85_filters = e85_predictor.predict_filters(x)
                e85_loss = downstream_loss(e85_filters, entry["filters"], DEFAULT_GRID)
            except Exception as exc:
                log.debug("E85 prediction failed for %s: %s", title, exc)

        # Determine split membership.
        if idx in train_indices:
            split = "train"
        elif idx in test_indices:
            split = "held-out"
        else:
            split = "unknown"

        # Catalogue filter stats.
        cat_filters = entry["filters"]
        cat_filter_count = len(cat_filters)
        cat_summed_gain = sum(abs(float(f.get("gain", 0))) for f in cat_filters)

        results_list.append({
            "title": title,
            "author": author,
            "tmdb_id": tmdb_id,
            "split": split,
            "e82_loss_db": round(e82_loss, 4),
            "e85_loss_db": round(e85_loss, 4) if e85_loss is not None else None,
            "catalogue_filter_count": cat_filter_count,
            "catalogue_summed_gain_db": round(cat_summed_gain, 2),
        })

        progress.update(idx + 1, label=title)

    progress.finish(f"scored {len(results_list)} titles")

    if not results_list:
        raise RuntimeError("No titles with filters found for evaluation")

    # Sort by E82 loss (worst first for the "worst 10" report).
    results_list.sort(key=lambda r: r["e82_loss_db"])

    # --- Build summary statistics ---
    all_e82 = [r["e82_loss_db"] for r in results_list]
    train_e82 = [r["e82_loss_db"] for r in results_list if r["split"] == "train"]
    held_out_e82 = [r["e82_loss_db"] for r in results_list if r["split"] == "held-out"]

    summary = {
        "total_titles": len(results_list),
        "train_count": len(train_e82),
        "held_out_count": len(held_out_e82),
        "e82": {
            "all_mean": round(float(np.mean(all_e82)), 4),
            "all_max": round(float(np.max(all_e82)), 4),
            "train_mean": round(float(np.mean(train_e82)), 4) if train_e82 else None,
            "held_out_mean": round(float(np.mean(held_out_e82)), 4) if held_out_e82 else None,
            "overfitting_gap": round(
                float(np.mean(train_e82)) - float(np.mean(held_out_e82)), 4
            ) if train_e82 and held_out_e82 else None,
        },
    }

    if has_e85:
        all_e85 = [r["e85_loss_db"] for r in results_list if r["e85_loss_db"] is not None]
        train_e85 = [r["e85_loss_db"] for r in results_list
                     if r["split"] == "train" and r["e85_loss_db"] is not None]
        held_out_e85 = [r["e85_loss_db"] for r in results_list
                        if r["split"] == "held-out" and r["e85_loss_db"] is not None]
        summary["e85"] = {
            "all_mean": round(float(np.mean(all_e85)), 4) if all_e85 else None,
            "all_max": round(float(np.max(all_e85)), 4) if all_e85 else None,
            "train_mean": round(float(np.mean(train_e85)), 4) if train_e85 else None,
            "held_out_mean": round(float(np.mean(held_out_e85)), 4) if held_out_e85 else None,
            "overfitting_gap": round(
                float(np.mean(train_e85)) - float(np.mean(held_out_e85)), 4
            ) if train_e85 and held_out_e85 else None,
        }

    # Per-author breakdown.
    by_author = defaultdict(list)
    for r in results_list:
        by_author[r["author"]].append(r["e82_loss_db"])
    per_author = {
        author: {
            "count": len(losses),
            "mean": round(float(np.mean(losses)), 4),
            "max": round(float(np.max(losses)), 4),
        }
        for author, losses in sorted(by_author.items(), key=lambda x: -len(x[1]))
    }

    total_time = time.time() - t_global

    return {
        "metadata": {
            "model_path": str(model_path),
            "has_e85": has_e85,
            "evaluation_time_s": round(total_time, 1),
        },
        "summary": summary,
        "per_author": per_author,
        "results": results_list,
    }


def _print_console_report(data: dict) -> None:
    """Print a Rich console report from the structured evaluation data."""
    summary = data["summary"]
    has_e85 = data["metadata"]["has_e85"]
    results = data["results"]
    per_author = data["per_author"]

    print()
    print("=" * 80)
    print(
        f"  PRODUCTION MODEL EVALUATION - {summary['total_titles']} titles "
        f"({summary['train_count']} train, {summary['held_out_count']} held-out)"
    )
    print(f"  Model: {Path(data['metadata']['model_path']).name}")
    print("=" * 80)
    print()

    # Per-title table (sorted by loss ascending - best first).
    e85_header = "  E85 dB" if has_e85 else ""
    print(f"  {'Title':<40s}  {'Author':<12s}  {'Split':<9s}  {'E82 dB':>7s}{e85_header}")
    print(f"  {'-' * (40 + 12 + 9 + 7 + (8 if has_e85 else 0) + 6)}")
    for r in results:
        e85_col = f"  {r['e85_loss_db']:7.2f}" if has_e85 and r["e85_loss_db"] is not None else ""
        title_trunc = r["title"][:40]
        print(
            f"  {title_trunc:<40s}  {r['author']:<12s}  {r['split']:<9s}"
            f"  {r['e82_loss_db']:7.2f}{e85_col}"
        )
    print()

    # Summary statistics.
    print("  Summary:")
    e85_header_short = "   E85 mean" if has_e85 else ""
    print(f"  {'':>20s}  {'E82 mean':>9s}{e85_header_short}")
    e82 = summary["e82"]

    def _fmt_e85(key):
        if not has_e85:
            return ""
        val = summary.get("e85", {}).get(key)
        return f"  {val:9.2f}" if val is not None else "       n/a"

    print(f"  {'All (' + str(summary['total_titles']) + ')':<20s}  {e82['all_mean']:9.2f}{_fmt_e85('all_mean')}")
    if e82["train_mean"] is not None:
        print(f"  {'Train (' + str(summary['train_count']) + ')':<20s}  {e82['train_mean']:9.2f}{_fmt_e85('train_mean')}")
    if e82["held_out_mean"] is not None:
        print(f"  {'Held-out (' + str(summary['held_out_count']) + ')':<20s}  {e82['held_out_mean']:9.2f}{_fmt_e85('held_out_mean')}")
    if e82["overfitting_gap"] is not None:
        e85_gap = ""
        if has_e85 and summary.get("e85", {}).get("overfitting_gap") is not None:
            e85_gap = f"  {summary['e85']['overfitting_gap']:9.2f}"
        print(f"  {'Gap (train-held)':<20s}  {e82['overfitting_gap']:9.2f}{e85_gap}  <-- overfitting indicator")
    print()

    # Per-author breakdown.
    print("  Per-author:")
    print(f"  {'Author':<15s}  {'Count':>6s}  {'Mean dB':>8s}  {'Max dB':>8s}")
    print(f"  {'-' * 45}")
    for author, stats in per_author.items():
        print(f"  {author:<15s}  {stats['count']:6d}  {stats['mean']:8.2f}  {stats['max']:8.2f}")
    print()

    # Worst 10 titles.
    worst_10 = sorted(results, key=lambda r: -r["e82_loss_db"])[:10]
    print("  Worst 10 titles:")
    for r in worst_10:
        e85_str = f" / E85={r['e85_loss_db']:.2f}" if has_e85 and r["e85_loss_db"] is not None else ""
        print(f"    {r['title'][:50]:<50s}  E82={r['e82_loss_db']:.2f}{e85_str}  ({r['split']})")
    print()

    print(f"  Evaluation completed in {data['metadata']['evaluation_time_s']:.1f}s")
    print()


def _save_csv(data: dict, csv_path: Path) -> None:
    """Save per-title results to CSV."""
    results = data["results"]
    has_e85 = data["metadata"]["has_e85"]

    fieldnames = [
        "title", "author", "tmdb_id", "split",
        "e82_loss_db",
    ]
    if has_e85:
        fieldnames.append("e85_loss_db")
    fieldnames.extend(["catalogue_filter_count", "catalogue_summed_gain_db"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            if not has_e85:
                row.pop("e85_loss_db", None)
            writer.writerow(row)
    log.info("CSV written: %s", csv_path)


def _save_json(data: dict, json_path: Path) -> None:
    """Save full structured results to JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    log.info("JSON written: %s", json_path)


def main(argv: list[str] | None = None):
    """Entry point for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production model on all catalogue-matched titles. "
            "Measures real-world performance - how the deployed model performs "
            "on your library."
        ),
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip retrain prompt (proceed with current model).",
    )
    args = parser.parse_args(argv)

    # Configure logging only when called as main entry point.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(name)s - %(message)s",
        )

    # Late imports so --help is fast.
    from model.wav_discovery import beq_shared_dir

    auto_yes = args.yes or os.environ.get("BEQ_AUTO_YES") == "1"
    shared_dir = beq_shared_dir()

    data = _evaluate_pipeline(shared_dir, auto_yes=auto_yes)

    # Output to console.
    _print_console_report(data)

    # Save CSV and JSON.
    csv_path = shared_dir / "evaluation_results.csv"
    json_path = shared_dir / "evaluation_results.json"
    _save_csv(data, csv_path)
    _save_json(data, json_path)

    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print()


if __name__ == "__main__":
    main()
