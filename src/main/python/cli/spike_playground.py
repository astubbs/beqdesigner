#!/usr/bin/env python3
"""CLI playground for the auto-BEQ spike.

Secondary deliverable - the primary one is the pytest integration test at
src/test/python/auto_beq/test_auto_beq.py. Use this script to poke at things
interactively without editing test fixtures.

Examples:
    # Synthetic roundtrip from the committed catalogue snapshot:
    bin/beq-designer dev playground \
        --entry-title "Battle: Los Angeles" --filter-count 1

    # Synthetic + optional matplotlib plot:
    bin/beq-designer dev playground \
        --entry-title "Battle: Los Angeles" --filter-count 1 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # cli/ -> python/ -> main/ -> src/ -> repo
sys.path.insert(0, str(REPO_ROOT / "src" / "main" / "python"))


from model.auto_beq import (  # noqa: E402
    DEFAULT_GRID,
    compute_match_metrics,
    evaluate_filter_chain,
    format_match_report,
    propose_filters,
)

SNAPSHOT_PATH = REPO_ROOT / "src" / "test" / "resources" / "auto_beq" / "database.json"


def load_snapshot_entry(title: str, filter_count: int | None) -> dict:
    with SNAPSHOT_PATH.open() as f:
        entries = json.load(f)
    matches = [
        e
        for e in entries
        if e.get("title") == title
        and (filter_count is None or len(e.get("filters", [])) == filter_count)
    ]
    if not matches:
        available = sorted({e["title"] for e in entries})
        raise SystemExit(
            f"No entry for title={title!r} filter_count={filter_count!r}.\n"
            f"Available titles: {', '.join(available)}"
        )
    return matches[0]


def maybe_plot(freqs, target, proposed_resp, ground_truth_resp, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freqs, target, label="target (rolloff)", linewidth=2)
    ax.semilogx(freqs, ground_truth_resp, label="catalogue filters", linestyle="--")
    ax.semilogx(freqs, proposed_resp, label="proposed filters", linestyle="-.")
    ax.semilogx(freqs, target + proposed_resp, label="residual (target+proposed)", alpha=0.7)
    ax.axvspan(20, 80, alpha=0.1, color="gray", label="scoring band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Auto-BEQ: {title}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry-title", required=True, help="title in the catalogue snapshot")
    p.add_argument("--filter-count", type=int, default=None,
                   help="disambiguate when multiple entries share a title")
    p.add_argument("--fs", type=int, default=1000, help="sample rate for filter construction")
    p.add_argument("--plot", action="store_true", help="show matplotlib overlay plot")
    args = p.parse_args()

    entry = load_snapshot_entry(args.entry_title, args.filter_count)
    freqs = DEFAULT_GRID

    ground_truth_resp = evaluate_filter_chain(entry["filters"], freqs, fs=args.fs)
    target = -ground_truth_resp  # rolled-off input

    proposed = propose_filters(target, freqs, fs=args.fs)
    metrics = compute_match_metrics(target, proposed, freqs, fs=args.fs)
    report = format_match_report(
        args.entry_title,
        entry["filters"],
        proposed,
        metrics,
        target_depth_db=float(target[0] - target[-1]),
    )
    print(report)

    if args.plot:
        proposed_resp = evaluate_filter_chain(proposed, freqs, fs=args.fs)
        maybe_plot(freqs, target, proposed_resp, ground_truth_resp, args.entry_title)


if __name__ == "__main__":
    main()
