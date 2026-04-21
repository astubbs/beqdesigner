"""Model metadata utilities - load and format model provenance info.

Stdlib-only - no scipy, numpy, or PyQt dependencies. Safe to import
from standalone scripts (Docker containers, NAS extraction, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, date
from pathlib import Path

log = logging.getLogger(__name__)


def load_model_metadata(model_path: Path) -> dict | None:
    """Load the .meta.json sidecar for a model file.

    Parameters
    ----------
    model_path : Path
        Path to the .joblib model file.

    Returns
    -------
    dict or None
        Parsed metadata dict, or None if the sidecar is missing or corrupt.
    """
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("corrupt metadata sidecar %s: %s", meta_path, exc)
        return None


def format_age(unix_timestamp: float) -> str:
    """Return a human-readable relative time string.

    Examples: "just now", "2 minutes ago", "3 hours ago", "2 days ago",
    "2 weeks ago", "3 months ago".
    """
    delta_s = time.time() - unix_timestamp
    if delta_s < 60:
        return "just now"
    minutes = int(delta_s // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(delta_s // 3600)
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(delta_s // 86400)
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    if days < 60:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    return f"{months} month{'s' if months != 1 else ''} ago"


def format_model_banner(metadata: dict | None, model_path: Path | None = None) -> str:
    """Format a multi-line banner showing model provenance.

    Parameters
    ----------
    metadata : dict or None
        Parsed metadata from the .meta.json sidecar.
    model_path : Path or None
        Path to the model file (used for the filename line).

    Returns
    -------
    str
        Multi-line string suitable for logging or printing.
    """
    filename = model_path.name if model_path else "unknown"

    if metadata is None:
        return f"Model: {filename} (no metadata sidecar - retrain to generate)"

    lines = [f"Model: {filename}"]

    # Trained timestamp + age
    trained_at = metadata.get("trained_at")
    if trained_at is not None:
        dt = datetime.fromtimestamp(trained_at)
        age = format_age(trained_at)
        lines.append(f"Trained: {dt:%Y-%m-%d %H:%M} ({age})")

    # Features
    n_features = metadata.get("n_features")
    feature_config = metadata.get("feature_config", "")
    foundation = metadata.get("foundation_model")
    if n_features is not None:
        if foundation:
            feat_desc = f"{n_features} ({feature_config}, foundation: {foundation})"
        else:
            feat_desc = f"{n_features} ({feature_config}, no foundation model)"
        lines.append(f"Features: {feat_desc}")

    # Training data
    n_real = metadata.get("n_real")
    n_synth = metadata.get("n_synth")
    real_weight = metadata.get("real_weight")
    if n_real is not None and n_synth is not None:
        weight_str = ""
        if real_weight is not None:
            ratio = int(real_weight) if real_weight == int(real_weight) else real_weight
            weight_str = f" ({ratio}:1 weight)"
        lines.append(f"Training data: {n_real} real{weight_str} + {n_synth} synthetic")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Champion history - track best model across tier1 comparison runs
# ---------------------------------------------------------------------------


def load_champion_history(history_path: Path) -> dict | None:
    """Read champion_history.json from the given path.

    Parameters
    ----------
    history_path : Path
        Path to the champion_history.json file.

    Returns
    -------
    dict or None
        Parsed history dict with ``current`` and ``history`` keys,
        or None if the file is missing or corrupt.
    """
    if not history_path.exists():
        return None
    try:
        return json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("corrupt champion history %s: %s", history_path, exc)
        return None


def record_champion(
    history_path: Path,
    experiment: str,
    mean_db: float,
    wav_count: int,
) -> dict:
    """Record a new champion if *mean_db* improves on the current best.

    Only updates when *mean_db* is strictly lower than the current champion
    (or when no champion exists yet). Writes atomically via a .tmp file
    to avoid corruption from interrupted writes.

    Parameters
    ----------
    history_path : Path
        Path to champion_history.json (created if absent).
    experiment : str
        Name of the experiment (e.g. "E85 diff-DSP").
    mean_db : float
        Mean downstream loss in dB.
    wav_count : int
        Number of WAV pairs used in the comparison.

    Returns
    -------
    dict
        The (possibly updated) history dict.
    """
    history = load_champion_history(history_path)
    if history is None:
        history = {"current": None, "history": []}

    entry = {
        "experiment": experiment,
        "mean_db": round(mean_db, 4),
        "date": date.today().isoformat(),
        "wav_count": wav_count,
    }

    current = history.get("current")
    if current is None or mean_db < current.get("mean_db", float("inf")):
        history["current"] = entry
        history.setdefault("history", []).append(entry)
    # else: no improvement, keep current champion unchanged

    # Atomic write: write to .tmp, then rename.
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(history, indent=2) + "\n")
    os.replace(str(tmp_path), str(history_path))

    return history


def format_champion_comparison(
    history: dict | None,
    results: list[tuple],
) -> str:
    """Format a human-readable champion progression summary.

    Parameters
    ----------
    history : dict or None
        Previously loaded champion history (from ``load_champion_history``).
    results : list[tuple]
        List of (name, mean_db, max_db, train_time, per_author, verdict)
        tuples from the current tier1 comparison run.

    Returns
    -------
    str
        Multi-line comparison string suitable for printing.
    """
    if not results:
        return ""

    best = min(results, key=lambda r: r[1])
    best_name, best_mean = best[0], best[1]

    if history is None or history.get("current") is None:
        return "First comparison run - establishing baseline"

    current = history["current"]
    prev_name = current["experiment"]
    prev_mean = current["mean_db"]
    prev_date = current.get("date", "unknown")

    delta = best_mean - prev_mean
    if delta < -0.01:
        return (
            f"Previous champion: {prev_name} ({prev_mean:.2f} dB mean, {prev_date})\n"
            f"New champion:      {best_name} ({best_mean:.2f} dB mean, {delta:.2f} dB improvement)"
        )
    else:
        today = date.today().isoformat()
        return f"Current champion unchanged: {prev_name} ({prev_mean:.2f} dB mean, {today})"


# ---------------------------------------------------------------------------
# Training staleness detection
# ---------------------------------------------------------------------------


def check_training_staleness(
    metadata: dict | None,
    current_wav_mtime: float,
) -> dict:
    """Check whether training data has changed since the model was trained.

    Parameters
    ----------
    metadata : dict or None
        Parsed model metadata (from ``load_model_metadata``).
    current_wav_mtime : float
        Current WAV cache mtime (from ``_latest_wav_mtime()``).

    Returns
    -------
    dict
        Keys: ``stale`` (bool), ``reason`` (str).
    """
    if metadata is None or "wav_cache_mtime" not in metadata:
        return {
            "stale": False,
            "reason": "no model metadata - retrain to enable staleness detection",
        }
    if current_wav_mtime == 0.0:
        return {"stale": False, "reason": "WAV cache not configured"}
    if current_wav_mtime != metadata["wav_cache_mtime"]:
        return {
            "stale": True,
            "reason": "WAV cache has changed since model was trained",
        }
    return {"stale": False, "reason": "up to date"}


def format_staleness_warning(staleness: dict, metadata: dict | None = None) -> str:
    """Format a staleness warning for display.

    Parameters
    ----------
    staleness : dict
        Result from ``check_training_staleness``.
    metadata : dict or None
        Model metadata (unused currently, reserved for future detail).

    Returns
    -------
    str
        Multi-line warning if stale, single-line info if metadata is missing,
        or empty string if up to date.
    """
    if staleness["stale"]:
        return (
            "WARNING: Training data has changed since the production model was trained.\n"
            "  Consider retraining: bin/beq-designer dev train"
        )
    if "no model metadata" in staleness["reason"]:
        return "INFO: No model metadata available - retrain to enable staleness detection."
    # "up to date" or "WAV cache not configured" - no output needed
    return ""
