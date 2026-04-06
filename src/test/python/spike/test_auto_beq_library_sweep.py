"""Library sweep: run the auto-BEQ pipeline over every film in the
user's library that has a matching catalogue entry.

Reads ``~/.config/beqdesigner/auto_beq_sweep.json`` (produced by
``python -m spike.sweep_discover``) and runs the full
extract-smooth-advise-fit-grade pipeline on the top N films (N from
``test_limit`` in the config, overridable via
``AUTO_BEQ_SWEEP_LIMIT``).

Skipped when the config file is absent — run the discovery CLI to
initialise it. CI stays green; a clear skip message points the user
at the next step.

No assertions: the sweep is informational. Each run appends a CSV row
per film to ``.pytest_cache/auto_beq_sweep.csv``.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from model.auto_beq import (
    DEFAULT_GRID,
    compute_match_metrics,
    evaluate_filter_chain,
)
from model.auto_beq_advisor import MediaMetadata, get_advisor

from spike._auto_beq_helpers import (
    _extract_lfe_wav,
    _have_tool,
    _probe_audio_stream,
    load_and_smooth,
)
from spike.sweep_discover import bucket_rating, load_catalogue_by_digest, load_config

log = logging.getLogger("auto_beq_sweep")

_DEFAULT_TEST_LIMIT = 10
_STALE_CONFIG_DAYS = 30
_SWEEP_REPORT_PATH = Path(os.environ.get(
    "AUTO_BEQ_SWEEP_REPORT", ".pytest_cache/auto_beq_sweep.csv",
))
_CONFIG_PATH = Path(os.environ.get(
    "AUTO_BEQ_SWEEP_CONFIG",
    str(Path.home() / ".config" / "beqdesigner" / "auto_beq_sweep.json"),
))


@dataclass(frozen=True)
class SweepFilm:
    """Parametrised-test payload: one film from the sweep config."""
    path: Path
    title: str
    year: int | None
    rating: float | None
    catalogue_entry: dict[str, Any]


def _warn_if_stale(generated_at: str) -> None:
    try:
        ts = dt.datetime.fromisoformat(generated_at)
    except ValueError:
        return
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - ts
    if age.days > _STALE_CONFIG_DAYS:
        msg = (
            f"auto_beq_sweep config is {age.days} days old "
            f"(generated {generated_at}); consider re-running: "
            "poetry run python -m spike.sweep_discover --refresh"
        )
        warnings.warn(msg, stacklevel=2)
        print(f"\n[auto_beq_sweep] WARNING: {msg}", file=sys.stderr)


def _load_films() -> tuple[list[SweepFilm], str | None]:
    """Return (films, skip_reason). films may be empty; skip_reason set iff skipped."""
    config = load_config(_CONFIG_PATH)
    if config is None:
        return [], (
            "sweep config not initialised at "
            f"{_CONFIG_PATH} — run: poetry run python -m spike.sweep_discover"
        )
    _warn_if_stale(config.get("generated_at", ""))
    limit = int(os.environ.get(
        "AUTO_BEQ_SWEEP_LIMIT",
        str(config.get("test_limit", _DEFAULT_TEST_LIMIT)),
    ))
    # v3: films store catalogue_digest, look up from cached catalogue.
    # v1/v2 fallback for older configs.
    catalogue_entries = config.get("catalogue_entries", [])
    digest_index = load_catalogue_by_digest()
    films: list[SweepFilm] = []
    for entry in config.get("films", [])[:limit]:
        if "catalogue_digest" in entry:
            cat_entry = digest_index.get(entry["catalogue_digest"])
            if cat_entry is None:
                log.warning("catalogue entry not found for digest %s (%s) — skipping",
                            entry.get("catalogue_digest", "")[:12], entry.get("title"))
                continue
        elif "catalogue_entry_idx" in entry:
            cat_entry = catalogue_entries[entry["catalogue_entry_idx"]]
        else:
            cat_entry = entry["catalogue_entry"]
        films.append(SweepFilm(
            path=Path(entry["path"]),
            title=entry["title"],
            year=entry.get("year"),
            rating=entry.get("rating"),
            catalogue_entry=cat_entry,
        ))
    # Limit to ceil(n/2) episodes per title so we test every title but
    # don't burn time on all 18 Spawn or 16 Pantheon episodes.
    import math
    from collections import defaultdict
    by_title: dict[str, list[SweepFilm]] = defaultdict(list)
    for f in films:
        by_title[f.title].append(f)
    capped: list[SweepFilm] = []
    for title, eps in by_title.items():
        keep = math.ceil(len(eps) / 2)
        capped.extend(eps[:keep])
    return capped, None


_SWEEP_FILMS, _SKIP_REASON = _load_films()


def _sweep_film_id(film: SweepFilm) -> str:
    r = f"{bucket_rating(film.rating):.1f}"
    return f"r{r}|{film.year}|{film.title}"


def _summed_low_shelf_gain(entry: dict[str, Any]) -> float:
    """Sum of ``gain`` values across LowShelf filters (approx 'boost depth')."""
    total = 0.0
    for f in entry.get("filters", []):
        if f.get("type") == "LowShelf":
            try:
                total += float(f.get("gain", 0.0))
            except (TypeError, ValueError):
                continue
    return total


def _append_sweep_report(
    film: SweepFilm,
    advisor_name: str,
    metrics: Any,
    catalogue_filter_count: int,
) -> None:
    is_new = not _SWEEP_REPORT_PATH.exists()
    _SWEEP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SWEEP_REPORT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "rating_bucket", "rating", "release_year", "title",
                "catalogue_filter_count", "advisor", "verdict",
                "mean_err_db", "max_err_db", "summed_catalogue_gain_db",
            ])
        w.writerow([
            f"{bucket_rating(film.rating):.1f}",
            f"{film.rating:.2f}" if film.rating is not None else "",
            film.year if film.year is not None else "",
            film.title,
            catalogue_filter_count,
            advisor_name,
            metrics.verdict,
            f"{metrics.mean_abs_err_db:.2f}",
            f"{metrics.max_abs_err_db:.2f}",
            f"{_summed_low_shelf_gain(film.catalogue_entry):.1f}",
        ])


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[_sweep_film_id(f) for f in _SWEEP_FILMS] or None,
)
def test_library_sweep(film: SweepFilm, caplog):
    """Run the full auto-BEQ pipeline for one discovered film.

    No assertions — the verdict is written to the CSV report and
    logged. Failures here are informational (they show which films
    the current advisor struggles with).
    """
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    if not film.path.exists():
        pytest.skip(f"media file not accessible: {film.path}")

    from model.auto_beq import propose_or_lookup

    fs = 1000
    wav_path = _extract_lfe_wav(film.path, target_fs=fs)
    measured = load_and_smooth(wav_path, fs=fs, freqs=DEFAULT_GRID)

    stream_info = _probe_audio_stream(film.path)
    advisor = get_advisor()
    metadata = MediaMetadata(
        title=film.title,
        year=film.year,
        audio_codec=stream_info.get("codec_name") if stream_info else None,
        channel_layout=stream_info.get("channel_layout") if stream_info else None,
    )

    # Catalogue-first: look up the catalogue entry by title+year+codec.
    # If found, use the expert's chain directly. If not, auto-generate.
    proposed, source = propose_or_lookup(
        title=film.title,
        measured_curve_db=measured,
        freqs_hz=DEFAULT_GRID,
        year=film.year,
        audio_codec=stream_info.get("codec_name") if stream_info else None,
        advisor=advisor,
        metadata=metadata,
        fs=fs,
    )

    ground_resp = evaluate_filter_chain(
        film.catalogue_entry["filters"], DEFAULT_GRID, fs=fs,
    )
    metrics = compute_match_metrics(-ground_resp, proposed, DEFAULT_GRID, fs=fs)

    _append_sweep_report(
        film=film,
        advisor_name=f"{source}",
        metrics=metrics,
        catalogue_filter_count=len(film.catalogue_entry["filters"]),
    )
    log.info(
        "%s (%s): source=%s verdict=%s mean=%.2f max=%.2f",
        film.title, film.year, source, metrics.verdict,
        metrics.mean_abs_err_db, metrics.max_abs_err_db,
    )
