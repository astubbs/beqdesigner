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
    MatchMetrics,
    compute_match_metrics,
    evaluate_filter_chain,
)
from model.auto_beq_advisor import MediaMetadata, get_advisor

from spike._auto_beq_helpers import (
    _extract_lfe_wav,
    _have_tool,
    load_and_smooth,
    load_and_smooth_chunked,
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


def _run_one_film(film: SweepFilm) -> tuple[SweepFilm, MatchMetrics, str] | None:
    """Run the auto-BEQ pipeline for one media file. Thread-safe.

    Returns (film, metrics, advisor_name) or None if skipped.
    """
    from model.auto_beq import propose_filters_from_measured

    if not film.path.exists():
        log.warning("media file not accessible: %s", film.path)
        return None

    fs = 1000
    wav_path = _extract_lfe_wav(film.path, target_fs=fs)
    measured = load_and_smooth(wav_path, fs=fs, freqs=DEFAULT_GRID)

    advisor = get_advisor()
    metadata = MediaMetadata(title=film.title, year=film.year)

    proposed = propose_filters_from_measured(
        measured, DEFAULT_GRID, fs=fs,
        advisor=advisor, metadata=metadata,
    )

    ground_resp = evaluate_filter_chain(
        film.catalogue_entry["filters"], DEFAULT_GRID, fs=fs,
    )
    metrics = compute_match_metrics(-ground_resp, proposed, DEFAULT_GRID, fs=fs)

    log.info(
        "%s (%s): advisor=%s verdict=%s mean=%.2f max=%.2f",
        film.title, film.year, advisor.name, metrics.verdict,
        metrics.mean_abs_err_db, metrics.max_abs_err_db,
    )
    return film, metrics, advisor.name


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[_sweep_film_id(f) for f in _SWEEP_FILMS] or None,
)
def test_library_sweep(film: SweepFilm, caplog):
    """Run the full auto-BEQ pipeline for one discovered media file.

    No assertions — the verdict is written to the CSV report and
    logged. Failures here are informational (they show which media
    the current advisor struggles with).
    """
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    result = _run_one_film(film)
    if result is None:
        pytest.skip(f"media file not accessible: {film.path}")
    film, metrics, advisor_name = result
    _append_sweep_report(
        film=film,
        advisor_name=advisor_name,
        metrics=metrics,
        catalogue_filter_count=len(film.catalogue_entry["filters"]),
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
def test_library_sweep_parallel():
    """Run the full auto-BEQ pipeline across all media in parallel.

    Concurrency = number of configured Ollama hosts (so each host
    gets one request at a time). Falls back to 1 if no Ollama hosts
    configured (e.g. MeasurementAdvisor doesn't need Ollama).

    This test is an ALTERNATIVE to the parametrised test_library_sweep
    above — run one OR the other, not both. Use this when you want
    speed; use the parametrised one when you want per-media pytest
    output.

    Select with: SPIKE_TEST=...::test_library_sweep_parallel
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from model.auto_beq_advisor import OllamaAdvisor, _load_ollama_hosts

    if not _SWEEP_FILMS:
        pytest.skip("no films to test")

    n_hosts = len(_load_ollama_hosts())
    n_workers = max(1, n_hosts)
    log.info("parallel sweep: %d media files, %d workers (%d Ollama hosts)",
             len(_SWEEP_FILMS), n_workers, n_hosts)

    verdicts: dict[str, int] = {"PASS": 0, "MARGINAL": 0, "FAIL": 0}
    skipped = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_film, film): film for film in _SWEEP_FILMS}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                skipped += 1
                continue
            film, metrics, advisor_name = result
            verdicts[metrics.verdict] = verdicts.get(metrics.verdict, 0) + 1
            _append_sweep_report(
                film=film,
                advisor_name=advisor_name,
                metrics=metrics,
                catalogue_filter_count=len(film.catalogue_entry["filters"]),
            )

    total = sum(verdicts.values())
    log.info(
        "parallel sweep done: %d PASS, %d MARGINAL, %d FAIL, %d skipped",
        verdicts["PASS"], verdicts["MARGINAL"], verdicts["FAIL"], skipped,
    )
    print(
        f"\nSweep results: {verdicts['PASS']} PASS, "
        f"{verdicts['MARGINAL']} MARGINAL, {verdicts['FAIL']} FAIL, "
        f"{skipped} skipped (out of {total + skipped} media files, "
        f"{n_workers} workers)"
    )
    OllamaAdvisor.print_host_stats()


# ---------------------------------------------------------------------------
# E18: Chunked percentile sweep — compare chunked spectrum extraction
# against the Welch baseline across the full library.
# ---------------------------------------------------------------------------

_CHUNKED_REPORT_PATH = Path(os.environ.get(
    "AUTO_BEQ_CHUNKED_REPORT", ".pytest_cache/auto_beq_sweep_chunked.csv",
))


def _run_one_film_chunked(
    film: SweepFilm, chunk_s: float,
) -> tuple[SweepFilm, MatchMetrics, MatchMetrics, str, float] | None:
    """Run both Welch and chunked pipelines for one film.

    Returns (film, baseline_metrics, chunked_metrics, advisor_name, chunk_s)
    or None if skipped.
    """
    import numpy as np

    from model.auto_beq import propose_filters_from_measured

    if not film.path.exists():
        log.warning("media file not accessible: %s", film.path)
        return None

    fs = 1000
    freqs = DEFAULT_GRID
    wav_path = _extract_lfe_wav(film.path, target_fs=fs)

    # Baseline: whole-film Welch.
    baseline_curve = load_and_smooth(wav_path, fs=fs, freqs=freqs)
    # Chunked: STFT peak per chunk → P90.
    chunked_curve = load_and_smooth_chunked(
        wav_path, fs=fs, freqs=freqs, chunk_s=chunk_s,
    )

    # Log delta at 10 Hz and 20 Hz.
    idx_10 = int(np.argmin(np.abs(freqs - 10.0)))
    idx_20 = int(np.argmin(np.abs(freqs - 20.0)))
    log.info(
        "%s chunk_s=%.0f | 10Hz delta=%+.1f dB, 20Hz delta=%+.1f dB",
        film.title, chunk_s,
        chunked_curve[idx_10] - baseline_curve[idx_10],
        chunked_curve[idx_20] - baseline_curve[idx_20],
    )

    advisor = get_advisor()
    metadata = MediaMetadata(title=film.title, year=film.year)

    ground_resp = evaluate_filter_chain(
        film.catalogue_entry["filters"], freqs, fs=fs,
    )

    # Baseline metrics.
    baseline_proposed = propose_filters_from_measured(
        baseline_curve, freqs, fs=fs, advisor=advisor, metadata=metadata,
    )
    baseline_metrics = compute_match_metrics(
        -ground_resp, baseline_proposed, freqs, fs=fs,
    )

    # Chunked metrics.
    chunked_proposed = propose_filters_from_measured(
        chunked_curve, freqs, fs=fs, advisor=advisor, metadata=metadata,
    )
    chunked_metrics = compute_match_metrics(
        -ground_resp, chunked_proposed, freqs, fs=fs,
    )

    log.info(
        "%s chunk_s=%.0f | baseline=%s(%.2f/%.2f) chunked=%s(%.2f/%.2f)",
        film.title, chunk_s,
        baseline_metrics.verdict, baseline_metrics.mean_abs_err_db,
        baseline_metrics.max_abs_err_db,
        chunked_metrics.verdict, chunked_metrics.mean_abs_err_db,
        chunked_metrics.max_abs_err_db,
    )
    return film, baseline_metrics, chunked_metrics, advisor.name, chunk_s


def _append_chunked_report(
    film: SweepFilm,
    advisor_name: str,
    baseline_metrics: MatchMetrics,
    chunked_metrics: MatchMetrics,
    chunk_s: float,
    catalogue_filter_count: int,
) -> None:
    is_new = not _CHUNKED_REPORT_PATH.exists()
    _CHUNKED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CHUNKED_REPORT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "title", "year", "chunk_s", "advisor",
                "baseline_verdict", "baseline_mean", "baseline_max",
                "chunked_verdict", "chunked_mean", "chunked_max",
                "mean_delta", "catalogue_filters", "summed_gain",
            ])
        mean_delta = chunked_metrics.mean_abs_err_db - baseline_metrics.mean_abs_err_db
        w.writerow([
            film.title,
            film.year if film.year is not None else "",
            f"{chunk_s:.0f}",
            advisor_name,
            baseline_metrics.verdict,
            f"{baseline_metrics.mean_abs_err_db:.2f}",
            f"{baseline_metrics.max_abs_err_db:.2f}",
            chunked_metrics.verdict,
            f"{chunked_metrics.mean_abs_err_db:.2f}",
            f"{chunked_metrics.max_abs_err_db:.2f}",
            f"{mean_delta:+.2f}",
            catalogue_filter_count,
            f"{_summed_low_shelf_gain(film.catalogue_entry):.1f}",
        ])


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize("chunk_s", [30.0, 60.0, 90.0])
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[_sweep_film_id(f) for f in _SWEEP_FILMS] or None,
)
def test_library_sweep_chunked(film: SweepFilm, chunk_s: float, caplog):
    """E18: chunked-percentile sweep — compare vs Welch baseline per film.

    No assertions. Both Welch and chunked results are logged side-by-side
    and appended to the chunked sweep CSV report. Run with:

        SPIKE_TEST=...::test_library_sweep_chunked bash scripts/run-spike-tests.sh
    """
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    result = _run_one_film_chunked(film, chunk_s)
    if result is None:
        pytest.skip(f"media file not accessible: {film.path}")
    film, baseline_metrics, chunked_metrics, advisor_name, chunk_s = result
    _append_chunked_report(
        film=film,
        advisor_name=advisor_name,
        baseline_metrics=baseline_metrics,
        chunked_metrics=chunked_metrics,
        chunk_s=chunk_s,
        catalogue_filter_count=len(film.catalogue_entry["filters"]),
    )
