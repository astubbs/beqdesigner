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
    _strategy_from_env,
    load_and_smooth,
    load_and_smooth_blended,
    load_and_smooth_chunked,
    load_measured,
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
    extraction_strategy: str = "",
) -> None:
    is_new = not _SWEEP_REPORT_PATH.exists()
    _SWEEP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SWEEP_REPORT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "rating_bucket", "rating", "release_year", "title",
                "catalogue_filter_count", "advisor", "extraction",
                "verdict", "mean_err_db", "max_err_db",
                "summed_catalogue_gain_db",
            ])
        w.writerow([
            f"{bucket_rating(film.rating):.1f}",
            f"{film.rating:.2f}" if film.rating is not None else "",
            film.year if film.year is not None else "",
            film.title,
            catalogue_filter_count,
            advisor_name,
            extraction_strategy,
            metrics.verdict,
            f"{metrics.mean_abs_err_db:.2f}",
            f"{metrics.max_abs_err_db:.2f}",
            f"{_summed_low_shelf_gain(film.catalogue_entry):.1f}",
        ])


def _run_one_film(film: SweepFilm) -> tuple[SweepFilm, MatchMetrics, str] | None:
    """Run the auto-BEQ pipeline for one media file. Thread-safe.

    Returns (film, metrics, advisor_name) or None if skipped.
    Uses the configured extraction strategy (default: blend-a0.7-P90).
    """
    from model.auto_beq import propose_filters_from_measured

    if not film.path.exists():
        log.warning("media file not accessible: %s", film.path)
        return None

    fs = 1000
    strategy = _strategy_from_env()
    wav_path = _extract_lfe_wav(film.path, target_fs=fs)
    measured = load_measured(wav_path, fs=fs, freqs=DEFAULT_GRID, strategy=strategy)

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
        extraction_strategy=_strategy_from_env().label,
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
                extraction_strategy=_strategy_from_env().label,
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


# ---------------------------------------------------------------------------
# E18b: Strategy sweep — percentile and blending variations at 60s chunks.
#
# Sub-experiments:
#   - Percentile: P75, P80, P90 (lower percentile = less peak bias)
#   - Blended: alpha=0.3, 0.5, 0.7 (Welch weight; higher = more Welch)
# All at 60s chunk length (E18 sweet spot).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionStrategy:
    """One spectrum extraction strategy to test."""
    name: str
    percentile: float
    alpha: float | None  # None = pure chunked, float = blended weight

    @property
    def label(self) -> str:
        if self.alpha is not None:
            return f"blend-a{self.alpha:.1f}-P{self.percentile:.0f}"
        return f"chunked-P{self.percentile:.0f}"


_STRATEGIES = [
    ExtractionStrategy("chunked-P75", percentile=75.0, alpha=None),
    ExtractionStrategy("chunked-P80", percentile=80.0, alpha=None),
    ExtractionStrategy("chunked-P90", percentile=90.0, alpha=None),
    ExtractionStrategy("blend-0.3", percentile=90.0, alpha=0.3),
    ExtractionStrategy("blend-0.5", percentile=90.0, alpha=0.5),
    ExtractionStrategy("blend-0.7", percentile=90.0, alpha=0.7),
]

_STRATEGY_REPORT_PATH = Path(os.environ.get(
    "AUTO_BEQ_STRATEGY_REPORT", ".pytest_cache/auto_beq_sweep_strategies.csv",
))


def _run_one_film_strategy(
    film: SweepFilm, strategy: ExtractionStrategy, chunk_s: float = 60.0,
) -> tuple[SweepFilm, MatchMetrics, MatchMetrics, str, ExtractionStrategy] | None:
    """Run Welch baseline + one extraction strategy for a film."""
    import numpy as np

    from model.auto_beq import propose_filters_from_measured

    if not film.path.exists():
        return None

    fs = 1000
    freqs = DEFAULT_GRID
    wav_path = _extract_lfe_wav(film.path, target_fs=fs)

    baseline_curve = load_and_smooth(wav_path, fs=fs, freqs=freqs)

    if strategy.alpha is not None:
        test_curve = load_and_smooth_blended(
            wav_path, fs=fs, freqs=freqs,
            chunk_s=chunk_s, percentile=strategy.percentile,
            alpha=strategy.alpha,
        )
    else:
        test_curve = load_and_smooth_chunked(
            wav_path, fs=fs, freqs=freqs,
            chunk_s=chunk_s, percentile=strategy.percentile,
        )

    advisor = get_advisor()
    metadata = MediaMetadata(title=film.title, year=film.year)
    ground_resp = evaluate_filter_chain(
        film.catalogue_entry["filters"], freqs, fs=fs,
    )

    baseline_proposed = propose_filters_from_measured(
        baseline_curve, freqs, fs=fs, advisor=advisor, metadata=metadata,
    )
    baseline_metrics = compute_match_metrics(
        -ground_resp, baseline_proposed, freqs, fs=fs,
    )

    test_proposed = propose_filters_from_measured(
        test_curve, freqs, fs=fs, advisor=advisor, metadata=metadata,
    )
    test_metrics = compute_match_metrics(
        -ground_resp, test_proposed, freqs, fs=fs,
    )

    log.info(
        "%s %s | baseline=%s(%.2f) test=%s(%.2f) delta=%+.2f",
        film.title, strategy.label,
        baseline_metrics.verdict, baseline_metrics.mean_abs_err_db,
        test_metrics.verdict, test_metrics.mean_abs_err_db,
        test_metrics.mean_abs_err_db - baseline_metrics.mean_abs_err_db,
    )
    return film, baseline_metrics, test_metrics, advisor.name, strategy


def _append_strategy_report(
    film: SweepFilm,
    advisor_name: str,
    baseline_metrics: MatchMetrics,
    test_metrics: MatchMetrics,
    strategy: ExtractionStrategy,
    catalogue_filter_count: int,
) -> None:
    is_new = not _STRATEGY_REPORT_PATH.exists()
    _STRATEGY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STRATEGY_REPORT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "title", "year", "strategy", "percentile", "alpha", "advisor",
                "baseline_verdict", "baseline_mean", "baseline_max",
                "test_verdict", "test_mean", "test_max",
                "mean_delta", "catalogue_filters", "summed_gain",
            ])
        mean_delta = test_metrics.mean_abs_err_db - baseline_metrics.mean_abs_err_db
        w.writerow([
            film.title,
            film.year if film.year is not None else "",
            strategy.label,
            f"{strategy.percentile:.0f}",
            f"{strategy.alpha:.1f}" if strategy.alpha is not None else "",
            advisor_name,
            baseline_metrics.verdict,
            f"{baseline_metrics.mean_abs_err_db:.2f}",
            f"{baseline_metrics.max_abs_err_db:.2f}",
            test_metrics.verdict,
            f"{test_metrics.mean_abs_err_db:.2f}",
            f"{test_metrics.max_abs_err_db:.2f}",
            f"{mean_delta:+.2f}",
            catalogue_filter_count,
            f"{_summed_low_shelf_gain(film.catalogue_entry):.1f}",
        ])


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "strategy",
    _STRATEGIES,
    ids=[s.label for s in _STRATEGIES],
)
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[_sweep_film_id(f) for f in _SWEEP_FILMS] or None,
)
def test_library_sweep_strategies(
    film: SweepFilm, strategy: ExtractionStrategy, caplog,
):
    """E18b: strategy sweep — percentile and blending variations.

    No assertions. Run with:
        SPIKE_TEST=...::test_library_sweep_strategies bash scripts/run-spike-tests.sh
    """
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    result = _run_one_film_strategy(film, strategy)
    if result is None:
        pytest.skip(f"media file not accessible: {film.path}")
    film, baseline_metrics, test_metrics, advisor_name, strategy = result
    _append_strategy_report(
        film=film,
        advisor_name=advisor_name,
        baseline_metrics=baseline_metrics,
        test_metrics=test_metrics,
        strategy=strategy,
        catalogue_filter_count=len(film.catalogue_entry["filters"]),
    )


# ---------------------------------------------------------------------------
# E19: MeasurementAdvisor constant calibration sweep.
#
# One-at-a-time sweep of cascade_gain_ratio, cascade_q,
# multi_knee_slope_threshold, and multi_knee_q. Each config varies one
# parameter from the baseline while holding the rest at defaults.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdvisorConfig:
    """One MeasurementAdvisor configuration to test."""
    cascade_gain_ratio: float = 7.0
    cascade_q: float = 0.9
    multi_knee_slope_threshold: float = 10.0   # E19 default
    multi_knee_q: float = 0.9                  # E19 default
    max_total_chain_gain_db: float = 30.0
    max_shelves: int = 2

    @property
    def label(self) -> str:
        return (
            f"g{self.cascade_gain_ratio:.0f}"
            f"-cQ{self.cascade_q:.1f}"
            f"-s{self.multi_knee_slope_threshold:.0f}"
            f"-mQ{self.multi_knee_q:.1f}"
            f"-cap{self.max_total_chain_gain_db:.0f}"
            f"-sh{self.max_shelves}"
        )


_BASELINE_CONFIG = AdvisorConfig()  # E19 defaults: s10, mQ0.9

_E19_CONFIGS = [
    _BASELINE_CONFIG,
    # Vary cascade_gain_ratio
    AdvisorConfig(cascade_gain_ratio=5.0),
    AdvisorConfig(cascade_gain_ratio=6.0),
    AdvisorConfig(cascade_gain_ratio=8.0),
    AdvisorConfig(cascade_gain_ratio=9.0),
    # Vary cascade_q
    AdvisorConfig(cascade_q=0.7),
    AdvisorConfig(cascade_q=0.8),
    AdvisorConfig(cascade_q=1.0),
    AdvisorConfig(cascade_q=1.2),
    # Vary multi_knee_slope_threshold (from pre-E19 baseline 15.0)
    AdvisorConfig(multi_knee_slope_threshold=12.0),
    AdvisorConfig(multi_knee_slope_threshold=15.0),
    AdvisorConfig(multi_knee_slope_threshold=18.0),
    AdvisorConfig(multi_knee_slope_threshold=20.0),
    # Vary multi_knee_q (from pre-E19 baseline 0.8)
    AdvisorConfig(multi_knee_q=0.6),
    AdvisorConfig(multi_knee_q=0.7),
    AdvisorConfig(multi_knee_q=0.8),
    AdvisorConfig(multi_knee_q=1.0),
]

_E19_REPORT_PATH = Path(os.environ.get(
    "AUTO_BEQ_E19_REPORT", ".pytest_cache/auto_beq_sweep_e19.csv",
))


def _run_one_film_e19(
    film: SweepFilm, config: AdvisorConfig,
) -> tuple[SweepFilm, MatchMetrics, MatchMetrics, str, AdvisorConfig] | None:
    """Run baseline + one advisor config for a film."""
    from model.auto_beq import propose_filters_from_measured
    from model.auto_beq_advisor import MeasurementAdvisor

    if not film.path.exists():
        return None

    fs = 1000
    freqs = DEFAULT_GRID
    wav_path = _extract_lfe_wav(film.path, target_fs=fs)
    measured = load_measured(wav_path, fs=fs, freqs=freqs)

    metadata = MediaMetadata(title=film.title, year=film.year)
    ground_resp = evaluate_filter_chain(
        film.catalogue_entry["filters"], freqs, fs=fs,
    )

    # Baseline: default MeasurementAdvisor.
    baseline_advisor = MeasurementAdvisor()
    baseline_proposed = propose_filters_from_measured(
        measured, freqs, fs=fs, advisor=baseline_advisor, metadata=metadata,
    )
    baseline_metrics = compute_match_metrics(
        -ground_resp, baseline_proposed, freqs, fs=fs,
    )

    # Test: configured MeasurementAdvisor.
    test_advisor = MeasurementAdvisor(
        cascade_gain_ratio=config.cascade_gain_ratio,
        cascade_q=config.cascade_q,
        multi_knee_slope_threshold=config.multi_knee_slope_threshold,
        multi_knee_q=config.multi_knee_q,
        max_total_chain_gain_db=config.max_total_chain_gain_db,
        max_shelves=config.max_shelves,
    )
    test_proposed = propose_filters_from_measured(
        measured, freqs, fs=fs, advisor=test_advisor, metadata=metadata,
    )
    test_metrics = compute_match_metrics(
        -ground_resp, test_proposed, freqs, fs=fs,
    )

    log.info(
        "%s %s | baseline=%s(%.2f) test=%s(%.2f) delta=%+.2f",
        film.title, config.label,
        baseline_metrics.verdict, baseline_metrics.mean_abs_err_db,
        test_metrics.verdict, test_metrics.mean_abs_err_db,
        test_metrics.mean_abs_err_db - baseline_metrics.mean_abs_err_db,
    )
    return film, baseline_metrics, test_metrics, "measurement", config


def _append_e19_report(
    film: SweepFilm,
    baseline_metrics: MatchMetrics,
    test_metrics: MatchMetrics,
    config: AdvisorConfig,
    catalogue_filter_count: int,
) -> None:
    is_new = not _E19_REPORT_PATH.exists()
    _E19_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _E19_REPORT_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "title", "year", "config", "cascade_gain_ratio", "cascade_q",
                "multi_knee_slope_threshold", "multi_knee_q", "max_chain_gain",
                "baseline_verdict", "baseline_mean", "baseline_max",
                "test_verdict", "test_mean", "test_max",
                "mean_delta", "catalogue_filters", "summed_gain",
            ])
        mean_delta = test_metrics.mean_abs_err_db - baseline_metrics.mean_abs_err_db
        w.writerow([
            film.title,
            film.year if film.year is not None else "",
            config.label,
            f"{config.cascade_gain_ratio:.1f}",
            f"{config.cascade_q:.1f}",
            f"{config.multi_knee_slope_threshold:.0f}",
            f"{config.multi_knee_q:.1f}",
            f"{config.max_total_chain_gain_db:.0f}",
            baseline_metrics.verdict,
            f"{baseline_metrics.mean_abs_err_db:.2f}",
            f"{baseline_metrics.max_abs_err_db:.2f}",
            test_metrics.verdict,
            f"{test_metrics.mean_abs_err_db:.2f}",
            f"{test_metrics.max_abs_err_db:.2f}",
            f"{mean_delta:+.2f}",
            catalogue_filter_count,
            f"{_summed_low_shelf_gain(film.catalogue_entry):.1f}",
        ])


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "config",
    _E19_CONFIGS,
    ids=[c.label for c in _E19_CONFIGS],
)
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[_sweep_film_id(f) for f in _SWEEP_FILMS] or None,
)
def test_library_sweep_e19(film: SweepFilm, config: AdvisorConfig, caplog):
    """E19: MeasurementAdvisor constant calibration sweep.

    No assertions. Run with:
        SPIKE_TEST=...::test_library_sweep_e19 bash scripts/run-spike-tests.sh
    """
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    result = _run_one_film_e19(film, config)
    if result is None:
        pytest.skip(f"media file not accessible: {film.path}")
    film, baseline_metrics, test_metrics, advisor_name, config = result
    _append_e19_report(
        film=film,
        baseline_metrics=baseline_metrics,
        test_metrics=test_metrics,
        config=config,
        catalogue_filter_count=len(film.catalogue_entry["filters"]),
    )


# ---------------------------------------------------------------------------
# E20: Multi-knee advisor improvements — 3-shelf cascade + gain cap sweep.
#
# Tests whether a 3-shelf chain (splitting the deficit across 5→10→20→peak
# instead of just 10→20→peak) improves high-gain catalogue entries.
# Also tests raising the gain cap from 30 to 35/40 dB.
# ---------------------------------------------------------------------------

_E20_CONFIGS = [
    AdvisorConfig(),                                                    # baseline: 2-shelf, 30 dB
    AdvisorConfig(max_shelves=3),                                       # 3-shelf, 30 dB
    AdvisorConfig(max_shelves=3, max_total_chain_gain_db=35.0),         # 3-shelf, 35 dB
    AdvisorConfig(max_shelves=3, max_total_chain_gain_db=40.0),         # 3-shelf, 40 dB
    AdvisorConfig(max_total_chain_gain_db=35.0),                        # 2-shelf, 35 dB
    AdvisorConfig(max_total_chain_gain_db=40.0),                        # 2-shelf, 40 dB
]


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
@pytest.mark.skipif(not _have_tool("ffmpeg"), reason="ffmpeg not on PATH")
@pytest.mark.skipif(not _have_tool("ffprobe"), reason="ffprobe not on PATH")
@pytest.mark.parametrize(
    "config",
    _E20_CONFIGS,
    ids=[c.label for c in _E20_CONFIGS],
)
@pytest.mark.parametrize(
    "film",
    _SWEEP_FILMS,
    ids=[_sweep_film_id(f) for f in _SWEEP_FILMS] or None,
)
def test_library_sweep_e20(film: SweepFilm, config: AdvisorConfig, caplog):
    """E20: multi-knee improvements — 3-shelf cascade + gain cap sweep.

    No assertions. Run with:
        SPIKE_TEST=...::test_library_sweep_e20 bash scripts/run-spike-tests.sh
    """
    caplog.set_level(logging.INFO, logger="auto_beq_sweep")
    result = _run_one_film_e19(film, config)  # reuses same runner
    if result is None:
        pytest.skip(f"media file not accessible: {film.path}")
    film, baseline_metrics, test_metrics, advisor_name, config = result
    _append_e19_report(
        film=film,
        baseline_metrics=baseline_metrics,
        test_metrics=test_metrics,
        config=config,
        catalogue_filter_count=len(film.catalogue_entry["filters"]),
    )
