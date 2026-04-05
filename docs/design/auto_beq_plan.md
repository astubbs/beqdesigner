# Plan: LLM-assisted Advisor for auto-BEQ pipeline

**Companion docs:**
- [`auto_beq.md`](auto_beq.md) — vision + overall design
- [`auto_beq_experiments.md`](auto_beq_experiments.md) — running log of experiments

## Context

The spike's core question is "can we take the human out of BEQ-making?" —
produce BEQ filter chains automatically from a measured LFE curve. The
scipy N-filter fitter is sound (synthetic roundtrip tests green). The
procedural `classify_content` + peak-extension heuristic works for
moderate-catalogue films (John Wick PASSes) but fails on Edge of Tomorrow
because EoT's catalogue applies +28 dB extension that is **expert
judgment** about the film, not a function of the measured curve.

Hand-tuning more classes/thresholds per-film is overfitting to the 3
fixtures we have. The judgment step — "how aggressive should this title's
BEQ be?" — needs world knowledge about the film (genre, director,
sound-design reputation, era) plus informed interpretation of the
measured curve. That's exactly what an LLM can provide.

**Goal**: thread an optional `Advisor` between the measured-curve analysis
and the scipy fitter. The Advisor picks `max_gain_db` (and optionally
`knee_hz`) from film metadata + curve features. The fitter stays untouched.
Ollama runs locally (default for real-media tests). Tests that need to be
deterministic use a `MockAdvisor` with canned JSON responses.

**Success criterion**: Edge of Tomorrow grades PASS or MARGINAL when
Ollama recognises it as an aggressive sci-fi action title.

## Per user's scope decisions
- **Minimum viable plumbing**: Advisor abstraction + Heuristic + Mock +
  Ollama. Defer Anthropic.
- **No per-title hints in manifest**: LLMs already know film metadata, and
  user-supplied context doesn't scale.
- **Real Ollama default for real-media tests**, with MockAdvisor escape
  hatch for CI/offline.

## File layout

### New files
- `src/main/python/model/auto_beq_advisor.py` — single module containing:
  - Dataclasses: `MediaMetadata`, `CurveFeatures`, `Advice`
  - `Advisor` protocol (duck-typed interface)
  - `HeuristicAdvisor` — current rule-based classifier, preserved as
    fallback and for deterministic reproducibility
  - `MockAdvisor` — reads canned JSON from `src/test/resources/auto_beq/advisor_responses/{title-slug}.json`
  - `OllamaAdvisor` — HTTP POST to `http://localhost:11434/api/generate`
    with `format: "json"`, `temperature: 0.1`, single retry, 60s timeout
  - Feature extraction: `extract_curve_features(curve_db, freqs_hz, band)`
  - Prompt templates (system + user)
  - `get_advisor(name: str | None)` discovery factory

- `src/test/python/spike/test_auto_beq_advisor.py` — unit tests:
  feature extraction, prompt rendering, JSON parsing, MockAdvisor
  round-trip, `get_advisor` env/arg resolution, fallback on exception.

- `src/test/resources/auto_beq/advisor_responses/` — canned JSON, one per
  manifest title (edge-of-tomorrow.json, mad-max-fury-road.json,
  john-wick.json). Used by MockAdvisor.

### Modified files (minimal)
- `src/main/python/model/auto_beq.py`:
  - Add optional `advisor: Advisor | None = None` and
    `metadata: MediaMetadata | None = None` kwargs to
    `propose_filters_from_measured` and `infer_correction_from_measured`.
  - Inside `infer_correction_from_measured`, if `advisor` is provided:
    build `CurveFeatures`, call `advisor.advise(metadata, features)`,
    use `advice.max_gain_db` (overriding classifier) and
    `advice.knee_hz` (if present, overriding peak-freq detection).
    Log `advice.source`, `advice.reasoning`, `advice.confidence` at INFO.
  - If `advisor` is None: current behaviour is byte-identical to today.

- `src/test/python/spike/test_auto_beq.py`:
  - In `test_real_media_roundtrip`: read `AUTO_BEQ_ADVISOR` env var
    (default `"ollama"`). Build `MediaMetadata` from the manifest entry
    + catalogue snapshot (title, year from catalogue). Call
    `get_advisor(name)` and pass into `propose_filters_from_measured`.
  - If the advisor call raises (e.g. Ollama not running):
    `pytest.skip(f"advisor unavailable: {exc}")`. Don't fall back
    silently — that'd mask real failures.

### Unchanged
- scipy N-filter fitter (`propose_filters`, `_fit_low_shelf`,
  `_fit_peq_at_seed`)
- synthetic roundtrip tests
- catalogue snapshot
- media manifest schema (no new fields)

## Interface spec

```python
@dataclass(frozen=True)
class MediaMetadata:
    title: str
    year: int | None = None
    audio_codec: str | None = None       # from ffprobe
    channel_layout: str | None = None    # from ffprobe

@dataclass(frozen=True)
class CurveFeatures:
    shoulder_peak_db: float
    shoulder_peak_hz: float
    level_at_5hz_db: float
    level_at_10hz_db: float
    level_at_20hz_db: float
    rolloff_depth_db: float        # peak - min(band)
    rolloff_slope_db_per_oct: float  # between 10 and 20 Hz
    dynamic_range_db: float
    curve_sample_points: tuple[tuple[float, float], ...]  # 12 log-spaced (hz, db) pairs

@dataclass(frozen=True)
class Advice:
    max_gain_db: float                # 0..35, clamped
    knee_hz: float | None = None      # 5..80 if present
    reasoning: str = ""
    confidence: float = 0.5
    source: str = ""                  # "heuristic" | "mock" | "ollama"

class Advisor(Protocol):
    name: str
    def advise(self, metadata: MediaMetadata, features: CurveFeatures) -> Advice: ...
```

A small `_clamp_advice` wrapper in the module clamps `max_gain_db` to
0..35 and `knee_hz` to 5..80. Any exception in a real-LLM advisor
propagates (caller decides whether to skip the test or fall back).

## Ollama prompt design

**System prompt** (single paragraph, concrete):
> You are a BEQ (Bass EQ) advisor for home-theatre DSP. Given a film's
> metadata and the measured frequency-response features of its LFE
> channel, recommend two numbers: `max_gain_db` (how much to boost at
> the deep bass to extend infra-bass content) and `knee_hz` (the shelf
> corner frequency). Use your knowledge of the film, its sound design,
> director, era, and genre, along with the measured curve shape, to
> make these decisions. Typical BEQs apply 5-30 dB of low-shelf boost
> with knees between 10 and 30 Hz. Respond ONLY with strict JSON.

**User prompt** (templated):
```
Film: {title} ({year})
Audio: {codec} {channel_layout}

Measured LFE curve features (relative to 80 Hz anchor):
- Shoulder peak: {shoulder_peak_db:+.1f} dB at {shoulder_peak_hz:.0f} Hz
- Level at 5 Hz: {level_at_5hz_db:+.1f} dB
- Level at 10 Hz: {level_at_10hz_db:+.1f} dB
- Level at 20 Hz: {level_at_20hz_db:+.1f} dB
- Rolloff depth: {rolloff_depth_db:.1f} dB
- Rolloff slope (10-20 Hz): {rolloff_slope_db_per_oct:.1f} dB/octave
- Dynamic range: {dynamic_range_db:.1f} dB

Curve samples (Hz, dB): {curve_sample_points}

Respond with JSON: {"max_gain_db": N, "knee_hz": N, "confidence": 0-1, "reasoning": "..."}
```

Temperature=0.1 for determinism. `format: "json"` enforces valid JSON.
Model defaults to `llama3.1:8b` but reads `OLLAMA_MODEL` env var.

## Discovery / config

Priority for advisor selection (first hit wins):
1. Explicit `advisor` kwarg to `propose_filters_from_measured`
2. `AUTO_BEQ_ADVISOR` env var: `heuristic` | `mock` | `ollama`
3. Default: `heuristic` (production API), `ollama` (real-media test)

Env vars for Ollama: `OLLAMA_HOST` (default `http://localhost:11434`),
`OLLAMA_MODEL` (default `llama3.1:8b`).

## Testing strategy

**Unit tests (new, `test_auto_beq_advisor.py`)** — all fast, deterministic:
- `extract_curve_features` on a synthetic known curve returns expected values
- `HeuristicAdvisor.advise` matches legacy `classify_content` output bit-for-bit on sample features
- `MockAdvisor` loads canned JSON and returns expected `Advice`
- Prompt template renders with placeholders populated
- JSON parser handles clamping and rejects malformed responses
- `get_advisor("mock")` returns `MockAdvisor`; `get_advisor(None)` with env unset returns `HeuristicAdvisor`

**Real-media test (modified `test_real_media_roundtrip`)**:
- Default path: builds `MediaMetadata` from manifest+catalogue snapshot,
  calls `get_advisor(os.getenv("AUTO_BEQ_ADVISOR", "ollama"))`, passes
  into `propose_filters_from_measured`.
- If advisor construction or call fails: `pytest.skip` with explanatory
  message.
- Grading unchanged (proposed response vs catalogue response, <2 dB
  mean / <5 dB max).

**For CI / offline runs**: set `AUTO_BEQ_ADVISOR=mock` in env. Tests use
canned advice and remain deterministic.

## Critical reuse
- `propose_filters` in `src/main/python/model/auto_beq.py:402` — the
  scipy fitter. Called unchanged by the Advisor-aware
  `propose_filters_from_measured`.
- `classify_content` in `src/main/python/model/auto_beq.py:253` — wrapped
  by `HeuristicAdvisor` so legacy behaviour is preserved.
- `_band_mask`, `evaluate_filter_chain`, `smooth_fractional_octave` —
  reused as-is from `auto_beq.py`.
- `catalogue_snapshot` fixture in `src/test/python/conftest.py` — provides
  entry metadata (title, year) for `MediaMetadata` construction.

## Verification

1. `poetry run pytest src/test/python/spike/test_auto_beq_advisor.py -v` —
   all unit tests pass.
2. `AUTO_BEQ_ADVISOR=mock poetry run pytest src/test/python/spike/test_auto_beq.py::test_real_media_roundtrip -v` —
   uses canned MockAdvisor JSON; expect EoT to move to PASS/MARGINAL
   (canned advice is `max_gain_db=26, knee_hz=23`), John Wick and
   Mad Max still pass.
3. With Ollama running locally and `llama3.1:8b` pulled:
   `AUTO_BEQ_ADVISOR=ollama poetry run pytest src/test/python/spike/test_auto_beq.py::test_real_media_roundtrip -v` —
   real LLM advice flows through pipeline. Report the actual
   grades observed.
4. `AUTO_BEQ_ADVISOR=heuristic poetry run pytest src/test/python/spike/ -v` —
   verifies heuristic path still green on John Wick, marginal/fail on
   EoT and Mad Max (baseline preservation).
5. `poetry run ruff check src/main/python/model/auto_beq_advisor.py src/test/python/spike/test_auto_beq_advisor.py` — lint clean.

## Library sweep benchmark (additional deliverable)

User request: "scan a directory of my media, all tagged with imdb ids,
year and name, cross reference against the BEQ catalog, filter matches,
then sort and group by movie ratings by 0.5 points out of 10, then
iterate through the list by newest release date, run the tests on that
source file and compare as usual to the catalogue profiles."

Rationale: auto-discover a realistic test corpus from the user's actual
library. Avoids manual per-film manifest curation and gives us
statistics over a large number of titles in one run.

### Implementation

New file: `src/test/python/spike/test_auto_beq_library_sweep.py`

Session fixture walks `AUTO_BEQ_LIBRARY_ROOT` (env var pointing to the
user's Movies directory), parses each media file's name/year/tmdb-id
from standard Plex/Jellyfin filename conventions
(`Title (YEAR) [tmdb-NNNNN]`), matches against the catalogue by title
(title + year), and collects a sorted list of `(catalogue_rating,
release_year, title, media_path, catalogue_entry)` tuples.

Sorting: **primary** key = rating bucket at 0.5 resolution (desc),
**secondary** key = release date (desc / newest first). Bucket =
`floor(rating * 2) / 2` so 7.3 and 7.4 both go to bucket 7.0, 7.5-7.9
go to bucket 7.5, etc.

One parametrised pytest test per discovered match, IDs formatted as
`rating-N.N | YYYY | Title`. Each test runs the same full pipeline as
`test_real_media_roundtrip` (extract LFE → Signal pipeline → Advisor →
fitter → grade vs catalogue), using the same advisor selected by the
`AUTO_BEQ_ADVISOR` env var.

Skipped when `AUTO_BEQ_LIBRARY_ROOT` is not set. Cached WAV extractions
remain next to source files (existing behaviour).

### Rating source

The BEQ catalogue entries already carry a `rating` field (confirmed in
`model/catalogue.py`). No external API required. If the matched
catalogue entry lacks a rating, the match is bucketed as "unrated" and
run last.

### Match strategy

Primary match: filename-parsed title + year vs catalogue title + year.
Fallback: title-only, warn in log if multiple catalogue entries match
the same title (real user libraries typically have one release per
title, catalogue has multiple).

Users whose filenames don't follow the `(YYYY)` or `[tmdb-ID]`
conventions will get fewer matches; that's fine for this iteration. A
filename that doesn't parse is skipped with an INFO log.

### Output

In addition to pytest's per-test PASS/FAIL, the sweep writes a CSV
summary to `AUTO_BEQ_SWEEP_REPORT` (env var, default
`./.pytest_cache/auto_beq_sweep.csv`) with columns:
`rating_bucket, release_year, title, filters_in_catalogue, mean_err_db, max_err_db, grade, advisor_source, advisor_max_gain_db, notes`.

This lets the user scan the report and answer "how well does the
system do across my library" and "does accuracy correlate with
rating / era?" without reading individual pytest logs.

### Reused components
- `_extract_lfe_wav` in the existing test file — extraction helper,
  refactor to a module-level function shared between the two tests.
- `_ground_truth_curve`, `get_advisor`, `propose_filters_from_measured`,
  `compute_match_metrics`, `format_match_report` — all reused.
- `load_catalogue` in `model/catalogue.py` — loads the full catalogue
  JSON. We can either pin to the committed snapshot or fetch fresh.

### Verification
- `AUTO_BEQ_LIBRARY_ROOT=/path/to/Movies poetry run pytest src/test/python/spike/test_auto_beq_library_sweep.py -v`
  runs the sweep with default Ollama advisor.
- `cat .pytest_cache/auto_beq_sweep.csv` shows per-film results sorted
  by rating bucket then year.
- Without `AUTO_BEQ_LIBRARY_ROOT`: collection yields zero parametrised
  cases, test is skipped, CI unaffected.

## Non-goals (for this iteration)

- `AnthropicAdvisor` (plan has interface in place; implement in a
  follow-up commit by copying OllamaAdvisor and swapping the HTTP call)
- TMDB metadata lookup
- On-disk response caching
- Topology-hint-driven fitter changes
- Multi-advisor ensemble / confidence blending
- Per-title manifest hints

These are all straightforward additions once the plumbing is in place.
