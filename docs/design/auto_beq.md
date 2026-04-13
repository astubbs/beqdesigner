# Auto-BEQ — Automated Filter Suggestion

**Companion docs:**
- [`auto_beq_plan.md`](auto_beq_plan.md) — current iteration's implementation plan (LLM-assisted Advisor)
- [`auto_beq_experiments.md`](auto_beq_experiments.md) — append-only log of experiments tried, results, lessons

**Status:** spike, not shipped. Tests green on 5 synthetic + 2
real-media fixtures (Edge of Tomorrow and Mad Max: Fury Road). The
N-filter iterative fitter reproduces real catalogue entries'
response curves within 0.52 dB mean / 1.40 dB max across 5-80 Hz.
This answers the vision brief's go/no-go question (can scipy.optimize
match human expert output?) as **YES**.

The spike does NOT yet solve the harder problem of inferring a
catalogue-style correction target from a measured LFE curve - that
is supervised learning or expert heuristic work beyond the spike's
scope. The real-media tests verify the extraction/measurement
pipeline is real and working, then feed the catalogue curve directly
to the fitter as the target. See section 4 for the scoping.

**Audience:** developers working on the magic-wand initiative. This document
will be split into user-facing docs (feature overview) and implementation
docs (optimizer internals) once the feature graduates from spike to
production.

---

## 1. Overview

Extend BEQDesigner with automated IIR filter suggestion. Goal: a user loads
raw audio content (LFE track, direct media file), presses a button, and gets
a filter chain proposal populated into the existing filter slots. They
review, tweak from a head start, and send to their DSP.

Long-term this powers:

1. A **magic-wand button** in the GUI signal-analysis view (Tier 1 of the
   vision).
2. A **headless CLI** that batch-processes a folder of media files into
   per-episode / per-movie BEQ profiles (Tier 1).
3. A fully automated **Home Assistant** workflow that preprocesses a NAS
   library into sidecar `.beq` files and loads them at play time via ezBEQ
   (Tier 3).

None of those UIs exist yet. This spike answers the one prerequisite
question: **can `scipy.optimize` produce filter parameters that match
human-expert BEQ catalogue entries within tolerance?**

---

## 2. System architecture

```mermaid
flowchart LR
    A[Media file<br/>.mkv / .wav] -->|ffmpeg extract<br/>pan=c{LFE}, ar=1000| B[mono WAV @ 1 kHz]
    B -->|read_wav_data<br/>soundfile| C[numpy samples]
    C -->|Signal + avg_spectrum<br/>scipy.welch| D[magnitude curve<br/>dB vs Hz]
    C -->|load_and_smooth_chunked<br/>STFT peak per chunk → P90| D2[chunked-percentile curve<br/>dB vs Hz]
    D -->|interp + anchor normalise| E[target curve on log grid]
    D2 -->|interp + anchor normalise| E
    E -->|propose_filters| F[filter chain<br/>list of dicts]
    F -->|CompleteFilter.get_sos| G[DSP / ezBEQ]

    subgraph spike[Spike scope]
    E
    F
    end

    subgraph reused[Pre-existing code]
    B
    C
    D
    G
    end
```

The only genuinely new component is `propose_filters`. Everything else is
either existing BEQDesigner code (`model.signal`, `model.iir`,
`model.catalogue`) or a thin ffmpeg invocation.

---

## 3. Algorithm

`model.auto_beq.propose_filters(target_curve_db, freqs_hz, fs, band)`

Takes an in-band magnitude curve in dB (expressed relative to an 80 Hz
anchor) and returns a short filter chain whose response cancels the
curve across `band` (default 5-80 Hz). Real-media callers pre-smooth
the curve to 1/6-octave before feeding it in (see Stage 0 below).

**Pipeline:**

### Stage 0 — 1/6-octave smoothing (caller's responsibility)

Raw magnitude spectra contain narrow resonances and dither artefacts
that a broad IIR filter cannot (and should not) chase. Real-media
callers smooth the curve with
`smooth_fractional_octave(curve, freqs, octaves=1/6)` before handing
it to `propose_filters`. The synthetic test skips this because
catalogue-generated curves are already smooth by construction.

Smoothing is a log-frequency Gaussian kernel: for each bin, a Gaussian
weighted over `log2(freqs)` with `sigma = octaves/2.355`. 1/6-octave is
the SPL-measurement convention.

### Stage 1 — One LowShelf (bidirectional)

`scipy.optimize.minimize` with `method="L-BFGS-B"`:

| Param | Seed | Bounds |
|---|---|---|
| freq (Hz) | 25 Hz | [5, 120] |
| Q | 0.7 (Butterworth-ish) | [0.3, 2.0] |
| gain (dB) | negated mean of target in lowest octave of band | [-30, +30] |

The gain bound is **bidirectional** - the shelf can lift or cut the
low end. This captures broad trends in the target.

### Stage 2 — Iterative residual PEQs

Loop, adding one PEQ per iteration until either `max_filters` (default
6) is reached or the in-band max residual drops below `stop_max_err_db`
(default 0.5 dB):

1. Compute `err = target + evaluate(chain_so_far)`.
2. If `max(|err|) in band < stop_max_err_db`, stop.
3. Find the worst-residual frequency (`argmax(|err|)` in band); use it
   as the PEQ seed frequency, with gain = `-err` at that bin.
4. Fit the PEQ from **three Q seeds** (0.7, 1.5, 3.0) and keep the
   best. Bounds: freq ∈ band, Q ∈ [0.3, 4.0], gain ∈ [-30, +30].
5. If the new PEQ reduces in-band RMS error by less than 0.05 dB, stop
   (the optimizer has nothing useful to add).
6. Append the PEQ and continue.

Three Q seeds matter: the L-BFGS-B optimizer is local, and a single
seed can get stuck in a shallow minimum when the target has multiple
features nearby. Trying 0.7/1.5/3.0 covers "broad", "medium", and
"narrow" shapes.

### Stage 3 — Return

List of dicts matching the `CatalogueEntry.filters` schema:
`{"type": "LowShelf"|"PeakingEQ"|"HighShelf", "freq": float, "q": float, "gain": float}`

Directly consumable by `model.iir.CompleteFilter(fs, filters=...)`.

### Shape of the output

The algorithm is a **generic IIR curve-fitter** - it produces a chain
whose response matches the target in the band, using whatever
combination of filters works. It is NOT a "BEQ-aware" algorithm: the
filters it proposes are not guaranteed to look like what a human BEQ
expert would choose. The output can include high-gain shelves
(+29 dB on Mad Max), negative-gain shelves, notches, and arbitrary
PEQ chains. See "Real-media findings" and "Known limitations".

---

## 4. Validation methodology

### The catalogue-as-ground-truth idea

The BEQ catalogue (`beqcatalogue.readthedocs.io/database.json`, 14785
entries as of April 2026) contains filter chains produced by human experts
for thousands of titles. Each chain implicitly defines a target "correction
curve" `G(f)` — the dB response the filters impose.

If we negate that curve, we get `-G(f)`, which is what a perfectly
catalogued release would measure as raw rolloff on its LFE track before
correction. Feeding `-G(f)` back into `propose_filters` should return a
filter chain whose response closely approximates `G(f)` — i.e. we should
reconstruct the original correction.

This is the **synthetic roundtrip** test. It isolates the optimizer from
real-world confounds (room noise, rip quality, multi-channel bass
management) and answers: does the optimization math work?

### Algorithm-quality thresholds (per vision doc)

For each title, compute `err = target + proposed_response` across the
scoring band:

| Metric | PASS threshold | MARGINAL threshold |
|---|---|---|
| `mean(|err|)` | < 2.0 dB | < 3.0 dB |
| `max(|err|)` | < 5.0 dB | < 7.5 dB |

Topology match (same filter types as catalogue) is a **secondary** metric.
A proposed chain with different topology but equivalent in-band response is
acceptable — bass-frequency IIR filters have well-known equivalencies
(different `freq`/`Q`/`gain` triplets can produce near-identical in-band
curves).

### Synthetic results (April 2026, band 5-80 Hz, N-filter fitter)

| Title | # filters proposed | Mean err | Max err | Grade |
|---|---|---|---|---|
| Battle: Los Angeles (1-filter catalogue) | 1 | 0.09 dB | 0.18 dB | PASS |
| Captain America: TWS (1-filter catalogue) | 1 | 0.13 dB | 0.27 dB | PASS |
| Run Hide Fight (1-filter catalogue) | 1 | 0.02 dB | 0.06 dB | PASS |

The N-filter fitter still produces single-filter output on these
easy cases because one LowShelf already matches the target within
`stop_max_err_db=0.5`. For deep catalogue entries the same fitter
produces 6-filter chains (see real-media results).

### Real-media results (April 2026, band 5-80 Hz)

Run via the media manifest at
`~/.config/beqdesigner/auto_beq_media.json`. Each fixture exercises
the full pipeline (extract → smooth → fit) and grades the fitter's
output against the catalogue entry's response curve.

| Title | Proposed filters | Mean err | Max err | Grade |
|---|---|---|---|---|
| Edge of Tomorrow (5-filter catalogue, ~+28 dB @ 10 Hz) | 6 | 0.52 dB | 1.40 dB | PASS |
| Mad Max: Fury Road (5-filter catalogue, ~+15 dB @ 10 Hz) | 5 | 0.25 dB | 1.02 dB | PASS |

The proposed leading filters resemble the catalogue entries'
topology. EoT's catalogue cascades four `LowShelf @ 23 Hz Q=0.9
+6.9 dB` shelves (summed gain +27.6 dB); the fitter proposes one
`LowShelf @ 23 Hz Q=1.15 +28.42 dB` - same frequency, summed gain,
near-matching Q. Mad Max's mixed cascade distils to `LowShelf @ 17 Hz
Q=0.68 +13.66 dB` plus four residual PEQs, matching the catalogue's
total shape.

### What this test exercises

1. **Extraction pipeline**: `ffmpeg pan=c0=LFE` on a real Blu-ray/UHD
   LFE track, cached next to the source. Confirms the app can consume
   real media without GUI bootstrapping.
2. **Measurement pipeline**: `Signal.avg_spectrum()` (Welch average)
   interpolated to a log grid, normalised to the 80 Hz anchor, and
   smoothed to 1/6-octave. An alternative chunked-percentile path
   (`load_and_smooth_chunked()`) splits audio into fixed-length chunks,
   computes STFT peak per chunk, and takes the 90th percentile across
   chunks — more robust for short content with sparse bass (E18).
3. **Dynamic-range sanity check**: the measured curve must have
   ≥5 dB of in-band range, catching "extracted silence" or
   "wrong channel" bugs.
4. **Fitter capability**: `propose_filters` is given the
   catalogue entry's response curve as target and must reproduce it
   with ≤6 IIR filters to <2 dB mean / <5 dB max. This is the vision
   brief's go/no-go question.

### What this test does NOT exercise

The fitter is fed the **catalogue curve** as target, not the
measured curve. The gap between measured and catalogue is logged
informationally (Mad Max: 9.37 dB, EoT: 7.82 dB mean in 20-80 Hz) -
this represents the difference between "what the content has" and
"what the expert prescribed to add". Bridging this gap (turning a
measured curve into a catalogue-shaped correction target) is a
harder problem out of scope for the spike: it requires either a
supervised mapping trained on catalogue/content pairs, or an
expert-encoded heuristic that decides how aggressively to extend
bass per title. Neither is implemented.

---

## 5. Known limitations

### The big one: measured → catalogue-shaped target is unsolved
The fitter reproduces any smooth in-band curve well. Tests feed it
the catalogue curve directly, so it passes. In a real magic-wand
scenario we only have the MEASURED content curve and need to INFER
what catalogue-shaped correction to target. There is no supervised
mapping or heuristic for this yet. Two plausible paths for a follow-up
iteration:
- **Supervised**: train a small model that maps measured-curve
  features to correction-curve parameters, using catalogue/content
  pairs as training data.
- **Heuristic**: detect the natural rolloff slope in the measured
  curve, extend it by a tunable "aggressiveness" factor (so the user
  can pick "mild / medium / aggressive" rather than the algorithm
  guessing).

### Other limitations
- **No topology preference.** The fitter minimises response error
  only. Output chains can contain narrow notches, high-gain shelves,
  and oscillating PEQs because the target curve drives them there.
- **Mono only.** No bass-management awareness. The fitter sees one
  channel's curve at a time.
- **Scoring band 5-80 Hz.** The lower edge depends on the signal
  pipeline's ability to produce reliable magnitude data at 5-10 Hz.
  1/6-octave smoothing helps, but at fs=1000 Hz the Welch bins are
  ~0.5 Hz wide and 5 Hz lives in the first ~10 bins.
- **Six-filter cap.** `max_filters=6` is arbitrary. Very complex
  catalogue entries could need more.
- **Media file assumption.** Requires a local file to analyse; no
  streaming-only content support.
- **Single extraction.** Welch averages 100+ minutes into one curve,
  so loud scenes dominate. Catalogue experts often work from
  specific reference scenes instead. E18 adds a chunked-percentile
  alternative (`load_and_smooth_chunked`) that mitigates this by
  taking STFT peaks per chunk and aggregating via 90th percentile.

---

## 6. Expansion path

### Next iteration: infer the correction target from measured content

The fitter is proven. The open question is how to construct its
target from a measured curve alone.

Quickest useful step: **build a heuristic that ingests the measured
curve and outputs a target-correction curve**, then have the user
tune "aggressiveness" (how deep to extend). This gets us to a
shippable magic-wand button that produces reasonable starting points
the user can review.

Once that's in place, the catalogue becomes a validation corpus: for
each title, check how close the heuristic's proposal lands relative
to the catalogue entry. The numbers we get will tell us whether a
single heuristic is enough or we need per-genre / per-era tuning.

### Then: add test coverage

1. Identify titles in the catalogue: inspect
   `src/test/resources/auto_beq/database.json`, or filter the full
   catalogue
   (`https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json`).
2. Add the title + filter-count tuple to the snapshot.
3. Add an entry to `FIXTURES` in `test_auto_beq.py` with an expected
   grade.
4. Run `pytest src/test/python/spike/test_auto_beq.py -v`.

### Then: graduate the spike to a production feature

The three-tier roadmap from the vision brief:

1. **Magic wand button** - wire `propose_filters` to a QPushButton in
   the signal-analysis view.
2. **Unified CLI** (`scripts/beq.py`) - interactive menu + subcommands
   for profile generation, LFE extraction, cache management, and sweep
   analysis. Includes batch processing of directories. *(done)*
3. **ezBEQ send** - HTTP POST of the filter chain to ezBEQ's `/api/`
   endpoint.

### Metrics we'll track as the spike grows

- **PASS-grade rate across the catalogue** - target ≥80% per the
  vision doc.
- **Distribution of `mean_abs_err_db`** - where does the long tail live?
- **Failure-mode taxonomy** - which content types fail, and why.
- **Wall-clock per title** - currently ~10 ms per synthetic roundtrip,
  ~70 s for a full feature-length LFE extraction (cached on first run).

---

## 7. Future features

- **Pre-extract audio for uncatalogued media.** The library discovery
  config records all media files (matched + unmatched). Currently
  audio extraction only happens on-demand when a test or the magic
  wand requests it. Optional background pre-extraction for unmatched
  media would make later profile construction instant instead of
  waiting 30-120s per title for ffmpeg. Low priority — extraction is
  cached after first run anyway.
- **AnthropicAdvisor** — API-key-based advisor using Claude for the
  gain-multiplier question (narrower than E8-E13's full-tier
  classification).
- **TMDB metadata lookup** — auto-fetch genre/director/year from
  TMDB to augment advisor context.
- **Topology-hint-driven fitter** — advisor returns a preferred
  filter topology (cascade vs shelf+PEQ) and the fitter respects it.
- **On-disk advisor response caching** — avoid re-running LLM for
  the same title+features.

---

## 8. File layout

### Core model

| Path | Role |
|---|---|
| `src/main/python/model/auto_beq.py` | Filter optimizer (grid, smoothing, evaluate_filter_chain) |
| `src/main/python/model/auto_beq_nn.py` | NN model (XGBoost late fusion, feature vectors, label encoding) |
| `src/main/python/model/auto_beq_advisor.py` | Metadata structures (MediaMetadata, CurveFeatures, Advice) |
| `src/main/python/model/auto_beq_catalogue.py` | BEQ catalogue fetch + disk cache |
| `src/main/python/model/auto_beq_metadata.py` | TMDb metadata enrichment |
| `src/main/python/model/iir.py` | Biquad coefficient computation (LowShelf, HighShelf, PeakingEQ) |
| `src/main/python/model/signal.py` | Audio I/O (Signal class, read_wav_data) |
| `src/main/python/model/wav_integrity.py` | WAV header validation |
| `src/test/python/spike/_auto_beq_helpers.py` | Shared helpers: WAV cache, config, audio probing, extraction |

### CLI scripts (user-facing)

| Path | Role | In unified CLI? |
|---|---|---|
| `scripts/beq.py` | **Unified CLI** — single entry point, interactive menu + subcommands | Entry point |
| `scripts/cli_common.py` | Shared CLI utilities (filterable_select, config, banner) | Library |
| `scripts/beq_profile_cli.py` | Profile generation CLI (menus, progress, directory browser) | `beq.py profile` |
| `scripts/generate_beq_profile.py` | End-to-end profile generation pipeline | Called by beq_profile_cli |
| `scripts/extract_lfe.py` | LFE extraction to portable WAV cache (standalone, Docker-safe) | `beq.py extract` |
| `scripts/wav_cache_status.py` | WAV cache summary: counts, titles, author breakdown | `beq.py cache-status` |
| `scripts/verify_wav_cache.py` | WAV cache integrity check, optional corrupt file deletion | `beq.py verify` |
| `scripts/nn_comparison_report.py` | Compare NN-predicted vs hand-coded BEQ filters (markdown) | `beq.py nn-report` |
| `scripts/sweep_report.py` | Experiment sweep comparison report from CSV results | `beq.py sweep report` |

### Shell wrappers (orchestration)

| Path | Role | In unified CLI? |
|---|---|---|
| `scripts/run-sweep-discover.sh` | Discover media + match catalogue (sets PYTHONPATH, calls module) | `beq.py sweep discover` |
| `scripts/run-sweep-tests.sh` | Run auto-BEQ pipeline on discovered media (pytest wrapper) | `beq.py sweep run` |
| `scripts/run-spike-tests.sh` | Run spike test suite (unit + integration) | No (dev tooling) |
| `scripts/run-advisor-comparison.sh` | Compare all advisor implementations side-by-side | No (research) |

### Internal / dev-only

| Path | Role |
|---|---|
| `scripts/spike_auto_beq.py` | CLI playground for testing filter proposals on synthetic data |
| `scripts/regen_ui.py` | Regenerate Python source from Qt Designer `.ui` files |

### Tests and resources

| Path | Role |
|---|---|
| `src/test/python/spike/test_auto_beq.py` | Primary deliverable — parametrised integration test |
| `src/test/python/spike/test_beq_profile_cli.py` | CLI integration tests (25 tests) |
| `src/test/python/conftest.py` | `catalogue_snapshot` session fixture |
| `src/test/resources/auto_beq/database.json` | Committed catalogue snapshot (~55 KB) |
| `docs/design/auto_beq.md` | This document |
