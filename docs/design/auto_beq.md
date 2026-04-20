# Auto-BEQ - Automated Filter Suggestion

**Companion docs:**
- [`auto_beq_experiments.md`](auto_beq_experiments.md) - append-only log of experiments tried, results, lessons
- [`auto_beq_plan.md`](auto_beq_plan.md) - original advisor design plan (historical context)
- [FAQ and Glossary](../faq.md) - terminology, developer commands, key numbers
- [LFE Extractor](../lfe_extractor.md) - Docker-based extraction for NAS deployment

**Audience:** a competent Python developer who has never touched machine
learning, digital signal processing, or audio engineering. All domain
terms are defined on first use and in the [glossary](../faq.md#glossary).

---

## 1. What Auto-BEQ does

Movies are often mastered with rolled-off bass - the low-frequency
content is intentionally reduced compared to the original mix. A BEQ
(Bass EQ) profile adds correction filters to restore that lost bass.
The [BEQ catalogue](https://beqcatalogue.readthedocs.io/) is a
community-maintained database of hand-crafted correction profiles for
thousands of titles.

Auto-BEQ generates these correction profiles automatically. Given a
media file (MKV, WAV), it extracts the LFE (Low Frequency Effects)
channel, analyses the frequency response, and produces a set of IIR
(Infinite Impulse Response) filters that correct the bass rolloff.

The system is in production with CLI commands, Docker deployment, and
a trained machine learning model.

### Current state

- **Production model:** E82 - an XGBoost gradient-boosted tree model
  trained on 50:1 weighted hybrid data (real audio + synthetic). Achieves
  ~2.8 dB mean error against hand-crafted catalogue entries.
- **Current champion:** E85 - a differentiable DSP approach that trains
  a neural network to directly output filter parameters, optimised with
  an acoustic loss function. Achieves ~2.3 dB mean error. Not yet the
  production default because it requires PyTorch.
- **WAV cache:** ~1000+ extracted LFE files used for training and
  evaluation.
- **Catalogue coverage:** 14,785+ entries in the BEQ catalogue (as of
  April 2026).

### CLI commands

| Command | Purpose |
|---|---|
| `bin/beq-designer profile` | Generate a BEQ correction profile for a media file |
| `bin/beq-designer extract` | Extract LFE audio from media files into the WAV cache |
| `dev reassess` | Train each experiment from scratch on 80/20 split, compare generalisation |
| `dev evaluate` | Run the saved production model on all titles, measure real-world performance |
| `dev train` | Train the production XGBoost model (E82) |
| `dev train-torch` | Train the differentiable DSP model (E85) |
| `dev benchmark` | Compare multiple advisors side-by-side on the same library |
| `dev test-advisor` | Test a single advisor on one title interactively |

See the [FAQ](../faq.md#developer-commands) for detailed guidance on
when to use `dev reassess` vs `dev evaluate`.

### Deployment

The LFE extraction pipeline ships as a Docker image for NAS deployment.
This lets users extract LFE audio from their full media library on the
server where the files live, without installing Python, scipy, or GUI
dependencies. See [`lfe_extractor.md`](../lfe_extractor.md) for setup.

---

## 2. System architecture

The pipeline has five stages: extract, analyse, enrich, advise, fit.

```mermaid
flowchart LR
    A[Media file<br/>.mkv / .wav] -->|ffmpeg extract<br/>pan=c{LFE}, ar=1000| B[mono WAV @ 1 kHz]
    B -->|read_wav_data<br/>soundfile| C[numpy samples]
    C -->|blend-a0.7-P90<br/>Welch + chunked STFT| D[magnitude curve<br/>dB vs Hz]
    D -->|feature extraction<br/>102-dim vector| E[CurveFeatures +<br/>MediaMetadata]
    E -->|Advisor| F[filter parameters<br/>freq, gain, Q per filter]
    F -->|scipy.optimize<br/>refinement| G[filter chain<br/>list of dicts]
    G -->|CompleteFilter.get_sos| H[DSP / ezBEQ]

    subgraph extraction[Audio extraction]
    A
    B
    C
    end

    subgraph analysis[Analysis + prediction]
    D
    E
    F
    G
    end
```

### Stage 1 - Extract

`ffmpeg` extracts the LFE channel from a media file and resamples it
to 1000 Hz mono WAV. The result is cached in the WAV cache
(`{shared_dir}/wav-cache/tmdb/{shard}/{id}/...`) so subsequent runs
skip extraction. See [`lfe_extractor.md`](../lfe_extractor.md) for the
Docker deployment path.

### Stage 2 - Analyse

The WAV is analysed to produce a frequency-domain magnitude curve. Two
methods are blended (the `blend-a0.7-P90` strategy from experiment E18):

- **Welch average** (`Signal.avg_spectrum()`) - splits the audio into
  overlapping windows, computes the FFT (Fast Fourier Transform - an
  algorithm that converts audio from time-domain to frequency-domain)
  of each, and averages. Good for long content but biased toward loud
  scenes.
- **Chunked STFT percentile** (`load_and_smooth_chunked()`) - splits
  audio into fixed-length chunks, computes the STFT (Short-Time Fourier
  Transform) peak per chunk, and takes the 90th percentile across
  chunks. More robust for short content with sparse bass.

The blended curve is interpolated to a logarithmic frequency grid,
normalised to an 80 Hz anchor point, and smoothed to 1/6-octave
resolution.

### Stage 3 - Enrich

Audio features (102 dimensions - spectral statistics, rolloff
characteristics, band energies) are extracted from the magnitude curve
and combined with metadata from TMDb (The Movie Database - genre, year,
runtime, content rating) to form the input feature vector for the
advisor.

### Stage 4 - Advise

An advisor takes the feature vector and produces filter parameters. The
system uses the strategy pattern - different advisors implement the same
interface but use different approaches.

| Advisor | How it works | Notes |
|---|---|---|
| `TrainedModelAdvisor` | XGBoost model predicting all filter params at once | **Production default** (E82) |
| `MeasurementAdvisor` | Heuristic rules based on audio frequency analysis | No ML dependency |
| `OllamaAdvisor` | Asks a local LLM about the film's likely bass profile | Requires Ollama |
| `TopologyAdvisor` | Classifies rolloff shape and applies a template | No ML dependency |
| `SlopeExtensionAdvisor` | Extends the measured rolloff slope | No ML dependency |

Select an advisor with `AUTO_BEQ_ADVISOR` env var or `--advisor` CLI flag.

### Stage 5 - Fit

The advisor's filter parameters are refined using `scipy.optimize` to
minimise the difference between the proposed filter chain's response
and the target correction curve. This stage uses the same N-filter
iterative fitter described in section 3 below.

---

## 3. The N-filter iterative fitter (optimizer internals)

`model.auto_beq.propose_filters(target_curve_db, freqs_hz, fs, band)`

Takes an in-band magnitude curve in dB (expressed relative to an 80 Hz
anchor) and returns a short filter chain whose response cancels the
curve across `band` (default 5-80 Hz). Real-media callers pre-smooth
the curve to 1/6-octave before feeding it in (see Stage 0 below).

**Pipeline:**

### Stage 0 - 1/6-octave smoothing (caller's responsibility)

Raw magnitude spectra contain narrow resonances and dither artefacts
that a broad IIR filter cannot (and should not) chase. Real-media
callers smooth the curve with
`smooth_fractional_octave(curve, freqs, octaves=1/6)` before handing
it to `propose_filters`. The synthetic test skips this because
catalogue-generated curves are already smooth by construction.

Smoothing is a log-frequency Gaussian kernel: for each bin, a Gaussian
weighted over `log2(freqs)` with `sigma = octaves/2.355`. 1/6-octave is
the SPL (Sound Pressure Level) measurement convention.

### Stage 1 - One LowShelf (bidirectional)

`scipy.optimize.minimize` with `method="L-BFGS-B"` (a quasi-Newton
optimiser that handles bounds on parameters):

| Param | Seed | Bounds |
|---|---|---|
| freq (Hz) | 25 Hz | [5, 120] |
| Q | 0.7 (Butterworth-ish) | [0.3, 2.0] |
| gain (dB) | negated mean of target in lowest octave of band | [-30, +30] |

The gain bound is **bidirectional** - the shelf can lift or cut the
low end. This captures broad trends in the target.

### Stage 2 - Iterative residual PEQs

PEQ = Parametric Equalizer - a filter that boosts or cuts a specific
frequency band, with adjustable centre frequency, bandwidth (Q), and
gain.

Loop, adding one PEQ per iteration until either `max_filters` (default
6) is reached or the in-band max residual drops below `stop_max_err_db`
(default 0.5 dB):

1. Compute `err = target + evaluate(chain_so_far)`.
2. If `max(|err|) in band < stop_max_err_db`, stop.
3. Find the worst-residual frequency (`argmax(|err|)` in band); use it
   as the PEQ seed frequency, with gain = `-err` at that bin.
4. Fit the PEQ from **three Q seeds** (0.7, 1.5, 3.0) and keep the
   best. Bounds: freq in band, Q in [0.3, 4.0], gain in [-30, +30].
5. If the new PEQ reduces in-band RMS (Root Mean Square) error by less
   than 0.05 dB, stop (the optimizer has nothing useful to add).
6. Append the PEQ and continue.

Three Q seeds matter: the L-BFGS-B optimizer is local, and a single
seed can get stuck in a shallow minimum when the target has multiple
features nearby. Trying 0.7/1.5/3.0 covers "broad", "medium", and
"narrow" shapes.

### Stage 3 - Return

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
PEQ chains.

---

## 4. Evaluation modes

The system has two evaluation modes for measuring model quality:

- **Reassess** (`dev reassess`) - trains each experiment approach from
  scratch on an 80/20 held-out split and compares them. Measures
  generalisation. Use during research.
- **Evaluate** (`dev evaluate`) - loads the saved production model and
  runs it on all catalogue-matched titles. Measures real-world
  performance. Use for quality assurance.

See the [FAQ](../faq.md#dev-reassess-vs-dev-evaluate) for detailed
guidance on when to use each.

### Quality thresholds

For each title, compute `err = target + proposed_response` across the
scoring band (5-80 Hz):

| Metric | PASS threshold | MARGINAL threshold |
|---|---|---|
| `mean(\|err\|)` | < 2.0 dB | < 3.0 dB |
| `max(\|err\|)` | < 5.0 dB | < 7.5 dB |

Topology match (same filter types as catalogue) is a **secondary**
metric. A proposed chain with different topology but equivalent in-band
response is acceptable - bass-frequency IIR filters have well-known
equivalencies (different freq/Q/gain triplets can produce near-identical
in-band curves).

### Champion tracking

Each `dev reassess` run records the best-performing experiment in
`{shared_dir}/champion_history.json`. See the
[FAQ](../faq.md#champion-tracking) for details.

---

## 5. Experiment history

The system has evolved through 85+ experiments across several research
families. The full details are in the
[experiment log](auto_beq_experiments.md). Here is a summary of the
major phases:

| Phase | Experiments | Approach | Outcome |
|---|---|---|---|
| Fitter validation | E1-E6 | scipy.optimize iterative fitter | Proven: fitter reproduces catalogue curves within 0.52 dB mean |
| LLM advisors | E7-E13 | Ollama llama3.1:8b for gain estimation | Dead end: small LLMs too weak for numeric calibration |
| Measurement heuristics | E14-E17 | Deficit + slope extension | 41% non-FAIL ceiling; topology classifier adopted |
| Spectrum extraction | E18-E22 | Chunked STFT, blended methods | `blend-a0.7-P90` adopted as default extraction |
| Initial ML | E25-E40 | XGBoost, late fusion, metadata encoding | 2.45 dB on 220 titles |
| Feature augmentation | E41-E52 (F) | Synthetic augmentation, clustering | F1 synthetic augmentation was breakthrough (-0.73 dB) |
| Hyperparameter tuning | E53-E59 (G) | Alpha sweep, sigma sweep, ensembles | Per-author alpha hits oracle ceiling |
| Multi-author | E60-E67 (H) | Response averaging, marginalisation | Dead end: multi-author disagreement is signal, not noise |
| Author selection | E68-E70 (I) | Metadata classifier for author routing | I1b soft-blend adopted at 2.37 dB |
| Scale-up validation | E71-E75 | 932-WAV corpus re-validation | Rankings preserved; +0.3-0.5 dB honest measurement |
| Real-audio regime | E77-E82 | 50:1 weighted hybrid training | **Production model: ~2.0 dB** |
| Differentiable DSP | E85 | Neural network with acoustic loss | **Champion: ~2.3 dB** (reassess), improves with more data |

### Key architectural lesson

Early experiments (E1-E40) tried to bridge the gap between a measured
LFE curve and a catalogue-shaped correction using heuristics and LLM
prompts. The breakthrough came from reframing the problem as supervised
learning: train a model on (audio features, catalogue filters) pairs
extracted from real media. The model learns the implicit mapping from
"what the content sounds like" to "what correction a human expert would
apply" without needing to encode that knowledge as rules.

---

## 6. Known limitations

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
- **E85 requires PyTorch.** The differentiable DSP champion (E85) needs
  PyTorch, which is a heavy dependency. It is not the production default
  for this reason. E82 (XGBoost) runs with lighter dependencies.

---

## 7. Future features

- **Magic-wand button** - wire `propose_filters` to a QPushButton in
  the GUI signal-analysis view (Tier 1 of the original vision).
- **ezBEQ send** - HTTP POST of the filter chain to ezBEQ's `/api/`
  endpoint for direct DSP control.
- **AnthropicAdvisor** - API-key-based advisor using Claude for the
  gain-multiplier question.
- **TMDB metadata lookup** - auto-fetch genre/director/year from
  TMDb to augment advisor context.
- **Topology-hint-driven fitter** - advisor returns a preferred
  filter topology (cascade vs shelf+PEQ) and the fitter respects it.
- **On-disk advisor response caching** - avoid re-running LLM for
  the same title+features.
- **Pre-extract audio for uncatalogued media.** The library discovery
  config records all media files (matched + unmatched). Optional
  background pre-extraction for unmatched media would make later profile
  construction instant instead of waiting 30-120s per title for ffmpeg.

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
| `scripts/beq.py` | **Unified CLI** - single entry point, interactive menu + subcommands | Entry point |
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
| `src/test/python/spike/test_auto_beq.py` | Primary deliverable - parametrised integration test |
| `src/test/python/spike/test_beq_profile_cli.py` | CLI integration tests (25 tests) |
| `src/test/python/conftest.py` | `catalogue_snapshot` session fixture |
| `src/test/resources/auto_beq/database.json` | Committed catalogue snapshot (~55 KB) |
| `docs/design/auto_beq.md` | This document |
