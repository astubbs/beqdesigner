# Auto-BEQ — Automated Filter Suggestion

**Status:** spike, not shipped. Tests green on 5 synthetic + 2
real-media fixtures (Edge of Tomorrow and Mad Max: Fury Road). The
N-filter iterative fitter + 1/6-octave smoothing can reproduce
measured LFE curves within 2 dB mean / 5 dB max across 5-80 Hz.
**However:** the fitter currently produces filters that NEUTRALISE
measured content rather than EXTENDING it - it passes its tests but
does not yet produce BEQ-appropriate output. See section 4
"Real-media results" and "Known limitations" for the gap. Next
iteration: construct a target-correction curve so the same fitter
produces usable DSP profiles.

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
    D -->|interp + anchor normalise| E[target curve on log grid]
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
`~/.config/beqdesigner/auto_beq_media.json` (two entries, both with
`expected_grade: PASS`). Both extractions are cached next to the
source files after the first run.

| Title | Proposed filters | Mean err | Max err | Grade |
|---|---|---|---|---|
| Edge of Tomorrow (5-filter catalogue) | 6 | 1.29 dB | 3.82 dB | PASS |
| Mad Max: Fury Road (5-filter catalogue) | 6 | 0.74 dB | 1.91 dB | PASS |

Pipeline: ffmpeg extracts LFE to 1 kHz mono WAV (cached), `Signal.avg_spectrum()`
produces a Welch-averaged curve, the test interpolates it onto the
log-spaced 5-200 Hz grid, normalises to 0 dB at 80 Hz, smooths to
1/6-octave, and feeds that to `propose_filters`.

### What this proves (and what it doesn't)

**Proven:** given a smoothed in-band target curve, `scipy.optimize`
plus an iterative greedy fitter can find an N-filter IIR chain that
reproduces it within the 2 dB mean / 5 dB max thresholds, for
realistically complex LFE content.

**NOT proven:** that the proposed filters are BEQ-appropriate. The
optimizer is a generic curve-fitter. Given raw LFE content, it
currently produces filters that **neutralise** the content's natural
shape rather than **extending** it. Look at the Mad Max output: a
LowShelf at 13 Hz with **+29 dB gain** plus a stack of PEQs that
oscillate around the band - these cancel the measured curve, but
would ruin the listening experience if applied to a DSP.

For BEQ this remains a genuinely hard problem. The spike's test gate
now ensures `scipy.optimize + N filters` can FIT real content curves.
The remaining work is wiring that fitter up to a target curve that
represents the *correction* we actually want (infra-bass extension),
not the measured content's raw shape.

---

## 5. Known limitations

### The big one: proposed filters neutralise rather than extend
The fitter minimises `|target + candidate_response|` - it treats
"flat target" as the goal. Applied to raw measured LFE, it produces
filters that CANCEL the natural content shape rather than extending
it deeper. **Real BEQ work is infra-bass extension, not flattening.**
The tests green-light the math (we CAN fit real curves) but the
filter chains produced aren't usable DSP profiles. A future iteration
needs to construct a DIFFERENT target - e.g. "what the content should
look like after correction" (measured + desired-extension) - and then
use this same fitter against that target.

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
  specific reference scenes instead.

---

## 6. Expansion path

### Next spike iteration: produce usable filters

The N-filter fitter is numerically correct but its OUTPUT is not
what we want. Next iteration needs to construct a target curve that
represents "what to correct TO", not just "what is here now":

1. **Derive a target-correction curve from the measured curve.**
   Heuristic: take the measured curve, identify the natural rolloff
   knee at the bottom of the LFE passband (e.g. by fitting a line to
   the slope between 10 and 20 Hz), and construct a synthetic
   extension that continues that slope down to 5 Hz with a shelf. The
   fitter then fits against `extension - measured` (what to add),
   which should produce sane LowShelf + small-PEQ chains.
2. **Cap gains and Qs more aggressively.** A +29 dB shelf is never a
   BEQ. Clamp shelf gain to ~12 dB, shelf Q to ~1.2, PEQ Q to ~2.5.
   The fitter must do more with less - which should nudge the output
   toward BEQ-shaped filters.
3. **Re-grade against the catalogue.** After generating the proposed
   chain, compare its in-band response to the catalogue's correction
   curve (not to the measured curve). PASS means "our proposal
   resembles what an expert prescribed", not "our proposal cancels
   the measured curve".

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
2. **Batch CLI** - thin wrapper that walks a folder of media files,
   calls `propose_filters` per file, writes sidecar `.beq` files.
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

## 7. File layout

| Path | Role |
|---|---|
| `src/main/python/model/auto_beq.py` | Optimizer module (pure Python, Qt-free) |
| `src/test/python/spike/test_auto_beq.py` | Primary deliverable — parametrised integration test |
| `src/test/python/conftest.py` | `catalogue_snapshot` session fixture |
| `src/test/resources/auto_beq/database.json` | Committed catalogue snapshot (~55 KB) |
| `scripts/spike_auto_beq.py` | CLI playground (secondary) |
| `docs/design/auto_beq.md` | This document |

No changes to existing `model/*.py` modules.
