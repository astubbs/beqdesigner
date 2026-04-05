# Auto-BEQ — Automated Filter Suggestion

**Status:** spike (single-title proof-of-concept). Not shipped to users.
The first real-media run exposed that the current objective function
("make the measured curve flat") is not what BEQ actually does - see
section 4 "Real-media findings". The next iteration needs a different
algorithm (knee-extension rather than curve-inversion). See
"Expansion path" for the route to a production feature.

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

Takes a magnitude curve representing a rolloff (low-frequency attenuation,
expressed in dB relative to an 80 Hz anchor) and returns a small filter
chain whose response cancels that rolloff across `band` (default 5-80 Hz).

> **Heads up:** the "cancel the rolloff" framing is what the current
> code does, and section 4 now explains why that framing doesn't match
> real BEQ work. This section describes the algorithm *as
> implemented*. The next iteration replaces it.

**Staged approach:**

### Stage 1 — Rolloff knee detection

Find the frequency where the curve first crosses -3 dB relative to the band
upper edge. This becomes the shelf-frequency seed.

- If the band's low-end depth is under 1 dB, return `[]` (no correction
  needed).
- If the curve never crosses -3 dB inside the band, use the band's lowest
  frequency as the knee seed.

### Stage 2 — Single low shelf fit

`scipy.optimize.minimize` with `method="L-BFGS-B"`:

| Param | Seed | Bounds |
|---|---|---|
| freq (Hz) | knee from stage 1 | [10, 120] |
| Q | 0.7 (Butterworth-ish) | [0.3, 2.0] |
| gain (dB) | measured rolloff depth, clamped to [1, 30] | [0, 30] |

**Objective:** RMS of `(target + shelf_response)` across the band. We're
looking for the shelf whose response cancels the target, so minimising
their sum is minimising residual error.

The shelf response is evaluated at the exact target frequencies via
`scipy.signal.freqz(b, a, worN=freqs_hz, fs=fs)` on the `b`/`a` coefficients
of a `model.iir.LowShelf`. No interpolation, no FFT bin mismatch.

### Stage 3 — Residual PEQ (conditional)

If the shelf's final objective value exceeds `residual_threshold_db`
(default 1.0 dB), fit one additional `PeakingEQ` on the residual
`target - (-shelf_response)`:

| Param | Seed | Bounds |
|---|---|---|
| freq (Hz) | bin of max \|residual\| in the band | [20, 80] |
| Q | 1.5 | [0.5, 4.0] |
| gain (dB) | `-residual` at worst bin | [-12, +12] |

Stop after one PEQ. Multi-PEQ chains are out of scope for the spike.

### Stage 4 — Return

List of dicts matching the `CatalogueEntry.filters` schema:
`{"type": "LowShelf"|"PeakingEQ"|"HighShelf", "freq": float, "q": float, "gain": float}`

Directly consumable by `model.iir.CompleteFilter(fs, filters=...)`.

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

### Algorithm-quality grade vs test outcome (terminology)

Two different "PASS/FAIL" concepts are in play and the word overlap is
confusing:

- **Algorithm quality grade** — `MatchMetrics.verdict`, with values
  `PASS` / `MARGINAL` / `FAIL`. This grades *how well the optimizer's
  output matches the ground truth* against the band-error thresholds
  below. It is printed in the text report.
- **Test outcome** — pytest's `PASSED`/`FAILED` for the test function.
  This depends on the test's hard `assert` statements, which are
  deliberately loose for real-media runs (we want diagnostics, not a
  red test, while we're exploring). You can see `Verdict: FAIL` and
  `PASSED` in the same run without contradiction.

A future rename may disambiguate (e.g. rename `verdict` to `grade`).

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

### Synthetic results (April 2026, band 5-80 Hz)

Three single-LowShelf titles, synthetic roundtrip:

| Title | Mean err | Max err | Grade |
|---|---|---|---|
| Battle: Los Angeles (+4 dB @ 28 Hz Q=0.9) | 0.09 dB | 0.18 dB | PASS |
| Captain America: TWS (+3.8 dB @ 22 Hz Q=1.1) | 0.13 dB | 0.27 dB | PASS |
| Run Hide Fight (+6 dB @ 17 Hz Q=0.7) | 0.02 dB | 0.06 dB | PASS |

All three comfortably pass. The **fixture set was too easy** though -
see "Real-media findings" below. All three entries are single
LowShelves with freq inside 17-28 Hz and gain ≤ 6 dB, so most of the
correction falls cleanly inside the scoring band. The optimizer's
single-shelf-plus-one-PEQ scope was never stressed by these fixtures.

### Real-media findings (April 2026) — the spike actually moved

Ran `test_real_media_roundtrip` against Edge of Tomorrow UHD 2160p
(DTS-HD MA 7.1, LFE channel extracted via `pan=c0=LFE` at 1 kHz).
Compared against the 5-filter catalogue entry (4x LowShelf @23 Hz +
PEQ @54 Hz, ~+28 dB correction at 10 Hz).

**Measured LFE curve (relative to 80 Hz anchor):**

| Freq | Measured dB | Catalogue correction dB |
|---|---|---|
| 10 Hz | -6.8 | +28.1 |
| 20 Hz | +7.5 | +19.4 |
| 80 Hz | 0 | 0 |

The measured LFE has a **hump around 20 Hz** then drops sharply both
ways. It is NOT a simple rolloff. The optimizer (minimising
`target + response`) proposed a -12 dB notch at 20 Hz to flatten the
hump, and an 8 dB boost at 10 Hz. Algorithm grade: FAIL.

**The real insight: the objective function is wrong.**

- Minimising `target + candidate_response` says "make the media flat".
- BEQ is *not* about making media flat. Real LFE content has
  mastering-specific shape (20 Hz humps, 80 Hz rolloff) that should be
  preserved.
- Human BEQ experts do something different: they identify the
  **natural rolloff slope below the LFE passband** (e.g. the
  14 dB/octave drop from 20 Hz to 10 Hz visible above) and extend it
  deeper with a shelf. The goal is infra-bass extension, not flattening.

**What this means:**

- The spike proved the optimization math works (synthetic passes).
- The spike proved the framing is wrong for BEQ (real-media FAIL).
- The next iteration needs a fundamentally different algorithm:
  knee-extension rather than curve-inversion.

### Real-media caveats that ALSO apply

- Release variants matter: a UHD master may have different LFE than the
  Blu-ray a catalogue entry was built against. The 7.37 dB 20-80 Hz
  divergence we observed is partly this.
- Welch averaging collapses a 113-minute movie into one curve; loud
  scenes dominate. This may or may not match what the catalogue expert
  measured (they often work on specific reference scenes).
- Multi-channel bass management was not exercised - mono LFE only.

---

## 5. Known limitations

### The big one: objective function is wrong for real BEQ work
The current `propose_filters` minimises `|target + candidate_response|` - i.e.
it tries to produce a chain whose response cancels the input curve and
leaves zero. That treats "flat" as the goal. **Real BEQ work isn't
flattening** - it's detecting the natural rolloff slope at the very low
end of the content and extending that slope with a shelf so deeper bass
is audible. The real-media Edge of Tomorrow run exposed this directly
(see section 4). Fixing this is the headline item for the next spike
iteration.

### Other limitations (all still apply)
- **Shelf + one PEQ only.** Multi-PEQ catalogue entries (the majority of
  the real catalogue) will fit imperfectly even after the objective is
  fixed.
- **No topology preference.** The objective minimises response error
  only, so a narrow PEQ might be chosen where a shelf belongs.
- **Mono only.** No bass-management awareness. The optimizer sees one
  channel's curve at a time.
- **Scoring band 5-80 Hz.** Extended down from 20-80 Hz after the
  real-media run showed infra-bass is where BEQ lives. Lower edge
  depends on the signal pipeline's ability to produce reliable
  magnitude data at 5-10 Hz - not yet validated end-to-end.
- **Synthetic fixtures are too easy.** Three hand-picked single-shelf
  entries with most action inside the band. Need to add a deep
  cascaded-shelf entry (e.g. Edge of Tomorrow 5-filter) as a known
  FAIL fixture so we can track algorithm improvements against it.
- **No streaming-only content.** Requires a local file to analyse.

---

## 6. Expansion path

### Next spike iteration: fix the objective

The real-media run proved the current algorithm answers the wrong
question. The next iteration needs:

1. **Knee-extension algorithm, not curve-inversion.** Detect the
   natural rolloff slope in the content at the bottom of the LFE
   passband (where the curve starts sloping down sharply - e.g. below
   15 Hz in the EoT measurement) and propose a shelf that extends that
   slope deeper. Don't try to modify what happens above the knee.
2. **Possibly: fit against the catalogue curve's shape, not the
   measured curve.** Treat the catalogue as labelled training data -
   given a measured rolloff shape, what shelf+PEQ combination did a
   human expert prescribe? This is essentially a supervised-learning
   framing of the same problem. May be overkill for an IIR fit but is
   the cleanest formulation of "do what the expert would do".
3. **Add a deep entry to synthetic fixtures as a FAIL canary.** E.g.
   Edge of Tomorrow 5-filter. Stops us from regressing on hard cases
   when we tune the optimizer for easy ones.
4. **Revisit knee detection on real LFE shapes.** The measured curve
   had a hump at 20 Hz and the current `detect_rolloff_knee` tries to
   scan from the band low edge upward. Real LFE needs a detector that
   finds the downward slope at the *very bottom* of the passband.

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
