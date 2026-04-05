# Auto-BEQ — Automated Filter Suggestion

**Status:** spike (single-title proof-of-concept). Not shipped to users. See
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
chain whose response cancels that rolloff across `band` (default 20-80 Hz).

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
| freq (Hz) | knee from stage 1 | [15, 120] |
| Q | 0.7 (Butterworth-ish) | [0.3, 2.0] |
| gain (dB) | measured rolloff depth, clamped to [1, 18] | [0, 18] |

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

### Pass thresholds (per vision doc)

For each title, compute `err = target + proposed_response` across 20-80 Hz:

| Metric | PASS threshold | MARGINAL threshold |
|---|---|---|
| `mean(|err|)` | < 2.0 dB | < 3.0 dB |
| `max(|err|)` | < 5.0 dB | < 7.5 dB |

Topology match (same filter types as catalogue) is a **secondary** metric.
A proposed chain with different topology but equivalent in-band response is
acceptable — bass-frequency IIR filters have well-known equivalencies
(different `freq`/`Q`/`gain` triplets can produce near-identical 20-80 Hz
curves).

### Spike results (April 2026)

Three single-LowShelf titles, synthetic roundtrip:

| Title | Mean err | Max err | Verdict |
|---|---|---|---|
| Battle: Los Angeles (+4 dB @ 28 Hz Q=0.9) | 0.02 dB | 0.09 dB | PASS |
| Captain America: TWS (+3.8 dB @ 22 Hz Q=1.1) | 0.02 dB | 0.16 dB | PASS |
| Run Hide Fight (+6 dB @ 17 Hz Q=0.7) | 0.08 dB | 0.38 dB | PASS |

All three comfortably pass. "Run Hide Fight" converges to a different
parameter triplet (26.78 Hz Q=0.82 +2.01 dB) than the catalogue's, but the
resulting in-band response is within 0.4 dB — the equivalence noted above.

### Real-media caveats (not yet validated)

The synthetic test does NOT prove:

- That a real LFE track from a Blu-ray matches its catalogue entry's
  implied curve. Release variants, stem-mastering differences, and
  bass-management pre-processing can all introduce divergence.
- That measurement noise (room rumble in the original mix, mastering
  dither) doesn't push the knee detector off.
- That multi-channel bass-managed content (5.1 where the LFE carries only a
  band-passed subset) produces a clean curve to feed the optimizer.

The test file includes an optional `test_real_media_roundtrip` gated on
`AUTO_BEQ_MEDIA_PATH`. Running it on a known title exercises the ffmpeg →
signal → optimizer pipeline end-to-end. A media-vs-catalogue divergence
assertion (<10 dB, soft) catches the "wrong rip" failure mode.

---

## 5. Known limitations

- **Shelf + one PEQ only.** Multi-PEQ catalogue entries (the majority of
  the real catalogue) will fit imperfectly. Expected MARGINAL/FAIL on
  entries with 4+ filters.
- **No topology preference.** The objective minimises response error only,
  so a narrow PEQ might be chosen where a shelf belongs. Acceptable for the
  spike; future work adds a penalty term.
- **Mono only.** No bass-management awareness. The optimizer sees one
  channel's curve at a time.
- **Single-title scope.** We've validated on three clean single-shelf
  cases. The 30-50 title benchmark from the vision doc is future work.
- **No streaming-only content.** Requires a local file to analyse.

---

## 6. Expansion path

### To add more test titles

1. Identify titles in the catalogue: inspect
   `src/test/resources/auto_beq/database.json`, or filter the full catalogue
   (`https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json`).
2. Add the title + filter-count tuple to the snapshot (the shell one-liner
   used to build the snapshot is documented in the spike's plan file).
3. Add an entry to `FIXTURES` in `test_auto_beq.py` with an expected
   verdict.
4. Run `pytest src/test/python/spike/test_auto_beq.py -v` — MARGINAL/FAIL
   cases surface immediately.

### To graduate the spike to a production feature

The three-tier roadmap from the vision brief:

1. **Magic wand button** — wire `propose_filters` to a QPushButton in the
   signal-analysis view. Take the currently-loaded `Signal`'s
   `avg_spectrum`, normalise to 80 Hz anchor, call `propose_filters`,
   populate the filter-table UI.
2. **Batch CLI** — thin wrapper that walks a folder of media files, calls
   `propose_filters` per file, writes sidecar `.beq` files.
3. **ezBEQ send** — HTTP POST of the filter chain to ezBEQ's `/api/`
   endpoint. ~100 LOC module; no optimizer changes.

### Metrics we'll track as the spike grows

- **PASS rate across the catalogue** — target ≥80% per the vision doc.
- **Distribution of `mean_abs_err_db`** — where does the long tail live?
- **Failure-mode taxonomy** — which content types fail, and why. This
  drives objective-function iteration.
- **Wall-clock per title** — currently ~10 ms per synthetic roundtrip.
  Comfortable for an interactive button.

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
