# E14-E22: Measurement advisor and spectrum extraction

**Status:** Adopted - topology classifier (E17c) and blend-a0.7-P90 extraction (E18b)

---

## 2026-04-08: Spectrum extraction experiments

### E18 - Chunked percentile spectrum (STFT peak per chunk → P90)

**Hypothesis**: Chunking the audio and taking the 90th percentile of
STFT peak levels across chunks produces a more robust rolloff ceiling
estimate than whole-film Welch average — resistant to outlier scenes
(E15c: EoT showcase scenes inflate 10 Hz by 16-19 dB) and sparse bass
content in short TV episodes.

**Method**: Split LFE WAV into fixed-length chunks (30/60/90 s
sub-experiment). For each chunk, compute STFT and take max amplitude
at each frequency across all time frames (peak curve). Stack all chunk
peaks into a matrix and take the 90th percentile across chunks at each
frequency bin. Then normalise + smooth identically to `load_and_smooth()`.
Feed resulting curve into `propose_filters_from_measured()` with
MeasurementAdvisor (same pipeline as E17d baseline).

**Implementation**: `load_and_smooth_chunked()` in `_auto_beq_helpers.py`,
`test_chunked_percentile_roundtrip()` in `test_auto_beq.py`.

**Results (2 titles from media manifest, MeasurementAdvisor)**:

| Title | chunk_s | 10Hz delta | 20Hz delta | Mean err | Max err | Verdict |
|---|---|---|---|---|---|---|
| Mad Max: Fury Road | 30 | +8.9 dB | +0.2 dB | 8.80 | 20.21 | FAIL |
| Mad Max: Fury Road | 60 | +7.9 dB | -0.1 dB | 8.68 | 20.11 | FAIL |
| Mad Max: Fury Road | 90 | +7.4 dB | +0.1 dB | 8.53 | 19.91 | FAIL |
| John Wick | 30 | +6.0 dB | +5.3 dB | 3.39 | 7.99 | FAIL |
| John Wick | 60 | +4.9 dB | +5.0 dB | 2.32 | 6.57 | MARGINAL |
| John Wick | 90 | +5.0 dB | +4.2 dB | 1.92 | 6.91 | MARGINAL |

(Delta = chunked curve minus Welch baseline at that frequency, positive
means chunked sees MORE energy.)

**E17d baseline comparison** (same titles, MeasurementAdvisor):
- Mad Max: FAIL in both (multi-knee catalogue, not a spectrum issue)
- John Wick: E17d was FAIL with Welch; E18 improves to MARGINAL at
  60s/90s chunks

**Key observations**:
1. **Chunking consistently lifts 10 Hz by 5-9 dB** vs Welch average.
   The STFT peak captures transient bass events that Welch averages out.
   This is expected — peak ≠ average.
2. **20 Hz delta is small for Mad Max** (~0 dB) but **large for John
   Wick** (+4-5 dB). JW has sparser bass content; chunked peak captures
   its intermittent bass events that Welch dilutes.
3. **Mad Max still fails** because the failure mode is catalogue shape
   (multi-knee at 10+18 Hz), not spectrum extraction. The chunked curve
   is "better" (higher 10 Hz) but MeasurementAdvisor over-estimates gain
   from the larger deficit, leading to worse overshooting.
4. **Chunk length sensitivity**: 90s slightly better than 30s for both
   titles (less noisy per-chunk estimates). Sweet spot appears to be
   60-90s.
5. **The chunked curve fundamentally changes the MeasurementAdvisor's
   input**: it sees a larger deficit (because peak > average), which
   pushes gain higher. For titles where the catalogue wants aggressive
   correction (like JW: +13 dB), this helps. For titles where the
   catalogue is moderate, it overshoots.

**Lesson**: chunked-percentile spectrum extraction IS a different
signal than Welch average — it captures transient bass events that
averaging dilutes. But feeding this higher-energy curve into the
existing MeasurementAdvisor produces mixed results because the advisor's
deficit formula was tuned to Welch input. The chunked curve needs
either (a) a recalibrated gain formula, or (b) a different advisor
that accounts for the peak-vs-average gap.

**Kept**: yes, as a selectable spectrum extraction path. Run with
`test_chunked_percentile_roundtrip` in `test_auto_beq.py`.

#### E18 library sweep (11 titles, 54 tests, MeasurementAdvisor)

Extended E18 to the full library sweep via
`test_library_sweep_chunked` in `test_auto_beq_library_sweep.py`.
Both Welch baseline and chunked curves tested side-by-side.

**Verdict summary by chunk length**:

| chunk_s | Baseline P/M/F | Chunked P/M/F | Improved | Degraded | Same |
|---------|----------------|---------------|----------|----------|------|
| 30s     | 3/4/10         | 3/1/13        | 0        | 3        | 14   |
| 60s     | 3/4/11         | 5/1/12        | 2        | 1        | 15   |
| 90s     | 4/4/11         | 4/4/11        | 2        | 2        | 15   |

**Mean delta (chunked − baseline) across all titles**:
- 30s: +0.74 dB (worse on average)
- 60s: +0.25 dB (slightly worse)
- 90s: +0.20 dB (neutral)

**Per-title highlights** (best chunk_s per title):

| Title | Best chunk_s | Baseline avg | Chunked avg | Delta | Verdict change |
|---|---|---|---|---|---|
| Blue Eye Samurai | 90s | 3.8 dB (0P) | 3.7 dB (1P) | -0.2 dB | MARGINAL→PASS (2 eps) |
| Splinter Cell | 90s | 4.5 dB (0P) | 3.8 dB (0P) | -0.7 dB | FAIL→MARGINAL (1 ep) |
| Flow | 60s | 5.0 dB (0P) | 4.2 dB (0P) | -0.8 dB | no grade change |
| KPop Demon Hunters | 90s | 6.9 dB (0P) | 6.6 dB (0P) | -0.3 dB | no grade change |
| Pantheon | 90s | 0.9 dB (3P) | 1.2 dB (3P) | +0.3 dB | stays PASS |
| X-Men '97 | 60s | 1.1 dB (1P) | 2.7 dB (1P) | +1.6 dB | MARGINAL→FAIL (1 ep) |
| Elio | 30s | 6.0 dB (0P) | 7.1 dB (0P) | +1.1 dB | no change (low-gain catalogue) |

**Key findings from sweep**:
1. **60s chunks are the sweet spot**: only chunk length with net
   positive grade changes (2 improved, 1 degraded). 30s is too noisy;
   90s trades sensitivity for stability.
2. **Blue Eye Samurai benefits most**: 2 episodes flip from MARGINAL
   to PASS at 60s chunks. BES has sparse, punchy bass events that
   Welch dilutes — exactly the failure mode chunking was designed for.
3. **Titles with sustained bass (Pantheon, X-Men '97) slightly degrade**:
   chunked peak > Welch average → advisor over-estimates gain →
   overshooting. The +1.6 dB regression on X-Men '97 shows the
   MeasurementAdvisor's deficit formula is calibrated to Welch input.
4. **Low-gain catalogues (Elio: 5 dB summed gain) always degrade**:
   chunked curve sees "more" bass → larger deficit → advisor applies
   too much gain. The opposite of what a modest catalogue entry needs.
5. **Net effect is neutral to slightly negative** on the current
   advisor. The chunked curve IS a better signal but the
   MeasurementAdvisor wasn't tuned for it. A recalibrated formula
   (or a separate advisor) is needed to exploit the chunked signal.

**Lesson**: chunked-percentile extraction reveals real signal
improvements (Blue Eye Samurai PASS flips prove it), but feeding the
higher-energy curve into an advisor tuned for Welch averages produces
mixed results. Next step: either recalibrate MeasurementAdvisor's
deficit formula for chunked input, or build a ChunkedAdvisor that
accounts for the peak-vs-average gap.

### E18b - Strategy sweep: percentile and blending variations

**Hypothesis**: The E18 sweep showed chunked-P90 is too aggressive for
some content. Two mitigation strategies: (a) lower percentile (P75/P80)
to reduce peak bias, (b) blend Welch + chunked in dB domain.

**Method**: 6 strategies tested at 60s chunk length across 31 film/episode
tests (11 unique titles):
- `chunked-P75`, `chunked-P80`, `chunked-P90` (percentile sweep)
- `blend-a0.3-P90`, `blend-a0.5-P90`, `blend-a0.7-P90` (alpha=Welch weight)

**Implementation**: `load_and_smooth_blended()` in `_auto_beq_helpers.py`,
`test_library_sweep_strategies` in `test_auto_beq_library_sweep.py`.

**Results (186 tests, MeasurementAdvisor)**:

| Strategy | Baseline P/M/F | Test P/M/F | Improved | Degraded | Avg Δ |
|---|---|---|---|---|---|
| chunked-P75 | 7/6/18 | 7/1/23 | 1 | 5 | +0.39 dB |
| chunked-P80 | 7/6/18 | 7/2/22 | 1 | 5 | +0.37 dB |
| chunked-P90 | 7/6/18 | 9/5/17 | 5 | 2 | -0.10 dB |
| blend-a0.3 | 7/6/18 | 9/4/18 | 3 | 1 | -0.06 dB |
| blend-a0.5 | 7/6/18 | 8/4/19 | 2 | 2 | -0.03 dB |
| **blend-a0.7** | **7/6/18** | **9/4/18** | **2** | **0** | **+0.01 dB** |

**Key findings**:

1. **`blend-a0.7-P90` is the safest strategy**: 2 grade improvements
   (Blue Eye Samurai MARGINAL→PASS, Scavengers Reign MARGINAL→PASS),
   **zero degradations**. Average delta is +0.01 dB — essentially
   neutral on error while strictly improving grades.

2. **Lower percentiles (P75/P80) backfire**: they produce *more*
   degradations (5 each) than P90 (2). The lower percentile apparently
   pushes the curve into a range that confuses the MeasurementAdvisor's
   deficit formula. The P90 peak provides more signal, not less.

3. **`chunked-P90` has the most raw improvements (5)** but also 2
   degradations. The risk-reward is +3 net grade changes, best on that
   metric but with X-Men '97 MARGINAL→FAIL as collateral.

4. **`blend-a0.3-P90` is the aggressive blend**: 3 improvements, 1
   degradation (X-Men '97). Adds Super Mario Bros FAIL→MARGINAL flip
   vs blend-a0.7. More power but not zero-risk.

5. **Blending works because it preserves Welch's calibrated level**
   while incorporating the chunked curve's transient-event sensitivity.
   Higher alpha (more Welch) = safer. α=0.7 is the conservative
   sweet spot; α=0.3 is the aggressive option.

**Recommendation**: default to `blend-a0.7-P90` for production —
strictly non-regressing with meaningful improvements. Offer
`blend-a0.3-P90` as a "more aggressive" option for power users.

**Kept**: `load_and_smooth_blended()` added to `_auto_beq_helpers.py`.
Strategy sweep CSV: `.pytest_cache/auto_beq_sweep_strategies.csv`.

### E19 - MeasurementAdvisor constant calibration (one-at-a-time sweep)

**Hypothesis**: The cascade and multi-knee constants were tuned for
Welch input. With blended extraction as default, recalibrating them
may improve results.

**Method**: Made MeasurementAdvisor constants configurable via
constructor args. Swept 18 configurations (one-at-a-time + combined
winner) across 31 film/episode tests with blended-a0.7 extraction.

**Parameters swept**:
- `cascade_gain_ratio`: 5.0, 6.0, **7.0** (baseline), 8.0, 9.0
- `cascade_q`: 0.7, 0.8, **0.9** (baseline), 1.0, 1.2
- `multi_knee_slope_threshold`: **10.0**, 12.0, **15.0** (baseline), 18.0, 20.0
- `multi_knee_q`: 0.6, 0.7, **0.8** (baseline), **0.9**, 1.0

**Results (558 tests)**:

| Config | P/M/F | Imp | Deg | Avg Δ |
|---|---|---|---|---|
| baseline (g7/cQ0.9/s15/mQ0.8) | 9/4/18 | 0 | 0 | +0.00 |
| **s10** (slope threshold 10) | 10/3/18 | 1 | 0 | -0.26 |
| **mQ0.9** (multi-knee Q 0.9) | 9/5/17 | 1 | 0 | -0.04 |
| **s10+mQ0.9** (combined) | **10/4/17** | **2** | **0** | **-0.32** |
| all cascade_gain_ratio variants | 9/4/18 | 0 | 0 | +0.00 |
| all cascade_q variants | 9/4/18 | 0 | 0 | +0.00 |

**Key findings**:

1. **`cascade_gain_ratio` and `cascade_q` have ZERO effect** across
   all variants tested (5.0-9.0 and 0.7-1.2 respectively). The
   cascade construction is completely insensitive to these parameters
   in the current test set. This makes sense: the cascade is a target
   for the fitter, and the fitter adjusts to match regardless of how
   the target was constructed.

2. **Lowering `multi_knee_slope_threshold` to 10** (from 15) flips
   Blue Eye Samurai MARGINAL→PASS (-1.03 dB). By triggering the
   multi-knee path for less steep rolloffs, more titles get the
   explicit 2-shelf chain which the fitter can match better than
   a single-knee cascade.

3. **Raising `multi_knee_q` to 0.9** (from 0.8) flips Super Mario
   Bros FAIL→MARGINAL (-0.55 dB). Tighter Q in the multi-knee shelves
   produces a steeper correction knee that better matches the
   catalogue's shape.

4. **The two winners are complementary** (different parameters) and
   combine cleanly: **s10+mQ0.9 gives 2 improvements, 0 degradations,
   avg Δ=-0.32 dB**. Several other titles improve substantially
   (Elio -2.02, KPop -1.91, Garfield -1.91) but remain FAIL.

**Combined config detail (s10+mQ0.9)**:
- Blue Eye Samurai: MARGINAL(1.95) → PASS(0.92) ✓
- Super Mario Bros: FAIL(3.22) → MARGINAL(2.66) ✓
- Elio: FAIL(6.38) → FAIL(4.37) — big improvement, still FAIL
- KPop Demon Hunters: FAIL(6.91) → FAIL(5.00)
- The Garfield Movie: FAIL(5.15) → FAIL(3.24)

**Recommendation**: Update MeasurementAdvisor defaults to
`multi_knee_slope_threshold=10.0` and `multi_knee_q=0.9`. These
are strictly non-regressing with meaningful improvements.

**New baseline after E19**: 10 PASS / 4 MARGINAL / 17 FAIL
(was 9/4/18).

**Kept**: configurable constructor in `MeasurementAdvisor`,
`test_library_sweep_e19` in sweep file.
CSV: `.pytest_cache/auto_beq_sweep_e19.csv`.

### E20 - Multi-knee improvements: 3-shelf cascade + gain cap

**Hypothesis**: High-gain catalogue entries (Splinter Cell 37-42 dB,
KPop 36 dB) fail because the 2-shelf chain with 30 dB cap can't
reach them. A 3-shelf cascade (splitting deficit across 5→10→20→peak)
and/or a higher gain cap should improve these titles.

**Method**: Extended `_measurement_chain()` to support 3-shelf mode
(adds inner shelf at shoulder/3 for the 10→5 Hz deficit). Swept 6
configs across 31 tests:

| Config | P/M/F | Imp | Deg | Avg Δ |
|---|---|---|---|---|
| baseline (2-shelf, 30 dB) | 10/4/17 | 0 | 0 | +0.00 |
| 3-shelf, 30 dB | 9/3/19 | 0 | 2 | +0.30 |
| **2-shelf, 35 dB** | **10/5/16** | **1** | **0** | **-0.27** |
| 3-shelf, 35 dB | 10/4/17 | 2 | 1 | -0.01 |
| 2-shelf, 40 dB | 11/4/16 | 2 | 1 | -0.43 |
| 3-shelf, 40 dB | 11/3/17 | 3 | 2 | -0.20 |

**Key findings**:

1. **3-shelf HURTS**: Blue Eye Samurai regresses PASS→FAIL in every
   3-shelf config (+2.34 dB). The 3rd shelf at 5-10 Hz produces a
   target shape the fitter can't match — the added degree of freedom
   in the target makes the greedy shelf+PEQ fitter less effective.
   Verdict: keep `max_shelves=2`.

2. **Raising gain cap to 35 is safe**: Flow flips FAIL→MARGINAL
   (-2.38 dB), 0 degradations. The higher cap lets the multi-knee
   chain reach deeper without cliff-scaling down the gains.

3. **Cap 40 is too aggressive**: gains Flow FAIL→PASS and one
   Splinter Cell FAIL→MARGINAL, but also regresses a different
   Splinter Cell episode MARGINAL→FAIL. Net +1, but not zero-risk.

**Recommendation**: raise `_MAX_TOTAL_CHAIN_GAIN_DB` from 30 to 35.
Zero degradations, one grade improvement.

**New baseline after E20**: 10 PASS / 5 MARGINAL / 16 FAIL
(was 10/4/17 after E19).

**Kept**: `max_shelves` constructor param (default stays 2),
`_MAX_TOTAL_CHAIN_GAIN_DB` raised to 35.
`test_library_sweep_e20` in sweep file.

---

### E21 - Self-feedback loop (iterative gain adjustment)

**Hypothesis**: The single-pass pipeline faithfully reproduces a wrong
target if the advisor's gain is off. An iterative loop that evaluates
corrected-curve flatness and adjusts gain should improve results.

**Method**: `propose_filters_with_feedback()` wraps
`propose_filters_from_measured()` in a loop. After each pass, it
computes `corrected = measured + chain_response` and checks flatness
in-band. If residual deficit or overshoot exceeds the threshold, it
adjusts gain by half the error via `GainAdjustedAdvisor` and re-runs.

**Configs tested** (155 tests, 5 configs × 31 films):

| Config | P/M/F | Imp | Deg | Avg Δ |
|---|---|---|---|---|
| single-pass (baseline) | 10/5/16 | 0 | 0 | +0.00 |
| iter=3, thr=3 | 8/3/20 | 3 | 8 | +0.94 |
| iter=3, thr=2 | 8/3/20 | 3 | 8 | +0.94 |
| iter=3, thr=1 | 8/3/20 | 3 | 8 | +0.94 |
| iter=5, thr=2 | 8/3/20 | 3 | 8 | +0.94 |

**Key findings**:

1. **All feedback configs produce identical results** regardless of
   threshold (1/2/3 dB) or iteration count (3/5). This means the
   loop converges in exactly one adjustment — the first iteration's
   gain offset is the same regardless of threshold, and subsequent
   iterations don't improve further.

2. **The feedback loop is net negative**: 3 improvements but 8
   degradations. It improves under-corrected titles (Garfield, two
   Scavengers Reign episodes) by boosting gain, but ALSO boosts
   gain on already-correct titles, pushing them into over-correction
   (Pantheon +4.59 dB, Flow +5.45 dB, X-Men +3.92 dB).

3. **Root cause**: the flatness metric evaluates "how flat is the
   corrected curve" — but the BEQ goal is NOT a flat curve. It's to
   match the catalogue's intended correction, which often leaves
   deliberate rolloff below 10 Hz. The feedback loop sees residual
   rolloff at 5-10 Hz as "under-correction" and boosts gain, but
   that rolloff was CORRECT. The metric is fundamentally wrong for
   BEQ.

4. **The existing single-pass pipeline is better because** the
   advisor's gain estimate (deficit at 10 Hz) is inherently
   conservative — it doesn't try to flatten below 10 Hz. The
   feedback loop breaks this conservatism.

**Verdict**: E21 feedback loop is **not adopted**. The flatness
metric needs to evaluate against the INTENDED correction shape
(which we don't have at inference time), not against "perfectly
flat." Without ground truth, the loop has no way to know when to
stop adding gain.

**Possible future direction**: if we ever have a "correction shape
template" (e.g., typical BEQ rolloff profile learned from catalogue
entries), the feedback loop could evaluate residual against that
template instead of flat. But that's essentially supervised learning,
which is a different approach.

**Kept**: `GainAdjustedAdvisor` and `propose_filters_with_feedback()`
remain in the codebase for future experimentation.
`test_library_sweep_e21` in sweep file.

---

## Full experiment history (E1–E21)

### Progression table

| # | Name | What changed | Adopted? | Cumulative baseline |
|---|------|-------------|----------|---------------------|
| **E1** | Single shelf + residual PEQs | Baseline: flatten measured curve with shelf+PEQ fitter | No — wrong objective (BEQ extends, doesn't flatten) | 0/0/3 (3 titles) |
| **E2** | N-filter iterative fitter | Greedy shelf+PEQ fitter, max 6 filters, Q 0.3–4.0 | **Yes** — fitter math proven | — (synthetic only) |
| **E3** | Rolloff-depth classifier | Classify mild/middle/cliff, cap gain per class | No — overfit to 3 fixtures | 1/1/1 |
| **E4** | Advisor interface + MockAdvisor | Advisor supplies max_gain+knee; single LowShelf Q=0.7 target | **Yes** (architecture) | 1/0/2 |
| **E5** | Q=0.9 correction target | Raise shelf Q 0.7→0.9 | No — helps EoT, hurts MM | 1/1/1 |
| **E6** | Cascaded shelf target | N shelves at Q=0.9, ~7 dB each (matches catalogue construction) | **Yes** | 2/0/1 |
| **E7** | Ollama llama3.1:8b | Use local LLM for gain+knee recommendation | No — numeric calibration poor | 0/0/3 |
| **E8** | Few-shot prompt | Add 4 labelled examples to LLM prompt | No — anchored LLM to low numbers | 1/0/2 |
| **E9** | Aesthetic-over-measurement prompt | Prompt: "BEQ is aesthetic, not measurement-derived" + film list | Partial — cheating with film names | 2/0/1 |
| **E10** | Advice.filters field | Advisor can prescribe explicit multi-knee filter chains | **Yes** (plumbing) | 2/0/1 (Mock) |
| **E11** | Multi-step Ollama + looks_multi_knee() | 3-call LLM (tier→numbers→chain) + programmatic cliff detection | No — prompts overfit to fixtures | 2/1/0 (overfit) |
| **E12** | Self-feedback refinement loop | LLM evaluates chain response, adjusts in loop (max 3×) | No — can't rescue wrong tier | 1/0/2 |
| **E13** | De-overfit prompts | Strip film names, use general principles only | **Yes** (honest baseline) | 1/1/1 |
| **E14** | Pure-measurement advisor | deficit + slope extension formula, no LLM | No — 3 distinct failure modes | 0/0/3 |
| **E15** | Absolute dBFS diagnostic | Log un-normalised levels before anchor normalisation | **Yes** (diagnostic) | — |
| **E15c** | EoT codec mismatch | Discovered test used wrong catalogue entry (Atmos vs DTS-HD) | Bug fix | — |
| **E16** | Catalogue-first pipeline | propose_or_lookup: catalogue match primary, auto fallback | **Yes** (production) | — |
| **E17a** | Knee = rolloff-start | Use 3 dB-below-peak frequency instead of shoulder peak | **Yes** | 1/3/0 (4 titles) |
| **E17b** | Rolloff threshold sweep | Test 3/4/6 dB thresholds | Kept 3 dB default | — |
| **E17c** | Topology classification | gentle/moderate/cliff classes with per-class gain formula | **Yes** | 10/2/22 (34 eps) |
| **E17d** | Expert formula (gain=peak−L10) | Simplify to deficit-only, raise chain cap 18→30 dB | **Yes** (baseline) | **10/4/20** (34 eps) |
| | | *— library sweep expanded to 31 titles —* | | |
| **E18** | Chunked-percentile extraction | STFT peak per chunk → P90, chunk sizes 30/60/90s | Partial | mixed |
| **E18b** | Blended extraction strategy | 70% Welch + 30% chunked P90 @ 60s (6 strategies tested) | **Yes** (default) | 9/4/18 → +2/0 grade changes |
| **E19** | Advisor constant calibration | slope threshold 15→10, multi-knee Q 0.8→0.9 (18 configs) | **Yes** | **10/4/17** |
| **E20** | Multi-knee gain cap | Cap 30→35 dB; 3-shelf tested but rejected (6 configs) | **Cap 35 yes** | **10/5/16** |
| **E21** | Self-feedback loop | Iterative gain adjustment via corrected-curve flatness | **No** (+3/−8) | unchanged |
| **E22** | NN + chunked audio | XGBoost validated with blended extraction (4 strategies) | Partial — blend-a0.3 best (-0.88 dB) | NN: 0P/1M/13F |

### Phases

1. **E1–E3** (procedural heuristics): failed to generalise beyond fixtures
2. **E4–E6** (advisor abstraction + cascade targets): shelf decomposition works
3. **E7–E12** (LLM-based tier classification): multi-step Ollama can work but prompts overfit
4. **E13–E15c** (pure-measurement baseline + diagnostics): 41% ceiling; found EoT codec bug
5. **E16–E17d** (catalogue-first + formula refinement): topology classification unlocked gains
6. **E18–E20** (spectrum extraction + calibration): blended extraction + recalibrated params
7. **E21** (feedback loop): rejected — metric fundamentally wrong for BEQ
8. **E22** (NN + chunked extraction): blended features close 32% of real-vs-synthetic gap

### Current baseline

**10 PASS / 5 MARGINAL / 16 FAIL** across 31 test cases (48% non-FAIL).

Key architectural components:
- `ExtractionStrategy` enum + `load_measured()` dispatcher (default: blend-a0.7-P90)
- `MeasurementAdvisor` with configurable constants (slope threshold, Q, gain cap)
- `AdvisorConfig` dataclass for parameterised sweeps
- `GainAdjustedAdvisor` wrapper for future feedback experiments
- Unified test infrastructure: `test_library_sweep_e19`, `_e20`, `_e21`,
  `_strategies`, `_chunked` — all share `_SWEEP_FILMS` and grading thresholds
- CSV reports per experiment + unified `scripts/sweep_report.py` generator

Remaining bottleneck: the advisor's deficit formula (`peak - L10`) is a
good first approximation but can't capture catalogue entries with
aesthetic choices divorced from the measured curve. Supervised learning
or catalogue-pattern templates are the next frontier.

### E22 — NN training with chunked audio features

**Hypothesis**: The XGBoost model (E18 NN) validates on real audio using
Welch-only feature extraction. E18b showed that blended/chunked
extraction captures transient bass events that Welch dilutes. If the
NN receives more accurate input features at validation time, downstream
loss should decrease — the real-vs-synthetic gap should narrow.

**Method**: Same XGBoost pipeline as E18 real-audio validation. Train
once on full catalogue (synthetic features). Build **four** separate
validation feature sets from the same WAV files, each using a different
extraction strategy:
- `welch` — baseline (identical to E18)
- `blend-a0.7-P90` — conservative blend (E18b winner)
- `blend-a0.3-P90` — aggressive blend
- `chunked-P90-60s` — pure chunked percentile

Compare downstream loss, verdict counts, and grade changes vs the
Welch baseline across all strategies.

**Key question**: Does the real-vs-synthetic performance gap shrink
when we use chunked/blended extraction? If so, the gap was partly
caused by Welch averaging out transient bass events that the synthetic
features (perfect inverse of catalogue filters) always capture.

**Implementation**: `test_nn_chunked_strategy_comparison()` in
`test_auto_beq_nn_chunked.py`. Uses `load_measured()` from
`_auto_beq_helpers.py` to dispatch to the appropriate extraction
function based on strategy.

**Results (14 titles, XGBoost trained on 8219 synthetic entries)**:

| Strategy | Mean loss | PASS | MARGINAL | FAIL | Δ vs Welch |
|---|---|---|---|---|---|
| welch (baseline) | 7.09 dB | 0 | 1 | 13 | — |
| blend-a0.7-P90 | 6.26 dB | 0 | 1 | 13 | -0.83 dB |
| **blend-a0.3-P90** | **6.21 dB** | 0 | 1 | 13 | **-0.88 dB** |
| chunked-P90 | 7.17 dB | 0 | 1 | 13 | +0.08 dB |

Synthetic-vs-real gap analysis:
- Synthetic mean loss: 4.33 dB
- Welch real-audio gap: 2.76 dB (7.09 - 4.33)
- Best (blend-a0.3) gap: 1.88 dB (6.21 - 4.33)
- **Blended extraction closes 32% of the real-vs-synthetic gap**

**Key findings**:
1. **Blended extraction helps the NN** — both blend strategies reduce
   mean downstream loss by ~0.85 dB. The improvement comes from more
   accurate input features at validation time.
2. **Pure chunked is neutral** — too aggressive for NN inputs, same
   as E18b found for MeasurementAdvisor. The blending preserves
   Welch's calibrated baseline while adding chunked's transient
   sensitivity.
3. **No verdict flips** — the 0.88 dB improvement isn't enough to
   cross grading thresholds. The NN's overall accuracy (7+ dB mean
   loss) is the bottleneck, not the extraction method.
4. **blend-a0.3 slightly beats blend-a0.7** for NN (opposite of E18b
   where a0.7 was safer for MeasurementAdvisor). The NN can tolerate
   more chunked signal because it learned feature patterns, whereas
   the MeasurementAdvisor's fixed formula overshoots.
5. **The real-vs-synthetic gap (2.76 dB) confirms E18's finding**: the
   "perfect inverse" assumption costs ~3 dB. Blended extraction
   recovers about a third of that.

**Lesson**: blended extraction is a free ~0.9 dB improvement for NN
validation. But the NN's overall accuracy needs improvement before
extraction method becomes the limiting factor. Next: improve the model
(more features, better architecture) rather than tuning extraction.

---

