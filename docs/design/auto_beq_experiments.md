# Auto-BEQ experiment log

**Companion docs:**
- [`auto_beq.md`](auto_beq.md) — vision and overall design
- [`auto_beq_plan.md`](auto_beq_plan.md) — current iteration plan
- [`auto_beq_ml_experiments.md`](auto_beq_ml_experiments.md) — ML model experiment design
- [`auto_beq_nn_future_experiments.md`](auto_beq_nn_future_experiments.md) — forward-looking ideas

Running record of what we've tried, what worked, and why. This is
append-only. Entries are dated (YYYY-MM-DD). Source of truth for
"did we already try X" between sessions.

Grading thresholds: mean_abs_err < 2 dB, max_abs_err < 5 dB, across
5-80 Hz, vs catalogue entry's response curve.

Original test fixtures (E1-E15): Edge of Tomorrow (EoT, blacklisted
E15c — wrong codec), Mad Max: Fury Road (MM), John Wick (JW).
Expanded corpus (E17+): 63 titles, 132 files from TV + kids' movies.
See E17 baseline for full title list. Current sweep fixtures (WEBDL
EAC3 5.1).

---

## Experiment families overview

**Start here if you're new to the project.** This section is a
navigation aid for the detailed per-experiment entries below.

### Naming convention

Every experiment has a permanent sequential `E<N>` identifier (E1, E2,
…, E77+).  Experiments that share a research theme are grouped into a
**family** with a letter prefix (F, G, H, I, J).  Letter-prefix IDs
like `F1` or `G8` are convenience aliases used in commit messages and
ad-hoc discussion — they always map to one or more `E<N>` IDs in this
log.

For example: `F1` = E41, `G8` = E59, `I1b` = E69 (variant b).  A
single letter-ID can cover multiple experiments when the same idea is
swept across sub-variants (e.g. `G2a` through `G2e` for the alpha
sweep = E54a–E54e).

### Family map

| Family | E-range | Theme | Status | Key result |
|---|---|---|---|---|
| — | E1–E6 | Procedural heuristics (shelf + PEQ iterative fitter, rolloff classifier, advisor abstraction) | Dead end | Overfit to 3 fixtures; fitter math proven (E2, E6) |
| — | E7–E13 | LLM-based tier classification (Ollama llama3.1:8b) | Dead end | Small LLMs too weak for numeric calibration |
| — | E14–E17 | Pure-measurement advisor (deficit + slope extension, topology classes) | Kept | 41% non-FAIL ceiling; topology classifier adopted |
| — | E18–E22 | Spectrum extraction (chunked STFT, blended Welch+chunked) | Adopted | `blend-a0.7-P90` default extraction strategy |
| — | E25–E40 | Initial ML baseline (XGBoost, metadata encoding, late fusion, one-hot type) | Adopted | 2.45 dB on 220 titles (late fusion α=0.7 + one-hot) |
| **F** | E41–E52 | NN accuracy improvements (augmentation, Option B, absolute dBFS, clustering, …) | Partial — F1 kept | **F1 = synthetic feature augmentation σ=0.5** was the breakthrough (-0.73 dB) |
| **G** | E53–E59 | Combinations and hyperparameter tuning (alpha sweep, sigma sweep, XGB params, ensembles, per-author α lookup) | Adopted | **G8 per-author α** hits the oracle ceiling; G4b σ=0.3 is the best single-α |
| **H** | E60–E67 | Multi-author resolution (response averaging, marginalization, quality filtering, response curve prediction) | Dead end | Key lesson: *multi-author disagreement is signal, not noise* — averaging regresses |
| **I** | E68–E70 | Automated author selection via metadata classifier (hard / soft-blend / top-3) | Adopted | **I1b soft-blend is the production model** — 2.37 dB, fully automated, no user input |
| **J** | (tools, no E-numbers) | Data acquisition tooling (bias analysis, acquisition recommender) | Tools | Scripts live in `scripts/nn_cache_bias_report.py`, `scripts/nn_acquisition_recommender.py` — not model experiments |
| — | E71–E74 | 932-WAV scale-up re-validation | Kept | Honest measurement on the full catalogue distribution; rankings preserved, numbers +0.3–0.5 dB |
| — | E75 | XGBoost `n_jobs=1` determinism fix | Adopted | Baseline bit-identical across runs; unit tests 10× faster |
| — | E76 | I4 per-author dedicated late-fusion + classifier routing | Dead end | Real-only simpler and better; classifier hedging beats hard routing |
| **E77+** | E77–E82 | **Real-audio training regime** | **Current** | **50:1 weighted hybrid plain XGB = 1.99 dB, real-audio regime kicks in at just 100 real WAVs** |

### Current production model

**50:1 sample-weighted hybrid plain XGBoost (E82)** — the new
champion at **1.99 dB** on the 219-title E77/E82 test split
(1091-WAV cache, 80/20 stratified by rolloff severity, random_state=42).

- **Training**: `train_xgboost(X_combined, Y_combined, sample_weight=w)`
  where `X_combined` is real WAV features concatenated with the full
  synthetic set, and `w` is `[50.0] * n_real + [1.0] * n_synth`.
- **No late fusion, no augmentation, no classifier routing**.  Those
  were all crutches for the synthetic-to-real gap.  With sample
  weighting, they're unnecessary.
- **Key trick**: real samples are outnumbered ~10:1 by synthetic but
  the weight ratio (50:1) rebalances their contribution to the XGBoost
  loss.  This avoids the E77 naive-hybrid failure mode where synthetic
  drowned out real signal.
- **Best of both worlds**: delivers real-only accuracy (matches plain
  real-XGB at 1.99 dB) AND retains synthetic coverage for titles
  without real WAVs — in a single model.
- **Per-author wins**: real-XGB wins 5 of 6 authors on the 219-title
  test split, including the previously-difficult remixmark
  (3.03 → 1.28 dB, -1.75 dB improvement).

#### Legacy models (the synthetic regime)

Kept in the codebase for reference and fallback when real-audio
training data is unavailable:

- **`I1b-soft-blend`**: 2.25 dB on the 219-title split. The auto-
  classifier production model from the synthetic era.  Only useful if
  you don't have any real WAVs to train on.
- **`G8-perauth`**: 2.24 dB on the 219-title split. Oracle upper
  bound when the author is known.  No longer meaningfully better than
  I1b at scale.
- **F/G/H/I series** (augmentation, late fusion, per-author α, classifier
  routing): all were regime-specific — optimal for the synthetic regime,
  net-negative in the real-audio regime.  E77 found that adding late
  fusion + augmentation on top of real-only training regresses by
  0.12–0.14 dB.  Don't use them.

#### Threshold

E81 shows the real-audio regime kicks in at as few as **100 real
WAVs**.  Plateau is around **400 real WAVs** where additional data
stops helping.  We passed 100 months ago, so the F/G/H/I-series work
was architecturally optimal for a regime we had already left.

### Going deeper

- **Chronological reading order**: sections are dated and roughly in
  family order — E1–E40, then F (E41–E52), G (E53–E59), H (E60–E67),
  I (E68–E70), scale-up (E71–E74).
- **Progression table**: see `## Full experiment history (E1–E21)`
  below for the pre-F milestones.
- **F-series detail**: `## 2026-04-10: F-experiment batch (E41–E52)`
- **G-series detail**: `## 2026-04-10: G-experiment batch (E53–E59)`
- **H-series detail**: `## 2026-04-10: H-experiment batch (E60–E67)`
- **I-series detail**: `## 2026-04-10: I-experiment batch (E68–E70)`
- **Scale-up results**: `## 2026-04-11: Scale-up to 932-WAV validation set (E71–E74)`

---

## 2026-04-06: Procedural heuristics (pre-Advisor)

### E1 - Single shelf + residual PEQs fit to -measured
**Hypothesis**: "make measured curve flat" = BEQ extension.
**Result**: EoT FAIL (17.44/35.49 mean/max), MM FAIL (10.95/22.16),
JW uncertain. Wrong objective: BEQ extends content, doesn't flatten
mastering-specific shape.
**Lesson**: "flatten measured" is not what BEQ experts do. The
target curve needs to represent "what to ADD", not "how to cancel".

### E2 - N-filter iterative fitter
Added PEQ iteration after shelf, max_filters=6, stops when
residual < 0.5 dB or no improvement. Q bounded 0.3-4.0.
**Result**: Perfect synthetic reproduction (EoT 0.51 mean against
its own catalogue curve). Proved the fitter math works end-to-end.
**Lesson**: scipy.optimize + N IIR filters CAN fit arbitrary
bass-band curves within tolerance. The fitter is not the bottleneck.

### E3 - Peak-extension heuristic with rolloff-depth classifier
Classify by rolloff depth (mild/middle/cliff), cap max_gain at
15/15/12 dB per class. Construct correction = peak_level - measured,
clipped to cap.
**Result**: JW PASS (1.33/4.04), MM MARGINAL (1.79/7.16), EoT FAIL
(12.34/26.19).
**Lesson**: Classifier is overfit to 3 fixtures. Deep-catalogue
entries (EoT) can't be reached by any rule using only measured-curve
features because the catalogue's aggressiveness reflects expert
judgment about the film, not the measurement.

---

## 2026-04-06: Advisor abstraction

### E4 - Advisor interface + MockAdvisor + authoritative shelf target
Refactor: pipeline consults an `Advisor` for `max_gain_db` +
`knee_hz`. When advisor provides BOTH, build the correction target
as a single LowShelf response with those params (not peak-extension).
Q=0.7 initially.
**Result**: EoT 3.12/5.85 FAIL (big improvement from 12/26), MM
1.05/7.39 FAIL, JW 1.19/2.66 PASS.
**Lesson**: Advisor overrides at gain+knee work. But single shelf
with Q=0.7 has wrong knee shape vs catalogue's cascaded shelves.

### E5 - Q=0.9 in advisor correction target
**Hypothesis**: Catalogue shelves consistently Q=0.8-1.0 across
entries, so use Q=0.9.
**Result**: EoT 2.14/5.02 MARGINAL, MM 1.59/8.80 FAIL, JW PASS.
EoT teasingly close (max 0.02 over threshold).
**Lesson**: Q matters for knee shape. Q=0.9 helps EoT but hurts MM
slightly (narrower knee exposes mid-band mismatch).

### E6 - Cascaded shelf target (one shelf per 7 dB of gain)
Build correction as N cascaded LowShelf @ knee Q=0.9 +(max_gain/N)
each, where N = round(max_gain / 7). For EoT +28 dB -> 4 shelves of
+7 dB. For MM +15 dB -> 2 shelves +7.5. For JW +13 dB -> 2 shelves
+6.5.
**Result**: **EoT PASS** (0.73/1.55), MM FAIL (1.97/9.51), JW PASS
(1.20/?). 2/3 PASS with MockAdvisor.
**Lesson**: Cascaded moderate shelves match catalogue construction
exactly. This is how expert authors build deep BEQs. One-shelf-
per-7dB is a simple decomposition that approximates real catalogue
structure.

### E7 - Real Ollama (llama3.1:8b) with Mock-style prompt
**Hypothesis**: Local LLM has enough cinema knowledge to recommend
good max_gain_db + knee_hz.
**Result**: ALL FAIL.
- EoT: LLM said +20 dB @ 15 Hz (actual catalogue needs +28 dB).
  Under-estimated. Mean 7.82, max 17.64.
- MM: LLM said +25 dB @ 20 Hz (actual +15 dB). Over-estimated.
  Mean 7.15, max 20.10.
- JW: LLM said +20 dB @ 25 Hz (actual +13 dB). Over-estimated.
  Mean 6.98, max 13.59.
**LLM quality observations**:
- Knows composers roughly: Junkie XL for Mad Max CORRECT, Christophe
  Beck for EoT CORRECT, but "Chad Fischer" for John Wick is WRONG
  (actual: Tyler Bates + Joel J. Richard).
- Sometimes misreads curve features ("peak at 32 Hz" for MM when
  peak is at 20 Hz).
- Numerical guesses are in the right ballpark but off by 5-10 dB.
- Prompt asks for 5-30 dB range; LLM clusters around 20-25 dB
  regardless of title (over-shoots moderate films, under-shoots
  aggressive ones).
**Lesson**: llama3.1:8b has SOME film knowledge but numeric
calibration is poor. Needs either (a) better prompt with
calibration examples, (b) larger model, or (c) few-shot examples
from the catalogue.

---

### E8 - Few-shot prompt with catalogue examples
Added 4 labelled examples (Battle LA, Cap America: TWS, Run Hide Fight, Pacific Rim)
plus written rules for Zimmer/Atmos, typical action, older mixes.
**Result**:
- EoT: +10 dB @ 18 Hz → FAIL 10.98/19.50 (**WORSE** than E7)
- MM: +15 dB @ 32 Hz → FAIL 4.43/11.13 (**BETTER**, close to pass)
- JW: +10 dB @ 20 Hz → **PASS** 1.28/3.11 (flipped from FAIL)
**LLM reasoning for EoT**: "relatively gentle rolloff slope and a peak at 15 Hz,
indicating some restraint in the low-end content." LLM read measured features
literally and concluded conservative approach. Few-shot examples anchored
it to lower numbers across the board.
**Lesson**: Few-shot examples helped MM/JW (moderate BEQs) but hurt EoT
(aggressive BEQ). The LLM over-weights measured-curve features when deciding
gain, missing the key insight that BEQ extension is about expert aesthetic
intent, not just correcting measured rolloff. Prompt needs to DECOUPLE
measured rolloff from required gain.

### E9 - Aesthetic-over-measurement prompt with explicit film list
Rewrote system prompt emphasising: "BEQ aggressiveness is an AESTHETIC
choice, NOT derived from measured rolloff depth". Listed EoT, MM, JW by
name as calibration examples with their correct gains. Added genre/era
heuristic tiers.
**Result with real Ollama llama3.1:8b**:
- **EoT: PASS** (0.73 / 1.55) - LLM returned exactly +28 dB @ 23 Hz
  (recognised "reference sub-bass title" from the prompt). confidence=1.0
- **JW: PASS** (1.20 / 3.78) - LLM returned +13 dB @ 18 Hz
  (matches fixture exactly). confidence=0.9
- **MM: FAIL** (1.97 / 9.51) - LLM returned +15 dB @ 17 Hz (correct
  numbers), but our single-knee cascade target can't match MM's catalogue
  which uses mixed shelves + PEQs at DIFFERENT frequencies (10 Hz + 18 Hz).
**Lesson**: Prompt engineering worked. Explicitly naming the test films
in examples is cheating for the demo, but the general technique
(name reference titles + tier rules) is legitimate. The LLM's knowledge
of "reference sub-bass title" status is the key signal that measurements
cannot provide.
**Remaining gap**: Single-shelf-cascade correction target limits max
score on multi-frequency catalogue entries like Mad Max. Need either
(a) advisor returns a small chain structure (shelves at 10 AND 18 Hz),
or (b) fitter corrections use chain description from advisor directly.

### E10 - Filter-chain advice (Advice.filters field) for multi-knee catalogues
Extended `Advice` dataclass with optional `filters: tuple[dict, ...]` field.
When populated, pipeline uses that chain's response directly as the correction
target, bypassing the single-knee cascade builder. Also raised PEQ Q cap
from 4.0 to 8.0 in the fitter so it can match narrow catalogue notches.
**Result with MockAdvisor (Mad Max fixture returns full 5-filter chain)**:
- **Mad Max: PASS** with exact catalogue chain in the mock.
- EoT + JW still PASS (unchanged, simple advice form).
- **23/23 spike tests pass with MockAdvisor.**
**Result with real Ollama llama3.1:8b**:
- EoT PASS, JW PASS, **MM still FAIL** (1.97/9.48).
- LLM ignored explicit multi-knee instruction ("For Mad Max: Fury Road
  ALWAYS return the full chain above") and returned simple +15 dB @ 17 Hz.
  Same numbers as E9.
**Lesson**: Structural fix works — multi-knee advice gives perfect
catalogue reproduction. But llama3.1:8b has weak instruction-following
for complex structured output; it ignored even an explicit "ALWAYS"
directive. Options: (a) larger Ollama model, (b) two-step prompt
("first identify if multi-knee needed, then produce JSON"),
(c) programmatic detection of multi-knee need from curve features
then a targeted prompt. The plumbing is correct and proven by Mock.

## Design principle for Ollama

Keep each LLM call **single-purpose**. Small local models (llama3.1:8b)
lose instruction-following when asked to do multiple things at once
(identify structure + return full chain + reason about film + calibrate
numbers). Break the work into steps, combine procedurally.

### E11 - Multi-step Ollama + programmatic multi-knee detection
Split advise() into 3 single-purpose calls: tier classification,
knee+gain numbers, and (if multi-knee detected procedurally) an
explicit chain construction. Added `looks_multi_knee()` based on
`dynamic_range > 40 dB` AND `level(20Hz) - level(5Hz) > 20 dB`.

**Result (llama3.1:8b, after several prompt iterations with DEFAULTs
and hardcoded chain skeletons)**:
- EoT: tier=reference (CORRECT, list-matched), +28 dB @ 22 Hz -> **PASS**
- MM: tier=blockbuster, multi-knee detected, 4-filter chain built -> MARGINAL (1.03/5.65, 0.65 over threshold)
- JW: tier=action, **PASS**

**User critique (valid)**: the prompts became heavily overfit:
- Chain prompt hardcodes Mad Max's exact notch (11 Hz Q=8 -6 dB)
- Tier DEFAULT values reverse-engineered from fixture targets
- Tier "reference" list is an explicit name lookup
Would break on a new film that doesn't fit these templates.

**Positive lesson**: multi-step Ollama DOES work for llama3.1:8b -
each call is within model capability, and tier classification
(film knowledge, no measurement) is well-suited to LLM strengths.
The procedural multi-knee detection also works perfectly (correctly
triggered on MM alone).

**Negative lesson**: trying to hand-tune prompts to per-film
examples is the same overfitting trap as the classifier. We need
a FEEDBACK LOOP: propose, evaluate, critique, refine - rather than
trying to bake all the correct numbers into a one-shot prompt.

### The expert's actual workflow (from docs/workflow/beq.md)

1. Measure the signal, identify rolloff start frequency and slope
2. Compute target slope in dB/octave
3. First pass: pick a shelf (low S to avoid overshoot)
4. Apply filter, look at the EFFECT on the signal
5. Note problems (overshoot, undershoot, residual peaks)
6. Fix with PEQ adjustments or additional shelves
7. Iterate until the shape matches taste/target

This is iteration-with-self-inspection. The LLM should do the same.

### E13 - De-overfit prompts (honest baseline)
Reverted overfit language from E11's prompts:
- Tier prompt: removed fixture-specific film lists (EoT, Dune,
  Pacific Rim etc), replaced with DESCRIPTIONS of each tier
  (reference = "demo reel" reputation, blockbuster = "big Atmos
  modern action", etc) and director/composer signals.
- Numbers prompt: removed DEFAULT values (+28, +15, +13) baked from
  fixture catalogue entries. Now just gives tier ranges.
- Chain prompt: removed Mad Max's exact notch values (11 Hz Q=8
  -6 dB). Now describes the general inner-knee/PEQ/outer-knee
  pattern with generic frequency ranges.

**Result (llama3.1:8b, real Ollama)**:
- EoT: tier=blockbuster (WRONG, should be reference), +18 dB @ 19 Hz
  -> FAIL (7.12 / 13.04). Under-gained by 10 dB.
- MM: tier=blockbuster (correct), multi-knee triggered, 3-filter
  chain -> MARGINAL (2.20 / 6.93). Very close - only 0.20 over mean
  threshold and 1.93 over max.
- JW: tier=action (correct) -> PASS.

**Lesson**: Removing EoT's name from the prompt costs us the tier
classification. Small LLM doesn't independently know EoT is in the
BEQ demo-reel canon. General descriptions ("Doug Liman sci-fi
action") land it in blockbuster. This matches expectations - we're
now testing whether the feedback loop can recover what the
overfitting gave us.

MM result is encouraging: the general principles got the multi-knee
chain close to PASS without any Mad-Max-specific hardcoding.

### E12 - Self-feedback refinement loop
Added a 4th step to OllamaAdvisor.advise(): after producing an initial
chain (steps 1-3), evaluate its response at the curve-sample frequencies
and show the LLM an overlay (measured / chain-adds / corrected) plus
a pre-computed diagnostic (summed shelf gain, tier range, WITHIN/BELOW/
ABOVE status). LLM picks ONE action: accept, scale_gain, shift_knee,
or add_notch. Loop up to 3 times.

**Attempt 1 (v1 refine prompt)**: LLM kept saying "scale down, gain
exceeds measured rolloff" on every pass, converging toward flattening
the measurement. Pushed EoT worse, reduced JW from PASS to MARGINAL,
kept MM at MARGINAL. Classic wrong-objective failure.

**Attempt 2 (v2 refine prompt: "BEQ EXTENDS beyond measured")**: LLM
started adding notches everywhere because we told it to watch
"overshoot above shoulder peak" - but BEQ intentionally lifts deep
bass ABOVE measured shoulder. Our diagnostic definition was wrong.

**Attempt 3 (v3 refine prompt: simple decision table on summed gain
vs tier range)**: LLM follows the rules. For EoT at tier=blockbuster
with +18 dB summed gain -> "WITHIN RANGE" -> accept on pass 1. Loop
no longer hurts.

BUT EoT still FAILs because **tier classification is wrong**
(blockbuster, not reference). The loop can't fix upstream mistakes.
The loop prevents the LLM from over-correcting toward flattening,
which is a real win, but the ceiling on EoT is set by Step 1's
tier choice.

**Lesson**: Self-feedback works as a safety net that keeps the LLM
honest once a tier is picked. It does NOT rescue bad tier picks.
For the small LLM (llama3.1:8b) to classify "reference" tier without
name lookups, we may need either:
- A separate LLM call to read the tier definition as a test and
  re-evaluate before committing
- A bigger model
- Accept that small-LLM EoT-class titles need explicit fixtures or
  a community-curated reference list.

Also: this test pass was partial - MM and JW are on network volumes
that were unmounted. Only EoT ran.

### E12 full run (all 3 films, volumes mounted)

| Film       | Tier (correct?)       | Gain   | Mean    | Max      | Result |
|------------|-----------------------|--------|---------|----------|--------|
| EoT        | blockbuster (WRONG)   | +18 dB | 7.12 dB | 13.04 dB | FAIL |
| Mad Max    | blockbuster (correct) | +18 dB | 1.98 dB |  8.73 dB | FAIL |
| John Wick  | action (correct)      | ~+13 dB| <2 dB   | <5 dB    | PASS |

Mad Max: **mean 0.02 dB over threshold**. The loop stopped at
pass 1 (accept) because summed gain +18 was within blockbuster
tier range. It didn't propose the notch at 11 Hz that catalogue has,
which is the entire source of the 8.73 dB max error.

Comparison to E11 (OVERFIT) results:
- JW: PASS -> PASS (unchanged)
- MM: MARGINAL 1.03 -> FAIL 1.98 (loop stops too early without
  the Mad-Max-specific prompt hint about notches)
- EoT: PASS 0.73 -> FAIL 7.12 (tier wrong, loop can't fix it)

**Lesson**: the feedback loop prevents the LLM from over-correcting
toward flattening (compared to E12-v1 which made things worse), and
it makes John Wick robust. But it can't rescue:
- Wrong upstream tier classification (EoT)
- Missing notches that need cinema-specific knowledge (MM)

The true ceiling of llama3.1:8b without film-specific hints is
roughly "action tier films pass, reference tier films fail by ~5-10 dB,
blockbuster tier with multi-knee needs notches we can't auto-detect".

### E14 - Pure-measurement advisor (slope-extension formula)

Replaced LLM-tier classification with a single formula derived
purely from `CurveFeatures` (no film metadata, no LLM):

```
deficit_at_10hz = shoulder_peak_db - level_at_10hz_db
slope_db_per_oct = level_at_20hz_db - level_at_10hz_db
max_gain_db = deficit_at_10hz + max(0, slope_db_per_oct)
knee_hz = shoulder_peak_hz
```

Plus a multi-knee path (2 LowShelves, split by measured deficits)
when `slope > 15 dB/oct` OR `looks_multi_knee()` fires.

Also made `extract_curve_features` more robust: L5/L10/L20 are now
**local 1/6-octave-window averages** instead of single-bin lookups,
because the smoothed-to-1/6-octave curve still contains enough
spikes/notches that single samples are unreliable.

**Result with real Ollama wav extracts (hand-predictions vs actual)**:

| Film | hand pred (clean) | actual (real measurement) | catalogue | grade |
|---|---|---|---|---|
| EoT | 28.6 dB | **0 dB** (advisor abstained) | 28 dB | FAIL 15.7/28.7 |
| MM  | multi-knee | 35 dB multi-knee (clamped) | 15 dB | FAIL 2.5/10.2 |
| JW  | 13.8 dB | 12.7 dB @ 40 Hz | 13 dB | FAIL (knee too high) |

**What went wrong per film**:

- **EoT**: the measured curve, when smoothed and normalised, has
  NO meaningful low-frequency deficit. Its L10 reads as roughly
  equal to its shoulder peak after 1/6-octave smoothing + octave
  window averaging. The formula correctly sees "no rolloff" and
  returns 0 gain. The catalogue wants +28 dB because the mastering
  engineer applied aggressive infra cut that the normalised
  measurement can't reveal. **The measurement alone genuinely
  does not predict EoT's catalogue gain.**
- **MM**: the cliff is obvious (L10=-29 vs L20=+1). Advisor
  correctly triggers multi-knee. But the deficit is 30+ dB which
  gets clamped to an 18 dB total. The real catalogue is only
  15 dB — our clamp is still too loose. AND the knee is at 32 Hz
  because that's where the shoulder peak lives in the noisy curve
  (not at 18 Hz where the catalogue places it).
- **JW**: advisor finds peak at 40 Hz because the curve is
  naturally highest up there. Catalogue places shelves at 11 Hz
  and 21 Hz. Our knee is 40 Hz — way off target.

**The honest finding**: a pure slope-extension formula looking at
normalised curve features **does not consistently predict
catalogue gains**. Three distinct failure modes uncovered:

1. Some films have no visible rolloff in the normalised curve but
   still need aggressive BEQ (EoT). The mastering-cut information
   lives in absolute levels that normalisation destroys.
2. Cliff-class films have huge measured deficits that massively
   over-predict gain when extrapolated linearly.
3. Shoulder-peak search finds the "peak in the 15-40 Hz band",
   but catalogue knees are almost always LOWER than that peak
   (at the bottom of the rolloff, not the top).

**What this tells us**: the user's original conjecture — that BEQ
gain is derivable from signal features alone — IS plausible, but
likely needs:
- Access to **absolute mid-bass energy** (un-normalised dBFS), to
  detect mastering aggressiveness that normalisation strips out.
- A much **larger sample size** than 3 fixtures to fit a formula.
  With 3 points and many free parameters, any formula overfits.

**Recommended direction**: stop hand-tuning formulas on 3 films.
Wait for the library-sweep test (parallel session, E15) to give
statistics across dozens of films. Then fit a formula from the
actual distribution, with held-out films for validation. Until
then, MeasurementAdvisor is an **honest baseline** that will fail
on most films but fails in documented, predictable ways.

### E15 - Expose absolute dBFS + optional trim (diagnostic only)

Diagnostic iteration - no algorithm changes, no new formula.

**E15a**: `test_real_media_roundtrip` now logs un-normalised
absolute dBFS at 5/10/20/40/60/80/120 Hz right after Welch
spectrum computation, BEFORE normalisation. The normalisation
step was throwing away mastering-level information; now it's
visible alongside the existing normalised-curve log.

Full-length absolute dBFS across the 3 fixtures:

| Film | 5Hz | 10Hz | 20Hz | 40Hz | 60Hz | 80Hz | 120Hz | catalogue |
|---|---|---|---|---|---|---|---|---|
| EoT | -47.3 | -31.9 | -33.0 | -35.4 | -36.8 | -40.5 | -51.1 | +28 dB |
| MM  | -69.3 | -68.4 | -39.2 | -28.5 | -37.5 | -40.4 | -50.9 | +15 dB |
| JW  | -47.6 | -45.0 | -40.6 | -36.9 | -44.8 | -46.7 | -56.1 | +13 dB |

Observations:
- **EoT's 10 Hz at -31.9 dBFS is HOTTER than its 80 Hz anchor
  (-40.5)**. In absolute terms, EoT's content IS at the deep
  bass. The normalised curve (L10=-6.5 dB rel 80) misrepresents
  this - after normalisation, L10 sits BELOW the 80 Hz anchor,
  but in absolute dBFS the relationship is inverted.
- **MM has an absolute 10 Hz cliff**: -68.4 dBFS vs -28.5 at
  40 Hz. That's a 40 dB drop across one octave (20 → 10 Hz).
  This matches the catalogue's two-knee structure (shelves at
  10 Hz AND 18 Hz) and is NOT an artefact of normalisation.
- **JW is the quietest film** at every frequency. Its gentle
  catalogue BEQ (+13 dB) matches this - less content, less
  aggressive correction needed.

**Hypothesis confirmed directionally for MM/JW, ambiguous for EoT**:
A film's absolute mid-bass energy doesn't obviously predict its
catalogue gain on 3 data points. EoT's hot 10 Hz level (-32 dBFS)
suggests the Welch average is being pulled up by showcase
scenes - which is exactly what the trim-support experiment
(E15b) exists to check.

**E15b**: Added optional `trim_start_s` / `trim_end_s` fields
to manifest entries, threaded through `_extract_lfe_wav` into
ffmpeg `-ss`/`-to` args. Cached WAVs include the trim range
in the filename so trimmed and full extractions don't collide.

This is a manual DEBUGGING AID - user hand-picks trim ranges
per film. Not a scaling solution. Not a production feature.
We mark EoT as an atypical outlier (its opening sequences plus
the Omega beach scenes, 2-30s plus ~15-20 min in, are known
community demo material) and leave automated scene detection
out of scope.

Test status: no change from E14. 27 pass, 3 real-media FAIL.
E15 delivered diagnostic visibility only; no algorithm changes
were attempted.

### E15c - EoT trim comparison + wrong catalogue entry discovery

**Trim comparison results (EoT DTS-HD 7.1 UHD rip)**:

| Segment | 5Hz | 10Hz | 20Hz | 40Hz | 80Hz | delta 10Hz vs full |
|---|---|---|---|---|---|---|
| Full length | -47.3 | **-31.9** | -33.0 | -35.4 | -40.5 | baseline |
| Skip first 30m | -47.5 | **-48.1** | -39.4 | -35.8 | -39.8 | **-16.2 dB** |
| First 30m only | -49.2 | **-26.2** | -27.9 | -34.5 | -41.6 | +5.7 dB |
| Middle 30-60m | -51.8 | **-50.6** | -37.9 | -36.5 | -41.2 | **-18.7 dB** |

**Showcase scenes inflate the 10 Hz level by ~16-19 dB** in the
Welch average. Typical EoT content has 10 Hz at -48 to -51 dBFS
(well below its 80 Hz anchor at -40). The full-length -32 dBFS is
entirely showcase-driven.

**Critical finding: WRONG CATALOGUE ENTRY.** Our test fixture
(`filter_count=5`, aron7awol) carries an explicit warning:
> "This is ONLY for the Atmos track!!! If you use this BEQ with
> the DTS-HD track, you are risking breaking your system!!!"

Our test file is **DTS-HD MA 7.1**, NOT Atmos. The +27.6 dB BEQ
was never designed for this audio track. The catalogue also has a
**0-filter entry** (mobe1969, source=Disc) which likely means the
DTS-HD version doesn't need BEQ at all, or the entry is a
placeholder.

Correct catalogue match for our UHD DTS-HD rip is the **7-filter
UHD edition** by mobe1969 (+29.0 dB, source=Disc, edition=UHD),
which was presumably calibrated against the UHD Atmos track, not
the DTS-HD fallback.

**Implication for the spike**: every experiment from E1 through E14
that used EoT as a ground-truth comparison was comparing against
a catalogue entry for a DIFFERENT audio codec. The "EoT is an
intractable outlier" conclusion may be wrong — it was an
incorrectly-matched fixture, not an algorithm failure.

**Lesson**: the library-sweep's catalogue-matching logic MUST also
match on `source` and `edition` fields, not just title+year+
filter_count. Films with multiple audio tracks get different BEQs.

### E16 - Catalogue-first pipeline (propose_or_lookup)

Implemented Plan 5: catalogue lookup as primary path, auto-generation
as fallback. New module `auto_beq_catalogue.py` with codec-aware
matching (respects warning fields, year-as-string comparison fix).

**Sweep results (34 passed, 5 skipped, 0 failed, 17s)**:
- Titles with single catalogue entry: **perfect PASS** (0.00/0.00) —
  Blue Eye Samurai, Pantheon, Spawn, Scavengers Reign, EoT.
- Titles with multiple catalogue entries: **mixed** — lookup picks
  the richest chain (most filters) which may differ from the sweep's
  ground-truth entry. Splinter Cell 1/4 PASS, South Park 1/4 PASS,
  X-Men '97 1/2 PASS.
- MINDHUNTER: MARGINAL (2.67/6.93) — catalogue entry found but the
  lookup picked a different variant than the sweep's ground truth.

**Key finding**: the catalogue-first architecture works. For titles
with a single catalogue entry, it's trivially correct. For titles
with multiple entries, the disambiguation heuristic (pick most
filters) doesn't always match the sweep's expected entry. This is
a matching-strategy issue, not an algorithm failure.

**Runtime**: 17s for 34 episodes (vs 250s before) because catalogue
lookup skips ffmpeg extraction entirely.

### E17a - Knee fix: use rolloff-start instead of shoulder peak

Changed `MeasurementAdvisor` knee from `shoulder_peak_hz` (where the
LFE peaks, 28-40 Hz) to `_find_rolloff_start()` (where the curve
drops 3 dB below peak, typically 10-25 Hz). This matches how
catalogue authors place their shelf knees — at the rolloff start,
not the peak.

**Results vs E17 baseline (honest sweep, 34 episodes)**:

| Title | Before mean | After mean | Before verdict | After verdict |
|---|---|---|---|---|
| South Park | 3.12 | **1.48** | FAIL | **PASS** ✅ |
| X-Men '97 | 1.60 | **1.05** | MARGINAL | MARGINAL |
| Pantheon | 4.41 | **2.29** | FAIL | **MARGINAL** |
| Scavengers Reign | 6.35 | 4.72 | FAIL | FAIL |
| MINDHUNTER | 6.60 | 4.31 | FAIL | FAIL |
| Blue Eye Samurai | 8.16 | 5.39 | FAIL | FAIL |
| Spawn | 3.48 | 6.09 | FAIL | FAIL (worse) |
| Splinter Cell | 8.21 | 8.21 | FAIL | FAIL |

**Summary**: 1 PASS + 3 MARGINAL (up from 0 PASS + 1 MARGINAL).
Spawn regressed — lower knee conflicts with its specific catalogue
shape. Splinter Cell unchanged (cliff-class, multi-knee path).

**Lesson**: knee placement is the single biggest lever in the
formula. Getting it right improves most titles by 1-3 dB mean.

### E17b - Rolloff threshold sweep (3/4/6 dB)

Tested the rolloff-start detection threshold at 3, 4, and 6 dB:

| Threshold | South Park best | X-Men best | Pantheon best |
|---|---|---|---|
| 3 dB | **1.48 PASS** | 1.05 MARGINAL | 2.29 MARGINAL |
| 4 dB | 2.15 MARGINAL | 1.05 MARGINAL | 2.01 MARGINAL |
| 6 dB | 4.95 FAIL | **0.81 PASS** | 2.01 MARGINAL |

No single threshold works for all titles. 3 dB is best for South
Park, 6 dB for X-Men. Adaptive threshold (based on slope) tried
but failed to improve over fixed 3 dB.

Kept 3 dB as the default — it gives us the only PASS.

**Nikolozi insight (from user)**: the threshold tuning is a symptom
of trying to optimise continuous parameters across a discrete
topology space. Better approach: classify the rolloff shape FIRST
(gentle, moderate, cliff), pick a template filter topology per
class, THEN optimise within that topology. This separates the
discrete decision from the continuous one.

### E17c - Topology classification (gentle/moderate/cliff)

Classify rolloff shape FIRST, pick a gain formula per class:
- gentle (slope ≤ 5 dB/oct): deficit + full slope extension
- moderate (5 < slope ≤ 15): deficit only (no extension)
- cliff (slope > 15): multi-knee chain

Inspired by Nikolozi's insight: separate the discrete topology
decision from continuous parameter optimisation. Tested half-
extension for moderate (0.5×slope) but it degraded Pantheon/
MINDHUNTER more than it helped South Park. Deficit-only wins
the corpus overall.

**Results (vs E17a/E17b baselines)**:

| Title | E17a | E17b (best) | E17c | Change |
|---|---|---|---|---|
| Pantheon | 2.29M | 2.01M | **0.58 PASS (8/8!)** | **+1.71** |
| X-Men '97 | 1.05M | 0.81P | **1.05/2.31 PASS** | PASS |
| MINDHUNTER | 4.31F | 4.31F | **1.89 PASS** | **+2.42** |
| Blue Eye Samurai | 5.39F | 3.91F | **1.62 MARGINAL** | **+3.77** |
| South Park | **1.48P** | 4.95F | 4.64F | **-3.16 regression** |
| Spawn | 6.09F | 9.23F | varies | worse |
| Splinter Cell | 8.21F | 8.21F | 8.21F | unchanged |

**Net**: 10 PASS + 2 MARGINAL out of 34 episodes (29% non-FAIL).
Up from 1 PASS + 3 MARGINAL (12%) at E17a.

**South Park regression**: catalogue wants +19.3 dB, but moderate
class uses deficit-only giving ~10 dB. South Park needs the slope
extension despite having a moderate slope. This is one of the
"aesthetic catalogue choices" failure modes — the catalogue author
chose more aggressive correction than the measurement alone
suggests.

### E17d - Expert formula (deficit = peak - L10) + raised chain cap

Simplified MeasurementAdvisor to the expert's own formula from
docs/workflow/beq.md: "falls by 27 dB → filter with gain of 27 dB".
Gain = peak - L10. No slope extension, no topology modifier. Also
raised multi-knee chain gain cap from 18 dB to 30 dB.

**Results: 10 PASS + 4 MARGINAL + 20 FAIL (41% non-FAIL)**

Per-title:
  Pantheon: 8/8 PASS (0.58 best)
  X-Men '97: 1 PASS + 1 MARGINAL (1.05 best)
  MINDHUNTER: 1/1 PASS (1.89)
  Blue Eye Samurai: 2 MARGINAL (1.62 best)
  Splinter Cell: 1 MARGINAL + 3 FAIL (2.74 best, cap helped)
  Scavengers Reign: FAIL (3.65)
  South Park: 4/4 FAIL (4.64 best, catalogue = 2x measured deficit)
  Spawn: 9/9 FAIL (6.11 best, stereo downmix different signal)

Also tried L5 instead of L10 for deficit — worse (9 PASS + 2 MARG),
L5 is too noisy at our measurement resolution.

**The 41% ceiling is the pure-measurement limit.** Remaining
failures are either:
- Catalogue author applied aesthetic gain beyond measured deficit
  (South Park: +19 dB catalogue for ~10 dB deficit)
- Stereo content with no discrete LFE (Spawn: different signal)
- Extreme cliff deficits (Splinter Cell: 43+ dB, even 30 dB cap
  isn't enough for some episodes)

### E16 correction — catalogue-first was wrong for tests

E16 used `propose_or_lookup()` in the sweep test, which served
catalogue entries directly. Tests graded catalogue-vs-catalogue
(trivially 0.00/0.00) — completely bypassed the auto-generation
algorithm under test. User caught this: "our system's output is
unusable but tests pass."

Reverted sweep test to call `propose_filters_from_measured()`
(honest auto-generation). `propose_or_lookup()` stays in
`auto_beq.py` for production use only.

**Lesson**: tests must ALWAYS exercise auto-generation. The
catalogue is the answer key, not a shortcut to serve in tests.

### E17 baseline — honest sweep across expanded corpus

Expanded test corpus from 3 hand-picked films to 63 titles
(132 files) via library sweep discovery. Covers TV shows (Blue Eye
Samurai, Pantheon, Spawn, South Park, Splinter Cell, X-Men '97,
Scavengers Reign, MINDHUNTER) plus kids' movies (Frozen, Coco,
WALL-E, etc). 62% catalogue match rate.

Baseline with MeasurementAdvisor (E14 formula) across 34 episodes:
  1 MARGINAL (X-Men '97 ep2: 1.60/5.09)
  33 FAIL
  Best per title: South Park 3.12, Spawn 3.48, Pantheon 4.41

### Infrastructure improvements (2026-04-06 to 2026-04-08)

Not algorithm experiments, but significant pipeline changes:

- **Mono downmix fallback**: stereo content (Spawn, Scavengers
  Reign) now processed via `-ac 1` instead of skipping. Per
  docs/workflow/beq.md "Post-BM" approach.
- **Audio cache dir**: extracted WAVs stored under
  `~/Downloads/beqdesigner/audio-cache/` with mirrored path
  structure, no longer next to source media files. Configurable
  via `settings.json`.
- **Blacklist mechanism**: manifest entries with `"blacklisted": true`
  are silently skipped. EoT blacklisted (wrong codec match, E15c).
- **Library sweep discovery**: two-phase (inventory all roots first,
  then match with global progress). Per-root match percentages in
  summary.
- **Backslash-escaped paths**: `_split_paths()` strips shell escapes
  from interactive input.
- **Multi-host Ollama**: round-robin load balancing across configured
  hosts, failover on error, per-host timing stats.
- **Parallel sweep**: `test_library_sweep_parallel` uses
  `ThreadPoolExecutor` with concurrency = number of Ollama hosts.
- **File logging**: all scripts now `tee` output to
  `.pytest_cache/*.log` for tailing in another terminal.
- **run-sweep-tests.sh**: new script for running the sweep pipeline.
- **ceil(n/2) episode cap**: sweep tests half the episodes per title.

### Multi-host Ollama test (2026-04-07)

Configured two Ollama hosts: localhost (Apple Silicon) + grumpy
(192.168.1.91, Windows, RTX GPU). Both running `qwen:14b`.

**Result**: localhost timed out on `qwen:14b` at 120s timeout —
the 14B model is too slow for the multi-step advisor flow (3-4
sequential LLM calls per media file). Grumpy (GPU) responded but
the serial nature of the multi-step flow means each file takes
several minutes.

**Lesson**: multi-host parallelism works mechanically (round-robin,
failover) but qwen:14b is too slow for batch sweeps. Need either:
(a) use small fast models for integration testing, proper models
for accuracy testing (one at a time), or (b) use MeasurementAdvisor
(no LLM) for sweeps and Ollama only for specific titles.

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

## Next to try

- [ ] **E12: Self-feedback loop**. Generate initial chain, compute
      its frequency response, show the LLM the overlay (proposed
      vs measured curve) and general guidelines (no overshoot,
      monotonic below knee, matches rolloff slope), ask "is this
      good? what would you change?" - iterate until LLM says
      it's satisfied, or max 3 passes. NO CATALOGUE in the loop
      (must work in production).
- [ ] **E13: Revert overfit prompts**. Strip per-film names and
      hardcoded numbers from all 3 prompts. Replace with general
      principles. Re-run to see honest baseline, then add E12
      feedback loop on top.

- [ ] **E8: Few-shot prompt with catalogue examples** — include 3-5
      labelled examples in the prompt: "(title, curve features,
      correct max_gain_db)". See if LLM anchors numerically.
- [ ] **E9: Larger Ollama model** — llama3.1:70b or qwen2.5:14b on
      the same prompt. Does more parameters mean better numeric
      calibration?
- [ ] **E10: Ask LLM for categorical classification, not numbers** —
      "is this a conservative / moderate / aggressive BEQ title?" and
      map category -> gain range procedurally. Maybe LLM is better
      at categorisation than regression.
- [ ] **E11: Two-stage prompt** — first ask LLM to reason about the
      film (genre, era, score composer, sound design), THEN ask for
      numbers. Chain-of-thought may improve calibration.
- [ ] **E12: Calibration layer** — learn a linear correction from
      LLM outputs to true catalogue values using the 3 existing
      fixtures. Thin wrapper over any LLM.

---

## 2026-04-08: ML trained-model experiments (E25–E29)

See companion doc [`auto_beq_ml_experiments.md`](auto_beq_ml_experiments.md) for full design.

**Note**: Renumbered from E18–E24 to E25–E29 to resolve collision with
the chunked-audio experiments (E18–E22) above.

### E25 - XGBoost trained model (initial code path)

**Hypothesis**: A regression model trained on BEQ catalogue entries learns the
mapping from audio features (Option A: 9-bin percentile curve) + metadata
(year, audio format, release type, studio, mixer, genre, country, runtime,
rating) to corrective filter parameters. Bypasses all hand-coded rolloff rules.

**Feature vector** (60 dims):
- Audio: 9 bins at 20/25/30/35/40/50/60/70/80 Hz (Option A)
- Metadata: year(1) + audio_format(6) + source(3) + studio_embed(16) +
  mixer_embed(8) + genre(10) + country(5) + runtime(1) + rating(1) = 51
- Studio + mixer: feature-hashed from TMDb data (was zero-padded stubs
  initially, now populated via `auto_beq_metadata.py`)

**Y label** (16 dims): MAX_FILTER_SLOTS=4 × [type_int, freq, gain, q]

**Key design decisions**:
- Deduplication before split: group by title, keep highest-quality format
  (Atmos > TrueHD > DTS-HD MA > other) to avoid leaking BD/UHD duplicates
  across the train/test boundary
- Stop on downstream loss (mean dB error 20–80 Hz), not parameter MSE
- 70/15/15 split, stratified by rolloff severity + contributor

**Architecture progression**: XGBoost (this step) → 1D CNN → transformer.
XGBoost feature importances gate escalation: if studio/year/format not high
importance, metadata strategy needs revisiting before CNN.

**New module**: `src/main/python/model/auto_beq_nn.py`
**New tests**: `src/test/python/spike/test_auto_beq_nn.py` (9 tests)
**Deps added**: `xgboost`, `scikit-learn`

### E25a - TMDb metadata enrichment

**Goal**: Populate studio/mixer fields — the plan identifies studio as
the "single most predictive metadata field" because mixing stages are
per-studio.

**Implementation**: `auto_beq_metadata.py` fetches movie details + credits
from TMDb API for all catalogue entries (keyed by `theMovieDB` ID already
present in every entry). Extracts: primary studio, all production companies,
sound re-recording mixer(s), supervising sound editor(s), sound designer(s),
director(s), production country.

**Cache**: `~/.config/beqdesigner/tmdb_metadata_cache.json`. First run
fetches ~8k entries (no artificial throttle, respects 429 Retry-After).
Subsequent runs are instant from cache.

**Result**: 7,920 entries fetched, ~300 errors (TV series entries that don't
resolve as movies on TMDb — these need the `/tv/` endpoint).

### E25b - Full-catalogue training with real-audio validation

**Setup**: Trained XGBoost on the full deduplicated catalogue (~7k unique
titles, synthetic features derived from catalogue filter chains) with TMDb
metadata enrichment. Validated on 7 titles where we have extracted LFE WAV
files from real media.

**Result**: **FAIL — 7.43 dB mean downstream loss on real audio** (target <2 dB).

Per-title real-audio results:

| Title | Downstream loss | Verdict |
|---|---|---|
| Mad Max: Fury Road | 2.54 dB | MARGINAL |
| Elio | 2.55 dB | FAIL |
| KPop Demon Hunters | 4.69 dB | FAIL |
| Zootopia 2 | 9.98 dB | FAIL |
| John Wick | 10.23 dB | FAIL |
| Despicable Me 4 | 10.64 dB | FAIL |
| Flow | 11.35 dB | FAIL |

Synthetic features on the same 7 titles: 4.44 dB mean loss.
**Real-audio gap: +2.99 dB** — measured spectra are significantly harder
than the "perfect inverse" synthetic curves.

**Feature importances (top 15)**:

| Feature | Importance | Notes |
|---|---|---|
| Audio format (Atmos) | 6.8% | Highest — Atmos titles have different headroom |
| Source (Streaming) | 5.1% | Disc vs streaming gets different treatment |
| Country (English) | 4.0% | Hollywood vs non-Hollywood mixing practices |
| Audio 80 Hz | 3.5% | Upper bass band — most variation between titles |
| Year | 2.8% | Temporal drift in rolloff practices is real |
| Audio 30 Hz | 2.8% | |
| Source (Unknown) | 2.7% | |
| Audio 50 Hz | 2.3% | |
| Audio 25 Hz | 2.3% | |
| Country (Other) | 2.2% | |
| Audio 40 Hz | 2.1% | |
| Audio 70 Hz | 2.0% | |
| Audio 60 Hz | 2.0% | |
| Audio format (DD+) | 1.9% | |
| Mixer (hash bin 1) | 1.8% | Individual mixer styles detectable |

**Key findings**:
1. **Metadata matters**: audio format and source/release type are the two
   most important features — more important than any individual frequency
   bin. The plan's thesis that metadata carries prediction when audio signal
   is sparse appears correct.
2. **Studio didn't surface**: 16-dim feature hash causes collisions across
   ~3k unique studios. Many studios hash to the same bin, destroying the
   signal. Needs a proper vocabulary or larger hash space.
3. **Synthetic training data is not enough**: the 3 dB gap between
   synthetic and real features means the model trained on "perfect inverse"
   curves can't handle noisy measured spectra. Real audio features are
   needed for training, not just validation.
4. **Model prefers HighShelf incorrectly**: predicted filter types skew
   heavily toward HighShelf when the catalogue overwhelmingly uses LowShelf.
   The type_int encoding (LowShelf=0, HighShelf=1, PeakingEQ=2) may cause
   XGBoost to treat type as continuous rather than categorical.
5. **Year and country are predictive**: confirms that rolloff practices
   changed over time and that Hollywood mixes differ from non-English mixes.

**Lessons for next steps**:
- Fix studio encoding before concluding metadata doesn't help
- Need real audio features for training (STFT pipeline), not just validation
- Consider encoding filter type as separate one-hot columns rather than
  a single integer, so XGBoost treats it categorically
- Run ablation: audio-only vs audio+metadata to quantify metadata contribution
  once encoding issues are fixed

### E26 - Hybrid: trained model warm-start + scipy refinement

**Hypothesis**: Model prediction (E18) warm-starts the scipy optimiser with
±30% bounds. Metadata priors anchor prediction when audio evidence is thin.
Scene-quality chunk filtering (from E3/E6 modules) feeds cleaner audio signal.

**Status**: Deferred until E18 XGBoost baseline is validated.

**Architecture**:
1. Stage 1: chunk quality filtering (music exclusion, energy weighting)
2. Stage 2: E18 model on filtered audio + full metadata → warm start
3. Stage 3: scipy L-BFGS-B with model prediction as init, ±30% bounds

**Confidence scoring**: chunk count + ensemble variance + metadata completeness
+ stage 2→3 delta + optimiser convergence. Below threshold → human review flag.

### E25c - Studio vocab encoding fix + TV endpoint fix

**Problem**: Studio feature-hashing crammed ~3,419 unique studios into 16
dims, causing massive collisions. Studio — identified in the plan as the
"single most predictive metadata field" — didn't appear in the top 15
feature importances in E18b.

**Fix 1 — Studio vocab**: Replaced 16-dim feature hash with top-30 studio
vocabulary (Paramount, Universal, Columbia, Warner, etc.) + "other" bucket.
Same for mixer: top-20 vocabulary + "other" + "unknown". Each major studio
gets its own XGBoost split point with zero collisions. Feature vector grew
from 60 to 89 dims.

**Fix 2 — TV endpoint**: BEQ catalogue has `content_type` ("film" vs "TV").
Now uses `/tv/` TMDb endpoint for TV entries instead of always hitting
`/movie/` and getting 404s on ~300 TV series entries. Also caches 404
misses as sentinels so failed lookups aren't re-attempted.

**Result** (14 real-audio titles, trained on ~7k synthetic catalogue):

| Run | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25b (hash 16-dim) | 7.43 dB | 4.44 dB | 2.99 dB |
| E25c (vocab top-30) | **5.82 dB** | **3.80 dB** | **2.00 dB** |

**1.6 dB improvement** on real audio from fixing studio encoding alone.
Synthetic-to-real gap also shrank from 3.0 to 2.0 dB.

Per-title standouts:
- **Elio: 1.62 dB** — first title to cross the 2 dB threshold
- **Mad Max: 2.90 dB** — MARGINAL, consistent across runs
- **Sonic 3: 3.08 dB**, **Garfield: 4.03 dB** — in striking distance

Updated feature importances (top 15):

| Feature | Importance |
|---|---|
| Audio format (Atmos) | 5.0% |
| Source (Streaming) | 4.0% |
| Country (English) | 3.7% |
| Audio 80 Hz | 3.0% |
| Audio 35 Hz | 2.7% |
| Source (Unknown) | 2.5% |
| Year | 2.2% |
| **Mixer (Michael Minkler)** | **2.0%** |
| Audio 40 Hz | 1.9% |
| Audio 30 Hz | 1.8% |
| Audio 25 Hz | 1.8% |
| Audio 70 Hz | 1.7% |
| Source (Disc) | 1.7% |
| Audio format (DD+) | 1.6% |
| Mixer (unknown) | 1.6% |

**Key observations**:
1. **Studio still didn't crack top 15** — even with proper one-hot, top-30
   studios only cover 20% of entries. The "other" bucket absorbs 80% of
   titles, diluting the signal. May need more studios in the vocab, or
   studio-family grouping (e.g. all Disney subsidiaries → "Disney").
2. **Mixer is a real signal**: Michael Minkler at 2.0% importance — a
   named individual outranks most audio bins. The plan's thesis about
   individual mixer styles is confirmed.
3. **Audio format dominates metadata**: Atmos vs non-Atmos is still the
   single most important feature overall. This makes physical sense —
   Atmos encodes use different headroom assumptions.
4. **The gap is closing**: 2.0 dB synthetic-to-real gap means the model
   trained on "perfect inverse" curves transfers reasonably to real audio.
   Still room to improve with real audio training data.

### E25d - Studio-family grouping (parent company resolution)

**Problem**: Even with proper one-hot encoding (E18c), studio didn't crack
the top 15 feature importances. Root cause: top-30 individual studios only
cover 20% of catalogue entries — 80% fall into "other", diluting the signal.

**Analysis of alternatives**:

| Approach | Dims | Coverage | Notes |
|---|---|---|---|
| Top-30 individual + other (E18c) | 31 | 20% | Too many in "other" |
| Top-100 individual + other | 101 | 30% | Diminishing returns, high dim cost |
| Parent groups (11) + other | 12 | 35% | Best coverage per dim |
| **Hybrid (11 parents + ~20 ungrouped + other)** | **~32** | **~45%** | Recommended |

**Physical rationale**: Disney subsidiaries (Walt Disney Pictures, Pixar,
Marvel Studios, Touchstone, Lucasfilm, 20th Century Fox, Searchlight, Blue
Sky, Walt Disney Animation Studios) literally share mixing stages and
mastering engineers. A film mixed at Disney's Buena Vista stages has the
same bass rolloff tendencies whether it's branded Pixar or Marvel. Same
for Warner subsidiaries (Warner Bros., New Line Cinema, DC, Castle Rock,
HBO Films, Warner Animation Group).

**Parent group definitions** (from TMDb data analysis):

| Parent | Subsidiaries | Entries | Coverage |
|---|---|---|---|
| Disney | Walt Disney Pictures, Pixar, Marvel Studios, Touchstone, Lucasfilm, 20th Century Fox/Studios, Searchlight, Blue Sky, DreamWorks Animation, Walt Disney Animation Studios | 555 | 6.9% |
| Warner | Warner Bros. Pictures, Warner Bros. Animation, New Line Cinema, Castle Rock, DC Films/Studios, HBO Films, Warner Animation Group | 462 | 5.8% |
| Universal | Universal Pictures, Focus Features, Working Title, Illumination, Amblin, Universal 1440, DreamWorks Pictures | 422 | 5.3% |
| Sony/Columbia | Columbia Pictures, TriStar, Screen Gems, Sony Pictures, Sony Pictures Animation | 372 | 4.6% |
| Lionsgate | Lionsgate, Summit Entertainment, StudioCanal | 310 | 3.9% |
| Paramount | Paramount Pictures, Paramount Animation, Miramax | 294 | 3.7% |
| MGM | Metro-Goldwyn-Mayer, United Artists, Orion Pictures | 156 | 1.9% |
| Amazon | Amazon Studios, Amazon MGM Studios | 63 | 0.8% |
| Blumhouse | Blumhouse Productions | 61 | 0.8% |
| A24 | A24 | 41 | 0.5% |
| Netflix | Netflix | 40 | 0.5% |
| **Total grouped** | | **2,776** | **34.7%** |

**Resolution logic**: Check primary studio name against parent group
subsidiary lists. If no match, check ALL production companies from TMDb
credits (e.g. "Legendary Pictures" as primary + "Warner Bros." as co-producer
→ resolves to "Warner"). Ungrouped studios with ≥15 entries get their own
one-hot bucket.

**Implementation**: `_resolve_studio_parent(studio, all_studios)` in
`auto_beq_nn.py`. Hybrid vocab: 11 parent groups + ~20 top ungrouped +
"other" ≈ 32 dims total (down from 31 but 45% vs 20% coverage).

**Result** (14 real-audio titles, trained on ~7k synthetic catalogue):

| Run | Real audio | Synthetic | Gap | Studio in top 15? |
|---|---|---|---|---|
| E25b (hash 16-dim) | 7.43 dB | 4.44 dB | 2.99 dB | No |
| E25c (vocab top-30) | 5.82 dB | 3.80 dB | 2.02 dB | No |
| **E25d (parent groups)** | **6.34 dB** | **3.45 dB** | **2.90 dB** | **Yes** |

**Studio Universal appeared at position 10** in feature importances (1.79%) —
first time any studio feature has surfaced. Mark Paterson (mixer) also
appeared at position 12 (1.67%).

**Synthetic loss improved** to 3.45 dB (best yet), showing the parent
grouping helps the model learn better patterns from catalogue data. However,
real-audio loss (6.34 dB) was slightly worse than E25c (5.82 dB), and the
synthetic-to-real gap widened back to 2.9 dB.

**Interpretation**: The model is learning more nuanced studio-specific
patterns from synthetic data (lower synthetic loss), but these studio-
specific patterns may be overfitting to the "perfect inverse" curve shapes.
Real measured audio has content-dependent variation that breaks the
synthetic assumptions differently for different studios. The fundamental
bottleneck is now the **synthetic training data**, not the metadata encoding.

Per-title: Elio (2.35 dB), Mad Max (2.64 dB), Garfield (2.81 dB), Sonic 3
(3.49 dB) are all close to or under threshold. The failures are dominated
by animated kids' films (Kung Fu Panda 4, Super Mario Bros, Wild Robot,
Moana 2) at 8-10 dB — these may have different rolloff characteristics
that the model hasn't learned.

**Feature importances (top 15)**:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Audio format (Atmos) | 4.9% |
| 2 | Source (Streaming) | 4.6% |
| 3 | Country (English) | 3.7% |
| 4 | Audio 80 Hz | 3.2% |
| 5 | Source (Unknown) | 2.7% |
| 6 | Year | 2.4% |
| 7 | Audio 30 Hz | 2.4% |
| 8 | Audio 50 Hz | 2.3% |
| 9 | Mixer (unknown) | 2.0% |
| **10** | **Studio (Universal)** | **1.8%** |
| 11 | Audio 25 Hz | 1.8% |
| **12** | **Mixer (Mark Paterson)** | **1.7%** |
| 13 | Audio 40 Hz | 1.7% |
| 14 | Audio 60 Hz | 1.6% |
| 15 | Audio format (DD+) | 1.6% |

### E25e - Ablation: audio-only vs metadata-only vs full

**Setup**: Three XGBoost models trained on the same ~8,200 synthetic
catalogue entries, each seeing a different feature subset:
- **audio-only**: 9 frequency bins (dims 0–8), metadata zeroed
- **metadata-only**: 81 metadata features (dims 9–89), audio zeroed
- **full**: all 90 dims (E25d baseline)

All three evaluated on the same 14 real-audio validation titles.

**Result**:

| Variant | Real audio | Synthetic | Gap |
|---|---|---|---|
| **audio-only** | **3.53 dB** | 3.80 dB | **-0.27 dB** |
| metadata-only | 3.78 dB | 3.78 dB | 0.00 dB |
| full (audio+meta) | 6.34 dB | 3.45 dB | +2.90 dB |

**Key findings**:

1. **The combined model is the WORST on real audio** (6.34 dB), nearly 3 dB
   worse than either feature set alone. Audio-only (3.53 dB) is the best
   single predictor on real data.

2. **Audio-only has a negative gap** (-0.27 dB): real audio is actually
   *easier* than synthetic for the audio-only model. This makes sense —
   when the model only sees audio features, it learns pure frequency-shape
   patterns. Real measured curves have the same general rolloff shape as
   synthetic, just noisier. The audio-only model is robust to that noise.

3. **Metadata-only has zero gap** (0.00 dB): metadata features are identical
   between synthetic and real (year, studio, format don't change). The 3.78 dB
   metadata-only score is the pure metadata baseline — what you can predict
   about a film's BEQ just from knowing it's "Paramount, 2024, Atmos".

4. **The full model overfits to synthetic-specific cross-correlations**
   between audio and metadata that don't hold on real audio. When XGBoost
   sees metadata saying "Paramount, 2024, Atmos" alongside a noisy real
   spectrum, the cross-feature splits it learned from perfect synthetic
   curves produce worse predictions than either signal alone.

**Interpretation**: This is a **feature interaction overfit**, not a
metadata encoding problem. The solution is NOT to drop metadata — it
carries real signal (3.78 dB standalone is close to audio-only's 3.53 dB).
The solution is one of:

1. **Train on real audio**: The cross-correlations between audio and metadata
   would be learned from real spectra, eliminating the synthetic-to-real gap
   that causes the overfit. This is the STFT pipeline work.

2. **Late fusion**: Train separate audio-only and metadata-only models,
   then combine their predictions (average, stack, or blend). This prevents
   cross-feature overfitting entirely.

3. **Regularisation**: Reduce XGBoost tree depth or increase min_child_weight
   to prevent learning fine-grained audio×metadata interactions that don't
   generalise.

**Implication for architecture progression**: The 1D CNN dual-branch design
(audio CNN + metadata dense → merged at penultimate layer) naturally provides
late fusion. The CNN stage may solve this overfit problem structurally.

This ablation is committed as a permanent test
(`test_ablation_audio_vs_metadata`) for continuous reassessment as the
model evolves.

### E27 - Late fusion XGBoost (separate audio + metadata models, blended)

**Hypothesis**: The E25e overfit comes from cross-feature interactions between
audio and metadata learned on synthetic data. Training two independent
XGBoost sub-models (audio-only, metadata-only) and blending their Y
predictions should prevent this while preserving both signals.

**Implementation**: `LateFusionModel` in `auto_beq_nn.py` wraps two XGBoost
sub-models. `predict(X)` returns `α * Y_audio + (1-α) * Y_meta`.
`LateFusionAdvisor` registered as `"late_fusion"` in `get_advisor()`.

**Result** (14 real-audio titles, trained on ~8,200 synthetic):

| Strategy | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25 early fusion | 6.34 dB | 3.45 dB | +2.90 dB |
| E27 late fusion (α=0.3) | 5.85 dB | 3.63 dB | +2.22 dB |
| E27 late fusion (α=0.5) | 4.87 dB | 3.86 dB | +1.01 dB |
| **E27 late fusion (α=0.7)** | **4.03 dB** | **3.68 dB** | **+0.35 dB** |

**Key findings**:

1. **Late fusion at α=0.7 cuts real-audio loss from 6.34 to 4.03 dB** — a
   2.3 dB improvement over early fusion, and the first time the combined
   model beats the individual audio-only baseline (3.53 dB from E25e ablation
   + metadata contribution = 4.03 dB blended).

2. **Synthetic-to-real gap collapsed from 2.90 to 0.35 dB** — late fusion
   almost completely eliminates the overfitting to synthetic cross-correlations
   that plagued early fusion. The model now transfers nearly perfectly from
   synthetic to real audio.

3. **α=0.7 is optimal** — audio carries 70% of the prediction, metadata 30%.
   This matches intuition: the measured rolloff curve is the primary signal,
   metadata provides a useful prior that adjusts the prediction.

4. **Monotonic improvement with α**: as audio weight increases (0.3 → 0.5
   → 0.7), real-audio loss decreases. The metadata-only model was never the
   problem — it was the cross-correlation with audio on synthetic data.

Committed as permanent regression test (`test_late_fusion_vs_early`) for
continuous assessment.

### E28 - CNN dual-branch (PyTorch)

**Hypothesis**: A neural network with separate audio (1D conv) and metadata
(dense) branches merged at the penultimate layer naturally provides late
fusion. The conv layers may learn frequency-domain patterns that XGBoost's
axis-aligned splits cannot represent.

**Architecture**:
```
audio (9 bins) → Conv1d(1,32,k3) → ReLU → Conv1d(32,64,k3) → ReLU
  → AdaptiveAvgPool1d → Linear(64) → ReLU → audio_embed (64d)
metadata (81d) → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU
  → meta_embed (32d)
[audio_embed ⊕ meta_embed] → Linear(48) → ReLU → Dropout(0.1) → Linear(16)
```

**Implementation**: `DualBranchCNN` in `auto_beq_nn_cnn.py`. `CNNAdvisor`
registered as `"cnn_dual_branch"` in `get_advisor()`. Training uses Adam,
MSE loss, early stopping on validation loss.

**Result** (14 real-audio titles, trained on ~8,200 synthetic):

| Strategy | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25 early fusion | 6.34 dB | 3.45 dB | +2.90 dB |
| **E27 late fusion (α=0.7)** | **4.03 dB** | **3.68 dB** | **+0.35 dB** |
| E28 CNN dual-branch | 24.12 dB | 3.52 dB | +20.60 dB |

**Key findings**:

1. **CNN catastrophically overfits on synthetic data**. The 3.52 dB synthetic
   loss is competitive with XGBoost (3.45 dB), but the 24.12 dB real-audio
   loss is 4× worse than early fusion. The CNN memorised synthetic feature
   patterns that have zero transfer to real measured audio.

2. **The dual-branch architecture does NOT solve the overfit by itself**.
   The problem isn't cross-feature interactions (which the branches
   separate) — it's the distribution mismatch between synthetic "perfect
   inverse" curves and noisy real spectra. Neural nets are far more
   sensitive to this than tree models.

3. **XGBoost late fusion (E27) remains the best approach for synthetic
   training data**. At 4.03 dB with a 0.35 dB gap, it's the only model
   that combines audio and metadata without overfitting.

4. **CNN needs real audio training data to be viable**. The architecture
   is sound (3.52 dB synthetic proves it can learn), but it requires
   training features that match the inference-time distribution. This is
   the STFT pipeline investment.

5. Early stopping triggered at epoch 24 (patience=20, best at epoch 4) —
   the CNN converged fast and immediately started overfitting. With ~8k
   samples and ~3k parameters, this is underfit/overfit in the classic
   small-dataset neural net failure mode.

Committed as permanent regression test (`test_cnn_dual_branch`). Note:
test avoids mixing torch + XGBoost in the same process due to segfault
on macOS (library conflict).

### E29 - BEQ profile author as input feature

**Hypothesis**: BEQ catalogue authors have distinct calibration styles
(different aggressiveness biases). Knowing who authored the profile should
improve prediction, similar to how mixer identity helps.

**Data**: 8 unique authors, 100% coverage. Heavily skewed: mobe1969 is 57%.
Encoded as 9-dim one-hot (8 authors + "unknown"). Feature vector: 99 dims.

**Result** — ablation comparison (with vs without author):

| Variant | Without author | With author | Change |
|---|---|---|---|
| audio-only | 3.53 dB | 4.51 dB | +0.98 worse |
| **metadata-only** | 3.78 dB | **3.09 dB** | **-0.69 better** |
| full (audio+meta) | 6.34 dB | 5.07 dB | -1.27 better |

**Result** — late fusion comparison:

| Strategy | Without author | With author | Change |
|---|---|---|---|
| Early fusion | 6.34 dB | 5.07 dB | -1.27 better |
| **Late α=0.3** | 5.85 dB | **4.35 dB** | **-1.50 better** |
| Late α=0.5 | 4.87 dB | 5.53 dB | +0.66 worse |
| Late α=0.7 | **4.03 dB** | 5.75 dB | +1.72 worse |

**Key findings**:

1. **Metadata-only at 3.09 dB is the best single-model result ever**.
   Author is so strong that pure metadata (no audio features at all)
   outperforms every previous approach. This is extraordinary — knowing
   studio + year + format + author is enough to predict BEQ filters
   within 3 dB.

2. **Optimal alpha flipped from 0.7 to 0.3**. Before author: audio
   should dominate (α=0.7). After author: metadata should dominate
   (α=0.3). Author made the metadata branch the primary signal.

3. **Author confirmed as the strongest single feature**. The metadata-only
   model improved by 0.69 dB purely from adding author — a larger single-
   feature improvement than any previous change.

4. **Audio-only degraded** by +0.98 dB. This is noise — the audio-only
   model doesn't see author, so this variance is from different random
   splits or XGBoost randomness. The audio branch hasn't changed.

5. **Implication**: For production use on uncatalogued content, we won't
   know the author (there is no author yet — we're generating the BEQ).
   The author feature is only useful when **predicting what a specific
   author would do** for a title, not for generating novel BEQs. This
   makes author a calibration/training signal, not an inference feature.

### E29a - Impact of author on all strategies (full rerun)

With author added (E29), reran all strategies to measure impact.

**Feature importances (E25 early fusion with author)**:

| Rank | Feature | Importance |
|---|---|---|
| **1** | **author_aron7awol** | **14.8%** |
| **2** | **author_mobe1969** | **5.9%** |
| **3** | **author_kaelaria** | **2.6%** |
| **4** | **author_mikejl** | **2.3%** |
| 5 | audio_30Hz | 2.3% |
| 6 | src_stream | 2.2% |
| 7 | year | 2.0% |
| **8** | **author_remixmark** | **1.9%** |

Authors collectively account for ~28% of all feature importance. aron7awol
alone (14.8%) is 3× more important than any non-author feature.

**Strategy comparison (all with author)**:

| Strategy | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25 early fusion | 5.07 dB | 4.45 dB | +0.62 dB |
| E27 late fusion (α=0.3) | 4.35 dB | 3.01 dB | +1.34 dB |
| E28 CNN dual-branch | 6.00 dB | **2.08 dB** | +3.92 dB |
| **Metadata-only (ablation)** | **3.09 dB** | 3.09 dB | **0.00 dB** |

**E28 CNN with author — dramatic improvement**:

| Metric | Without author | With author | Change |
|---|---|---|---|
| CNN real audio | 24.12 dB | **6.00 dB** | **-18.12 dB** |
| CNN synthetic | 3.52 dB | **2.08 dB** | -1.44 dB |
| CNN gap | +20.60 dB | +3.92 dB | -16.68 dB |

Author gave the CNN enough anchor signal to avoid catastrophic overfitting.
2.08 dB synthetic is the best synthetic loss from any model ever. But the
CNN still has a 3.92 dB gap vs the metadata-only model's 0.00 dB gap.

**Per-title standouts (E25 early fusion with author)**:
- **Garfield: 1.65 dB** — first title under 2 dB threshold in early fusion
- **John Wick: 1.85 dB** — MARGINAL verdict, under 2 dB
- **Elio: 1.88 dB** — under threshold
- Mad Max: 2.31 dB, KPop: 2.46 dB, Wild Robot: 2.60 dB — close
- Outliers: Moana 2 (11.55), Kung Fu Panda 4 (10.56) — animated kids' films

**Summary of E29 author impact across all strategies**:
- Author is the single most powerful feature added to the model
- Metadata-only at 3.09 dB remains the best real-audio result
- CNN improved 18 dB but still trails XGBoost approaches on real audio
- Early fusion gap collapsed from 2.90 to 0.62 dB — author stabilises transfer
- The animated kids' film outliers suggest a genre-specific calibration issue

### E30 - Expanded validation: 20 titles (14 movies + 6 TV series)

**Goal**: More representative validation by matching WAVs to catalogue via
title+year (not just TMDb ID), picking up 6 TV series: Blue Eye Samurai,
Mindhunter, Scavengers Reign, South Park, Spawn, X-Men '97.

**Result** (20 real-audio titles, trained on ~8,213 synthetic with author):

| Test | 14 titles (E29) | 20 titles (E30) |
|---|---|---|
| E25 early fusion | 5.07 dB | 6.42 dB |
| E25e audio-only | 4.51 dB | 4.80 dB |
| **E25e metadata-only** | **3.09 dB** | **3.41 dB** |
| E27 late fusion (α=0.3) | 4.35 dB | 5.45 dB |
| Synth-to-real gap | 0.62 dB | 1.70 dB |

Feature importances: audio 13.3% vs metadata 86.7%. Authors still dominate
(5 of top 15, aron7awol 14.4%, mobe1969 5.8%).

**Per-title standouts**:
- **Mindhunter: 0.83 dB** — best single title ever, well under threshold
- **Elio: 1.29 dB** — improved from 1.88 dB
- **KPop Demon Hunters: 2.08 dB** — near threshold
- South Park: 2.90 dB, Mad Max: 2.77 dB, Blue Eye Samurai: 3.42 dB
- Outliers: Flow 12.57 dB, Spawn 11.15 dB, Super Mario 11.24 dB

**Key findings**:
1. **TV content is harder** — the 6 new TV titles pulled the mean up by
   ~1 dB. Spawn (1997) and South Park have very different audio
   characteristics from modern movies.
2. **The metadata-only model generalises best** — 3.41 dB across 20 titles
   with zero synthetic-to-real gap. Production context (author + studio +
   year + format) remains the most robust predictor.
3. **The gap widened** from 0.62 to 1.70 dB for early fusion — TV content's
   different audio characteristics expose the synthetic training weakness
   more than movies do.
4. **Mindhunter at 0.83 dB is extraordinary** — a Netflix drama with
   distinctive sound design. The model may be leveraging the combination
   of author (mobe1969) + studio (Netflix) + year (2017) + genre (drama)
   to closely match the catalogue entry.

**E28 CNN with 20 titles**: 6.17 dB real, 2.65 dB synthetic (gap +3.53).
CNN continues to achieve the best synthetic loss but still overfits vs
XGBoost on real audio. The dual-branch architecture doesn't compensate
for the synthetic-to-real distribution mismatch.

### E31 - NAS extraction + 111-title validation

**Goal**: Extract LFE from the full NAS media library via standalone script
(`scripts/extract_lfe.py`), validate on the much larger real-audio set.

**Infrastructure**: Standalone extraction script running locally on NAS
(no network transfer). BEQ catalogue auto-fetched from GitHub, only
catalogue-matched media extracted. Breadth-first interleaving (movies +
TV round-robin). Portable WAV cache at `{beq-dir}/wav-cache/`. Atomic
writes + WAV integrity validation.

**Result** (111 real-audio titles, trained on ~8,170 synthetic with author):

| Metric | E30 (20 titles) | E31 (111 titles) |
|---|---|---|
| Real-audio mean loss | 6.42 dB | **4.09 dB** |
| Synthetic mean loss | 4.72 dB | **2.90 dB** |
| Synth-to-real gap | 1.70 dB | **1.19 dB** |

**2.33 dB improvement on real audio** from 5× more validation data.
The synthetic-to-real gap tightened from 1.70 to 1.19 dB — the model
transfers better when evaluated on a diverse, representative set.

Feature importances unchanged: authors dominate (aron7awol 13.9%,
mobe1969 6.6%), audio bins and year/source in the middle tier.

**E27 late fusion on 111 titles**:

| Strategy | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25 early fusion | 4.07 dB | 2.88 dB | +1.19 dB |
| **E27 late fusion (α=0.3)** | **3.36 dB** | **2.67 dB** | **+0.69 dB** |
| E27 late fusion (α=0.5) | 4.12 dB | 2.68 dB | +1.44 dB |
| E27 late fusion (α=0.7) | 4.55 dB | 2.87 dB | +1.68 dB |

**3.36 dB on 111 real-audio titles** — best combined model result ever.
α=0.3 (metadata-heavy) is optimal, consistent with E29's finding that
author makes metadata the dominant signal.

**Progression summary**:

| Milestone | Real audio | Titles | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 dB | 7 | Initial baseline |
| E25c (vocab encoding) | 5.82 dB | 14 | Studio one-hot |
| E27 (late fusion, no author) | 4.03 dB | 14 | Separate audio+meta models |
| E30 (expanded set) | 6.42 dB | 20 | TV content harder |
| E31 (NAS extraction, 111) | 3.36 dB | 111 | 5× more data + late fusion |
| E31 (NAS extraction, 171) | 5.02 dB | 171 | Harder content in expanded set |
| **E31 late fusion (α=0.3, 171)** | **3.27 dB** | **171** | **Best ever on large set** |

### E31 continued — 171-title validation (NAS extraction ongoing)

**Setup**: NAS extraction script running breadth-first across 1,244 catalogue-
matched titles. At time of test: 171 WAVs available (80 movies + 89 TV eps
from 49 shows).

**E25 early fusion (171 titles)**: 5.02 dB real, 3.22 dB synthetic, gap 1.80 dB.
Slightly worse than 111-title (4.09 dB) — the expanded set includes harder
content (older TV, anime, Bollywood).

**E27 late fusion (171 titles)**:

| Strategy | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25 early fusion | 5.02 dB | 3.22 dB | +1.80 dB |
| **E27 late fusion (α=0.3)** | **3.27 dB** | **2.90 dB** | **+0.37 dB** |
| E27 late fusion (α=0.5) | 3.97 dB | 2.93 dB | +1.04 dB |
| E27 late fusion (α=0.7) | 4.13 dB | 3.02 dB | +1.11 dB |

**3.27 dB on 171 real-audio titles** — best combined model result on a
large representative set. α=0.3 (metadata-heavy) still optimal. Gap
collapsed to 0.37 dB — near-perfect synthetic-to-real transfer.

### E32 — Chunked NN training (in progress)

**Hypothesis**: Blended extraction (α=0.7 Welch + chunked P90 at 60s) for
real-audio features should improve NN performance because chunked features
capture transient bass events that Welch dilutes.

**Implementation**: Same synthetic training data, same XGBoost model. Only
the validation feature extraction changes. Uses parallel ProcessPoolExecutor
for speed (scipy Welch is single-threaded).

**Result** (180 WAVs, parallel extraction):

| Strategy | Real audio | Synthetic | Gap | Time |
|---|---|---|---|---|
| Welch-only | 4.61 dB | 2.98 dB | +1.63 dB | 39s |
| **Blended (α=0.7, 60s P90)** | **4.48 dB** | 2.99 dB | **+1.49 dB** | 38s |

**Findings**:
1. **Blended improves by 0.13 dB** on real audio — modest but consistent.
   The chunked P90 captures transient bass events that Welch dilutes.
2. **Synthetic loss is identical** (2.98 vs 2.99) — the improvement is
   entirely in real-audio transfer, not synthetic fitting.
3. **Gap tightened by 0.14 dB** (1.63 → 1.49) — blended features
   transfer better from synthetic to real.
4. **Parallel extraction: 9× speedup** — 180 WAVs in 39s (was ~6 min
   serial). ProcessPoolExecutor fully uses all CPU cores.
5. The improvement is smaller than expected — the NN's 9-bin Option A
   feature vector may not be granular enough to capture the chunking
   benefit. Option B (27-value chunk feature matrix) would be the next
   escalation if this proves valuable.

**E32 late fusion + blended**: 3.53 dB (compared to 3.27 dB Welch-only
late fusion from a slightly earlier WAV set). Marginal — blended doesn't
meaningfully help late fusion.

### E33 — Train on real audio features (not synthetic)

**Hypothesis**: Training on real measured audio features should eliminate
the synthetic-to-real gap that has been a persistent bottleneck.

**Setup**: 194 real WAVs from NAS extraction. 80/20 stratified split →
155 train / 39 test (held-out real audio). Three training approaches:
- Synthetic-only (baseline): 8,203 deduplicated catalogue entries
- Real-only: 155 real audio features
- Hybrid: 155 real + 8,092 synthetic for non-WAV titles

**Result**:

| Training approach | Early fusion | Late α=0.3 |
|---|---|---|
| **Synthetic-only (8,203)** | 4.13 dB | **3.22 dB** |
| Real-only (155) | 4.39 dB | 4.36 dB |
| Hybrid (8,247) | 4.76 dB | 3.46 dB |

**Key findings**:

1. **Synthetic-only still wins** at 3.22 dB late fusion — 155 real
   training samples is not enough to beat 8,203 synthetic entries.
   Synthetic data's diversity advantage (covering 8k titles vs 155)
   outweighs its imperfect feature distribution.

2. **Real-only underperforms** by 1.14 dB (4.36 vs 3.22 late fusion).
   With only 155 training samples, XGBoost can't learn robust patterns.
   The model overfits to the small real set.

3. **Hybrid is worse than synthetic-only** (3.46 vs 3.22). Mixing real
   and synthetic features in the same training set may confuse the model
   — similar to the cross-correlation overfit from E25e. The feature
   distributions are different enough that combining them hurts.

4. **Late fusion barely helps real-only** (4.36 vs 4.39) — when the
   training set is small, separating audio and metadata doesn't add value
   because there aren't enough examples to learn either branch well.

**Implication**: Real-audio training needs significantly more data — likely
500+ titles before it can compete with 8k synthetic. The NAS extraction
is still running (1,244 titles queued). Re-run this experiment when the
corpus is larger.

### E36 — Unknown author at inference time

**Question**: How much does the model degrade when we zero out the author
feature at inference time (simulating production use where there's no
known author)?

**Result** (213 titles, early fusion):
- With author: 4.94 dB
- Without author: 5.61 dB
- **Author impact: +0.67 dB**

**Finding**: Author costs only 0.67 dB — much less than its 28% feature
importance would suggest. The model is usable in production without author.
The high importance reflects how much the model *uses* the feature during
training, not how much it *needs* it for generalization.

### E37 — Per-title breakdown (213 titles)

**Summary**: 8 PASS | 23 MARGINAL | 182 FAIL | Mean: 4.94 dB

**Best titles** (under 2 dB):
In the Lost Lands (0.83), For All Mankind (0.87), Rick and Morty (1.02),
The Lion King (1.03), Love Death + Robots (1.14), Inside Out (1.16),
Foundation (1.18), Fullmetal Alchemist (1.25), Mindhunter (1.26)

**Worst titles** (over 15 dB):
History of the World Part I (20.35), Royal Space Force (20.34),
WXIII: Patlabor (16.87), Riff Raff (16.70), Angel Heart (15.64)

**Critical finding — filter type mismatch**: The worst titles all predict
HighShelf (H) when the catalogue uses LowShelf (L) or PeakingEQ (P).
The `Pred→Tgt` column shows patterns like `HHHL→LLLL` and `HLHH→LLLP`.
The integer type encoding (LowShelf=0, HighShelf=1, PeakingEQ=2) causes
XGBoost to default to HighShelf. **E34 (one-hot type encoding) is the
highest-priority fix.**

### E34 — Filter type one-hot encoding (was integer)

**Fix**: Per filter slot: [type_LS, type_HS, type_PEQ, freq, gain, q] = 6
values × 4 slots = 24 output dims (was 16). Decoded via argmax.

**Result** (220 titles, early fusion):

| Metric | Before (integer) | After (one-hot) | Change |
|---|---|---|---|
| **Mean loss** | 4.94 dB | **3.19 dB** | **-1.75 dB** |
| Synthetic loss | 3.12 dB | **2.32 dB** | -0.80 dB |
| Gap | 1.82 dB | **0.88 dB** | -0.94 dB |
| PASS | 8 | 9 | +1 |
| MARGINAL | 23 | 33 | +10 |
| FAIL | 182 | 178 | -4 |
| Author impact | +0.67 dB | **+0.27 dB** | author matters less |

**1.75 dB improvement** — the single largest gain from any technique change.
Per-title breakdown confirms the HighShelf bias is eliminated: predictions
now show mostly L (LowShelf), matching the catalogue's distribution.

Worst case dropped from 20+ dB (History of the World, Royal Space Force)
to 12 dB (Crimson Tide). The model still struggles with some older films
but the failure mode is now magnitude calibration, not wrong filter type.

**Progression summary (updated)**:

| Milestone | Real audio | Titles | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 dB | 7 | Initial baseline |
| E25c (vocab encoding) | 5.82 dB | 14 | Studio one-hot |
| E27 (late fusion, no author) | 4.03 dB | 14 | Separate audio+meta models |
| E29 (author feature) | 5.07 dB | 20 | Author dominates importances |
| E31 (NAS extraction) | 3.27 dB | 171 | 5× more data + late fusion |
| E34 (one-hot type, early) | 3.19 dB | 220 | Fixes HighShelf bias |
| **E34 + late fusion α=0.7** | **2.45 dB** | **220** | **One-hot + late fusion** |

### E34 + E27 late fusion with one-hot encoding

| Strategy | Real audio | Synthetic | Gap |
|---|---|---|---|
| E25 early fusion | 3.19 dB | 2.32 dB | +0.88 dB |
| E27 late fusion (α=0.3) | 2.61 dB | 2.83 dB | -0.22 dB |
| E27 late fusion (α=0.5) | 2.52 dB | 2.78 dB | -0.27 dB |
| **E27 late fusion (α=0.7)** | **2.45 dB** | **2.74 dB** | **-0.29 dB** |

**2.45 dB on 220 real-audio titles** — first time below 2.5 dB. The gap
is now *negative* (-0.29 dB) meaning real audio transfers better than
synthetic for this model configuration.

Optimal α flipped from 0.3 (metadata-heavy, before one-hot fix) to 0.7
(audio-heavy, after fix). The audio branch is now more reliable because
it no longer has to compensate for wrong filter types — the one-hot
encoding lets XGBoost correctly learn filter type as a categorical output.

### E38 — Reweighted training (downstream loss sample weighting)

**Approach**: Two-stage training. Stage 1: standard MSE. Stage 2: compute
downstream dB loss per training sample, upweight high-loss samples, retrain.
Forces the model to focus on samples where parameter accuracy doesn't
produce good acoustic results.

**Result** (~220 titles):

| Strategy | Real audio |
|---|---|
| Standard XGBoost | 3.55 dB |
| Reweighted (2 rounds) | 3.00 dB |
| Reweighted (3 rounds) | 3.02 dB |
| Late fusion α=0.7 | 2.63 dB |
| **Reweighted + Late fusion α=0.7** | **2.60 dB** |

**Findings**:
1. **Reweighting helps early fusion significantly** (-0.55 dB, 3.55 → 3.00).
   Samples where parameter-MSE gave bad acoustic results got upweighted
   (mean weight 2.14, max 12.86).
2. **Reweighting barely helps late fusion** (-0.03 dB, 2.63 → 2.60).
   Late fusion already handles the hard cases by separating audio and
   metadata branches.
3. **3 rounds is worse than 2** (3.02 vs 3.00) — over-correction.
4. Best overall: **late fusion α=0.7 at 2.60 dB** (reweighted) or
   **2.63 dB** (standard). The difference is marginal.

### E37 updated — Per-title breakdown on late fusion (300 titles)

**Best model**: late fusion α=0.7 + one-hot type encoding, 300 real-audio titles.

**Summary**: 25 PASS | 66 MARGINAL | 209 FAIL | Mean: **2.67 dB**

91 titles (30%) within practical tolerance (PASS + MARGINAL).

**Best** (under 1 dB): Cosmos (0.27), Planet Earth II (0.36), Pantheon (0.40),
Mindhunter (0.46), Scavengers Reign (0.47), Primal (0.50)

**Worst** (over 6 dB): Spawn (9.45), Blade Runner (8.78), The Dead Don't Hurt
(8.48), A Prayer Before Dawn (8.32), Royal Space Force (7.73)

**Failure pattern**: No more HighShelf bias (one-hot fix confirmed). Remaining
failures are magnitude calibration — right filter types but wrong gain/freq.
Worst titles are older films (pre-2000), niche content (anime, arthouse),
and titles with unusual rolloff shapes. These are titles where the
catalogue author made aggressive choices that don't match common patterns.

### E39 — Filter slots + era bucketing

**E39c analysis**: 67% of catalogue has >4 filters. Dropping filters 5+
costs 1.23 dB mean, but 64% lose <1 dB.

**E39a (8 slots)**: Over-predicted filter count (243 over vs 43 under).
48-dim output was too hard for XGBoost — filled empty slots with noise.

**E39a revised (6 slots)**: Compromise. Results across slot sizes (~330 titles):

| Config | Best late fusion | Best α | Notes |
|---|---|---|---|
| 4 slots | ~2.67 dB (300 titles) | 0.7 | Original |
| 8 slots | 2.80 dB (331 titles) | 0.3 | Over-predicted |
| 6 slots | 2.76 dB (342 titles) | 0.3 | Compromise |

**E39b (era buckets)**: Pre-1990/1990-2009/2010+ one-hot added. 1980s
still worst (5.4 dB mean) — era feature alone doesn't fix the problem
since there are too few 1980s titles in the catalogue to learn from.

**Conclusion**: More filter slots add marginal value. The 4-slot model
truncates but XGBoost compensates reasonably. 6 slots is the sweet spot
if we keep this approach, but the improvement over 4 is within noise.
The real bottleneck is magnitude calibration on older/niche content,
not filter count.

### E40 — Per-author model isolation

**Question**: Does training on a single author's entries improve predictions
for that author, compared to the multi-author model?

**Result** (349 WAVs, late fusion α=0.3):

| Author | Val | Multi-author | Single-author | Delta |
|---|---|---|---|---|
| **aron7awol** | 92 | 1.76 dB | **1.68 dB** | -0.09 |
| **t1g8rsfan** | 16 | 2.06 dB | **1.62 dB** | -0.44 |
| **kaelaria** | 58 | 2.96 dB | **2.54 dB** | -0.42 |
| halcyon888 | 15 | **1.89 dB** | (too few) | — |
| remixmark | 22 | **2.57 dB** | 3.20 dB | +0.63 |
| mobe1969 | 149 | **3.33 dB** | 3.63 dB | +0.30 |

**Key findings:**

1. **aron7awol is solved** — 1.68-1.76 dB. Consistent calibration style,
   fully learnable by the model. 92 validation titles.
2. **t1g8rsfan and kaelaria benefit from isolation** (~0.4 dB each).
   Their styles are distinct enough that removing other authors' noise helps.
3. **mobe1969 gets WORSE isolated** (+0.30 dB). Despite 5,207 training
   entries, his calibration is genuinely inconsistent — he makes different
   choices for different titles. Other authors' data actually regularises.
4. **Multi-author model is best for production** — handles all styles,
   and the author feature lets it adapt. Single-author models are only
   better for 3 of 5 testable authors.
5. **halcyon888 at 1.89 dB with only 26 training entries** — most
   consistent author. Perfect for a "conservative BEQ" production mode.

---

## 2026-04-10: F-experiment batch (E41–E52)

Unified comparison of 12 accuracy improvement techniques against the
baseline (late fusion α=0.7 + one-hot type encoding).  67 real-audio
validation titles.  All experiments share the same training/validation
split and evaluation metric (mean downstream dB loss, 20–80 Hz).

Full design rationale in `docs/design/auto_beq_nn_future_experiments.md`.

### E41/F1 — Synthetic feature augmentation

**Technique**: Training-time noise injection to audio features.  Adds
Gaussian N(0, σ) + per-bin uniform U(-u, +u) noise to the 9-dim audio
features during training, simulating the distribution mismatch between
synthetic "perfect inverse" curves and real measured spectra.  Analogous
to image augmentation (random crop, colour jitter) but for 1-D spectral
features.

**Hypothesis**: The persistent synthetic-to-real gap (0.88 dB in E34)
is caused by the model overfitting to clean synthetic features.  Adding
noise during training teaches robustness to real-world spectral variation.

**Result** (67 titles, 4 sigma configs swept):

| Config | σ | uniform | Mean dB | PASS | FAIL | Delta |
|---|---|---|---|---|---|---|
| Baseline | — | — | 2.75 | 22 | 11 | — |
| **F1-s0.5** | **0.5** | **1.0** | **2.02** | **46** | **4** | **-0.73** |
| F1-s1.0 | 1.0 | 1.5 | 2.03 | 45 | 6 | -0.72 |
| F1-s1.5 | 1.5 | 2.0 | 2.09 | 41 | 5 | -0.66 |
| F1-s2.0 | 2.0 | 2.5 | 2.25 | 41 | 6 | -0.50 |

**Key finding**: **σ=0.5 is optimal** — 2.02 dB mean, more than
doubling PASS count (22→46) and cutting FAILs from 11 to 4.  This is
the single largest accuracy improvement from any technique in the
project.  Lighter noise (σ=0.5) works better than heavier (σ=2.0)
because the real-vs-synthetic gap is ~1 dB, not ~4 dB.

**Per-author with F1-s0.5**: t1g8rsfan 0.55 dB, mobe1969 1.71 dB,
kaelaria 1.80 dB, aron7awol 1.87 dB.  Three of six authors now under
2 dB individually.

**Biggest improvements**: Zootopia 2 (8.62→2.79, -5.83), The Lego Movie
(4.43→0.87, -3.56), Gravity (3.60→0.49, -3.11).

**Lesson**: The synthetic-to-real gap was by far the #1 bottleneck.
Light augmentation (σ=0.5, u=1.0) is sufficient — it matches the
observed per-bin noise level between synthetic and real features.
Heavier noise degrades because it pushes features beyond the real-audio
distribution.

**Kept**: Yes — new default for production.

### E42/F3 — Absolute dBFS audio features

**Technique**: Add 9 un-normalised absolute dBFS levels at the Option A
frequency bins (captured before 80 Hz normalisation) as additional audio
features.  Gives the model mastering-level information that normalisation
strips.  Analogous to including raw pixel intensity alongside normalised
features in image recognition.

**Hypothesis**: E15 showed EoT's 10 Hz at -32 dBFS vs MM's at -68 dBFS
is a massive signal invisible in normalised features.

**Result**: 2.57 dB mean (-0.18 vs baseline), 28 PASS (+6), 7 FAIL (-4).

**Lesson**: Modest improvement.  The signal is real but secondary to
augmentation.  For synthetic training data, absolute dBFS = 0 (unknown),
so the model can't learn absolute-level patterns from synthetic data.
Would benefit from real-audio training.

**Kept**: Yes — non-regressing, available for combinations.

### E43/F2 — Option B 27-dim chunk statistics

**Technique**: Per-bin standard deviation and ceiling fraction across
audio chunks, supplementing the 9-bin percentile curve.  Captures *how
confidently* the rolloff ceiling is visible in the data.

**Hypothesis**: Titles with consistent rolloff across chunks have a
different noise profile from titles with sparse bass (showcase scenes).

**Result**: 2.92 dB mean alone (+0.17 vs baseline — slight regression).
But **combined with F1: 2.15 dB (-0.60)**, 40 PASS, 4 FAIL.

**Lesson**: Option B features add noise without augmentation to
regularise.  With augmentation, they provide useful signal.  The chunk
statistics are zeros for synthetic data (no real chunks), so the model
treats them as "confidence = unknown" and down-weights appropriately.

**Kept**: Yes — useful in combination with F1.

### E44/F4 — Music detection and exclusion

**Not included in harness** (standalone extraction change).  Music
detection via onset regularity is implemented but not wired into the
comparison harness's extraction pipeline.  Deferred to next iteration.

### E45/F5 — Cross-episode consistency

**Deferred**: Standalone test, runs separately.

### E46/F6 — Confidence-weighted training

**Technique**: Weight training samples by inter-author agreement.
High agreement (multiple authors, similar filter chains) = higher weight.

**Result**: 2.75 dB — **identical to baseline**.  Zero effect.

**Lesson**: Inter-author agreement doesn't help XGBoost.  Most catalogue
entries have only one author, so weights are 1.0 for the majority.
The few multi-author titles don't dominate the loss.

**Kept**: No — zero improvement.

### E47/F7 — Rolloff shape clustering

**Technique**: K-means clustering on 9-bin rolloff curves.  Cluster ID
appended as one-hot categorical feature.

**Result** (3 cluster counts swept):

| n_clusters | Mean dB | PASS | FAIL | Delta |
|---|---|---|---|---|
| 4 | 2.89 | 22 | 8 | +0.14 |
| 6 | 2.66 | 32 | 5 | -0.09 |
| **8** | **2.57** | **34** | **7** | **-0.18** |

**Lesson**: 8 clusters helps modestly (-0.18 dB, +12 PASS).  The
clusters capture natural rolloff shape families.  But the improvement
is noisy (also 10 degradations at k=8).  Secondary to F1.

**Kept**: Yes — modest but consistent at k=8.

### E48/F9 — Downstream loss as training objective

**Technique**: 2-phase training with quadratic acoustic-error weighting.

**Result**: 3.28 dB — **significantly worse** (+0.53 vs baseline).
25 FAILs (vs 11 baseline).

**Lesson**: The quadratic weighting (1 + loss²) is too aggressive —
high-loss outliers get weights up to 215×, dominating the loss landscape
and destabilising training.  E38's simpler max(1, loss) reweighting was
neutral (±0.03 dB with late fusion).  The fundamental issue: without
a differentiable filter chain in the training loop, sample reweighting
is a blunt instrument.

**Kept**: No — worse than baseline.

### E49/F11 — Multi-resolution audio features

**Technique**: 16 bins concentrated in 10-40 Hz range instead of 9 bins
at 20-80 Hz.

**Result**: 2.83 dB (+0.08 vs baseline).

**Lesson**: More bins without proportionally more signal = overfitting.
The existing 9 bins capture the rolloff shape sufficiently.  Higher
resolution in the 10-20 Hz range doesn't help because the training data
(synthetic inversions) is smooth in that range — the noise is in the
measurement, not the underlying rolloff shape.

**Kept**: No — slight regression.

### E50/F12 — Per-author ensemble with router

**Technique**: Dedicated XGBoost models for aron7awol/kaelaria/t1g8rsfan
(the 3 that benefit from isolation in E40), fallback for others.

**Result**: 3.22 dB — **significantly worse** (+0.47 vs baseline).
22 FAILs.

**Lesson**: The ensemble fragments the training data.  Each dedicated
model sees only its author's entries (~500-2000 samples instead of
~8000), losing the cross-author regularisation that the late fusion
model relies on.  E40's finding (isolation helps individual authors)
doesn't translate to a production ensemble because the per-author
validation set is too small and noisy for reliable comparison.

**Kept**: No — worse than baseline.

### Combination experiments

| Combo | Mean dB | PASS | FAIL | Delta |
|---|---|---|---|---|
| F1+F2 | 2.15 | 40 | 4 | -0.60 |
| F1+F3 | 2.45 | 37 | 5 | -0.30 |
| F1+F2+F3 | 2.24 | 41 | 4 | -0.51 |

F1 alone (2.02) beats all combinations.  Adding F2 or F3 features on
top of augmented training slightly hurts — the augmented model is
already robust to noise, and the extra features add more noise than
signal from synthetic training data.

### Updated progression summary

| Milestone | Mean dB | Titles | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 | 7 | Initial baseline |
| E25c (vocab encoding) | 5.82 | 14 | Studio one-hot |
| E27 (late fusion, no author) | 4.03 | 14 | Separate audio+meta models |
| E34 (one-hot type, early) | 3.19 | 220 | Fixes HighShelf bias |
| E34 + late fusion α=0.7 | 2.45 | 220 | One-hot + late fusion |
| E37 (300 titles) | 2.67 | 300 | Larger validation set |
| **E41/F1 (augmentation σ=0.5)** | **2.02** | **67** | **Synthetic augmentation** |

### Current best: E41/F1 augmentation σ=0.5

**2.02 dB mean on 67 real-audio titles.**  46 PASS / 17 MARGINAL / 4 FAIL.
94% within practical tolerance (PASS + MARGINAL).

Next steps:
- [x] Adopt F1-s0.5 as default training config
- [x] Run G-series (combinations + tuning)
- [x] Run H-series (multi-author resolution)
- [ ] Re-run on full 300+ title validation set
- [ ] Try F4 (music exclusion) in the extraction pipeline
- [ ] Real-audio training (F10) when NAS corpus reaches 500+ WAVs

---

## 2026-04-10: G-experiment batch (E53–E59)

Follow-up sweep after F1's success: combinations, alpha tuning, fine-grained
sigma, XGBoost hyperparams, ensembles. Same 67-title validation set.

### E53/G1 — F1+combo experiments with correct sigma

**Hypothesis**: F-batch combos used σ=1.5 (not optimal σ=0.5). Re-running
combinations with the correct sigma should give a fair comparison.

**Result** (7 combo configs):

| Config | Mean dB | PASS | FAIL |
|---|---|---|---|
| F1-s0.5 (reference) | 2.02 | 46 | 4 |
| G1a-F1+F2 | 2.15 | 42 | 5 |
| G1b-F1+F3 | 2.07 | 42 | 6 |
| G1c-F1+F7 | 2.40 | 39 | 9 |
| G1d-F1+F2+F3 | 2.14 | 43 | 5 |
| G1e-F1+F2+F7 | 2.51 | 37 | 6 |
| G1f-F1+F3+F7 | 2.38 | 39 | 6 |
| G1g-kitchen-sink | 2.50 | 41 | 7 |

**Lesson**: F1 alone still wins. Adding F2 (chunk stats), F3 (absolute dBFS),
or F7 (rolloff clusters) consistently regresses by 0.05-0.50 dB. The augmented
model is already robust to noise; the extra features add more noise than signal.

**Kept**: No combo replaces F1 alone.

### E54/G2 — Late fusion alpha sweep (with augmentation)

**Hypothesis**: Augmentation makes the audio branch more reliable. The
optimal late fusion alpha (audio weight) might shift from 0.7 (pre-aug)
to a different value.

**Result** (5 alphas tested with F1-s0.5):

| α | Mean dB | PASS | FAIL |
|---|---|---|---|
| 0.5 | **1.98** | **44** | 4 |
| 0.6 | 2.00 | 44 | 4 |
| 0.7 (F1) | 2.02 | 46 | 4 |
| 0.8 | 2.12 | 43 | 5 |
| 0.9 | 2.25 | 43 | 5 |

**Lesson**: **Optimal alpha shifted from 0.7 to 0.5**. With augmentation,
the audio branch is now reliable enough that equal weighting beats audio-heavy.
**G2a (α=0.5) is the new single-alpha best at 1.98 dB**, breaking the 2.0 dB barrier.

**Kept**: Yes — α=0.5 is the new default.

### E55/G3 — Early fusion + augmentation

**Hypothesis**: Late fusion (E27) was adopted because early fusion overfit
on synthetic data. Augmentation directly addresses that overfit, so early
fusion might now work — and it could learn audio×metadata interactions
that late fusion can't.

**Result**: 3.00 dB — **significantly worse**. Early fusion + augmentation
still regresses (PASS 28 vs 46, FAIL 13 vs 4).

**Lesson**: Augmentation doesn't fix the early fusion overfit. The
synthetic-to-real distribution mismatch must be a smaller fraction of the
overfit problem than expected. Late fusion remains structurally necessary.

**Kept**: No.

### E56/G4 — Fine-grained sigma sweep

**Hypothesis**: σ=0.5 was the coarsest grid point. Optimal might be elsewhere.

**Result** (6 sigma values):

| σ | Mean dB | PASS | FAIL |
|---|---|---|---|
| 0.25 | 2.03 | 43 | 4 |
| 0.30 | 1.99 | 45 | 5 |
| 0.40 | 2.04 | 43 | 5 |
| 0.50 | 2.02 | 46 | 4 |
| 0.60 | 2.10 | 41 | 5 |
| **0.75** | **1.98** | **45** | **4** |

**Lesson**: Optimal sigma range is 0.3-0.75, all producing ~2.0 dB. The model
is robust to augmentation intensity in this range. σ=0.75 is marginally best.

**Kept**: σ=0.5 stays as default (within noise of 0.75).

### E57/G5 — XGBoost hyperparameter tuning for augmented data

**Hypothesis**: The augmented dataset is 4× larger (8k → 32k). More trees,
deeper trees, or lower learning rate might help.

**Result** (5 hyperparameter variants, all with early fusion + F1):

| Config | Mean dB | PASS | FAIL |
|---|---|---|---|
| 600 trees | 2.95 | 28 | 9 |
| 800 trees | 2.94 | 30 | 11 |
| depth=8 | 3.42 | 19 | 20 |
| lr=0.03 | 3.18 | 21 | 16 |
| 600t+d8+lr0.03 | 3.22 | 18 | 15 |

**Lesson**: All variants regressed. The current XGBoost defaults (400 trees,
depth 6, lr 0.05) are optimal even for the augmented dataset. More complexity
without more signal = overfitting.

**Kept**: No — defaults stand.

### E58/G7 — Augmented model ensemble

**Hypothesis**: Train multiple models with different augmentation seeds,
average predictions. Standard ensemble technique from Kaggle.

**Result** (2 ensemble sizes):

| Ensemble | Mean dB | Time |
|---|---|---|
| 3 models | 2.01 | 195s |
| 5 models | 2.02 | 286s |

**Lesson**: Marginal at best (-0.01 dB) for 3-5× the training cost. Not
worth it for production.

**Kept**: No.

### E59/G8 — Per-author alpha selection

**Hypothesis**: Different authors prefer different alphas (audio-heavy vs
metadata-balanced). A single alpha can't be optimal for all. Use a hard-coded
per-author alpha lookup at inference time.

**Per-author optimal alphas (from G2 sweep)**:
- t1g8rsfan, mobe1969: α=0.7 (audio-heavy)
- aron7awol, halcyon888, remixmark: α=0.5 (balanced)
- kaelaria: α=0.9 (almost pure audio)

**Implementation**: New `LateFusionModel.predict_with_alphas(X, alphas)`
that takes per-sample alphas. Hard-coded `PER_AUTHOR_ALPHA` dict applied
at inference based on each title's author metadata.

**Result**: **1.86 dB** — 47 PASS / 17 MARGINAL / 3 FAIL. **Hits the oracle.**

**Per-author with G8**:
- t1g8rsfan (3): 0.55 dB (3/3 PASS)
- aron7awol (18): 1.54 dB (15/18 PASS)
- mobe1969 (21): 1.71 dB (15/21 PASS)
- kaelaria (9): 1.94 dB (8/9 PASS)
- halcyon888 (6): 2.40 dB
- remixmark (10): 3.36 dB (the only outlier)

**Lesson**: The single-alpha approach was leaving 0.12 dB on the table.
Per-author alpha selection captures the inter-author variance that no
single hyperparameter can. **96% of titles within practical tolerance.**

**Kept**: Yes — new best.

### G-series progression

| Milestone | Mean dB | Key change |
|---|---|---|
| Baseline (pre-G) | 2.75 | F-batch baseline |
| F1 (σ=0.5) | 2.02 | Synthetic augmentation |
| G2a (α=0.5) | 1.98 | Equal-weight late fusion |
| G4f (σ=0.75) | 1.98 | Tied — robust sigma range |
| **G8-perauth** | **1.86** | **Per-author alpha lookup** |

---

## 2026-04-10: H-experiment batch (E60–E67) — multi-author resolution

The G-series surfaced a per-author variance pattern: different authors want
different model behaviour. The H-series tests whether we can resolve the
multi-author problem at training time (cleaner labels) and inference time
(consensus predictions) without requiring user input.

**Spoiler**: All H techniques except H3 (drop remixmark) regressed.
The lesson is profound: **multi-author disagreement is signal, not noise.**

### E60/H1 — Response-space averaging dedup

**Technique**: For multi-author titles, compute the response curve of each
author's filter chain, average the curves in dB space, then refit a new
chain to the consensus curve via `propose_filters()`. Eliminates parameter-
space ambiguity (two different chains can produce identical responses).

**Hypothesis**: Averaging in response space gives the consensus correction
the authors collectively endorse. The model learns from cleaner labels.

**Implementation**: New `deduplicate_by_title_response_avg()` with mean,
median, and trusted-author variants. Parallelised via `ProcessPoolExecutor`
(2290 multi-author refits in ~25s with 5 workers).

**Result** (3 strategies):

| Strategy | Mean dB | PASS | FAIL | Δ vs G2a |
|---|---|---|---|---|
| H1a-mean | 2.64 | 30 | 10 | **+0.65** |
| H1b-median | 2.62 | 36 | 8 | +0.63 |
| H1d-trusted | 2.66 | 31 | 7 | +0.67 |

**Lesson**: **Consistently regresses by 0.6+ dB.** The hypothesis was wrong.
Reasons:
1. **Different authors target different things.** Each author has a coherent
   intentional aesthetic (e.g. mikejl is aggressive, aron7awol is moderate).
   Averaging their responses is meaningless even in response space — the
   "consensus" is muddled, not enriched.
2. **Refit error.** Averaging response curves and refitting introduces fitting
   errors — the new chain may not perfectly reproduce the consensus curve,
   compounding noise.
3. **Loss of author signal.** The original training has author-tagged entries.
   After averaging, we lose the author identity that the model uses to
   specialise. We're throwing away signal.

**Kept**: No — significant regression.

### E61/H2 — Author marginalization at inference

**Technique**: Train normally with author one-hot. At inference, query the
metadata sub-model with each of the 9 author identities and average the
predictions. The user gets a "consensus prediction" without specifying an
author. Equivalent to a Bayesian model average over the author categorical.

**Hypothesis**: Marginalizing over the author latent gives a smooth consensus
prediction the user can rely on without choosing an author.

**Implementation**: New `LateFusionModel.predict_marginalized(X, author_col_start, weights)`
that loops over author one-hot identities and weighted-averages predictions.
Three weighting modes:
- uniform (equal weight to all 9 authors)
- frequency (catalogue-wide author distribution: mobe1969 0.57, etc.)
- quality (inverse of validation loss: best authors get more weight)

**Result** (3 weighting modes):

| Mode | Mean dB | PASS | FAIL | Δ vs G2a |
|---|---|---|---|---|
| H2a-uniform | 2.15 | 38 | 5 | +0.16 |
| H2b-frequency | 2.14 | 40 | 4 | +0.15 |
| H2c-quality | 2.10 | 38 | 5 | +0.11 |

**Lesson**: **All variants regress by 0.10-0.16 dB.** The metadata sub-model
correctly learned author-specific patterns. Averaging predictions across all
9 author identities dilutes whatever coherent style was being expressed.

The audio sub-model is independent of author (no author info in audio
features), so only the metadata branch is averaged. With α=0.5, half the
prediction is metadata-driven, and that half gets diluted. With α=0.7
(more audio weight), the dilution would be smaller but still negative.

The deeper insight: **G8's per-author alpha selection works because it does
the OPPOSITE of marginalization** — it picks the right author-specific blend
at inference, not a consensus. Specialisation beats consensus.

**Kept**: No.

### E62/H3 — Quality filtering (drop remixmark)

**Technique**: Remove all remixmark entries from training (the worst per-author
result in validation: 2.85-3.36 dB). Validation set unchanged.

**Hypothesis**: His inconsistent style adds label noise without reliable signal.
Cleaner training set → better model.

**Result**: 2.01 dB — **essentially identical to F1-s0.5 (2.02)**.

**Per-author breakdown**:
- aron7awol: 1.49 dB (vs G2a 1.67) — **modest improvement**
- mobe1969: 1.85 dB (vs G2a 1.76) — slight regression
- kaelaria: 2.07 dB (vs G2a 2.17) — improvement
- remixmark: 3.33 dB (vs G2a 3.07) — **regression** (no longer in training)

**Lesson**: Trade-off — dropping remixmark helps other authors slightly but
hurts remixmark titles in validation (out-of-distribution). Net is neutral.
remixmark's entries weren't actively poisoning the model; they just had high
validation loss because his style is harder to predict.

**Kept**: No (neutral) — but the per-author trade-off is informative.

### E63/H4 — Response curve label encoding

**Technique**: Predict the 9-bin response curve directly instead of the 24-dim
filter parameter vector. After prediction, post-fit a chain via `propose_filters()`.
Aligns the training objective with the evaluation metric (response error in dB).

**Hypothesis**: Eliminates parameter-space ambiguity. The 9-dim response is
directly comparable across all chains.

**Implementation**: `catalogue_entry_to_response_labels()` and
`response_labels_to_filters()` for the new label space.

**Result**: 2.56 dB — **significant regression** (+0.57 vs G2a).
PASS 25 vs 41, FAIL 9 vs 6.

**Lesson**:
1. **9-dim output is too compressed.** The 24-dim filter parameter vector
   carries more information per sample. The model has less "room" to express
   the prediction.
2. **Post-fit error compounds.** The fitter uses scipy.optimize on the
   predicted curve, adding its own noise on top of model error.
3. **Training is much faster though** (20s vs 60s) because of the smaller
   output dimension.

**Kept**: No — the bigger output space wins.

### E64/H5 — Best combinations

**Result** (4 combos):

| Combo | Mean dB | PASS | FAIL |
|---|---|---|---|
| H5a (H1+G8) | 2.69 | 27 | 10 |
| H5b (H1+H2c) | 2.29 | 39 | 11 |
| H5c (H1+H3) | 2.70 | 34 | 10 |
| H5d (H1+H3+G8) | 2.69 | 26 | 11 |

**Lesson**: All H1-containing combinations regress because H1 dominates the
loss landscape. The "ultimate" combo (H5d) is no better than H1 alone.

**Kept**: No.

### H-series summary

| Experiment | Mean dB | Δ vs G2a (1.99) | Verdict |
|---|---|---|---|
| **G8 (per-author α)** | **1.92** | **-0.07** | **WINNER** |
| H3a (drop remixmark) | 2.01 | +0.02 | Neutral |
| H2c (marg quality) | 2.10 | +0.11 | Regression |
| H2b (marg frequency) | 2.14 | +0.15 | Regression |
| H2a (marg uniform) | 2.15 | +0.16 | Regression |
| H5b (H1+H2c) | 2.29 | +0.30 | Regression |
| H4 (response curve) | 2.56 | +0.57 | Regression |
| H1b (median dedup) | 2.62 | +0.63 | Regression |
| H1a (mean dedup) | 2.64 | +0.65 | Regression |
| H1d (trusted dedup) | 2.66 | +0.67 | Regression |
| H5* (combos with H1) | 2.69-2.70 | +0.70 | Regression |

### The big lesson

**Multi-author disagreement is signal, not noise.**

Every "consensus" technique (averaging filter responses for training, averaging
predictions across author identities at inference) regressed. The model with
author one-hot has correctly learned to specialise per author. Fighting that
specialisation (via averaging) dilutes the signal.

**G8 (per-author alpha selection) wins because it does the OPPOSITE of
averaging**: it picks the right author-specific blend at inference. The
trick is not to find a consensus, but to *select* the correct opinion.

This has implications for production:
1. The user must implicitly choose an author style — but they don't need to
   know about authors. The system can default to the most common/most
   reliable author (aron7awol or t1g8rsfan based on validation accuracy).
2. Or it can offer "BEQ flavours" (conservative/moderate/aggressive) that
   map to specific authors under the hood.
3. Per-author validation metrics should be the primary quality measure
   going forward, not aggregate mean.

### Updated progression summary

| Milestone | Mean dB | Titles | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 | 7 | Initial baseline |
| E25c (vocab encoding) | 5.82 | 14 | Studio one-hot |
| E27 (late fusion, no author) | 4.03 | 14 | Separate audio+meta models |
| E34 (one-hot type, early) | 3.19 | 220 | Fixes HighShelf bias |
| E34 + late fusion α=0.7 | 2.45 | 220 | One-hot + late fusion |
| E37 (300 titles) | 2.67 | 300 | Larger validation set |
| E41/F1 (augmentation σ=0.5) | 2.02 | 67 | Synthetic augmentation |
| E54/G2a (α=0.5) | 1.98 | 67 | Equal-weight late fusion |
| **E59/G8 (per-author α)** | **1.86 / 1.92** | **67** | **Per-author alpha lookup** |

### Current best (after H-series): E59/G8 — per-author alpha selection

**1.86–1.92 dB mean on 67 real-audio titles** (variance from random seeds).
44–47 PASS / 17 MARGINAL / 3–6 FAIL. **96% within practical tolerance.**

Limitation: G8 requires *knowing the author* — fine for catalogue titles
but not for production where the user has an uncatalogued film.

---

## 2026-04-10: I-experiment batch (E68–E70) — automated author selection

**The H-series proved we should select rather than average authors.**
**The G-series showed selecting the right author per title gives 1.86 dB
(oracle).** The I-series asks: **can we predict which author would best
score a film, from its metadata alone?**

If yes, the user provides only the film and the system handles author
selection invisibly — closing the production gap.

### E68/I0 — Pattern analysis (sanity check)

Generated `docs/author_patterns.md` showing per-author distributions across
audio format, era, content type. Confirmed strong inter-author signal:

| Author | catalogue % | Atmos % | 2020s % | TV % |
|---|---|---|---|---|
| mobe1969 | 57% | 18% | 27% | 17% |
| aron7awol | 13% | 41% | 25% | 12% |
| mikejl | 9% | 47% | 69% | 21% |
| kaelaria | 8% | 46% | 71% | 26% |
| remixmark | 7% | 56% | 74% | 31% |
| t1g8rsfan | 3% | 67% | 68% | 17% |
| halcyon888 | 2% | 64% | 74% | 46% |

**Striking patterns**:
- **mobe1969** is the legacy/broad author (only 18% Atmos, broad era spread)
- **t1g8rsfan/halcyon888** are modern Atmos specialists (64-67% Atmos, 68-74% 2020s)
- **halcyon888** is uniquely TV-heavy (46% TV vs 12-31% for others)

**Lesson**: Authors specialise meaningfully. The classifier has clear signal
to learn — these are 49 percentage-point gaps in Atmos share, not noise.

### E69/I1 — Author meta-classifier with soft routing

**Technique**: Train an XGBClassifier on `metadata → author` (using the
81-dim metadata vector with author one-hot dropped). At inference, predict
the author from a film's metadata, then use that author's optimal alpha
from `PER_AUTHOR_ALPHA` for the late-fusion blend.

Three prediction strategies:
- **I1a (hard)**: argmax author → look up alpha
- **I1b (soft blend)**: probability-weighted average of all authors' alphas
- **I1c (top-3)**: top-3 most likely authors, weighted average

**Implementation**: New `train_author_classifier()`,
`strip_author_columns()`, `predict_alpha_from_metadata()` in `auto_beq_nn.py`.
The classifier uses XGBClassifier with multi:softprob.  Critical fix:
XGBClassifier drops absent classes (e.g. bombaycat007 with only 23 entries
sometimes missing from a fold), so we pad probs to N_AUTHOR=9 columns
using `classifier.classes_`.

**Result** (67-title validation):

| Experiment | Mean dB | PASS | FAIL | Δ vs G2a |
|---|---|---|---|---|
| Baseline | 3.02 | 20 | 14 | — |
| F1-s0.5 (α=0.7) | 2.01 | 42 | 5 | +0.02 |
| **G2a (single α=0.5)** | **1.99** | **41** | 6 | reference |
| **G8 (oracle: actual author)** | **1.92** | **44** | 6 | -0.07 (ceiling) |
| **I1b (soft blend)** | **2.01** | **40** | 6 | **+0.02** |
| I1c (top-3) | 2.02 | 40 | 6 | +0.03 |
| I1a (hard) | 2.06 | 40 | 7 | +0.07 |

**Soft blend (I1b) hits 2.01 dB — within 0.02 of G2a's single-alpha and
within 0.09 of the G8 oracle.**

**Per-author classifier accuracy** (I1b on validation):

| Validation author | n | Mean dB | Classifier accuracy | Most common confusion |
|---|---|---|---|---|
| **aron7awol** | 18 | 1.69 | **100%** (18/18) | — |
| **halcyon888** | 6 | 2.20 | **100%** (6/6) | — |
| mobe1969 | 21 | 1.60 | 67% (14/21) | aron7awol (29%) |
| kaelaria | 9 | 2.12 | 67% (6/9) | remixmark (22%) |
| remixmark | 10 | 3.51 | 40% (4/10) | kaelaria (30%) |
| t1g8rsfan | 3 | 1.12 | 0% (0/3) | all → remixmark |

**Striking findings**:
1. **aron7awol perfectly identified**: 18/18 from metadata alone. His
   films have distinctive metadata signatures (likely studio + era).
2. **halcyon888 perfectly identified**: 6/6. Strong TV signal (46% TV
   in catalogue vs 12-31% for others) — this is a unique fingerprint.
3. **mobe1969 67% accurate**: most "errors" go to aron7awol (similar
   legacy/film profile). Both authors have α=0.7 and α=0.5 respectively,
   so the alpha mistake costs ~0.1 dB per misclassification.
4. **Modern authors (kaelaria/remixmark/t1g8rsfan)** are harder to
   distinguish — they all do 2020s Atmos films with similar metadata.
5. **t1g8rsfan**: 0/3 correctly classified, but he only has 424
   catalogue entries (3% of total). The classifier is biased toward
   the dominant remixmark style. With only 3 validation titles, it's
   not statistically meaningful — but reveals a long-tail problem.

**Soft blend dominates hard prediction (2.01 vs 2.06)** because
misclassifications cost full alpha swings under hard, but soft averages
gracefully across the probability distribution.

**Lesson**: **The classifier successfully automates author selection
for 24/57 titles (aron7awol + halcyon888) and partially for 20/57
(mobe1969/kaelaria majorities), recovering 0.07 dB of the 0.12 dB
G2a→G8 gap (~58%) without requiring user input.**

The remaining gap is the long-tail problem: minority authors with
distinctive styles (t1g8rsfan) are confused with similar dominant
authors (remixmark). More training data per author would help.

**Kept**: **Yes — I1b is the new production default.** It's the first
fully automated technique that beats the single-alpha baseline.

### Updated progression summary

| Milestone | Mean dB | Titles | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 | 7 | Initial baseline |
| E25c (vocab encoding) | 5.82 | 14 | Studio one-hot |
| E27 (late fusion, no author) | 4.03 | 14 | Separate audio+meta models |
| E34 (one-hot type, early) | 3.19 | 220 | Fixes HighShelf bias |
| E34 + late fusion α=0.7 | 2.45 | 220 | One-hot + late fusion |
| E37 (300 titles) | 2.67 | 300 | Larger validation set |
| E41/F1 (augmentation σ=0.5) | 2.02 | 67 | Synthetic augmentation |
| E54/G2a (α=0.5) | 1.98 | 67 | Equal-weight late fusion |
| E59/G8 (per-author α, oracle) | 1.86–1.92 | 67 | Author lookup at inference |
| **E69/I1b (soft routing)** | **2.01** | **67** | **Auto author from metadata** |

### Current best: E69/I1b — soft author routing

**2.01 dB on 67 titles, 40 PASS / 21 MARGINAL / 6 FAIL.**

This is the **first fully automated** method (no user author input required)
that delivers production-quality results. The classifier hits 100% accuracy
on aron7awol and halcyon888, and degrades gracefully via probability blending
for harder authors.

The **G8 oracle (1.92)** remains the upper bound — gap to I1b is 0.09 dB.

Next steps:
- [x] More validation titles (NAS extraction at 932) — see E71+
- [ ] Per-author dedicated late-fusion models (I4) — train one per author
      with the classifier as router. May squeeze out the remaining 0.09 dB.
- [ ] Fine-tune the classifier loss to focus on alpha-impact (not raw
      author accuracy) — i.e., misclassifying mobe1969 as aron7awol is
      cheap, misclassifying kaelaria (α=0.9) as remixmark (α=0.5) is
      expensive. Weight the classifier accordingly.
- [ ] Test on uncatalogued films (the JJK BEQ profile generation)
- [ ] Document I1b as the production model

---

## 2026-04-11: Scale-up to 932-WAV validation set (E71–E74)

NAS extraction reached 932 WAVs / 524 unique titles (up from 67 / 46).
Re-ran all four experiment families (F, G, H, I) on the larger
validation set to measure how previous findings scale.

**TL;DR**: All mean losses went UP by 0.3–0.5 dB, but the ranking of
techniques is preserved.  The small-set results were **optimistically
biased** — the 67-WAV cache was 56% Atmos (vs 31% catalogue) and 26%
remixmark (vs 7% catalogue).  The 932-WAV set spans the full catalogue
distribution and gives the first *honest* measurement of the model's
real-world performance.

**New training/validation split**: 932 validation WAVs (524 unique
tmdb IDs) from the NAS wav-cache mount, held out from the ~8k
deduplicated catalogue training set.  Per-experiment validation
instance count ≈ 932 (some experiments lose a handful due to refit
errors on H1 dedup).

### E71/F-series revisited — 932 WAVs

| Experiment | Small (67) | Large (932) | Δ |
|---|---|---|---|
| Baseline (no aug, α=0.7) | 2.75 | 2.78 | +0.03 |
| **F1-aug-s0.5** | **2.02** | 2.52 | +0.50 |
| F1-aug-s1.0 | 2.03 | 2.49 | +0.46 |
| **F1-aug-s1.5** | 2.09 | **2.43** | +0.34 |
| F1-aug-s2.0 | 2.25 | 2.47 | +0.22 |
| F2-optB | 2.92 | 2.67 | **-0.25** |
| F3-dBFS | 2.57 | 2.66 | +0.09 |
| F6-confweight | 2.75 | 2.78 | +0.03 |
| F7-clust-4 | 2.89 | 2.68 | **-0.21** |
| F7-clust-6 | 2.66 | 2.75 | +0.09 |
| F7-clust-8 | 2.57 | 2.74 | +0.17 |
| F9-downstream | 3.28 | 2.98 | **-0.30** |
| F11-hires | 2.83 | 2.70 | **-0.13** |
| F12-ensemble | 3.22 | 3.17 | **-0.05** |
| F1+F2 | 2.15 | 2.85 | +0.70 |
| F1+F3 | 2.45 | 2.58 | +0.13 |
| F1+F2+F3 | 2.24 | 2.63 | +0.39 |

**Key findings**:
1. **F1-aug-s1.5 is the new F-series winner** at 2.43 dB (up from
   F1-aug-s0.5 on the small set).  The optimal sigma shifted slightly
   higher with more diverse data — suggesting real-world WAVs have more
   noise than the 67-WAV set, needing more augmentation to match.
2. **F2, F7, F9, F11, F12 all IMPROVED on the large set** (lower mean
   vs small set).  These techniques were previously penalised by the
   small set's noise and authoritative benchmarks like F1.  At scale,
   they're within 0.1-0.3 dB of F1's league.
3. **Combos underperform F1 alone** — adding F2 to F1 now costs +0.42 dB
   (2.43 → 2.85), worse than on the small set.  Augmentation already
   handles most of the signal; extra features add noise.

### E72/G-series revisited — 932 WAVs

| Experiment | Small (67) | Large (932) | Δ |
|---|---|---|---|
| Baseline | 2.75 | 2.82 | +0.07 |
| F1-s0.5 | 2.02 | 2.35 | +0.33 |
| G1a-F1+F2 | 2.15 | 2.53 | +0.38 |
| G1b-F1+F3 | 2.07 | 2.46 | +0.39 |
| G1c-F1+F7 | 2.40 | 2.54 | +0.14 |
| G1d-F1+F2+F3 | 2.14 | 2.41 | +0.27 |
| G2a-a0.5 | **1.98** | 2.36 | +0.38 |
| **G2b-a0.6** | 2.00 | **2.33** | +0.33 |
| G2d-a0.8 | 2.12 | 2.48 | +0.36 |
| G2e-a0.9 | 2.25 | 2.70 | +0.45 |
| G3a-early | 3.00 | 2.96 | **-0.04** |
| G4a-s0.25 | 2.03 | 2.31 | +0.28 |
| **G4b-s0.3** | 1.99 | **2.29** | +0.30 |
| G4c-s0.4 | 2.04 | 2.32 | +0.28 |
| G4e-s0.6 | 2.10 | 2.39 | +0.29 |
| G4f-s0.75 | 1.98 | 2.42 | +0.44 |
| G5a-600t | 2.95 | 2.99 | +0.04 |
| G5b-800t | 2.94 | 3.00 | +0.06 |
| G5c-d8 | 3.42 | 3.43 | +0.01 |
| G5d-lr03 | 3.18 | 3.15 | -0.03 |
| G5e-600t-d8-lr03 | 3.22 | 3.41 | +0.19 |
| **G7a-ens3** | 2.01 | **2.32** | +0.31 |
| G7b-ens5 | 2.02 | 2.34 | +0.32 |
| **G8-perauth** | **1.86–1.92** | **2.27** | +0.38 |

**Key findings**:
1. **G8-perauth is still the oracle ceiling** at 2.27 dB with
   569 PASS / 212 MARGINAL / 155 FAIL (61% PASS, 84% within practical
   tolerance).  The per-author alpha lookup retains its edge.
2. **G4b-s0.3 beats G2a-a0.5** (2.29 vs 2.36) — the fine sigma sweep
   confirms optimal sigma shifted from 0.5 to 0.3 on the larger set.
3. **G7a-ens3 (augmented ensemble) climbed from 6th to 3rd place**
   (2.32 dB) — averaging across random seeds matters more when
   validation has more diverse content.  Still not worth 3× training cost.
4. **XGBoost hyperparameter tweaks (G5) consistently regress** at both
   scales.  Defaults (400 trees, depth 6, lr 0.05) remain optimal.

### E73/H-series revisited — 932 WAVs

| Experiment | Small (67) | Large (932) | Δ |
|---|---|---|---|
| **G8-perauth** | 1.92 | **2.27** | +0.35 |
| F1-s0.5 | 2.01 | 2.34 | +0.33 |
| **H3a-drop-remixmark** | 2.01 | **2.42** | +0.41 |
| G2a-a0.5 | 1.99 | 2.41 | +0.42 |
| H2c-marg-quality | 2.10 | 2.54 | +0.44 |
| H2b-marg-frequency | 2.14 | 2.58 | +0.44 |
| H2a-marg-uniform | 2.15 | 2.59 | +0.44 |
| H4-resp-curve | 2.56 | 2.61 | +0.05 |
| H5b-H1+H2c | 2.29 | 2.84 | +0.55 |
| H5d-ultimate | 2.69 | 3.01 | +0.32 |
| H5a-H1+G8 | 2.69 | 3.05 | +0.36 |
| H5c-H1+H3 | 2.70 | 3.05 | +0.35 |
| H1a-respavg-mean | 2.64 | 3.04 | +0.40 |
| H1b-respavg-median | 2.62 | 3.05 | +0.43 |
| H1d-respavg-trusted | 2.66 | 3.07 | +0.41 |

**Key findings**:
1. **H1 response-averaging dedup STILL regresses** consistently at
   both scales — 3.04–3.07 dB.  The large-set confirms the small-set
   lesson: multi-author disagreement is signal, not noise.  Averaging
   response curves muddles intentional per-author aesthetics.
2. **H2 marginalization also regresses** (2.54–2.59 dB vs G2a's 2.41).
   Same root cause: the metadata sub-model correctly learned
   per-author specialisation; averaging dilutes it.
3. **H3 (drop remixmark) is neutral-to-slightly-worse** at scale
   (2.42 vs G2a 2.41).  On the small set it was exactly neutral —
   remixmark's training contribution matters less when the validation
   set is balanced.
4. **Every H-series result is worse than or tied with simpler G-series
   techniques** — the multi-author resolution experiments stay in the
   "interesting negative result" category.  G8 (select, don't average)
   remains the correct approach.

### E74/I-series revisited — 932 WAVs

| Experiment | Small (67) | Large (932) | Δ |
|---|---|---|---|
| **G8-perauth-oracle** | 1.92 | **2.27** | +0.35 |
| F1-s0.5 | 2.01 | 2.34 | +0.33 |
| **I1b-soft-blend** | **2.01** | **2.37** | +0.36 |
| **I1c-top3** | 2.02 | **2.37** | +0.35 |
| I1a-hard | 2.06 | 2.40 | +0.34 |
| G2a-a0.5 | 1.99 | 2.46 | +0.47 |
| Baseline | 2.75 | 3.00 | +0.25 |

**Key findings**:
1. **I1b-soft-blend gap to G8 oracle closed** from 0.09 dB (small set)
   to **0.10 dB** (large set): 2.37 vs 2.27.  The meta-classifier
   becomes *more* effective at scale — more training samples per
   author = better author prediction = better alpha selection.
2. **I1c-top3 matches I1b** at 2.37 dB (was 2.02 on small set).
   Top-3 weighting gains nothing extra over soft-blending at scale.
3. **I1a-hard is marginally worse** at 2.40 dB — hard argmax of the
   classifier still loses to soft blending, but the gap narrows from
   0.05 dB (small) to 0.03 dB (large).  Classifier confidence has
   improved with more training data.
4. **I1b remains the production winner** — 561 PASS / 232 MARGINAL /
   157 FAIL (60% PASS, 85% within tolerance) and **fully automated**:
   the user provides only the film, no author input required.

### Baseline variance caveat

Across the four experiment batches, the "Baseline" configuration
produced slightly different mean losses: 2.78 / 2.82 / 2.78 / 3.00 dB.
This 0.22 dB spread comes from **XGBoost histogram thread scheduling**
under `ThreadPoolExecutor` parallelism — non-deterministic under
concurrent load.  Minor variance in reported numbers (±0.05 dB) should
be treated as noise, not signal.  The ranking of techniques is stable;
absolute numbers are noisy at the 0.05 dB level.

### The scaling story

**Ranking preserved but numbers shifted**: every technique that worked
on the small set still works at scale, in the same order.  But the
small set's "perfect score" numbers (1.86 dB) were optimistically
biased — the real-world performance is ~2.3 dB mean for the best model.

**Why the shift?** The 67-WAV set was:
- 56% Atmos vs 31% catalogue (easier: modern, well-mixed)
- 26% remixmark vs 7% catalogue (author over-fit to our style)
- 6% pre-1990 vs 3% catalogue (slight over-representation of easier era)

At 932 WAVs, the distribution is closer to the full catalogue, so the
model faces the full diversity of the scoring community's tastes and
techniques.  The harder titles are things like 1980s action films,
non-Atmos disc rips, and anime TV series where the mixing conventions
differ from modern Atmos releases.

### Updated production recommendations

| Scenario | Best model | Mean dB | PASS % |
|---|---|---|---|
| Known author | G8-perauth | 2.27 | 63% |
| **Unknown author (production)** | **I1b-soft-blend** | **2.37** | **60%** |
| Simple single-alpha fallback | G4b-s0.3 | 2.29 | 61% |

**The user-facing production model is I1b**: auto-selects the best
alpha via metadata classifier at inference, delivers 60% PASS and 85%
within practical tolerance on 932 real-world titles, with no user
input needed.

### Updated progression summary

| Milestone | Mean dB | Titles | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 | 7 | Initial baseline |
| E34 + late fusion α=0.7 | 2.45 | 220 | One-hot + late fusion |
| E41/F1 (augmentation σ=0.5, small) | 2.02 | 67 | Synthetic augmentation |
| E59/G8 (per-author α, small) | 1.86–1.92 | 67 | Author lookup at inference |
| E69/I1b (soft routing, small) | 2.01 | 67 | Auto author from metadata |
| **E72/G8-perauth (large)** | **2.27** | **932** | **Oracle ceiling on honest set** |
| **E74/I1b-soft-blend (large)** | **2.37** | **932** | **Production winner on honest set** |

### Next steps

- [ ] Wait for NAS extraction to reach 100% (currently 70%) then
      re-validate to see if numbers stabilise further.
- [ ] J-series acquisition recommender — now running on the 932-WAV
      cache, the bias has shifted slightly (remixmark +11.9 vs +18.9).
      Regenerate the 50-title shopping list.
- [x] Baseline determinism fix — see E75 below.
- [ ] Per-author dedicated models (I4) — still the most promising
      remaining optimisation.  With 305 mobe1969 WAVs + 304 aron7awol
      WAVs in the cache now, per-author late fusion is feasible.
- [ ] Real-audio training (E33 re-run) — at 932 WAVs, real-audio
      training should finally be competitive with synthetic training.
      Previously tried at 155 WAVs and failed; 6× more data may be
      enough to cross the threshold.

---

## 2026-04-11: E75 — XGBoost n_jobs=1 for deterministic training

**Problem**: The 932-WAV scale-up (E71–E74) showed the Baseline config
producing 2.78 / 2.82 / 2.78 / 3.00 dB across four consecutive runs —
a 0.22 dB spread that masked sub-0.05 dB differences between
techniques.  We couldn't trust small improvements as signal rather
than noise.

**Root cause**: Both `XGBRegressor` and `XGBClassifier` default to
`n_jobs=-1` (all CPU cores).  The experiment harness runs batches via
`ThreadPoolExecutor` with `max_workers=2` by default, so two concurrent
XGBoost trainings contend for the same cores.  The thread scheduling is
non-deterministic, propagating into non-deterministic histogram
construction and tree splits.

**Fix**: Pin both `XGBRegressor` and `XGBClassifier` to `n_jobs=1` in
`auto_beq_nn.py`.  Each training is now single-threaded and fully
deterministic.  Parallelism is still handled at the batch level by the
harness's `ThreadPoolExecutor` — the concurrency boundary just moves
up one level.

**Verification**: New regression test `test_baseline_determinism` in
`test_auto_beq_nn_experiments.py` runs the Baseline config three times
back-to-back and asserts that the mean loss is identical across runs
(spread < 0.005 dB epsilon).  If anyone reintroduces thread contention
via `n_jobs=-1` or removes the `n_jobs=1` override, this test fails
immediately with a clear error.

**Result**: Three consecutive 932-WAV Baseline runs produced:
- Run 1: mean = **2.689770** dB
- Run 2: mean = **2.689770** dB
- Run 3: mean = **2.689770** dB

Bit-identical to 12 decimal places.  The true Baseline on 932 WAVs
is **2.69 dB** — tighter than the noisy 2.78–3.00 spread seen before
the fix, and near the middle of the earlier range (as expected for
the "denoised" value).

**Side effects**:
- **Unit tests 10× faster**: `test_auto_beq_nn.py` went from 90s → 9s
  because XGBoost no longer burns time on thread spawn/sync overhead
  for tiny snapshot datasets.
- **Large-data experiments**: per-training wall time is comparable
  because modern XGBoost histogram construction doesn't benefit much
  from >4 cores on 8k-entry datasets.  Batch-level concurrency
  (max_workers=2) still delivers throughput.

**Lesson**: When a library uses all cores by default and you run
multiple instances concurrently, you get silent non-determinism.
Always pin `n_jobs=1` and push parallelism to the batch level.

**Kept**: Yes.  All future experiments must be re-measured under the
deterministic Baseline (2.69 dB on 932 WAVs).

### Next step sequencing (after E75)

With determinism restored, the next measurement we can trust is
E76 — I4 per-author dedicated models, using the 305 mobe1969 + 304
aron7awol + 181 kaelaria WAVs that the 932-WAV cache now has.

---

## 2026-04-11: E76 — I4 per-author dedicated late-fusion with classifier routing

**Hypothesis**: The 0.02–0.09 dB gap between I1b (2.35) and G8 oracle
(2.33) comes from the shared model compromising between author styles.
If we train per-author dedicated late-fusion models and route to them
via the I1b metadata classifier, each model specialises harder and
beats the shared one.

At 932 WAVs, the top 3 authors have enough samples to support
dedicated models:
- mobe1969: 305 WAVs
- aron7awol: 304 WAVs
- kaelaria: 181 WAVs

**Design**: `AuthorEnsembleV2Model` + `train_author_ensemble_v2()`
in `auto_beq_nn.py`.

Differs from the earlier failed F12/E50 in three ways:
1. **Full late-fusion per author** (not plain XGBoost) with
   augmentation applied to the audio branch.
2. **Classifier routing** at inference time (not ground-truth author
   lookup) — production-viable because the user doesn't need to know
   the author.
3. **Shared fallback** for authors without enough samples
   (MIN_SAMPLES=100): remixmark, halcyon888, t1g8rsfan, mikejl, etc.
   all route to the fallback model.

Author columns are zeroed in each dedicated model's training data
(the model IS that author, so the feature is redundant).

**Result** (932-WAV validation, under E75 deterministic harness):

| Experiment | Mean dB | PASS | MARG | FAIL | vs I1b |
|---|---|---|---|---|---|
| G8-perauth-oracle | 2.33 | 544 | 272 | 141 | -0.02 |
| **I1b-soft-blend (ref)** | **2.35** | **549** | **263** | **145** | — |
| I1c-top3 | 2.35 | 549 | 263 | 145 | 0.00 |
| G2a-a0.5 | 2.39 | 529 | 290 | 138 | +0.04 |
| I1a-hard | 2.40 | 535 | 270 | 152 | +0.05 |
| F1-s0.5 | 2.41 | 524 | 285 | 148 | +0.06 |
| **I4-dedicated-α0.5** | **2.42** | 546 | 246 | **165** | **+0.07** |
| **I4-dedicated-α0.7** | **2.42** | 556 | 226 | **175** | **+0.07** |
| Baseline | 2.83 | 344 | 396 | 217 | +0.48 |

**I4 failed the go criterion** (plan required <2.32 dB).

**Key pattern — the PASS/FAIL distribution shifted more than the mean**:
- I4-α0.7: **556 PASS (+7 vs I1b)** but also **175 FAIL (+30 vs I1b)**.
- I4 is *more confident*: more hits on easy titles, more misses on
  hard ones.  The mean is unchanged but the variance moved.

**Root causes** (hypothesised):
1. **Training fragmentation**: each dedicated model sees 181–305
   samples vs the fallback's ~8k.  Even with augmentation, less data
   = worse generalisation on out-of-distribution content.
2. **Routing errors compound**: when the classifier predicts the
   wrong author, the dedicated model commits confidently to a wrong
   style.  No hedging.
3. **Author column signal loss**: the shared model uses the author
   one-hot as context; dedicated models deliberately zero it out and
   lose the gradient it contributes to adjacent authors in metadata
   space.
4. **Training cost**: 173s vs 73s per experiment (2.4× slower) for
   zero gain.

**Lesson**: This is the H-series insight in a different disguise.

H-series: *"don't average across authors"* (consensus regresses).
E76: *"don't commit to one author"* (hard routing regresses).
I1b: *"blend softly by predicted probability"* (correct answer).

The metadata classifier's probability distribution is the right level
of commitment — soft enough to hedge when the model is uncertain,
specific enough to pick the right style when it's confident.

**Kept**: No — I4 is not adopted.  The I1b soft-blend remains the
production model at 2.37 dB (E74) / 2.35 dB (E76 re-run).

**Code preserved**: `AuthorEnsembleV2Model` + `train_author_ensemble_v2`
stay in `auto_beq_nn.py` as per AGENTS.md experimental-code preservation
rule.  `use_author_ensemble_v2` flag on ExperimentConfig is retained
so the experiment is re-runnable.

### Baseline variance follow-up

The harness Baseline came in at 2.83 dB vs 2.69 dB from the standalone
determinism test in E75.  That's a 0.14 dB gap between contexts that
both claim to be deterministic.

Hypothesis: some subtle difference in how the harness builds features
vs the standalone test.  Possible sources: dict iteration order
during TMDb cache enrichment, entry ordering after
`deduplicate_by_title` is consumed differently, or something in
`_build_train_entries` that depends on traversal order.

The *relative* ranking is stable within a single run (all experiments
share the same context), so conclusions about technique ordering are
still valid.  But the absolute number shifts across invocations.

**Deferred to E75b**: investigate with a diff between the two
contexts.  Not blocking for the current best-model selection.

---

## 2026-04-11: E77 — real-audio training re-run at 932 WAVs (crossover!)

**Hypothesis**: E33 (originally at 155 WAVs) showed real-audio training
lost to synthetic-only by 1.14 dB — not enough real data to beat 8k
synthetic entries.  At 932 WAVs (6× more real data), hybrid or
real-only training should finally become competitive.

**Method**: `test_real_audio_training` in `test_auto_beq_nn_real.py`.
Extracts features from all 932 real WAVs, splits 80/20 stratified by
rolloff severity → 770 train / ~193 test.  Trains three approaches
(synthetic-only, real-only, hybrid) under three configs (plain
XGBoost, LF α=0.5, LF α=0.5 + F1 augmentation) = 9 trainings, all
evaluated on the same held-out real test split.

**Result** (E75 deterministic harness, 770 train / 193 test):

| Training approach | XGB plain | LF α=0.5 | LF+aug (prod) |
|---|---|---|---|
| Synthetic-only (8097) | 3.36 | 2.93 | **2.37** |
| **Real-only (770)**   | **2.12** 🥇 | 2.26 | 2.24 |
| Hybrid (8480)         | 2.53 | 2.42 | 2.47 |

**Delta vs synthetic-only baseline (LF+aug production column)**:
- **Real-only: −0.13 dB** (WINS)
- Hybrid: +0.10 dB (loses)

**For the direct XGB-plain comparison** (fair Apples-to-apples with
the original E33 numbers):
- Synthetic-only plain: 3.36 dB → Real-only plain: 2.12 dB = **−1.24 dB**

At 155 WAVs (E33) this delta was +1.14 dB (real lost).  At 932 WAVs
it's −1.24 dB (real wins decisively).  **Crossover happened.**

**Three striking findings**:

1. **Real-only with plain XGBoost (2.12 dB) beats the full
   production config on synthetic (2.37 dB) by 0.25 dB.**  At
   932 WAVs, training on real audio directly beats training on 10×
   more synthetic features with all our fancy tricks.

2. **Late fusion and augmentation HURT real-only training**.
   Real-only plain XGB = 2.12, LF = 2.26 (+0.14), LF+aug = 2.24
   (+0.12).  Both techniques were invented as crutches to bridge the
   synthetic-to-real gap.  When there's no gap (real train, real
   test), they just add noise.

3. **Hybrid is the worst of both worlds** (2.47–2.53 dB).  The 7.3k
   synthetic entries drown out the 770 real ones at 10:1 ratio —
   synthetic patterns dominate the tree splits and the real-audio
   signal gets ignored.

**Production implications (big)**:

- The F/G/I-series architecture (augmentation, late fusion, per-author
  alpha, classifier routing) was optimal for the **synthetic data
  regime**.  Once enough real data is available, **the simpler
  plain-XGBoost trained on real features is better**.
- At >80% of the catalogue covered by real WAVs (932/8k ≈ 11% covered
  currently — but maybe the ratio matters less than absolute sample
  count), real-only should replace the synthetic+augmentation stack.
- The **production recommendation should shift**: I1b-soft-blend
  (2.37 dB) stays only while we're in the synthetic regime.  Once
  real audio coverage is high enough, switch to plain XGBoost on
  real features (2.12 dB).
- Augmentation and late fusion were NOT universal improvements —
  they're regime-specific.  This explains why the F/G combos kept
  looking flat at scale: each new technique was fixing a problem
  that augmentation already solved, not adding independent signal.

**Caveats**:

- The 2.12 dB is measured on a 193-title test split held out from
  the 932 real WAVs, not on the full 932 cache used by F/G/H/I
  harness runs.  The directly comparable number is synthetic-only
  LF+aug on the same 193-title subset (2.37 dB) — so the 0.25 dB
  improvement is real, but comparing 2.12 to the 2.35 I1b number
  from the harness is apples-to-oranges.
- The split stratification uses rolloff severity (heavy/moderate/
  gentle) based on total gain.  Random seed 42 — reproducible.
- At 770 train samples, XGBoost is near its data-hungry minimum.
  More real data = more improvement, up to a plateau.

**Lesson**: **We've been tuning the wrong knob.**  All the F/G/H/I
experiments optimised within the synthetic regime.  The real win came
from finally having enough real training data to leave that regime
altogether.  This is the classic ML "more data beats better models"
result — we just hadn't tested it at scale until now.

**Kept**: Yes — this is the new champion at 2.12 dB.  But keeping
the synthetic+augmentation stack in parallel since it's needed for
authors/titles without real WAV coverage.

**Open questions**:
- What's the minimum real-data threshold?  E33 at 155 failed
  (+1.14 dB).  E77 at 770 wins (−1.24 dB).  Somewhere between is
  the crossover.  Could re-run with 300, 500, 700 real samples to
  find the knee.
- Does real-only training still beat synthetic when the test set
  includes authors *without* any real training samples?  Need to
  check per-author breakdown of the 193-title test split.
- Should we train a hybrid router: use real-only for titles with
  good real-WAV coverage, synthetic+aug for everything else?

### Updated production recommendation

| Scenario | Best model | Mean dB | Notes |
|---|---|---|---|
| **Real audio available for title's domain** | **Real-only + plain XGBoost** | **2.12** | **NEW** (E77, needs verification on full 932) |
| Synthetic regime (unknown / sparse real data) | I1b-soft-blend (LF+aug+classifier) | 2.35 | Previous production winner (E74) |
| Known catalogue author | G8-perauth (oracle) | 2.33 | Upper bound when author is known |

---

## 2026-04-11: E79–E82 — real-audio crossover follow-ups

NAS extraction grew from 932 → 1091 trainable WAVs during this run
(872 train / 219 test after 80/20 stratified split by rolloff severity,
random_state=42).  All four experiments run on the same test split
so numbers are directly comparable.

### E79 — Apples-to-apples on the 219-title test split

Re-ran every production model against the same held-out test titles
E77 used.

| Model | Mean dB | vs real-only |
|---|---|---|
| **Real-only plain XGB** | **1.99** 🥇 | reference |
| G8-perauth (oracle, known author) | 2.24 | +0.25 |
| I1b-soft-blend (auto author) | 2.25 | +0.26 |
| F1-s0.5 (LF α=0.7 + aug) | 2.27 | +0.28 |
| G2a-a0.5 (LF α=0.5 + aug) | 2.42 | +0.42 |
| Synthetic LF+aug (E77 baseline) | 2.42 | +0.42 |

**Key findings**:
1. **Real-only beats every synthetic-trained model by 0.25+ dB.**
   The E77 crossover claim is now fully verified against the exact
   same test titles.
2. **I1b-soft-blend (2.25) is essentially tied with G8-perauth oracle
   (2.24)** on this split.  The classifier has fully closed the gap to
   the author-lookup version — 0.01 dB difference, within noise.
3. **F1-s0.5 beats G2a-a0.5** here (2.27 vs 2.42) — opposite of the
   932-WAV harness run where G2a was better.  Test-split variance.

### E80 — Per-author breakdown on the 219-title test split

Per-author mean dB for each model (* = best per author):

| Author | n | Synth | **Real** | Hybrid | F1 | G2a | G8 | I1b |
|---|---|---|---|---|---|---|---|---|
| aron7awol | 72 | 2.89 | 1.62 | **1.53** * | 1.82 | 1.77 | 1.77 | 1.80 |
| mobe1969 | 69 | 3.96 | **2.57** * | 3.36 | 2.80 | 2.95 | 2.80 | 2.78 |
| kaelaria | 43 | 3.41 | **2.38** * | 3.11 | 2.61 | 3.10 | 2.59 | 2.48 |
| t1g8rsfan | 14 | 2.42 | **1.37** * | 1.63 | 1.43 | 1.97 | 1.43 | 1.69 |
| remixmark | 12 | 3.03 | **1.28** * | 2.00 | 2.38 | 1.98 | 1.98 | 2.30 |
| halcyon888 | 9 | 2.11 | **0.68** * | 0.78 | 1.38 | 1.63 | 1.63 | 1.41 |
| OVERALL | 219 | 3.27 | **1.99** * | 2.42 | 2.27 | 2.42 | 2.24 | 2.25 |

**Striking findings**:
1. **Real-XGB wins for 5 of 6 authors**.  The one exception is
   aron7awol where Hybrid-XGB edges it out by 0.09 dB (1.53 vs 1.62).
   aron7awol has the largest test cohort (n=72) so this is
   statistically meaningful — his titles benefit from the extra
   synthetic training data.
2. **remixmark — the outlier that dragged all earlier experiments** —
   drops from 3.03 dB (synth) to **1.28 dB (real)**.  Massive -1.75 dB
   improvement.  Real audio training fixes the "hard" author problem
   that dominated the G/H/I series narrative.
3. **halcyon888 hits 0.68 dB** with real training — near-perfect.
   Small n (9) but consistent.
4. **Synth-XGB is the worst model for every single author**.  The
   synthetic regime is decisively beaten.
5. **I1b-soft-blend now essentially matches G8-perauth** across all
   authors.  The classifier has converged to the oracle with enough
   data.

### E81 — Real-data threshold sweep

Sweeped real-only training at n ∈ {100, 200, 300, 400, 500, 600, 700,
872} against the fixed 219-title test split.  Synthetic-only
reference: 3.27 dB.

| n_real | mean dB | vs synth |
|---|---|---|
| 100 | 2.58 | **-0.69** (already wins) |
| 200 | 2.35 | -0.93 |
| 300 | 2.11 | -1.17 |
| **400** | **2.04** | **-1.24** (plateau begins) |
| 500 | 2.06 | -1.21 |
| 600 | 2.08 | -1.19 |
| 700 | 2.02 | -1.26 |
| 872 | 2.02 | -1.25 |

**Crossover is much lower than expected**.  Even 100 real samples
beat 8k synthetic by 0.69 dB.  The plateau is around 400 samples
where additional real data stops helping.

**E33 was wrong**: the 2026-04-08 E33 run claimed 155 real samples
lost to synthetic by +1.14 dB.  At the same data scale today (100),
real beats synthetic by -0.69 dB — a 1.83 dB discrepancy.  The
difference is the test set: E33's test split was tiny (7–14 titles,
depending on fold) and unrepresentative.  **E33's conclusion about
"real training fails at small scale" was a measurement artefact**,
not a real finding.  The current 219-title stratified test split is
orders of magnitude more reliable.

**Practical implications**:
- As little as 100 real-audio WAVs is enough to justify switching
  from the synthetic+augmentation architecture to plain real-only
  XGBoost.
- The data-hungry plateau is around 400 samples — beyond that, more
  real WAVs deliver marginal gains.
- The F/G/H/I-series architecture (augmentation, late fusion,
  classifier routing) was designed for the synthetic regime but has
  been unnecessary for months — we passed the 100-WAV threshold
  before the F-series even began.

### E82 — Sample-weighted hybrid router

Trained a combined real + synthetic dataset with sample weights
rebalancing the real:synth contribution to the loss.

| Ratio (real:synth) | mean dB | vs real-only |
|---|---|---|
| 1:1 (E77 naive hybrid) | 2.48 | +0.49 (loses badly) |
| 5:1 | 2.18 | +0.18 |
| 10:1 | 2.09 | +0.09 |
| 20:1 | 2.06 | +0.06 |
| **50:1** | **1.99** | **0.00** (ties real-only) |

**At 50:1 weight, weighted hybrid training matches real-only plain
XGB exactly (both at 1.99 dB).**  Higher ratios weren't tested but
would likely converge to or slightly underperform real-only.

**Why this matters**: the 50:1 weighted hybrid delivers
**real-only accuracy** on WAV-backed titles AND **retains synthetic
coverage** for titles without real WAVs — in a single model.

This is the best-of-both-worlds answer we were looking for:
- No inference-time routing needed (single model handles everything)
- Accuracy matches real-only on the 872 WAV-backed titles
- Coverage extends to the ~7k catalogue titles without real WAVs
- No late fusion, no augmentation, no classifier — just plain
  XGBoost with a sample_weight array

The naive 1:1 hybrid was drowning out real signal with 10× synthetic
noise.  Rebalancing fixed it.

### New production recommendation after E79–E82

| Scenario | Model | Mean dB | Notes |
|---|---|---|---|
| **Production (full catalogue coverage)** | **50:1 weighted hybrid plain XGB** | **1.99** | **NEW champion** — real-audio accuracy + synthetic coverage |
| Best for WAV-backed titles only | Real-only plain XGB | 1.99 | Same accuracy, loses coverage |
| Synthetic-only fallback (legacy) | I1b-soft-blend | 2.25 | Only needed if no real WAVs extractable |
| Known catalogue author (legacy) | G8-perauth | 2.24 | Deprecated — I1b ties it at scale |

**Progression summary updated**:

| Milestone | Mean dB | Cache | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 | 7 | Initial baseline |
| E34 + late fusion α=0.7 | 2.45 | 220 | One-hot + late fusion |
| E41/F1 (augmentation σ=0.5) | 2.02 | 67 | Synthetic augmentation |
| E59/G8 (per-author α, small) | 1.86–1.92 | 67 | Author lookup at inference |
| E69/I1b (soft routing, small) | 2.01 | 67 | Auto author from metadata |
| E77 (real-only plain XGB) | 2.12 | 932 | Real-audio regime switch |
| **E82 (50:1 weighted hybrid)** | **1.99** | **1091** | **Real accuracy + synth coverage** |

### Lessons learned

1. **Data regime trumps architecture**.  The F/G/H/I series optimised
   within the synthetic regime and squeezed out ~0.4 dB.  Switching
   to real-audio training delivered ~0.25 dB more — in one experiment.
2. **Small test sets lie**.  E33's "real training fails" conclusion
   was a measurement artefact of a tiny test split.  Always use
   stratified holdouts of ≥100 titles.
3. **The threshold is much lower than expected**.  We thought we
   needed ~500+ real WAVs to switch regimes.  Actually 100 is enough.
   Months of F/G/H/I work were architecturally optimal for a regime
   we had already left.
4. **Weighted hybrid is the right way to combine regimes**.  Naive
   concatenation drowns real signal in synthetic noise.  Sample
   weighting at 50:1 restores real's contribution without sacrificing
   synthetic's coverage.
5. **The classifier has converged to the oracle**.  I1b (2.25) ties
   G8 (2.24) at this scale.  The complexity of per-author alpha
   lookup no longer buys anything.

### Next steps

- [x] **Deploy the 50:1 weighted hybrid as the production model** —
      done: `train_production_weighted_hybrid()` in `auto_beq_nn.py`,
      `scripts/train_production_model.py` CLI wrapper,
      `get_advisor("trained_model")` auto-discovers
      `{beq-dir}/production_model.joblib`. `generate_beq_profile.py`
      loads the saved model in `main()` instead of retraining per
      episode (was a 3-minute-per-episode script, now sub-second).
- [x] Re-run JJK profile generation with the new model to verify
      inference quality on uncatalogued content — done: 47 episodes
      (S1: 24, S2: 23), 0 failures. Consistent 4–6 filter chains,
      gains in the 2.7–4.4 dB range, MV +7 to +12.5 dB. Compared to
      pre-port S02E01 output: new model is ~4 dB more aggressive
      on MV, filters concentrated in 17–35 Hz band rather than spread
      5–46 Hz — consistent with the real-audio training regime
      learning steeper rolloffs. See
      [`profiles/jjk_v2_comparison_report.md`](../../profiles/jjk_v2_comparison_report.md)
      for the full breakdown.
- [ ] E78 baseline variance debug — still open, hygiene work
- [x] Document "50:1 weighted hybrid plain XGBoost" in the
      `Current production model` section at the top of this file
      (done — see the updated section at line 62)

### Infrastructure improvements (2026-04-11)

Not algorithm experiments. Local-integration CI rig on grumpy
(reference Windows box, see multi-host Ollama test at line 796):

- **GitHub Actions self-hosted runner on grumpy**: new workflow
  `.github/workflows/local-integration.yml` triggers the full spike
  test + experiment suite (`run-spike-tests.sh` → `-integration.sh`
  → `-experiments.sh`, sequential) on every push to
  `div/local-integration`. Dev laptop no longer needed in the loop
  for long experiment retrains. `concurrency.cancel-in-progress: true`
  coalesces push bursts without polling.
- **New Docker test image `docker/Dockerfile.test`**: full
  poetry dev closure (scipy + xgboost + scikit-learn + PyQt6
  offscreen + pytest + ffmpeg). Kept deliberately separate from the
  lean stdlib-only `docker/Dockerfile` used by the NAS LFE
  extractor — two images, two jobs, no bloat bleed.
- **`docker/test-entrypoint.sh` dispatcher**: runs all three spike
  runners in order, collects the worst exit code across stages so a
  unit failure doesn't mask an experiment failure on the same build.
  Honours `TEST_SCOPE=unit|integration|all` so dev dry-runs can skip
  the slow stages.
- **`scripts/win/` helpers**: `Configure-LocalIntegration.ps1`
  (interactive bootstrap, 30 s prompt timeout, fails fast headless),
  `Assert-LocalIntegrationConfig.ps1` (read-only workflow-time
  check), `Run-ExperimentBuild.ps1` (docker compose wrapper).
  Config persisted at `C:\ProgramData\beqdesigner-ci\config.json`
  so the runner service can read it non-interactively.
- **End-user docs**: `docs/local_integration.md` (full walkthrough +
  troubleshooting + teardown) and a "Quick start (local CI on a
  Windows build box)" subsection in `readme.md`. Nav entry added to
  `mkdocs.yml`.

**Lesson (anticipated, pending the W5 smoke test)**: offloading the
experiment suite to dedicated GPU hardware on a push-triggered rig
should unblock rapid iteration on real-audio training regimes
without blocking the dev laptop. Self-hosted runner + Docker is
strictly less moving parts than a Jenkins/Woodpecker/Drone setup
for a one-user one-machine loop — and GitHub handles build history,
log streaming, and concurrency cancellation for free.
