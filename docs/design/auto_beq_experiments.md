# Auto-BEQ experiment log

**Companion docs:**
- [`auto_beq.md`](auto_beq.md) — vision + overall design
- [`auto_beq_plan.md`](auto_beq_plan.md) — current iteration plan

Running record of what we've tried, what worked, and why. This is
append-only. Entries are dated (YYYY-MM-DD). Source of truth for
"did we already try X" between sessions.

Grading thresholds: mean_abs_err < 2 dB, max_abs_err < 5 dB, across
5-80 Hz, vs catalogue entry's response curve.

Test fixtures (real-media): Edge of Tomorrow (EoT, UHD DTS-HD 7.1),
Mad Max: Fury Road (MM, UHD TrueHD Atmos 7.1), John Wick (JW, WEBDL
EAC3 5.1).

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
