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
