# Auto-BEQ experiment log

**Companion docs:**
- [`auto_beq.md`](auto_beq.md) — vision + overall design
- [`auto_beq_plan.md`](auto_beq_plan.md) — current iteration plan

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
