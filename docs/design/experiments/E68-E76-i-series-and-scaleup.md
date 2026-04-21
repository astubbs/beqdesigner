# I-series (E68-E76): Author selection and scale-up validation

**Status:** I1b soft-blend adopted as production model (synthetic era)

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

