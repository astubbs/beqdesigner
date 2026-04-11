# Future NN Experiments — Accuracy Improvement Plan

**Date**: 2026-04-10
**Context**: Current best = late fusion α=0.7 + one-hot type encoding = **2.45 dB on 220 titles** (E34+E27). Per-author best: aron7awol solved at 1.68 dB (E40).

This doc captures ideas not yet tried, derived from cross-referencing the
original experiment plan (`auto_beq_plan.md`) against the experiment log
(`auto_beq_experiments.md`, E1–E40). Ranked by expected impact.

---

## Coverage gap: plan experiments never attempted

These were in the original plan but never executed:

| Plan experiment | Description | Why still relevant |
|---|---|---|
| **E3 — Scene-description chunk weighting** | Weight chunks by bass-energy expectation from frame descriptions | Could help short content and sparse bass titles |
| **E5 — Cross-episode consistency** | Test whether one episode's profile transfers to all others in a season | Validates season-wide profiling for TV |
| **E6 — Music detection and exclusion** | Exclude music-dominated chunks from rolloff analysis | Could help heavily scored content |
| **E7 — LLM as output validator** | LLM flags suspicious model predictions as QA layer | Different from E7-E13 which used LLM as designer |

---

## New ideas (not in original plan)

### F1 — Synthetic feature augmentation

**Priority**: HIGH | **Effort**: Easy (30 min)

**Hypothesis**: The persistent synthetic-to-real gap (0.88 dB in E34, was
2.90 dB in E25) comes from the distribution mismatch between "perfect inverse"
synthetic curves and noisy real measured spectra. Augmenting synthetic features
during training to simulate real-audio noise should improve transfer.

**Method**:
- Add Gaussian noise (σ=1-2 dB) to the 9-bin synthetic audio features
  during training
- Randomly perturb individual bins by ±1-3 dB (simulates content-dependent
  spectral variation)
- Randomly smooth or blur the synthetic curve (simulates Welch averaging
  artifact)

**Rationale**: This is trivially implementable (a few lines in
`train_xgboost`), costs nothing at inference time, and directly targets the
distribution mismatch that killed the CNN (E28: 24 dB!) and hurt early fusion.
XGBoost is more robust to noise than NNs, but even its 0.88 dB gap suggests
room for improvement.

**Success metric**: Gap between synthetic and real-audio loss decreases.
Real-audio loss decreases without synthetic loss regressing.

---

### F2 — Option B features: 27-dim chunk statistics

**Priority**: MEDIUM-HIGH | **Effort**: Moderate

**Hypothesis**: The 9-bin percentile curve (Option A) discards information
about *how confidently* the rolloff ceiling is visible. Option B captures
this via per-bin statistics across chunks.

**Method** (from `auto_beq_ml_experiments.md`, never implemented):
- Per frequency bin across all chunks:
  - 90th percentile (rolloff ceiling estimate)
  - Standard deviation (content variability)
  - Fraction of chunks within 3 dB of ceiling (consistency)
- 3 statistics × 9 bins = 27 values

**Rationale**: The standard deviation captures exactly the signal that
distinguishes EoT-class content (high variance — showcase scenes inflate
the average) from consistent rolloff (every chunk shows the same level).
Titles where every chunk agrees on the 10 Hz level have high-confidence
rolloff; titles where it varies wildly need different treatment.

**Success metric**: Improved accuracy on titles with sparse/variable bass
content. Compare against Option A baseline head-to-head.

---

### F3 — Absolute dBFS as additional features

**Priority**: MEDIUM | **Effort**: Easy

**Hypothesis**: E15 proved absolute levels carry information that
normalisation destroys. EoT's 10 Hz at -32 dBFS vs MM's at -68 dBFS is a
massive signal that the normalised features cannot represent.

**Method**:
- Add 9 absolute dBFS values (same frequency bins as Option A) as
  additional audio features
- Total: 18 audio dims instead of 9
- The model can learn that "normalised rolloff looks flat but absolute
  level is hot" means different correction than "normalised rolloff looks
  flat and absolute level is quiet"

**Rationale**: Normalisation was introduced to make features comparable
across titles, but it strips mastering-level information. Absolute dBFS
restores this without removing normalised features. Two complementary views.

**Caveat**: Only available for real-audio features. Synthetic features would
need a plausible absolute-level model. Could train with absolute dBFS = 0
for synthetic entries and let the model learn to ignore it on synthetic data.

**Success metric**: Improvement on titles where normalised features hide
the rolloff (EoT-class). No regression on titles where normalisation works.

---

### F4 — Music detection and exclusion (Plan E6)

**Priority**: MEDIUM | **Effort**: Moderate

**Hypothesis**: Chunks dominated by score/soundtrack music are harmful to
rolloff detection because music bass is intentionally mixed and doesn't
reflect the rolloff ceiling.

**Method** (two options, compare both):
- **Cheap**: Onset regularity detection. Music has periodic rhythmic
  patterns; impacts are aperiodic. Compute autocorrelation of onset
  envelope per chunk, threshold on periodicity score.
- **Cheaper**: If frame descriptions exist, keyword-match for "music",
  "score", "song", "concert" and down-weight those chunks.

**Rationale**: E18b showed that what you feed into the model matters.
Titles heavily scored (musicals, concert films, Zimmer-scored action)
would benefit from excluding music chunks. The plan identified this as
potentially the bigger win than impact detection.

**Success metric**: Improvement on heavily scored content without
regression on sparse-bass content.

---

### F5 — Cross-episode consistency test (Plan E5)

**Priority**: LOW-MEDIUM | **Effort**: Very easy

**Hypothesis**: TV episodes from the same series share the same mixing
chain. A profile from one episode should transfer to all others.

**Method**:
- Select 5+ TV series from the validation set (Blue Eye Samurai, Pantheon,
  X-Men '97, South Park, Mindhunter, Scavengers Reign all have multiple
  episodes)
- Profile episode 1 using best model
- Apply that profile to episodes 2-N
- Measure error vs per-episode prediction error

**Rationale**: Validates the TV season-batch use case (JJK full season
generation). If cross-episode error ≈ per-episode error, season-wide
profiling is acoustically justified.

**Success metric**: Cross-episode error within 0.5 dB of per-episode error
for 4+ of 5 test series.

---

### F6 — Confidence-weighted training

**Priority**: MEDIUM | **Effort**: Easy

**Hypothesis**: Some catalogue entries have multiple authors agreeing on
similar profiles (high-quality ground truth). Others have wildly different
profiles from different authors (noisy ground truth). Weighting by
inter-author agreement should improve training signal quality.

**Method**:
- For each title, count how many distinct authors have entries
- For titles with multiple authors: compute pairwise downstream loss
  between their filter chains
- Low inter-author loss = high agreement = reliable ground truth
- Weight training samples by agreement score (high agreement = higher weight)

**Rationale**: mobe1969 makes inconsistent choices (E40: gets worse
isolated). Training on his entries with equal weight to aron7awol's
(consistent at 1.68 dB) adds noise. Down-weighting inconsistent entries
should improve the loss landscape.

**Success metric**: Improved accuracy on reliable-ground-truth titles
without regression on others.

---

### F7 — Rolloff shape clustering (Plan E4, full version)

**Priority**: LOW-MEDIUM | **Effort**: Moderate

**Hypothesis**: E17c used hand-crafted gentle/moderate/cliff classes.
Natural clusters in the data may capture finer patterns like "Disney 2010s
Atmos rolloff" vs "1990s action film rolloff".

**Method**:
- K-means (k=4-8) on the 9-bin normalised rolloff curves across the full
  catalogue
- Visualise clusters — expect natural groupings by studio/era/format
- Use cluster ID as a categorical feature for the model
- Alternative: use cluster centroid filter as warm-start for scipy
  refinement (Plan E4's original proposal)

**Rationale**: The model currently learns rolloff-shape patterns implicitly.
Explicit cluster features compress this into a single high-information
categorical feature that XGBoost can split on efficiently.

**Success metric**: Cluster ID appears in feature importances. Modest
accuracy improvement (0.2-0.5 dB expected).

---

### F8 — LLM as output validator (Plan E7)

**Priority**: LOW | **Effort**: Moderate (exploratory)

**Hypothesis**: An LLM given the prediction + title metadata + measured
rolloff can flag cases where the prediction is likely wrong.

**Method**:
- After model predicts, pass to LLM: predicted filter params + title
  metadata + detected rolloff shape
- Ask: "Does +15 dB at 20 Hz make sense for a 2024 Disney Atmos animated
  film?"
- LLM returns confidence score / flag
- Measure whether LLM flags correlate with high-error cases

**Rationale**: This is a QA layer, not core algorithm. Won't improve the
model itself, but could prevent bad predictions from reaching production.
Different from E7-E13 which used LLM as the filter *designer* — here the
model designs, the LLM validates.

**Success metric**: LLM correctly flags >50% of predictions with >5 dB
error, with <20% false positive rate on good predictions.

---

### F9 — Downstream loss as direct training objective

**Priority**: HIGH-RISK | **Effort**: Hard

**Hypothesis**: E38 tried sample reweighting (helped early fusion by
0.55 dB). The deeper version: make the training loss itself compute the
filter chain's frequency response and measure dB error, rather than MSE
on parameters.

**Method**:
- Custom XGBoost objective function that:
  1. Decodes predicted Y into filter parameters
  2. Evaluates the filter chain's frequency response
  3. Computes dB error vs the target response
  4. Returns gradient/hessian of that acoustic loss
- This is "downstream loss as training objective" — the plan says to use
  it as a stop condition, but using it as the objective is more powerful

**Rationale**: Parameter MSE penalises "wrong numbers" but two different
filter parameter sets can produce nearly identical acoustic results. The
current MSE objective may push the model away from acoustically equivalent
solutions. Direct acoustic loss avoids this.

**Risk**: Custom gradient computation for filter chain evaluation is
complex. XGBoost requires differentiable objectives (gradient + hessian).
Filter chain evaluation involves trigonometric functions (biquad coefficients)
which are differentiable but the implementation is nontrivial.

**Alternative**: Two-stage approach (E38 reweighting) is simpler. Or use
PyTorch with a differentiable filter evaluation layer for the CNN branch.

**Success metric**: >0.5 dB improvement over MSE objective. No training
instability.

---

### F10 — Larger real-audio corpus

**Priority**: HIGH (ongoing) | **Effort**: Waiting

**Status**: NAS extraction queued for 1,244 titles. E33 showed 155 WAVs
wasn't enough for real-audio training to beat synthetic. Expected threshold:
500+ titles.

**Action**: Re-run E33 (hybrid training) when NAS extraction reaches 500+
WAVs. The synthetic-to-real gap should continue shrinking as real audio
coverage grows.

---

### F11 — Multi-resolution audio features

**Priority**: LOW | **Effort**: Easy

**Hypothesis**: The current 9 bins are uniformly spaced in log-frequency
from 20-80 Hz. Higher resolution in the critical 10-30 Hz range (where
most BEQ correction happens) could capture rolloff knee shape more precisely.

**Method**:
- 18 bins: 10, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35, 40, 45, 50, 60,
  70, 80, 100 Hz
- Or: keep 9 bins but shift them to concentrate in 10-40 Hz range

**Rationale**: The catalogue's most common knee frequencies are 10-25 Hz.
The current bins (20, 25, 30, 35, 40, 50, 60, 70, 80) have only 2 bins
below 30 Hz. More resolution where it matters could help magnitude
calibration on low-frequency rolloffs.

**Success metric**: Improvement on titles with low-frequency rolloff knees
(< 25 Hz).

---

### F12 — Per-author ensemble with router

**Priority**: LOW | **Effort**: Moderate

**Hypothesis**: E40 showed 3 of 5 authors benefit from isolated models.
Instead of one model with author as input, train an ensemble of
author-specific models + a fallback multi-author model, with a router.

**Method**:
- Train separate XGBoost models for aron7awol, kaelaria, t1g8rsfan
  (the three that benefit from isolation)
- Keep the multi-author model for mobe1969, remixmark, halcyon888, unknown
- Router: if author specified and has dedicated model, use it; else fallback

**Rationale**: E40 showed isolated models improve by 0.09-0.44 dB for
3 authors. The ensemble approach captures this without losing the
multi-author model's regularisation benefit for inconsistent authors.

**Success metric**: Weighted-average improvement across all authors vs
single multi-author model.

---

## Recommended execution order

1. **F1 — Synthetic augmentation** (30 min, directly targets #1 bottleneck)
2. **F3 — Absolute dBFS features** (easy, proven signal from E15)
3. **F2 — Option B features** (moderate effort, captures rolloff confidence)
4. **F5 — Cross-episode consistency** (very easy, validates TV workflow)
5. **F6 — Confidence-weighted training** (easy, improves training signal)
6. **F4 — Music exclusion** (moderate, helps specific failure modes)
7. **F7 — Rolloff clustering** (moderate, explicit shape features)
8. **F9 — Downstream loss objective** (hard but highest theoretical ceiling)
9. **F11 — Multi-resolution features** (easy, low expected impact)
10. **F12 — Per-author ensemble** (moderate, diminishing returns)
11. **F8 — LLM validator** (exploratory, QA layer not core improvement)
12. **F10 — Real-audio corpus** (ongoing, re-run E33 at 500+ WAVs)

**Stop condition**: If F1 + F2 + F3 push the 220-title mean below 2.0 dB,
the remaining experiments are enhancements rather than requirements.
