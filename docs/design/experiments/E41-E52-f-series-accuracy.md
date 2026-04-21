# F-series (E41-E52): NN accuracy improvements

**Status:** Partial - F1 synthetic augmentation kept (-0.73 dB breakthrough)

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

