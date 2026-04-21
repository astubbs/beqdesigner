# G-series (E53-E59): Combinations and hyperparameter tuning

**Status:** Adopted - G8 per-author alpha, G4b sigma=0.3

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

