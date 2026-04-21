# H-series (E60-E67): Multi-author resolution

**Status:** Dead end - averaging multi-author disagreement regresses

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

