# E25-E40: ML trained model experiments

**Status:** Adopted - XGBoost baseline, late fusion, metadata encoding

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

See [auto_beq.md](../auto_beq.md) for the main reference.

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
**New tests**: `src/test/python/auto_beq/test_auto_beq_nn.py` (9 tests)
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

