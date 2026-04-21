# E83-E87: Paradigm-shift experiments (post-champion)

**Status:** Active - Whisper embeddings, differentiable DSP, self-training

---

## 2026-04-12: Paradigm-shift experiments (T1.x)

See [`auto_beq_nn_paradigm_shifts.md`](auto_beq_nn_paradigm_shifts.md)
for the full research menu. This section logs actual experiment
results against the E82 production champion.

### E84 — Semi-supervised self-training (T1.3)

**Hypothesis**: the WAV cache contains unlabelled real audio (WAVs we
have but the catalogue has no filter chain for). A teacher model can
generate pseudo-labels for them; filter by self-consistency
(prediction's acoustic response matches measured rolloff within 1.5
dB mean abs error); retrain with those pseudo-labels as an
additional training-data channel (weight 10, between real at 50 and
synth at 1). Iterate.

**Method**: E82-style weighted hybrid baseline on a fresh 80/20 split
of the 1279-WAV cache (1023 train / 256 test, stratified by rolloff
severity, random_state=42). Three iterations:
- iter 0: baseline = E82 weighted hybrid (real=50, synth=1)
- iter 1: teacher=iter 0, pseudo-label unmatched, retrain with pseudo=10
- iter 2: teacher=iter 1, pseudo-label again, retrain

**Result**:

| Iteration | mean dB | max dB | n_pseudo | vs baseline |
|---|---|---|---|---|
| iter 0 (E82 baseline) | **1.71** | 9.92 | 0 | — |
| iter 1 | 1.69 | 11.51 | **0** | −0.02 (noise) |
| iter 2 | 1.69 | 11.51 | **0** | −0.02 (noise) |

**Per-author (baseline → final)**:

| Author | baseline | final | Δ |
|---|---|---|---|
| aron7awol | 1.33 | 1.24 | −0.10 |
| halcyon888 | 0.54 | 0.47 | −0.07 |
| kaelaria | 1.92 | 1.85 | −0.06 |
| mobe1969 | 2.50 | 2.57 | +0.07 |
| remixmark | 0.88 | 0.87 | −0.00 |
| t1g8rsfan | 1.53 | 1.60 | +0.07 |

**Why it was a no-op**: the unmatched WAV pool has **only 11 WAVs**
— not the ~200 I estimated. `extract_lfe.py` only extracts WAVs for
catalogue-matched titles by design, so 1279 / 1290 WAVs in the cache
(99.1%) are already labelled. The 11 unmatched are edge cases where
the ID tag → catalogue entry lookup failed. All 11 failed the 1.5 dB
self-consistency gate across both iterations, so **zero pseudo-labels
were ever added to training** — iter 1 and iter 2's training data is
identical to iter 0's. The −0.02 dB drift is XGBoost non-determinism
across two train calls on the "same" data (cf E75 determinism work).

**Lesson**: self-training is the right mathematical idea but the
WRONG FIT for this data pipeline. `extract_lfe.py`'s current behaviour
makes the unmatched pool essentially empty — it's a closed-set
problem, not an open-set one. For E84 to actually pay off we'd need
to **expand extract_lfe.py to process titles that have NO catalogue
entry** (e.g. bulk-extract from a library root, not from catalogue
matches). That's a separate scope expansion and is out of scope for
Tier 1.

**Bonus finding**: the baseline itself is **1.71 dB on the new
1279-WAV split** — significantly better than E82's 1.99 dB. The 188
additional WAVs extracted since E82 (1091 → 1279) lowered the mean
error by 0.28 dB essentially for free. The E77+ observation that
"more real WAVs monotonically improve accuracy" still holds.

**Verdict**: **preserved as selectable alternative, not promoted**.
Code path lives at `auto_beq_nn.py::train_e84_self_trained` + the
`--self-train` CLI flag. If `extract_lfe.py` ever grows a
scan-everything mode, re-run E84 with a realistic unlabelled pool.

**Related commit**: (this commit)
**CSV**: `.pytest_cache/e84_self_trained.csv`

### E83 — Audio foundation-model features (T1.1)

**Hypothesis**: Whisper's encoder is trained on 680,000 hours of
speech + general audio. Even though its 80-bin log-mel front-end is
biased toward mid/high frequencies, its 384-dim pooled output might
carry some signal about spectral texture, compression, or mastering
character that our 9 Welch bins discard. Drop it in as additional
features for the E82 weighted hybrid, let XGBoost's feature
importance decide whether it helps.

**Method**: Same E82 pipeline, `AudioFeatureConfig(foundation_model=
"whisper-tiny")`. Extract one 384-dim embedding per real WAV
(upsample 1 → 16 kHz, 30-second chunks, encoder forward, mean-pool
across chunks). Concatenate to the existing 102-dim feature vector →
486 total dims. Synthetic samples get a zero-vector fallback.

Apples-to-apples comparison on the same 1279-WAV 80/20 stratified
split (random_state=42) as E84:
- baseline = E82 weighted hybrid with 102 features
- E83 = same pipeline with 486 features (102 + 384 Whisper)

**Result**:

| Model | mean dB | max dB | n_features | train time |
|---|---|---|---|---|
| baseline (E82) | **1.73** | 11.51 | 102 | 30.5 s |
| E83 + whisper-tiny | 1.90 | 14.37 | 486 | 941 s (31×) |
| **Δ mean** | **+0.17 (regression)** | **+2.86 (worse)** | — | — |

**Per-author** — **regresses on every single author**:

| Author | baseline | E83 | Δ |
|---|---|---|---|
| aron7awol | 1.32 | 1.45 | +0.13 |
| halcyon888 | 0.51 | 0.81 | +0.30 |
| kaelaria | 1.85 | 2.16 | +0.30 |
| mobe1969 | 2.59 | 2.72 | +0.13 |
| remixmark | 0.93 | 1.05 | +0.12 |
| t1g8rsfan | 1.57 | 1.75 | +0.18 |

**Timing cost**: Whisper embedding extraction took **63 minutes** for
the 1279 WAV cache on CPU (0.3 WAVs/s across 12 threads — the per-
WAV mel + encoder cost dominates). XGBoost training grew from 30 s
to 941 s (31× slowdown) because the 4.8× wider feature matrix needs
more split candidate evaluations per tree. Extraction was fully
cached after the first run, but the training-time cost persists.

**Why it failed** — the theoretical concern from
`auto_beq_nn_paradigm_shifts.md` was empirically confirmed:

1. Our LFE content is sub-500 Hz. When upsampled to 16 kHz for
   Whisper's input, everything above 500 Hz is silence.
2. Whisper's 80-bin log-mel filterbank puts its highest resolution
   in the 1–8 kHz speech range (where phonemes live) and has near-
   zero resolution below 100 Hz.
3. The resulting 384-dim embedding is dominated by whatever minimal
   structure Whisper's attention heads find in the "silence+some
   bass energy" input — essentially noise from the model's
   perspective (far from its training distribution).
4. XGBoost with 486 features and only 1022 real samples + 7797
   synthetic (heavily weighted) has enough capacity to fit the 384
   noise dimensions on training data. This overfits — visible in
   the max-error increase (+2.86 dB), where worst-case titles got
   materially worse.

**Verdict**: **preserved as selectable alternative, not promoted**.
Code path: `AudioFeatureConfig(foundation_model="whisper-tiny")`,
CLI flag `--foundation-model whisper-tiny`. Cached embeddings at
`{beq-dir}/foundation-embeddings/whisper-tiny/*.npy`. If a future
experiment wants to compare against a proper sub-bass-aware
foundation model (EnCodec, or a custom 1D CNN trained on LFE), the
plumbing is ready — just add a new entry to `FOUNDATION_MODEL_DIMS`
and wire up the extractor.

**Next foundation-model attempt** (if any) should use **EnCodec**
(24 kHz neural codec — trained to RECONSTRUCT full-bandwidth audio
including sub-bass, so its encoder has real incentive to preserve
low-frequency content) or a **custom 1D CNN trained from scratch**
on our LFE data directly. Both live in T3.x of the paradigm-shifts
doc as speculative bets; Whisper was meant to be the easy first win
and empirically wasn't.

**CSV**: `.pytest_cache/e83_foundation.csv`
**Experiment test**: `test_e83_foundation_features` in
`src/test/python/spike/test_auto_beq_nn_real.py`

### E85 — Differentiable DSP with acoustic loss (T1.2) — NEW CHAMPION

**Hypothesis**: the fundamental limitation of E82 and all prior
experiments is the training objective. XGBoost minimises mean squared
error on filter parameters (frequency, gain, Q values) — but two very
different parameter sets can produce nearly identical acoustic
responses. The model wastes capacity matching exact parameter values
instead of matching what matters: the sound. A neural network trained
through a differentiable biquad layer on the actual acoustic response
error should outperform any param-MSE model regardless of feature
engineering or data strategy.

**Method**: a small neural network (3-layer, 256-wide, ~200K params)
that consumes the same 102-dim feature vector as E82 and outputs 6
filter slots × 4 params (frequency, gain, Q, enabled). All outputs
are range-clamped via sigmoid/tanh activations (5–80 Hz, ±15 dB,
Q 0.3–4.0). Fixed LowShelf topology for all slots (no type selection
in v1).

The key innovation: a **differentiable biquad response layer** that
evaluates the predicted filter chain's log-magnitude response on the
BEQ frequency grid using the closed-form `|H(e^jw)|²` formula in
real arithmetic (no complex tensors). Gradients flow cleanly back
through sin/cos/sqrt of the biquad coefficients. The training loss is
the band-masked (5–80 Hz) squared dB difference between the predicted
response and the target response — this IS the production evaluation
metric, not a proxy.

Two-stage training:
- **Stage 1 — MSE warm-start** (10 epochs): clone the E82 XGBoost
  teacher's filter-param predictions. Gets the network into a
  reasonable region of parameter space before switching to the
  non-convex acoustic loss.
- **Stage 2 — acoustic-loss fine-tune** (20 epochs): minimise the
  direct acoustic match error through the differentiable biquad layer.

Total training time: **2.4 seconds** on CPU (vs E82's 30 seconds for
XGBoost). Model size: **661 KB** (vs E82's 8 MB).

**Tier 1 unified comparison result** (1023 train / 255 test,
stratified by rolloff severity, random_state=42, full 1279-WAV cache):

| Experiment | mean dB | max dB | Δ vs E82 | train time | verdict |
|---|---|---|---|---|---|
| E82 baseline (plain XGB) | 1.70 | 9.92 | — | 29.6 s | reference |
| E83 Whisper-tiny (+384 dims) | 2.53 | 12.85 | +0.83 | 11.6 min | regression |
| E84 self-training (11 unmatched) | 1.70 | 9.92 | +0.00 | 1.5 min | no-op |
| **E85 diff-DSP (acoustic loss)** | **1.49** | 11.45 | **−0.21** | **2.4 s** | **NEW CHAMPION** |

**Per-author breakdown** — E85 crushes the previously-hardest authors:

| Author | E82 | E85 | Δ | Winner |
|---|---|---|---|---|
| **mobe1969** | 2.47 | **1.74** | **−0.73** | E85 |
| **t1g8rsfan** | 1.53 | **0.81** | **−0.72** | E85 |
| kaelaria | 1.92 | 1.72 | −0.20 | E85 |
| aron7awol | 1.33 | 1.43 | +0.10 | E82 |
| halcyon888 | 0.54 | 0.61 | +0.07 | E82 |
| remixmark | 0.88 | 1.40 | +0.52 | E82 |

**Why it works**: the acoustic loss sidesteps the proxy-objective
problem. E82's MSE penalises "wrong numbers" even when two different
parameter sets produce the same sound. E85 is free to find ANY
parameter combination that produces the right acoustic result. This
matters most for:
- **mobe1969** (−0.73 dB): uses unusual parameter values that look
  "wrong" to MSE but are acoustically valid.
- **t1g8rsfan** (−0.72 dB): similar pattern — non-standard filter
  choices that MSE penalises but acoustic loss accepts.
- The easy authors (aron7awol, halcyon888) were already well-served
  by param-MSE, so E85's different optimisation landscape introduces
  slight regressions there.

**Caveats**:
- **Max error worse** (11.45 vs 9.92): some titles regress because
  the fixed LowShelf-only topology can't reach HighShelf/PeakingEQ
  targets in the catalogue. Adding softmax over filter types per slot
  (Phase 2 item 6) should fix this.
- **3/6 authors regress** slightly (+0.07 to +0.52 dB). A future
  ensemble router (Phase 2 item 8) could use E82 for those authors
  and E85 for the others.
- **Only 20 acoustic epochs** in 2.4s total. More epochs + learning
  rate scheduling (Phase 2 item 7) might squeeze another 0.1 dB.

**JJK S2 regeneration** (23 episodes, E85 production model):
E85 produces **fewer but more confident filters** than E82:
4.7 filters/episode (vs 5.0), max gain/filter 5.4 dB (vs 3.6 dB).
Instead of 5 overlapping small shelves, E85 places 3–4 decisive
shelves at the acoustically correct frequencies. Also occasionally
predicts **negative-gain filters** (cuts) — something E82 never did
— when a cut is part of matching the target response shape.

**Verdict**: **E85 is the new production champion** at 1.49 dB mean
(−0.21 dB vs E82). Passes the ≥0.1 dB decision gate. Deployed via
`AUTO_BEQ_ADVISOR=torch_differentiable` + `scripts/train_torch_model.py`.

**Code**: `src/main/python/model/auto_beq_torch.py` (BiquadResponseLayer
+ FilterChainPredictor + train_e85_differentiable_dsp), 7 unit tests in
`test_auto_beq_torch.py`, `TorchFilterAdvisor` in `auto_beq_advisor.py`,
`scripts/train_torch_model.py` CLI.

**CSV**: `.pytest_cache/tier1_comparison.csv`,
`.pytest_cache/e85_diff_dsp.csv`
