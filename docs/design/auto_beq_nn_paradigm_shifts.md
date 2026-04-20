# Paradigm-shift experiments for auto-BEQ (post-E82)

**Date**: 2026-04-11
**Context**: E82 parked the within-paradigm race at **1.99 dB mean
per-title error** on the 219-title test split (50:1 sample-weighted
hybrid plain XGBoost, 102 hand-crafted features → 36 filter params,
stratified random_state=42).

The F/G/H/I series (E41–E70) exhaustively swept the tabular-XGBoost
family: feature engineering (Option B chunk stats, absolute dBFS,
rolloff clustering, multi-resolution bins), training tricks
(augmentation, late fusion, confidence weighting, downstream
reweighting, per-author α lookup, classifier routing), ensemble
variants, and the real-audio regime crossover (E77) + weighted hybrid
finish (E82). Remaining levers inside this paradigm look like <0.1 dB
individual gains — diminishing returns.

**To keep progressing we need paradigm shifts, not more within-paradigm
tuning.** This doc is a curated research menu of techniques from the
wider world of audio engineering + ML that we have NOT tried yet,
ranked by expected impact and feasibility for this codebase.

Companion docs:
- [`auto_beq_nn_future_experiments.md`](auto_beq_nn_future_experiments.md)
  — older F-series brainstorm (largely implemented, within the old
  paradigm). Start there for historical context.
- [`auto_beq_experiments.md`](auto_beq_experiments.md) — append-only
  experiment log (E1–E82). Source of truth for what's been tried.
- [`auto_beq.md`](auto_beq.md) — vision and overall design.

**Hook points already in place** that any new experiment should reuse:

- `build_feature_vector()` → 102-dim concat
  (`src/main/python/model/auto_beq_nn.py:481`)
- `train_xgboost(X, Y, sample_weight=w)` (`auto_beq_nn.py:981`)
- `TrainedModelAdvisor` + `get_advisor("trained_model")` — works with
  any `.predict(X)`-compatible object, including PyTorch `nn.Module`
  wrapped in a `predict` method
- `discover_wav_catalogue_pairs()` + `_extract_features_parallel()` —
  canonical (path, catalogue_entry) pairs across the 1277-WAV cache
- `compute_match_metrics()` / `evaluate_filter_chain()` — the acoustic
  loss function that IS the real objective (vs MSE on params)
- `N_OUTPUT = 36` (6 slots × 6 params per slot) — stable label space
- `{beq-dir}/production_model.joblib` — current champion artifact
  (E82 50:1 weighted hybrid)

## Tier 1 — Paradigm shifts with the highest expected impact

These three target fundamental limitations of the current approach
(hand-crafted features, param-MSE loss, labelled-only data). Each one
could individually move the mean error 0.3–0.8 dB.

### T1.1 — Audio foundation model embeddings as features

**Technique**: Replace / augment the 9 hand-crafted audio bins with
embeddings from a pre-trained audio foundation model. Feed the frozen
embedding vector straight into the existing XGBoost pipeline (no
neural re-training needed, just a new feature extractor).

**Candidates, roughly in order of fit to this task**:

| Model | Why it might work | How to use |
|---|---|---|
| **Whisper encoder** (OpenAI) | Encoder-only half of Whisper is a powerful general audio encoder even for non-speech tasks. CPU-viable at tiny/base sizes. 16 kHz mel input. | `whisper.load_model("tiny").encoder`, mel-pad to 30 s, mean-pool. 384-dim (tiny) / 512-dim (base). |
| **OpenL3** | Purpose-built general audio embedder for downstream classification. 6-layer CNN. CPU-friendly. | `openl3.get_audio_embedding(audio, sr)` → 512-dim mean-pool. |
| **EnCodec** (Meta) | 24 kHz neural codec with discrete latent tokens — compresses waveform to a semantic latent. Trained on general audio incl. music. | `encodec` package, pool across time → 128-dim. |
| **BEATs** (Microsoft) | SSL audio transformer pre-trained on AudioSet. SOTA 2023 on audio tagging. | HuggingFace `microsoft/BEATs`. 768-dim pooled embedding. GPU preferred. |
| **AudioMAE** (Meta) | Masked autoencoder on mel-spectrograms. Captures spectral texture at many scales. | HuggingFace ViT-style; direct AudioMAE repo. |
| **wav2vec2-base** | SSL speech model. Lower fit for bass-band but very cheap to run. | HuggingFace `facebook/wav2vec2-base`. |
| **PANNs / YAMNet** | Pre-trained audio tagging CNNs. Classical but battle-tested. | TF-hub `yamnet` or `panns-inference`. |

**Hook**: add a helper
`extract_foundation_embedding(media_path, model) -> np.ndarray`
alongside `extract_curve_features()` in `auto_beq_advisor.py`. Feed
into a new `AudioFeatureConfig(foundation_model=...)` branch of
`build_feature_vector()`. XGBoost picks which features matter via
feature importance.

**Caching**: embeddings are small (hundreds of floats) vs WAVs (MBs).
Cache per-media under `{beq-dir}/foundation-embeddings/<model>/<key>.npy`
so reruns are free.

**Expected impact**: 0.3–0.8 dB. Foundation-model embeddings have
dominated audio ML benchmarks for 3 years; the signal they capture
(spectral texture, dynamics, codec artefacts) is precisely what 9
Welch bins discard.

**Risk**: GPU-heavy models may be too slow over 1277 WAVs. Mitigation:
Whisper-tiny, EnCodec and OpenL3 are all CPU-viable at small batch.

**Verification**: train the weighted-hybrid with the new features,
measure per-title error on the E82 219-title split, compare to 1.99 dB
baseline. Also report feature importance ranking — expect foundation
features to be in the top 10 by gain.

### T1.2 — Differentiable DSP: train on acoustic loss

**Technique**: Replace the MSE-on-params training objective with a
differentiable evaluation of the predicted filter chain's frequency
response, then compute dB error against the target response. This is
F9 from `auto_beq_nn_future_experiments.md` taken to its conclusion:
don't reweight MSE, replace it.

**Why it matters**: two very different filter parameter sets can
produce nearly identical acoustic responses. The current MSE objective
penalises "wrong numbers" even when the acoustic result is right,
pushing the model away from acoustically-equivalent solutions and
flattening the loss landscape in the wrong places.

**Implementation approach**:
1. Write a PyTorch module `BiquadResponseLayer` that takes filter
   params `(type_idx, freq, gain, q, enabled)` and returns the log-mag
   response on the `DEFAULT_GRID` (1000 Hz fs, log grid). The biquad
   coefficients for LowShelf / HighShelf / PeakingEQ are closed-form
   trig; PyTorch handles autograd through sin/cos.
2. Regression head: feature vector → (6, 5) filter chain params.
3. Loss = `|| BiquadResponse(predicted) - target_response ||_2` in dB
   space, weighted by the 5–80 Hz band mask (where BEQ matters).
4. Train with Adam. Start with MSE warm-start from the XGBoost teacher
   for stability, then switch to acoustic loss for fine-tuning.
5. Wrap in a `TorchFilterAdvisor` that satisfies the existing
   `Advisor` protocol.

**Hook**: new file `src/main/python/model/auto_beq_torch.py`. No
changes to the XGBoost pipeline — parallel model family that
`get_advisor("torch_differentiable")` selects at runtime per AGENTS.md
experiment-preservation rules.

**Expected impact**: 0.3–0.6 dB. The task's real metric IS the acoustic
match, and we've been optimising a proxy.

**Risk**: filter-type is categorical → softmax relaxation or
straight-through estimator adds optimisation complexity. Mitigation:
start with fixed 5-slot all-LowShelf topology (matches most of E82's
predictions) and only relax type selection if the fixed-topology
version works.

### T1.3 — Self-training on unlabelled real WAVs

**Technique**: Self-training with the E82 production model as the
teacher. For every real WAV in the cache that has NO catalogue entry
(~200 today, 1000+ if extraction runs against fresh content), generate
a pseudo-label via the teacher, confidence-filter, add to training
set, retrain. Iterate.

**Why it matters**: the E82 test split is 219 titles and the full WAV
cache is 1277. The difference is titles WITHOUT catalogue entries —
currently unusable for supervised training but usable as unlabelled
data. Self-training can 2-3× the effective training signal.

**Implementation approach**:
1. Run the production model on every unmatched real WAV in the cache.
2. For each pseudo-label, compute a **self-consistency score**: the
   predicted filter chain's acoustic response vs the measured curve
   itself, via `compute_match_metrics`. Low mean error = confident.
3. Select the top-K pseudo-labels (say, top 50%) as high-confidence.
4. Retrain the weighted hybrid with
   `X_combined = [real_labelled; synthetic; pseudo_labelled]`,
   `sample_weight = [50; 1; k]` where k < 50 (pseudo-labels are
   noisier than ground-truth real).
5. Iterate: the new model generates better pseudo-labels, etc.

**Hook**: new function `pseudo_label_unmatched_wavs()` in
`auto_beq_nn.py` alongside `train_production_weighted_hybrid()`. New
CLI flags on `scripts/train_production_model.py`.

**Expected impact**: 0.2–0.5 dB. Diminishing returns after one or two
iterations. Biggest win: per-author performance on underrepresented
authors, whose test-set bias is a major error source today.

**Risk**: pseudo-label noise can drift the model if confidence
filtering is too loose. Mitigation: use the self-consistency metric
(the model's own confidence in its own output) as the filter, tighten
iteratively.

## Tier 2 — Incremental wins and novel representations

Lower expected impact individually but cheap and cumulative.

### T2.1 — Classical DSP features we never tried

The 9-bin log-spaced Welch pixel is a lossy representation.
Alternatives from the audio-engineering canon worth trying as drop-in
additions to `build_feature_vector()`:

| Feature | What it captures | Est. cost |
|---|---|---|
| **Constant-Q transform (CQT)** | True log-frequency representation with equal relative bandwidth per bin (critical for bass). `librosa.cqt`. | 1 hr |
| **Gammatone filter bank** | Auditory-scale (ERB) bandpass energies — matches how humans hear sub-bass. `gammatone` package. | 1 hr |
| **Loudness (ITU-R BS.1770 / EBU R128)** | Short-term integrated loudness — broadcast mastering target (Netflix –27 LKFS, Disney+ –24, theatrical –23). A feature the model could use to predict mastering intent directly. `pyloudnorm`. | 30 min |
| **LPC residual envelope** | Linear-prediction residual captures rolloff slope shape via the LPC coefficients themselves. `librosa.lpc`. | 1 hr |
| **Spectral flatness, crest, rolloff point, centroid** | MIR classics — one scalar each. All in `librosa.feature`. | 30 min |
| **Modulation spectrum** | Amplitude modulation rate of bass — separates "sustained sub-bass" from "kick-drum driven" content. Classical in music-genre classification. | 2 hrs |
| **Time-domain stats (RMS, crest factor, ZCR)** | Captures compression/limiting — heavily compressed masters have very different BEQ needs than uncompressed. | 30 min |

**Hook**: extend `AudioFeatureConfig` with `use_cqt`, `use_gammatone`,
`use_loudness`, etc. Feature-importance analysis will show which ones
XGBoost actually picks up.

**Expected impact**: 0.1–0.3 dB cumulative. Compounds with T1.1.

### T2.2 — Uncertainty quantification + heuristic fallback

**Technique**: Quantile XGBoost (or conformal prediction wrapping the
production model) to emit a prediction interval for each filter param.
When the interval is wide (high uncertainty), fall back to the
`HeuristicAdvisor` or `MeasurementAdvisor` instead of trusting the
NN-predicted filter chain.

**Why it matters**: E82's 1.99 dB mean hides a long tail — some titles
have 4–5 dB error. Users would rather get a known-safe heuristic on
hard titles than a confidently-wrong NN prediction. This is
"calibrated abstention".

**Implementation approach**: train three XGBoost models with
`objective="reg:quantileerror"` and `alpha=[0.1, 0.5, 0.9]`. Uncertainty
for filter param = `q90 - q10`. Threshold on aggregated uncertainty;
above threshold, `get_advisor()` routes to the heuristic fallback.

**Hook**: new advisor class `UncertaintyGatedAdvisor` in
`auto_beq_advisor.py` that wraps a production advisor and a fallback
advisor with a confidence gate.

**Expected impact**: 0.0 dB on mean error, but **reduces the worst-case
error dramatically**. UX win rather than a metric win.

### T2.3 — LightGBM / CatBoost head-to-head

Retrain E82's exact pipeline with LightGBM (histogram-based GBM) and
CatBoost (ordered boosting, native categorical support). No code
changes beyond the training call. ±0.1 dB typical, worth the hour it
takes to confirm which GBM flavour handles this dataset best.

### T2.4 — Multi-task auxiliary heads

Train one XGBoost per auxiliary target (author, studio, content_type,
rolloff cluster from F7) and concatenate their predictions as
additional features for the main filter-params regressor. Classical
stacking. 0.1–0.2 dB expected.

## Tier 3 — Speculative high-variance bets

Exploratory or deeper infrastructure investment. Include-if-curious.

### T3.1 — Reinforcement learning with acoustic reward

RL agent proposes a filter, measures acoustic match, updates. Reward =
`-mean_abs_err_db`. Action space = continuous filter params. Handles
the non-differentiable categorical filter-type selection that T1.2 has
to hack around.

`stable-baselines3` PPO on a `gym.Env` wrapping `evaluate_filter_chain`.
Episodes are single-title, one filter per step, up to 6 steps.

**Risk**: sample inefficiency on 1277 titles. Likely needs expert
demonstrations (the catalogue) as offline warm-start → BC+RL.

### T3.2 — Conditional diffusion for filter-chain generation

Train a DDPM over the 36-dim filter param space, conditioned on the
102-dim input features (or T1.1 embedding). Sample multiple filter
chains per title, score with acoustic match, take the best.

**Why**: the output space is multimodal (multiple valid filter chains
per title). Regression models collapse to the mean, losing mode
diversity. Generative models preserve it.

**Risk**: 1277 samples is tiny for a diffusion model. Likely needs
pretraining on synthetic chains.

### T3.3 — Meta-learning for per-author few-shot adaptation

MAML (or simpler Reptile / ANIL) over the per-author split. Learn an
initialisation that adapts to a new author with 3–5 examples.

**Why**: mobe1969's inconsistent choices and remixmark's distinct style
are currently noise in the same training set. Meta-learning gives each
author their own weight update at inference.

### T3.4 — Large LLM as filter designer (revisit)

E7–E13 tried llama3.1:8b and failed — small LLMs couldn't do numeric
calibration. 2025-class LLMs (Claude Sonnet 4.5, GPT-4.1, Llama 3.1
70B) are materially better at numeric reasoning and tool use.

Give the LLM the measured curve summary + title metadata + a few
catalogue examples, ask it to output a JSON filter chain via
tool-calling. New `LLMAdvisor` class parallel to the existing
`OllamaAdvisor`. Even if mean error doesn't drop, the LLM might be
better at *explaining* its output — useful for debugging hard titles
and user trust.

**Risk**: latency (multi-second per prediction), cost (API calls),
and the LLM may regress to the catalogue mean.

### T3.5 — Graph neural net on the catalogue

Model the catalogue as a graph: titles ↔ authors ↔ studios ↔
composers ↔ genres. Learn node embeddings (GraphSAGE, GAT), use the
title embedding as a feature.

**Why**: some signal leaks between titles sharing a sound engineer or
mixing studio that hand-crafted features can't capture. GNN
propagation makes it explicit.

**Risk**: TMDb crew data quality is inconsistent; graph may be too
sparse.

## Cross-cutting: evaluation rules

Every experiment on this list MUST:

1. Use the **E82 219-title test split**, `stratify=severity,
   random_state=42`. The split is the constant across experiments.
2. Report **mean AND max per-title error**, plus **per-author
   breakdown** (E80 showed different techniques help different authors).
3. Produce a **reproducible CSV row** in `.pytest_cache/e<N>_<name>.csv`
   for future aggregation via the existing `scripts/nn_*_report.py`
   family.
4. Be **logged in `docs/design/auto_beq_experiments.md`** with a new
   `E<N>` ID (next free: E83+).
5. Live as a **selectable alternative** per AGENTS.md — new advisor /
   training function, not in-place replacement. Promotion to
   production happens only via an explicit commit after a decision gate.
6. Be marked `@pytest.mark.experiment` (if the test retrains models)
   so it runs only via `bash scripts/run-spike-experiments.sh`.

## Recommended execution order

Strictly within Tier 1 first (biggest expected impact), alternating
foundational infra and quick wins:

1. **T1.1 (foundation model features) — Whisper-tiny encoder or
   OpenL3**. Easiest entry into deep audio learning. Drop-in, no NN
   training needed. Direct concatenation with existing features.
2. **T1.3 (self-training / pseudo-labels)**. Zero new ML, maximises
   data utilisation. Compounds with T1.1.
3. **T2.1 (classical DSP features)**. Cheap half-day additions that
   compound with T1.1.
4. **T2.2 (uncertainty quantification + heuristic gate)**. Orthogonal
   UX improvement; worst-case errors, not mean.
5. **T1.2 (differentiable DSP, PyTorch)**. Larger investment, highest
   ceiling. Do this once T1.1 proves learned features help at all.
6. **T2.3 (LightGBM / CatBoost)**. Parallel sanity check, 1 hour.
7. **T3.x** only after Tier 1 is exhausted and we still haven't
   cracked 1.5 dB.

**Stop condition**: if one of T1.1/T1.3 drops mean error below 1.5 dB
on the E82 test split, downgrade the remaining Tier 1 items to
"validation" rather than "blocker" and move on to the Tier 2 UX
improvements (especially T2.2 uncertainty gating).

## Critical files (per experiment)

| Experiment | Files |
|---|---|
| T1.1 foundation features | `auto_beq_advisor.py` (`extract_foundation_embedding`), `auto_beq_nn.py` (`AudioFeatureConfig`, `build_feature_vector`), new test in `test_auto_beq_nn_real.py` |
| T1.2 differentiable DSP | NEW `src/main/python/model/auto_beq_torch.py` (BiquadResponseLayer + regression head), `auto_beq_advisor.py` (new advisor class), `pyproject.toml` (torch dep, experiment group) |
| T1.3 pseudo-labels | `auto_beq_nn.py` (`pseudo_label_unmatched_wavs`), `_auto_beq_helpers.py` (`discover_unmatched_wavs`), `scripts/train_production_model.py` (`--self-train` flag) |
| T2.1 DSP features | `auto_beq_advisor.py` (extract_*), `auto_beq_nn.py` (`AudioFeatureConfig`) |
| T2.2 uncertainty gate | `auto_beq_advisor.py` (`UncertaintyGatedAdvisor`), `auto_beq_nn.py` (quantile XGBoost) |
| T2.3 LightGBM/CatBoost | `auto_beq_nn.py` (`train_lightgbm`, `train_catboost`), `pyproject.toml` |
| T2.4 multi-task | `auto_beq_nn.py` (stacking training pipeline) |
| T3.1 RL | NEW `src/main/python/model/auto_beq_rl.py`, `pyproject.toml` (`stable-baselines3`) |
| T3.2 diffusion | NEW `src/main/python/model/auto_beq_diffusion.py` |
| T3.3 meta-learning | NEW `src/main/python/model/auto_beq_meta.py` |
| T3.4 LLM designer | `auto_beq_advisor.py` (new LLM advisor class) |
| T3.5 graph NN | NEW `src/main/python/model/auto_beq_gnn.py`, TMDb crew fetcher extension |

## Dependencies added (by tier)

| Tier | Package | Size/cost | Notes |
|---|---|---|---|
| T1.1 | `openai-whisper` OR `openl3` OR `encodec` | 500 MB – 1 GB (model weights) | Pick one primary; others on demand |
| T1.2 | `torch` (CPU wheel) | ~700 MB | New opt-in poetry group `experiment` |
| T2.1 | `librosa`, `pyloudnorm`, `gammatone` | ~100 MB total | |
| T2.3 | `lightgbm`, `catboost` | ~50 MB each | |
| T3.1 | `stable-baselines3`, `gymnasium` | ~50 MB | |
| T3.4 | `anthropic` or `openai` SDK | small, but API cost per call | |

All new deps land under
`[tool.poetry.group.experiment.dependencies]` (new opt-in poetry
group, not installed by default). Install locally with `poetry
install --with experiment`. CI and the main build image stay lean.

## What NOT to do

- Don't replace E82 production until a new experiment beats it on the
  E82 test split AND in a JJK regeneration spot-check. Experiment
  branches stay as `get_advisor("<name>")` alternatives per AGENTS.md.
- Don't invest in Tier 2 or Tier 3 before trying T1.1 and T1.3 — the
  expected-impact gap is too wide.
- Don't re-run F/G/H/I-style within-paradigm tuning on top of any new
  technique. E77 showed that stacking synthetic-era crutches on top of
  real-audio training regresses.
- Don't commit trained model artifacts. Only save/distribute the
  training script + logs + a joblib/pt the user can regenerate locally.
- Don't mix multiple Tier 1 experiments in one commit — each needs its
  own E-ID in the log and its own A/B measurement.
- Don't install PyTorch / Whisper / OpenL3 into the main poetry group
  — opt-in `experiment` group only, CI untouched.
