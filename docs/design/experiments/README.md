# Auto-BEQ experiment log

**See also:** [`auto_beq.md`](../auto_beq.md) - main auto-BEQ reference

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

## Experiment families overview

**Start here if you're new to the project.** This section is a
navigation aid for the detailed per-experiment entries below.

### Naming convention

Every experiment has a permanent sequential `E<N>` identifier (E1, E2,
…, E77+).  Experiments that share a research theme are grouped into a
**family** with a letter prefix (F, G, H, I, J).  Letter-prefix IDs
like `F1` or `G8` are convenience aliases used in commit messages and
ad-hoc discussion — they always map to one or more `E<N>` IDs in this
log.

For example: `F1` = E41, `G8` = E59, `I1b` = E69 (variant b).  A
single letter-ID can cover multiple experiments when the same idea is
swept across sub-variants (e.g. `G2a` through `G2e` for the alpha
sweep = E54a–E54e).

### Family map

| Family | E-range | Theme | Status | Key result |
|---|---|---|---|---|
| — | E1–E6 | Procedural heuristics (shelf + PEQ iterative fitter, rolloff classifier, advisor abstraction) | Dead end | Overfit to 3 fixtures; fitter math proven (E2, E6) |
| — | E7–E13 | LLM-based tier classification (Ollama llama3.1:8b) | Dead end | Small LLMs too weak for numeric calibration |
| — | E14–E17 | Pure-measurement advisor (deficit + slope extension, topology classes) | Kept | 41% non-FAIL ceiling; topology classifier adopted |
| — | E18–E22 | Spectrum extraction (chunked STFT, blended Welch+chunked) | Adopted | `blend-a0.7-P90` default extraction strategy |
| — | E25–E40 | Initial ML baseline (XGBoost, metadata encoding, late fusion, one-hot type) | Adopted | 2.45 dB on 220 titles (late fusion α=0.7 + one-hot) |
| **F** | E41–E52 | NN accuracy improvements (augmentation, Option B, absolute dBFS, clustering, …) | Partial — F1 kept | **F1 = synthetic feature augmentation σ=0.5** was the breakthrough (-0.73 dB) |
| **G** | E53–E59 | Combinations and hyperparameter tuning (alpha sweep, sigma sweep, XGB params, ensembles, per-author α lookup) | Adopted | **G8 per-author α** hits the oracle ceiling; G4b σ=0.3 is the best single-α |
| **H** | E60–E67 | Multi-author resolution (response averaging, marginalization, quality filtering, response curve prediction) | Dead end | Key lesson: *multi-author disagreement is signal, not noise* — averaging regresses |
| **I** | E68–E70 | Automated author selection via metadata classifier (hard / soft-blend / top-3) | Adopted | **I1b soft-blend is the production model** — 2.37 dB, fully automated, no user input |
| **J** | (tools, no E-numbers) | Data acquisition tooling (bias analysis, acquisition recommender) | Tools | Scripts live in `scripts/nn_cache_bias_report.py`, `scripts/nn_acquisition_recommender.py` — not model experiments |
| — | E71–E74 | 932-WAV scale-up re-validation | Kept | Honest measurement on the full catalogue distribution; rankings preserved, numbers +0.3–0.5 dB |
| — | E75 | XGBoost `n_jobs=1` determinism fix | Adopted | Baseline bit-identical across runs; unit tests 10× faster |
| — | E76 | I4 per-author dedicated late-fusion + classifier routing | Dead end | Real-only simpler and better; classifier hedging beats hard routing |
| **E77+** | E77–E82 | **Real-audio training regime** | **Current** | **50:1 weighted hybrid plain XGB = 1.99 dB, real-audio regime kicks in at just 100 real WAVs** |

### Current production model

**50:1 sample-weighted hybrid plain XGBoost (E82)** — the new
champion at **1.99 dB** on the 219-title E77/E82 test split
(1091-WAV cache, 80/20 stratified by rolloff severity, random_state=42).

- **Training**: `train_xgboost(X_combined, Y_combined, sample_weight=w)`
  where `X_combined` is real WAV features concatenated with the full
  synthetic set, and `w` is `[50.0] * n_real + [1.0] * n_synth`.
- **No late fusion, no augmentation, no classifier routing**.  Those
  were all crutches for the synthetic-to-real gap.  With sample
  weighting, they're unnecessary.
- **Key trick**: real samples are outnumbered ~10:1 by synthetic but
  the weight ratio (50:1) rebalances their contribution to the XGBoost
  loss.  This avoids the E77 naive-hybrid failure mode where synthetic
  drowned out real signal.
- **Best of both worlds**: delivers real-only accuracy (matches plain
  real-XGB at 1.99 dB) AND retains synthetic coverage for titles
  without real WAVs — in a single model.
- **Per-author wins**: real-XGB wins 5 of 6 authors on the 219-title
  test split, including the previously-difficult remixmark
  (3.03 → 1.28 dB, -1.75 dB improvement).

#### Legacy models (the synthetic regime)

Kept in the codebase for reference and fallback when real-audio
training data is unavailable:

- **`I1b-soft-blend`**: 2.25 dB on the 219-title split. The auto-
  classifier production model from the synthetic era.  Only useful if
  you don't have any real WAVs to train on.
- **`G8-perauth`**: 2.24 dB on the 219-title split. Oracle upper
  bound when the author is known.  No longer meaningfully better than
  I1b at scale.
- **F/G/H/I series** (augmentation, late fusion, per-author α, classifier
  routing): all were regime-specific — optimal for the synthetic regime,
  net-negative in the real-audio regime.  E77 found that adding late
  fusion + augmentation on top of real-only training regresses by
  0.12–0.14 dB.  Don't use them.

#### Threshold

E81 shows the real-audio regime kicks in at as few as **100 real
WAVs**.  Plateau is around **400 real WAVs** where additional data
stops helping.  We passed 100 months ago, so the F/G/H/I-series work
was architecturally optimal for a regime we had already left.

### Detailed experiment logs

Each file covers one experiment family. Read the one relevant to your work:

- [E1-E6: Procedural heuristics](E01-E06-procedural-heuristics.md) - dead end
- [E7-E13: LLM tier classification](E07-E13-llm-tier-classification.md) - dead end
- [E14-E22: Measurement advisor + extraction](E14-E22-measurement-and-extraction.md) - adopted
- [E25-E40: ML trained models](E25-E40-ml-trained-models.md) - adopted
- [F-series (E41-E52): Accuracy improvements](E41-E52-f-series-accuracy.md) - partial
- [G-series (E53-E59): Hyperparameter tuning](E53-E59-g-series-tuning.md) - adopted
- [H-series (E60-E67): Multi-author resolution](E60-E67-h-series-multi-author.md) - dead end
- [I-series (E68-E76): Author selection + scale-up](E68-E76-i-series-and-scaleup.md) - adopted
- [E77-E82: Real-audio training](E77-E82-real-audio-champion.md) - **champion**
- [E83-E87: Paradigm shifts](E83-E87-paradigm-shifts.md) - active

