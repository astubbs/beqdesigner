# Branch plan: `feats/neural-net-strat`

## Goal

End-to-end NN-based auto-BEQ: train a production model from real WAV
cache + catalogue metadata, wire it into the production inference path,
regenerate JJK S1/S2 profiles for visual sanity check.

## Parent branches

- `feats/audio-chunks-strat` — chunked P90 audio feature strategy (E62+).
- `feats/sharp-goldberg` — per-author alpha router (G8 oracle / I1b).

## Current state

### Completed

- **E77–E82 experiment series**: real-audio training regime, per-author
  isolation, weighted hybrid router. Champion: E82 **50:1 sample-weighted
  hybrid plain XGBoost, 1.99 dB mean per-title error** on the 219-title
  test split. Matches real-only accuracy AND retains synthetic coverage
  for the ~7k catalogue titles without real WAVs.
- **A1**: `train_production_weighted_hybrid()` in `auto_beq_nn.py` —
  canonical training function for the production model.
- **A2**: `scripts/train_production_model.py` — CLI wrapper. Discovers
  WAV cache, extracts features in parallel, trains, saves model +
  metadata sidecar to `{beq-dir}/production_model.joblib`.
- **A3**: `get_advisor("trained_model")` in `auto_beq_advisor.py` now
  falls back to `{beq-dir}/production_model.joblib` when
  `AUTO_BEQ_MODEL_PATH` is unset, and warns when the model is >7 days old.
- **A4**: Unit tests for the training function in
  `test_auto_beq_nn.py` (happy-path metadata + model shape, empty-real
  rejection). 24 → 26 unit tests, all passing in ~9 s.
- **B1**: Cherry-picked 4 commits from `feats/magic-wand` to port
  `scripts/generate_beq_profile.py` (end-to-end + temp-dir fix + Blu-ray
  ISO support + author in spectrograph title).
- **B2**: `generate_beq_profile.py` now loads the saved production
  model once in `main()` and passes it to `generate_profile()`. Falls
  back to inline late-fusion α=0.3 training + warning if the saved
  model is missing. Batch mode no longer retrains per-episode.

### In progress

- **B3**: Run `generate_beq_profile.py` on JJK S1/S2 episode directories,
  output to `profiles/jjk_s1_v2/` and `profiles/jjk_s2_v2/`. Prerequisite:
  train the production model on the current WAV cache via
  `train_production_model.py`.

### Pending

- **B4**: Visual compare new profiles against the earlier I1b-trained
  profiles (if they exist from earlier runs). Spectrograph overlays +
  filter-table diffs.
- **E78**: Baseline variance debug — 0.14 dB gap between harness
  (2.83 dB) and standalone (2.690 dB) baseline runs. Not blocking
  production deployment; debug when convenient.

## Key files

| File | Role |
|---|---|
| `src/main/python/model/auto_beq_nn.py` | `train_production_weighted_hybrid()` + `save_model()` + `load_model()` |
| `src/main/python/model/auto_beq_advisor.py` | `get_advisor("trained_model")` with `{beq-dir}/production_model.joblib` fallback |
| `scripts/train_production_model.py` | CLI: train + save production model |
| `scripts/generate_beq_profile.py` | CLI: load production model, generate per-episode BEQ profiles |
| `src/test/python/spike/test_auto_beq_nn.py` | 26 unit tests incl. production-weighted-hybrid coverage |
| `docs/design/auto_beq_experiments.md` | E77–E82 results documented |

## How to use the production model

```bash
# 1. Train (once per WAV cache update — takes a few minutes on 1k WAVs)
BEQ_WAV_CACHE=/Volumes/jetspeed/beqdesigner/wav-cache \
  poetry run python3 scripts/train_production_model.py

# 2. Generate profiles (sub-second per episode)
poetry run python3 scripts/generate_beq_profile.py \
  --media-dir "/path/to/Show (2020)/Season 01/" \
  --output-dir profiles/show_s1/
```

Or use from tests / other scripts via
`AUTO_BEQ_ADVISOR=trained_model` (no `AUTO_BEQ_MODEL_PATH` needed —
will auto-discover the saved model).
