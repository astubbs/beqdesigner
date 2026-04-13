# Branch plan: `feats/neural-net-strat`

## Goal

Train and deploy a production BEQ filter-prediction model from real
audio data, then explore paradigm-shift techniques beyond the initial
XGBoost baseline.

## Current champion

**E85 differentiable DSP — 1.49 dB mean** on the 1279-WAV full cache
(1023 train / 255 test stratified split). Trained in 2.4 seconds on
CPU. A 3-layer neural network (256 hidden, ~200K params) trained
through a differentiable biquad layer on the acoustic response match
error — directly optimising for the production metric instead of the
parameter-MSE proxy that all prior models used.

Deploy via:
```bash
# Train (once, ~40s total including XGBoost teacher):
BEQ_WAV_CACHE=/Volumes/jetspeed/beqdesigner/wav-cache \
  poetry run python3 scripts/train_torch_model.py

# Generate profiles (sub-second per episode):
AUTO_BEQ_ADVISOR=torch_differentiable \
  poetry run python3 scripts/generate_beq_profile.py --media-dir ... --output-dir ...
```

## Progression

| Milestone | Mean dB | Cache | Key change |
|---|---|---|---|
| E25b (hash encoding) | 7.43 | 7 | Initial baseline |
| E34 + late fusion α=0.7 | 2.45 | 220 | One-hot + late fusion |
| F1 (augmentation σ=0.5) | 2.02 | 67 | Synthetic augmentation |
| G8 (per-author α) | 1.86–1.92 | 67 | Author lookup at inference |
| I1b (soft routing) | 2.01 | 67 | Auto author from metadata |
| E77 (real-only plain XGB) | 2.12 | 932 | Real-audio regime switch |
| E82 (50:1 weighted hybrid) | 1.99 | 1091 | Real accuracy + synth coverage |
| E82 on 1279-WAV cache | 1.70 | 1279 | More data alone |
| **E85 (differentiable DSP)** | **1.49** | **1279** | **Acoustic loss via differentiable biquad** |

## Shipped this session

| Commit | What |
|---|---|
| `b94ad45` | Production model deployment + test segregation (unit/integration/experiment) |
| `d33e44e` | Paradigm-shift research menu (T1/T2/T3 brainstorm doc) |
| `1014d99` | E83 foundation-model infra + curve-feature/discovery caches (17.5× speedup) |
| `563a8c8` | E84 self-training (negative: only 11 unmatched WAVs) |
| `e347f87` | XGBoost native serializer (unblocks torch coinstall) |
| `04faac7` | extract_lfe.py uncatalogued-media mode + selection heuristic |
| `96a2454` | Logging cleanup + 10% progress ticks for long operations |
| `144804c` | E83 result: Whisper-tiny regresses +0.83 dB |
| `ec97207` | E85 module + 7 BiquadResponseLayer unit tests |
| `97aeabd` | Tier 1 unified comparison: E85 = 1.49 dB, new champion |
| `ff319e7` | E85 production promotion: TorchFilterAdvisor + CLI + JJK S2 regen |

## What's next

### Immediate (Phase 2)

- **Multi-type topology**: add softmax over (LowShelf, HighShelf,
  PeakingEQ) per slot. Targets the max-error regression (11.45 vs
  E82's 9.92) from titles needing non-LowShelf filters.
- **More epochs + LR schedule**: cosine annealing, 100 acoustic
  epochs. May squeeze another 0.1 dB.
- **E82+E85 ensemble router**: dispatch by author (aron7awol/
  halcyon888/remixmark → E82, mobe1969/t1g8rsfan/kaelaria → E85).
  Combines both models' strengths at inference via a routing table.

### Blocked on user action

- **NAS unmatched extraction**: re-run extract_lfe.py with
  `--extract-unmatched 50` on fatman. Then re-run E84 self-training
  with a real unlabelled pool.

### Later

- A/B profile comparison tool (overlay spectrographs + before/after
  WAV files for listening tests)
- Tier 2 experiments (classical DSP features, uncertainty gating,
  LightGBM/CatBoost, multi-task stacking)
- Tier 3 speculative bets (RL, diffusion, meta-learning, large LLM)

## Key files

| File | Role |
|---|---|
| `src/main/python/model/auto_beq_torch.py` | E85: BiquadResponseLayer + FilterChainPredictor + training |
| `src/main/python/model/auto_beq_nn.py` | E82 XGBoost + E84 self-training + feature pipeline |
| `src/main/python/model/auto_beq_advisor.py` | Advisor protocol + TorchFilterAdvisor + get_advisor() |
| `scripts/train_torch_model.py` | CLI: train E85 production model |
| `scripts/train_production_model.py` | CLI: train E82 production model |
| `scripts/generate_beq_profile.py` | End-to-end profile generation (E82 or E85) |
| `scripts/extract_lfe.py` | LFE WAV cache builder + uncatalogued extraction mode |
| `scripts/run_tier1_comparison.py` | Unified comparison runner (all 4 experiments) |
| `docs/design/auto_beq_nn_paradigm_shifts.md` | Research menu (T1/T2/T3) |
| `docs/design/auto_beq_experiments.md` | Experiment log (E1–E85) |
