# Branch plan: claude/sharp-goldberg (auto-BEQ spike)

## Branch goal

Validate whether BEQ filter chains can be proposed automatically from
measured LFE audio — "take the human out of BEQ-making". This is the
technical spike for the magic-wand vision (Tier 1 of feats/magic-wand).

## Current state (2026-04-06)

### What's built and working
- **N-filter scipy fitter** (`model/auto_beq.py`): given ANY target
  correction curve, fits LowShelf + PEQs within 0.5 dB. Proven.
- **Advisor abstraction** (`model/auto_beq_advisor.py`): pluggable
  interface. Implementations: Heuristic, Measurement, Mock, Ollama.
- **Library sweep** (`test_auto_beq_library_sweep.py` +
  `sweep_discover.py`): auto-discovers films, cross-refs catalogue,
  runs pipeline, writes CSV.
- **Test infrastructure**: 70+ tests.
- **Experiment log**: 15 experiments in
  `docs/design/auto_beq_experiments.md`.

### Key findings (E1-E15)
1. Fitter is NOT the bottleneck — scipy works perfectly.
2. LLM tier classification is a dead end (E8-E13).
3. Pure-measurement formulas don't generalise from 3 data points (E14).
4. EoT was a bad test case (E15c): wrong codec match. Blacklisted.
5. Catalogue-first is the right architecture: popular films already
   have profiles. Auto-generation is only for uncatalogued content.

## Next steps (prioritised)

### ML model path (E18-E19) — active
1. ✅ Design document: `docs/design/auto_beq_ml_experiments.md`
2. **In progress**: Build `auto_beq_nn.py` + `TrainedModelAdvisor` (E18 code path)
   - Synthetic training data from catalogue filter chains
   - XGBoost multi-output regressor
   - 60-dim feature vector: 9 audio (Option A) + 51 metadata
   - `MediaMetadata` extended with studio/genres/source/mixer fields
   - New tests: `src/test/python/spike/test_auto_beq_nn.py`
3. ✅ TMDb metadata fetcher — batch fetch studio/mixer, repo-committed mirror
4. Run ablation: audio-only vs audio+metadata (XGBoost feature importances)
5. Escalate to 1D CNN on RTX 3090
6. E19 hybrid (model warm-start + scipy) — after E18 validated
7. **TODO:** Persist TMDb-enriched catalogue into git DB catalogue output
   (include metadata alongside filter params in produced profiles)

### Immediate (pre-ML)
1. Run library sweep against user's library.
2. Pick 3-5 diverse sample films for ongoing algo development.
3. Catalogue-first pipeline (Plan 5 in `.claude/plans/`).

### Deferred
- Improve MeasurementAdvisor (needs sweep statistics).
- Absolute mid-bass energy hypothesis.
- Magic-wand button UI, batch CLI, ezBEQ integration.

## Key files

| File | Role |
|---|---|
| `src/main/python/model/auto_beq.py` | Pipeline + fitter |
| `src/main/python/model/auto_beq_advisor.py` | Advisor abstraction |
| `src/test/python/spike/test_auto_beq.py` | Integration tests |
| `src/test/python/spike/test_auto_beq_library_sweep.py` | Sweep test |
| `src/test/python/spike/sweep_discover.py` | Discovery CLI |
| `docs/design/auto_beq.md` | Living design doc |
| `docs/design/auto_beq_experiments.md` | Experiment log (E1-E19) |
| `docs/design/auto_beq_ml_experiments.md` | ML model experiments design (E18-E19) |
| `src/main/python/model/auto_beq_nn.py` | ML advisor (TrainedModelAdvisor) |
| `src/test/python/spike/test_auto_beq_nn.py` | ML advisor spike tests |
| `scripts/run-spike-tests.sh` | Test wrapper |
| `scripts/run-sweep-discover.sh` | Discovery wrapper |
