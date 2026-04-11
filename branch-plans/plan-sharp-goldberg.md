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

### ML model path (E25-E31) — active
1. ✅ E25: XGBoost trained model + TMDb metadata enrichment
2. ✅ E25a-e: Studio vocab/grouping, ablation, full-catalogue validation
3. ✅ E27: Late fusion (best combined approach: 4.03 dB at α=0.7)
4. ✅ E28: CNN dual-branch (overfits on synthetic, needs real audio)
5. ✅ E29: BEQ author feature (metadata-only hits 3.09 dB — best ever)
6. ✅ E30: Expanded validation (20 titles, Mindhunter 0.83 dB)
7. ✅ WAV integrity system (found+deleted 5 corrupt WAVs, atomic writes)
8. ✅ Standalone LFE extraction script (portable cache, NAS-ready)
9. **In progress**: E31 — extract 50 titles from library, validate
10. **TODO:** Persist TMDb-enriched catalogue into git DB catalogue output
11. **TODO:** Train on real audio features (not synthetic)
12. **TODO:** Dockerise extraction script to eliminate standalone code duplication
    — script imports from main modules, runs in container with ffmpeg, no scp needed

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
