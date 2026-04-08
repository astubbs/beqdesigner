# Branch plan: feats/audio-chunks-strat

## Branch goal

Validate whether chunked-percentile spectrum extraction produces a more
robust rolloff ceiling estimate than whole-film Welch average (the
E17d baseline). This is Experiment 2 from the audio chunking plan —
a new spectrum extraction code path, not a new advisor or optimizer.

Forked from: `feats/magic-wand` (which contains E1-E17d).

## Current state (2026-04-08)

### What's built
- **`load_and_smooth_chunked()`** in `_auto_beq_helpers.py`: splits
  WAV into fixed-length chunks, computes STFT peak per chunk, takes
  Nth-percentile across chunks, then normalizes+smooths identically
  to `load_and_smooth()`.
- **`test_chunked_percentile_roundtrip()`** in `test_auto_beq.py`:
  parametrized by `chunk_s=[30, 60, 90]`, logs delta vs baseline at
  10 Hz and 20 Hz, same manifest and grading gate as
  `test_real_media_roundtrip`.

### What's NOT changed
- No changes to `model/auto_beq.py`, `model/auto_beq_advisor.py`,
  `model/signal.py`, or any advisor implementation.
- Existing tests (`test_synthetic_roundtrip`, `test_real_media_roundtrip`)
  are unmodified.

## Hypothesis

The whole-film Welch average can be dominated by a single outlier scene
(E15c: EoT showcase scenes inflate 10 Hz by 16-19 dB). The 90th
percentile across chunk peaks asks "what level do almost all chunks
agree is the ceiling?" — a more stable estimate, especially for short
TV episodes with sparse bass content.

## Next steps

1. Run `test_chunked_percentile_roundtrip` across the media library.
2. Compare results to E17d baseline (10 PASS + 4 MARGINAL / 34 episodes).
3. Log results as E18 in `docs/design/auto_beq_experiments.md`.
4. Identify optimal chunk length from 30/60/90 s sub-experiment.

## Key files

| File | Role |
|---|---|
| `src/test/python/spike/_auto_beq_helpers.py` | `load_and_smooth_chunked()` |
| `src/test/python/spike/test_auto_beq.py` | `test_chunked_percentile_roundtrip()` |
| `docs/design/auto_beq_experiments.md` | Experiment log (append E18) |
| `branch-plans/plan-sharp-goldberg.md` | Parent branch plan (E1-E17d context) |
