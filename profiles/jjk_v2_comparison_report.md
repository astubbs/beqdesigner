# JJK v2 regeneration — new production model sanity check

**Model**: E82 50:1 weighted hybrid plain XGBoost
**Trained on**: 1277 real WAVs + 7710 synthetic catalogue entries
**Script**: `scripts/generate_beq_profile.py` (B2: loads saved production
model at boot, predicts per-episode in sub-second)

## Output

- **S1**: 24/24 episodes, 0 errors — `profiles/jjk_s1_v2/`
- **S2**: 23/23 episodes, 0 errors — `profiles/jjk_s2_v2/`

Both seasons are stereo-only (S1: Bluray FLAC 2.0, S2: WEBDL AAC 2.0).
Neither has a real LFE channel, so the extractor falls back to mono
downmix — the model is inferring correction from the mono low-frequency
content, not a discrete LFE track.

## Filter topology (all 47 episodes)

| Statistic | Value |
|---|---|
| Filters per episode | 4–6 (mostly 5) |
| Dominant topology | 3–4× LowShelf + 1–2× PeakingEQ |
| Frequency range | 7–46 Hz (sub-bass) |
| Max per-filter gain | 2.7–4.4 dB (all positive — correcting rolloff) |
| MV adjust | +7.3 to +12.5 dB |

## S1 vs S2 characteristic split

| Season | Source | Max filter Hz | Typical MV | Max gain / filter |
|---|---|---|---|---|
| S1 (24 eps) | Bluray FLAC 2.0 | 40–46 Hz | +7.3 to +11.6 | 2.7–3.4 dB |
| S2 (23 eps) | WEBDL AAC 2.0 | 34–38 Hz | +7.9 to +12.5 | 3.3–4.4 dB |

S2 (WEBDL) gets slightly more aggressive correction than S1 (Bluray),
consistent with lossy streaming usually rolling off harder than Bluray.

## Old vs new: S02E01 (one point of comparison)

Earlier pre-port run produced `profiles/jjk_s02e01_beq.json` using
inline `train_late_fusion(alpha=0.3)` — the old synthetic-only model
that `generate_beq_profile.py` retrained every single run.

| | OLD (late fusion α=0.3, synthetic) | NEW (E82 weighted hybrid, production) |
|---|---|---|
| note | `late fusion α=0.3, style=aron7awol` | `production, style=aron7awol` |
| filter count | 5 | 5 |
| topology | 1×PeakingEQ + 4×LowShelf | 4×LowShelf + 1×PeakingEQ |
| freq range | 5–46 Hz | 17–35 Hz |
| max gain/filter | +2.40 dB | +3.46 dB |
| MV adjust | +6.4 dB | +10.6 dB |

The new model:
1. Concentrates filters in the 17–35 Hz band instead of spreading
   from 5 to 46 Hz.
2. Predicts ~4 dB more total MV adjust (10.6 vs 6.4 dB).
3. Keeps the same filter-count topology.

The more aggressive boost is consistent with the real-audio training
regime (E77+): real LFE tracks in the training cache have steeper
rolloffs than the synthetic curves the old model was trained on, so
the new model learned to compensate harder.

## Caveats

- **Stereo content**: JJK S1/S2 are both stereo (no real LFE channel).
  The "rolloff" the model sees below 20 Hz is partly absence of deep
  bass in the stereo mix, not a mastering rolloff. A real +10 dB boost
  at 17 Hz on a stereo source could expose subwoofer artefacts that
  wouldn't appear with a proper LFE-sourced BEQ. Proceed with caution
  if playing back through a bass-managed system.
- **Only one "before" data point**: the earlier run only produced
  S02E01. Can't do a full-season before/after comparison.
- **No catalogue reference**: there are no hand-coded JJK TV entries
  in the BEQ catalogue (only the JJK 0 movie by mikejl). Can't
  benchmark against a known-good reference chain.

## Verdict

E82 production model produces consistent, plausible BEQ profiles across
47 JJK episodes with zero failures. Gains and frequencies are in the
expected range; topology is stable (always 4–6 filters dominated by
LowShelves in 10–45 Hz). Ready to ship as the default advisor.

The one open concern (stereo content getting aggressive correction) is
a model-coverage question, not a regression — the old model would have
produced the same issue, just with slightly less aggressive numbers.
