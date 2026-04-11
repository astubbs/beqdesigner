# Automated BEQ Filter Prediction — Neural Network Approach

## What is this?

A trained regression model that takes a movie or TV show's LFE audio
measurement and production metadata (studio, year, audio format, genre)
and predicts BEQ filter parameters — the same type of filter chains that
BEQ catalogue authors create by hand.

It's trained entirely on the existing BEQ catalogue. No new labelling
was required — the catalogue IS the training data.

## How does it work?

1. **Training data**: Every entry in the BEQ catalogue becomes a training
   sample. The model learns the relationship between "what a title's rolloff
   looks like" + "what we know about the title" → "what filter parameters
   the expert prescribed."

2. **Audio features**: The LFE channel is extracted and measured (spectral
   analysis at 9 frequency bins between 20-80 Hz). This captures the
   rolloff shape that the BEQ filter chain needs to correct.

3. **Metadata features**: Studio (parent company grouping), release year,
   audio format (Atmos/TrueHD/DTS-HD MA/DD+), source (disc/streaming),
   genre, country of origin, sound mixer, BEQ catalogue author, and era
   bucket. These provide production context that influences how aggressive
   the bass extension should be.

4. **Model**: XGBoost gradient-boosted trees with "late fusion" — separate
   models for audio features and metadata features, blended at prediction
   time. This prevents the model from learning spurious correlations between
   audio and metadata.

5. **Output**: Up to 6 IIR filter parameters (type, frequency, gain, Q)
   in the same format as existing BEQ catalogue entries.

## How well does it work?

Validated on **241 unique titles** with real extracted LFE audio from an
actual media library, compared against the hand-coded catalogue entries:

| Metric | Result |
|---|---|
| Mean error | **3.01 dB** across 20-80 Hz |
| Expert quality (< 2 dB) | **82 titles (34%)** |
| Good starting point (< 3 dB) | **146 titles (60%)** |
| Usable (< 5 dB) | **206 titles (85%)** |

### Per-author accuracy

The model learns each catalogue author's calibration style:

| Author | Titles | Mean error | Expert quality |
|---|---|---|---|
| halcyon888 | 5 | 1.74 dB | 80% |
| aron7awol | 64 | 2.13 dB | 52% |
| t1g8rsfan | 11 | 2.55 dB | 36% |
| mikejl | 3 | 2.88 dB | 33% |
| kaelaria | 38 | 3.15 dB | 34% |
| remixmark | 14 | 3.20 dB | 36% |
| mobe1969 | 106 | 3.58 dB | 21% |

aron7awol's calibration style is the most consistently predictable — the
model matches his hand-coded filters within 2 dB for over half of titles.

### What does "3 dB error" mean practically?

The error is the mean absolute difference between the predicted filter
chain's frequency response and the catalogue's, measured at each frequency
point across 20-80 Hz. In practical terms:

- **< 2 dB**: Indistinguishable to most listeners. Matches expert quality.
- **2-3 dB**: Audible difference if A/B compared, but still a good BEQ.
  Usable as-is for casual listening.
- **3-5 dB**: Noticeable — the bass extension is in the right direction
  but the magnitude is off. Good starting point for manual refinement.
- **> 5 dB**: Too far off. Affects mostly older films (pre-1990) and titles
  with unusual rolloff patterns.

## Key findings

1. **No real audio training data needed.** The model trains on synthetic
   data derived mathematically from the catalogue's filter chains. Real
   audio is only needed at inference time (measuring the movie's LFE).

2. **Metadata is surprisingly powerful.** Knowing the studio, year, format,
   and genre alone (without any audio measurement) predicts BEQ filters
   within 3.4 dB. Audio features improve this further.

3. **Author style is learnable.** The model can generate BEQ filters "in
   the style of" a specific catalogue author by setting the author feature.

4. **The main failure mode is magnitude calibration on older content.**
   1980s films average 6.0 dB error — the model hasn't seen enough examples
   of that era's rolloff patterns.

## What's next?

- **Production integration**: A "magic wand" button in BEQDesigner that
  measures the current media's LFE and proposes filter parameters.
- **More training data**: The NAS extraction pipeline is building a larger
  validation corpus (~1,200 titles queued). As it grows, we can better
  understand edge cases.
- **Confidence scoring**: Flag titles where the model is uncertain so the
  user knows when to trust the prediction vs. when to tweak manually.

## Technical details

See [`docs/design/auto_beq_ml_experiments.md`](design/auto_beq_ml_experiments.md)
for the full experiment history (E25-E40), and
[`docs/nn_comparison_report.md`](nn_comparison_report.md) for the per-title
comparison of predicted vs hand-coded filters.
