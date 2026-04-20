# How the auto-BEQ model works — a plain-language guide

**Audience**: developers who have NOT worked with machine learning or
audio engineering before. Every technical term is defined on first use
and linked where helpful. If you can read Python, you can understand
this system.

---

## The problem we're solving

Home cinema receivers often roll off (reduce) deep bass frequencies
below about 30 Hz. A **BEQ** ("Bass EQ") profile is a set of digital
filters that boosts those frequencies back to what the film's audio
engineers intended — flat response down to the subwoofer's limits.

Hundreds of volunteers have hand-crafted BEQ profiles for thousands
of films and published them in the [BEQ Catalogue](https://beqcatalogue.readthedocs.io/).
But there are far more films than volunteers. **Our goal**: predict a
BEQ filter chain for ANY film, automatically, from its actual audio.

---

## Key concepts (defined for non-specialists)

### Biquad filter

The **biquad** (short for "bi-quadratic") is the universal building
block of digital audio equalisers. It's a small formula with 6
numbers (called **coefficients**: b₀, b₁, b₂, a₀, a₁, a₂) that
processes audio one sample at a time:

```
output[n] = (b₀·input[n] + b₁·input[n-1] + b₂·input[n-2]
           - a₁·output[n-1] - a₂·output[n-2]) / a₀
```

By choosing the 6 coefficients differently, the same formula becomes:
- A **LowShelf** — boost or cut everything below a frequency
- A **HighShelf** — boost or cut everything above a frequency
- A **PeakingEQ** — boost or cut a narrow band around a frequency

The coefficients are computed from three human-readable parameters:
**frequency** (where the filter acts, in Hz), **gain** (how much to
boost/cut, in decibels), and **Q** (how wide/narrow the effect is).

A BEQ profile is typically 4–6 biquads chained together (cascaded).

> **Further reading**: Robert Bristow-Johnson's
> [Audio EQ Cookbook](https://www.w3.org/2011/audio/audio-eq-cookbook.html)
> — the industry standard reference for biquad coefficient formulas.

### XGBoost

[XGBoost](https://xgboost.readthedocs.io/) ("eXtreme Gradient
Boosting") is a machine learning algorithm that learns to predict
numbers from input data. Think of it as building a forest of simple
decision trees, where each new tree focuses on correcting the
mistakes the previous trees made. It's excellent at tabular data
(spreadsheet-like rows of numbers) and was our first production model.

In our system: XGBoost takes 102 numbers describing a film's bass
content + metadata (year, codec, genre, etc) and predicts 36 numbers
that encode a 6-filter BEQ chain.

### Neural network (MLP)

A **neural network** is a function built from layers of simple
"neurons" (multiply inputs by learned weights, add a bias, apply a
squishing function like sigmoid). An **MLP** ("multi-layer
perceptron") is the simplest type — just layers stacked on top of
each other with no special structure.

Our E85 model is a small MLP: 102 inputs → 3 hidden layers of 256
neurons each → 24 outputs (6 filters × 4 parameters each). Total
size: ~200,000 learned weights, stored in a 661 KB file.

> **Further reading**:
> [3Blue1Brown's neural network series](https://www.3blue1brown.com/topics/neural-networks)
> — visual, intuitive, no prerequisites.

### Mean Squared Error (MSE)

The simplest way to measure "how wrong is my prediction": take each
predicted number, subtract the target, square it (so negatives don't
cancel positives), average across all predictions. Smaller = better.

### Differentiable

A function is **differentiable** if you can compute its **gradient**
— the answer to "if I nudge input X up by a tiny amount, how much
does the output change, and in which direction?" Gradients are what
let neural networks learn: the training algorithm nudges each weight
in the direction that reduces the error, thousands of times, until
the predictions are good.

**"Differentiable biquad"** means: we rewrote the biquad coefficient
formula in [PyTorch](https://pytorch.org/) (a neural network
framework) so that gradients flow through it. The mathematical result
is identical to the standard audio formula — but PyTorch can now
answer "if I change the predicted frequency by 0.1 Hz, how much does
the acoustic output change?" That's what lets us train directly on
the sound.

---

## How the system evolved

### Era 1: XGBoost on hand-crafted features (E1–E82)

1. **Extract** the LFE (Low Frequency Effects) channel from a film
   via ffmpeg, downsample to 1000 Hz (we only care about bass).
2. **Measure** the bass spectrum using Welch's method — tells us "how
   much energy at each frequency" averaged over the whole film.
3. **Summarise** into 9 numbers (energy at 9 log-spaced frequency
   bins from 20–80 Hz) + 93 metadata features (year, codec, genre,
   author style, etc) = 102-number input vector.
4. **Train** XGBoost to predict the 36 filter-parameter numbers that
   match the hand-crafted BEQ catalogue entry for that film.
5. **At inference**: given a NEW film's 102-number vector, XGBoost
   predicts 36 numbers → decode into 6 filter dicts → apply via
   ezBEQ or any EQ system.

**Production result (E82)**: 1.70 dB mean error on 255 test films.
That means: on average, the predicted filter chain's frequency
response differs from the hand-crafted target by 1.70 dB at each
point in the bass band. For reference, <2 dB is generally inaudible
to most listeners.

### Era 2: Differentiable DSP (E85) — the breakthrough

The problem with Era 1: XGBoost minimises **parameter error** (MSE
on the 36 filter numbers). But two very different sets of filter
numbers can produce nearly identical sound:

```
Filter A: LowShelf at 18 Hz, +8.0 dB, Q=0.9
Filter B: LowShelf at 20 Hz, +7.5 dB, Q=1.1
   → Both produce almost the same bass boost curve
```

XGBoost says A and B are "wrong" relative to each other (different
numbers). But the **acoustic loss** says they're equivalent (same
sound). XGBoost wastes effort matching exact numbers instead of
matching what matters.

**The E85 fix**: train a neural network through a **differentiable
biquad layer** that evaluates the predicted filter chain's actual
frequency response, then minimise the dB difference between
"what my prediction sounds like" and "what the target sounds like":

```
                     ┌───���──────────┐
 102 features ─────►│  MLP (3×256) │─────► predicted filter params
                     └──────────────┘              │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │  Differentiable Biquad   │
                                    │  (same trig as any EQ,   │
                                    │   but gradients flow)    │
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                         predicted response curve
                                                   │
                                    ┌──────────────┴──────────────┐
                                    │  Loss = ||predicted - target||²  │
                                    │  (in dB, band-masked 5-80 Hz)    │
                                    └─────────────────────────────────┘
```

**Production result (E85)**: **1.49 dB mean error** — a 12% reduction
from E82's 1.70 dB, while training in 2.4 seconds (vs 30 seconds for
XGBoost) and producing a 661 KB model (vs 8 MB).

---

## What's NOT novel (prior art we build on)

Each individual technique in our system was invented by someone else.
Understanding the building blocks helps you learn each one
independently:

| Technique we use | Who invented it | Where to learn more |
|---|---|---|
| **Biquad filter coefficients** | Robert Bristow-Johnson (1998) | [Audio EQ Cookbook](https://www.w3.org/2011/audio/audio-eq-cookbook.html) — the exact formulas every EQ on earth uses |
| **Differentiable audio processing** | Google Magenta team (2020) | [DDSP paper](https://magenta.tensorflow.org/ddsp) — making synthesiser parameters learnable end-to-end |
| **XGBoost (gradient-boosted trees)** | Tianqi Chen (2016) | [XGBoost docs](https://xgboost.readthedocs.io/) — the workhorse of tabular ML |
| **Neural networks / MLPs** | Rosenblatt (1958), modern revival ~2012 | [3Blue1Brown series](https://www.3blue1brown.com/topics/neural-networks) — the best visual explanation |
| **Acoustic loss (training on what it sounds like)** | Standard in speech synthesis since ~2016 | [WaveNet paper](https://arxiv.org/abs/1609.03499) by DeepMind |
| **Teacher-student warm-start** | Hinton et al. "knowledge distillation" (2015) | [Distilling the Knowledge](https://arxiv.org/abs/1503.02531) |
| **Welch's method (spectrum estimation)** | Peter Welch (1967) | [Wikipedia: Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method) |
| **Softmax for categorical selection** | Standard neural network technique | [PyTorch softmax docs](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) |

None of these are our invention. We stand on shoulders.

## What IS novel (our specific contribution)

What's new is the **combination** of all the above, applied to a
**problem nobody else has solved**:

1. **Nobody does automated BEQ prediction.** The entire concept —
   "measure a film's LFE rolloff → predict correction filters" — is
   unique to this project. There's no academic paper, no commercial
   product, and no open-source tool that does what we do. The BEQ
   community does it by hand, one film at a time, by expert
   volunteers with measurement microphones and years of experience.

2. **The specific architecture doesn't exist elsewhere**: 9 hand-
   crafted sub-bass frequency features + 93 metadata features →
   neural network → differentiable biquad layer → acoustic dB-error
   loss. This pipeline is specific to predicting bass-extension EQ
   chains from measured content. Nobody needed it before because
   nobody was automating this task.

3. **XGBoost → neural network warm-start** (two completely different
   model families): we train a gradient-boosted tree first (fast,
   robust, well-understood), then use its predictions as the starting
   point for the neural network's acoustic-loss training. This solves
   a cold-start problem specific to filter prediction: randomly
   initialised networks produce garbage filter params whose biquad
   responses are wildly wrong, causing the acoustic loss landscape to
   be almost flat (no useful gradient). The XGBoost warm-start puts
   the network in a "reasonable" region of parameter space where the
   acoustic loss landscape IS informative.

4. **Soft type selection with hard-argmax inference**: during
   training, each filter slot blends all three types (LowShelf,
   HighShelf, PeakingEQ) weighted by a softmax so gradients flow
   smoothly. At inference, we commit to one type per slot via hard
   argmax (the winning type becomes the filter). This is specific to
   multi-type filter chain prediction and hasn't been needed in other
   DDSP work (which typically has fixed synthesis architectures).

5. **The production system end-to-end**: from NAS-based batch LFE
   extraction with atomic-write caching → bias-corrected selection
   of uncatalogued media → curve-feature disk cache (17× speedup) →
   model training in 2.4 seconds → sub-second inference → catalogue-
   compatible JSON output with biquad coefficients + spectrograph
   images. A complete tool that real users can deploy, not just a
   research experiment that lives in a Jupyter notebook.

**In summary**: if this were published, it would be an "application
paper" — demonstrating that known techniques, combined in the right
way, solve a real problem that previously required human expertise.
The novelty is in the combination and the domain, not in any single
algorithmic breakthrough.

---

## The numbers at a glance

| Model | Mean error | Max error | Training time | Model size |
|---|---|---|---|---|
| E82 (XGBoost, parameter MSE) | 1.70 dB | 9.92 dB | 30 s | 8 MB |
| **E85 (neural net, acoustic loss)** | **1.49 dB** | 11.45 dB | **2.4 s** | **661 KB** |

"Mean error" = average dB difference between predicted and target
frequency responses across 255 test films in the 5–80 Hz bass band.
Under ~2 dB is generally considered inaudible.

---

## Running the system

```bash
# 1. Extract LFE audio from your media library (once, cached):
poetry run python3 scripts/extract_lfe.py

# 2. Train the E85 model (uses the E82 XGBoost as a warm-start teacher):
poetry run python3 scripts/train_torch_model.py

# 3. Generate BEQ profiles for any media:
AUTO_BEQ_ADVISOR=torch_differentiable \
  poetry run python3 scripts/generate_beq_profile.py \
    --media-dir "/path/to/Show/Season 01/" \
    --output-dir profiles/
```

---

## Glossary

| Term | Definition |
|---|---|
| **BEQ** | Bass EQ — a set of filters that extends a film's bass response to compensate for receiver rolloff |
| **Biquad** | A 6-coefficient digital filter formula that can implement LowShelf, HighShelf, or PeakingEQ depending on the coefficients |
| **Catalogue** | The community-maintained database of hand-crafted BEQ profiles at [beqcatalogue.readthedocs.io](https://beqcatalogue.readthedocs.io/) |
| **Differentiable** | A function whose gradient (sensitivity to input changes) can be computed — required for neural network training |
| **dB (decibel)** | A logarithmic unit for measuring sound intensity. +3 dB ≈ twice the power; +10 dB ≈ 10× the power |
| **Frequency response** | A curve showing how much each frequency is boosted/cut by a filter — the "shape" of the EQ |
| **Gain** | How much a filter boosts (+) or cuts (−) the signal, in dB |
| **LFE** | Low Frequency Effects — the ".1" in "5.1 surround sound", the dedicated subwoofer channel |
| **MLP** | Multi-Layer Perceptron — the simplest neural network: stacked layers of neurons |
| **MSE** | Mean Squared Error — average of (prediction − target)² across all outputs |
| **PyTorch** | An open-source framework for building and training neural networks ([pytorch.org](https://pytorch.org/)) |
| **Q factor** | Controls how wide/narrow a filter's effect is. Low Q = broad; high Q = narrow peak |
| **Rolloff** | The gradual reduction of bass below a certain frequency — what BEQ corrects |
| **Welch's method** | A technique for estimating a signal's frequency content by averaging many short FFTs |
| **XGBoost** | A machine learning algorithm that builds an ensemble of decision trees, each correcting the previous ones' mistakes ([xgboost.readthedocs.io](https://xgboost.readthedocs.io/)) |
