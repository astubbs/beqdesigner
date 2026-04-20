---
title: Data regime changes beat architecture changes - switch to real data early
date: 2026-04-20
category: best-practices
module: ml-training
problem_type: best_practice
component: tooling
severity: high
applies_when:
  - Planning ML experiment roadmap
  - Deciding between model architecture changes and data strategy changes
  - Working with synthetic training data considering switch to real data
tags:
  - machine-learning
  - training-data
  - architecture
  - real-audio
  - synthetic
  - regime-change
---

# Data regime changes beat architecture changes - switch to real data early

## Context

The BEQ auto-profile project spent months optimizing model architecture on synthetic training data (experiments E1-E67, F/G/H/I series). Each architecture change yielded incremental improvements of 0.05-0.15 dB. Then experiment E77 switched the training data from synthetic to real audio - and immediately delivered 0.25 dB improvement in a single experiment, more than months of architectural optimization combined.

The threshold for the regime switch was lower than assumed: only ~100 real samples were needed, not the 500+ that was the assumed minimum.

## Guidance

- **Invest in real training data before optimizing architecture.** 100 real samples + a simple model will likely outperform a complex model trained on 10,000 synthetic samples.
- **Budget time for data collection, not just model iteration.** The highest-leverage work in the BEQ project was building the extraction pipeline to collect real WAV files, not tuning XGBoost hyperparameters.
- **Synthetic data is a bootstrap, not a destination.** Use it to validate the pipeline and establish baselines. Switch to real data as soon as you have 50-100 samples. Keep synthetic as a fallback for coverage gaps.
- **The 50:1 weighted hybrid** (E82) is the production pattern: train on both real and synthetic, but weight real samples 50x higher. This gives real-data accuracy with synthetic-data coverage.

## Why This Matters

Architecture optimization on synthetic data is a local optimum trap. The model learns to be good at synthetic data, but the distribution mismatch with real data means improvements don't transfer. Switching the data regime is a phase transition - it moves you to a fundamentally different (and better) part of the solution space.

## When to Apply

- When you've been iterating on model architecture for more than 5 experiments without meaningful improvement
- When your training data is synthetic/generated and real data exists but hasn't been collected
- When the gap between synthetic-data performance and real-world performance is unknown

## Examples

**Architecture optimization (months of work):**
```
E1-E67: Various architecture experiments
  F-series (feature engineering): +0.08 dB
  G-series (loss functions): +0.05 dB
  H-series (ensemble methods): -0.03 dB (regression!)
  I-series (hyperparameter search): +0.12 dB
Total architecture gain: ~0.22 dB over 67 experiments
```

**Data regime switch (one experiment):**
```
E77: Switch from synthetic to real audio training
  Gain: +0.25 dB in one experiment
  Required: ~100 real WAV files (built extraction pipeline)
  Final production model (E82): 1.49 dB mean error
```

## Related

- Experiment entries: E1-E67, E77, E82 in `docs/design/auto_beq_experiments.md`
- The extraction pipeline (`cli/extract.py`, `docker/`) was built specifically to collect real training data at scale
- `docs/design/auto_beq_how_it_works.md` describes the differentiable biquad layer that made the E85 architecture work - but even E85 needed real data to reach its best performance
