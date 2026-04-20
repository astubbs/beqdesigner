---
title: Small test sets produce misleading ML conclusions - require 100+ stratified holdouts
date: 2026-04-20
category: best-practices
module: ml-training
problem_type: best_practice
component: tooling
severity: high
applies_when:
  - Evaluating ML model accuracy on audio/media datasets
  - Comparing experiment results across different training strategies
  - Deciding whether a training approach "works" or "fails"
tags:
  - machine-learning
  - evaluation
  - test-set
  - stratified
  - holdout
  - statistics
---

# Small test sets produce misleading ML conclusions - require 100+ stratified holdouts

## Context

During the BEQ auto-profile experiment series, experiment E33 concluded that "real-audio training fails" based on evaluation against 155 samples. This conclusion was wrong. When E81 re-ran the same approach with 400+ samples using stratified holdouts, the conclusion flipped - real-audio training actually outperformed synthetic-only training by a significant margin (0.25 dB).

The small test set in E33 happened to over-represent edge cases that penalized real-audio training, leading to a false negative that delayed the adoption of real-audio training by several experiment iterations.

## Guidance

- **Minimum 100 unique titles in the evaluation holdout.** Below this threshold, individual outliers dominate the aggregate metrics and produce unstable conclusions.
- **Stratify the holdout** across key dimensions: content type (film vs TV), audio format (Atmos, TrueHD, DTS-HD, DD+), era (decade), and author. Unstratified random splits can accidentally cluster one dimension in the holdout.
- **Report per-stratum metrics alongside aggregates.** A 1.5 dB mean error that's 0.8 dB for films and 3.2 dB for TV episodes tells a very different story than the aggregate alone.
- **Never declare an approach "failed" from a single small-set evaluation.** Run at least two evaluations with different random seeds or holdout splits before drawing conclusions.

## Why This Matters

A wrong conclusion from a small test set doesn't just waste one experiment - it changes the direction of all subsequent experiments. E33's false "real-audio training fails" conclusion led to months of work optimizing synthetic-only approaches (F/G/H/I series) when the breakthrough was actually in using real audio data. The cost of a 100-title minimum holdout is one-time data collection; the cost of a wrong conclusion is unbounded engineering effort.

## When to Apply

- Any time you evaluate an ML model's accuracy and plan to make a training strategy decision based on the result
- When comparing two approaches (A vs B) and the difference is less than 0.5 dB
- When the evaluation set was created by convenience sampling (whatever was available) rather than deliberate stratification

## Examples

**Before (E33 - wrong conclusion):**
```
Evaluated on 155 samples (unstratified)
Result: real-audio 2.1 dB mean error vs synthetic 1.8 dB
Conclusion: "real-audio training fails" ← WRONG
```

**After (E81 - correct conclusion):**
```
Evaluated on 400+ samples (stratified by format/era/type)
Result: real-audio 1.49 dB mean error vs synthetic 1.74 dB
Per-stratum: films 1.35 dB, TV 1.82 dB, Atmos 1.21 dB
Conclusion: real-audio training outperforms synthetic
```

## Related

- Experiment log entries: E33, E77, E81, E82 in `docs/design/auto_beq_experiments.md`
- The E82 "50:1 weighted hybrid" production model was the direct result of correctly evaluating real-audio training with adequate sample sizes
