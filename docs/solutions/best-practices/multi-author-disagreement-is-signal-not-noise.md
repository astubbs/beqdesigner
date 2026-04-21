---
title: Multi-author disagreement is signal, not noise - don't average out intentional variation
date: 2026-04-20
category: best-practices
module: ml-training
problem_type: best_practice
component: tooling
severity: medium
applies_when:
  - Training models on human-authored labels from multiple experts
  - Considering response-space averaging or consensus methods
  - Observing high variance in labels for the same input
tags:
  - machine-learning
  - multi-author
  - consensus
  - label-noise
  - domain-knowledge
  - beq-catalogue
---

# Multi-author disagreement is signal, not noise - don't average out intentional variation

## Context

The BEQ catalogue contains bass correction profiles authored by different experts (aron7awol, mobe1969, kaelaria, etc.). When multiple authors provide profiles for the same title, their corrections differ - sometimes significantly. The H-series experiments (E60-E67) attempted to improve model accuracy by averaging these disagreements away through response-space consensus and deduplication. Every attempt regressed performance.

## Guidance

- **Don't average multi-author labels.** Different BEQ authors intentionally apply different correction philosophies (aesthetic preference vs measurement-driven, aggressive vs conservative). Averaging produces a compromise that no expert would actually choose.
- **Model the variation, don't suppress it.** Treat each author's profile as a valid independent label. The model learns the DISTRIBUTION of reasonable corrections, not a single "correct" answer.
- **Use author identity as a feature, not a grouping key.** Author style is a meaningful metadata signal that helps the model predict the right correction for a given audio characteristic + target style.
- **When deduplicating titles for training, keep the first (or random) author's profile** rather than averaging across authors. `deduplicate_by_title()` in `model/auto_beq_nn.py` implements this.

## Why This Matters

In many ML domains, label disagreement between experts IS noise (medical imaging, document classification). In creative/aesthetic domains like audio engineering, disagreement is intentional variation reflecting different valid approaches. Treating it as noise destroys the signal. The H-series experiments proved this empirically: every consensus/averaging method regressed accuracy.

## When to Apply

- When training on human-authored labels where the "correct" answer is subjective
- When multiple experts provide different labels for the same input and the differences are NOT random errors
- When considering inter-annotator agreement metrics - low agreement may mean the domain has legitimate variation, not that the labels are bad

## Examples

**What didn't work (H-series regressions):**
- Response-space averaging of multi-author profiles: regressed
- Consensus deduplication (keep only titles where authors agree): regressed
- Weighted averaging by author reputation: regressed

**What worked (E82 production model):**
- Keep one randomly-selected author's profile per title
- Include author identity as a training feature
- The model learns author-conditional predictions
- Result: 1.49 dB mean error across the full catalogue

## Related

- H-series experiments (E60-E67) in the [experiment archive](../../design/experiments/README.md)
- `model/auto_beq_nn.py:deduplicate_by_title()` implements the keep-one-author strategy
- Per-author analysis: `cli/nn_author_pattern_report.py` shows how each author's style differs
