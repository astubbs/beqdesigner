---
title: "refactor: Consolidate 14K lines of docs into agent-friendly structure"
type: refactor
status: active
date: 2026-04-21
origin: docs/brainstorms/2026-04-21-docs-consolidation-requirements.md
---

# Consolidate documentation into agent-friendly structure

## Overview

Restructure ~14,000 lines of documentation into a compact, navigable
layout optimized for agent consumption. Split the monolithic 4K-line
experiment log into per-family files, consolidate 7 design docs into
one reference, and delete stale plans and generated reports.

## Problem Frame

Documentation accumulated over 87 iterative experiments. Agents waste
context loading massive files when they only need one experiment's
details. Knowledge is scattered across 8 design docs, 5 plans, and
8 generated reports. (see origin: `docs/brainstorms/2026-04-21-docs-consolidation-requirements.md`)

## Requirements Trace

- R1. Split experiment log into per-family files in `docs/design/experiments/`
- R2. Provide index file with experiment family map
- R3. Consolidate 7 design docs into one ~500-line reference
- R4. Extract remaining golden learnings into `docs/solutions/`
- R5. Delete completed plans, generated reports, stale branch plans
- R6. Update AGENTS.md, mkdocs.yml, and cross-references
- R7. Total docs under 5,000 lines

## Scope Boundaries

- `docs/architecture.md`, `docs/faq.md`, `docs/lfe_extractor.md` - untouched
- `docs/ui/` and `docs/workflow/` - untouched
- `docs/solutions/` existing entries - untouched (only additions)
- `docs/install.md`, `docs/concepts.md`, `docs/index.md` - untouched
- Active plan `2026-04-21-003-refactor-review-fixes-robustness-plan.md` - keep

## Key Technical Decisions

- **Experiment files grouped by family, not individual E-number** -
  E1-E6 are one file, E7-E13 are one file, etc. Keeps file count
  manageable (~10 files) while each stays under 400 lines.

- **Index file is the family map table** - the existing overview section
  from the experiment log becomes `docs/design/experiments/README.md`.
  Agents scan this first, then load specific families.

- **Consolidated reference replaces 7 docs, not merges them** - write
  fresh from current state rather than awkwardly merging stale content.
  Covers: what auto-BEQ is, how the pipeline works, current champion,
  key architecture decisions, dead ends to avoid.

## Output Structure

    docs/design/experiments/
      README.md                              # family map index (~100 lines)
      E01-E06-procedural-heuristics.md       # dead end: shelf + PEQ fitter
      E07-E13-llm-tier-classification.md     # dead end: Ollama for numbers
      E14-E17-measurement-advisor.md         # kept: topology classifier
      E18-E22-spectrum-extraction.md         # adopted: blended extraction
      E25-E40-ml-trained-models.md           # evolved: XGBoost, late fusion
      E41-E68-f-g-h-series.md               # feature engineering sweeps
      E69-E76-i-series-metadata.md           # metadata features
      E77-E82-real-audio-training.md         # champion: weighted hybrid
      E83-E87-paradigm-shifts.md             # post-champion: Whisper, torch

    docs/design/
      auto_beq_reference.md                  # consolidated reference (~500 lines)

## Implementation Units

- [ ] **Unit 1: Split experiment log into per-family files**

**Goal:** Break 4068-line monolith into ~10 files of 100-400 lines each.

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Delete: `docs/design/auto_beq_experiments.md`
- Create: `docs/design/experiments/README.md`
- Create: `docs/design/experiments/E01-E06-procedural-heuristics.md`
- Create: `docs/design/experiments/E07-E13-llm-tier-classification.md`
- Create: `docs/design/experiments/E14-E17-measurement-advisor.md`
- Create: `docs/design/experiments/E18-E22-spectrum-extraction.md`
- Create: `docs/design/experiments/E25-E40-ml-trained-models.md`
- Create: `docs/design/experiments/E41-E68-f-g-h-series.md`
- Create: `docs/design/experiments/E69-E76-i-series-metadata.md`
- Create: `docs/design/experiments/E77-E82-real-audio-training.md`
- Create: `docs/design/experiments/E83-E87-paradigm-shifts.md`

**Approach:**
- Extract the "Experiment families overview" section as `README.md`
- Split remaining content at `## 2026-MM-DD:` date headers
- Each file gets a title, status line (dead end / adopted / champion),
  and the original content for its experiment range
- Preserve all content verbatim - this is archival, not rewriting

**Test scenarios:**
- Test expectation: none - pure documentation restructuring

**Verification:**
- `ls docs/design/experiments/` shows ~10 files
- `wc -l docs/design/experiments/*` totals ~4000 lines (no content lost)
- `docs/design/auto_beq_experiments.md` no longer exists

---

- [ ] **Unit 2: Consolidate design docs into auto_beq.md**

**Goal:** Enrich `auto_beq.md` (already well-written, 409 lines) by
folding in the best content from the other 7 docs, then delete them.
Add the result to mkdocs.yml navigation.

**Requirements:** R3, R6

**Dependencies:** Unit 1 (links to experiment files)

**Files:**
- Modify: `docs/design/auto_beq.md` - enrich with content from the
  docs being deleted. This is the anchor document.
- Delete: `docs/design/auto_beq_plan.md` (historical advisor plan)
- Delete: `docs/design/auto_beq_how_it_works.md` (beginner guide -
  fold key concepts section into auto_beq.md)
- Delete: `docs/design/auto_beq_ml_experiments.md` (ML experiment
  design - fold key findings into auto_beq.md, details in archive)
- Delete: `docs/design/auto_beq_nn_future_experiments.md` (forward
  ideas - fold actionable items into auto_beq.md "what's next")
- Delete: `docs/design/auto_beq_nn_paradigm_shifts.md` (post-E82
  experiments - summarize in auto_beq.md, details in archive)
- Delete: `docs/design/auto_beq_library_sweep_plan.md` (discovery
  pipeline - fold into auto_beq.md pipeline section)
- Delete: `docs/nn_introduction.md` (ML summary with results tables -
  fold results tables and "what does 3 dB mean" section into auto_beq.md)
- Modify: `mkdocs.yml` - add Auto-BEQ section to nav

**Approach:**
- Keep auto_beq.md's existing structure (it already has: what it does,
  architecture diagram, pipeline stages, CLI commands, deployment)
- Fold in from how_it_works.md: the "Key concepts" section (biquad,
  rolloff, etc.) as a glossary/concepts section
- Fold in from nn_introduction.md: the results tables (per-author
  accuracy, "what does 3 dB mean"), key findings section
- Fold in from nn_future_experiments.md and paradigm_shifts.md: a
  concise "what's next" section with the most promising ideas
- Fold in from library_sweep_plan.md: the discovery quick-start into
  the pipeline section
- Update all links: faq.md (5 links), lfe_extractor.md (3 links),
  experiment archive files
- Add to mkdocs.yml under a new "Auto-BEQ" section: the main doc +
  experiment archive index
- Target: auto_beq.md grows to ~600-700 lines (from 409), replacing
  ~2700 lines across 7 deleted docs + nn_introduction
- Maintain accessible tone (defined for devs with no ML/audio background)

**Patterns to follow:**
- `docs/architecture.md` for tone and structure
- Existing auto_beq.md style (already sets the right tone)

**Test scenarios:**
- Test expectation: none - pure documentation

**Verification:**
- `wc -l docs/design/auto_beq.md` is under 750 lines
- 7 old design docs + nn_introduction.md no longer exist
- All links from auto_beq.md resolve (faq.md, lfe_extractor.md,
  experiment archive files)
- mkdocs.yml has an Auto-BEQ section in the nav
- Content from nn_introduction.md results tables is present

---

- [ ] **Unit 3: Delete stale files**

**Goal:** Remove completed plans, generated reports, and stale branch plans.

**Requirements:** R5

**Dependencies:** None (can run in parallel with Units 1-2)

**Files:**
- Delete: `docs/plans/2026-04-20-001-refactor-cleanup-config-docs-dry-plan.md`
- Delete: `docs/plans/2026-04-21-001-refactor-duplication-reduction-plan.md`
- Delete: `docs/plans/2026-04-21-002-fix-test-overhaul-e2e-mps-restructure-plan.md`
- Delete: `docs/plans/reassess-pipeline-ux.md`
- Delete: `branch-plans/plan-audio-chunks-strat.md`
- Delete: `branch-plans/plan-neural-net-strat.md`
- Delete: `branch-plans/plan-sharp-goldberg.md`
- Delete: `docs/nn_comparison_report.md`
- Delete: `docs/acquisition_recommendations.md`
- Delete: `docs/f_experiment_results.md`
- Delete: `docs/g_experiment_results.md`
- Delete: `docs/h_experiment_results.md`
- Delete: `docs/i_experiment_results.md`
- Delete: `docs/wav_cache_bias.md`
- Delete: `docs/author_patterns.md`
- Keep: `docs/plans/2026-04-21-003-refactor-review-fixes-robustness-plan.md` (still active)

**Approach:**
- Straight deletion. Content is preserved in git history.
- Keep the active review-fixes plan

**Test scenarios:**
- Test expectation: none - file deletion

**Verification:**
- `docs/plans/` contains only the active plan + this plan
- `branch-plans/` is empty or deleted
- No generated report `.md` files remain in `docs/`

---

- [ ] **Unit 4: Fix remaining cross-references and verify**

**Goal:** Fix all broken cross-references from the deletions, update
AGENTS.md and readme.md, verify line count target.

**Requirements:** R6, R7

**Dependencies:** Units 1, 2, 3

**Files:**
- Modify: `AGENTS.md` - update doc references to point to auto_beq.md
  and experiment archive
- Modify: `readme.md` - update design docs links section
- Modify: any other files with broken links to deleted docs

**Approach:**
- Link graph is already handled by Unit 2 (auto_beq.md keeps its
  links to faq.md and lfe_extractor.md, and gains links to the
  experiment archive)
- Search for ALL references to deleted filenames across the repo
  and fix or remove each one
- Verify mkdocs build would succeed (nav entries match real files)
- Final line count check

**Test scenarios:**
- Test expectation: none - reference updates

**Verification:**
- No broken internal links (grep for deleted filenames returns 0 hits)
- `find docs/ branch-plans/ -name "*.md" -exec cat {} + | wc -l` < 5000
- mkdocs.yml nav entries match actual files

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Experiment content lost during split | Verify line counts match before and after |
| Cross-references to old design docs break | Grep for all deleted filenames and fix |
| Active plan accidentally deleted | Explicit keep list in Unit 3 |

## Sources & References

- **Origin document:** `docs/brainstorms/2026-04-21-docs-consolidation-requirements.md`
- Related: `AGENTS.md` accessible-docs rule
- Related: `docs/architecture.md` (style reference for consolidated doc)
