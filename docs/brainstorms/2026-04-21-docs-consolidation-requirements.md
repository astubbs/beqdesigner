---
title: "Documentation consolidation - reduce 14K lines to agent-friendly structure"
type: refactor
date: 2026-04-21
status: ready
---

# Documentation consolidation

## Problem

The repo has ~14,000 lines of markdown documentation accumulated over
iterative agentic experiment development. The content has real value
(experiment learnings, architecture decisions, dead-end warnings) but
the current structure causes four problems:

1. **Can't find things** - knowledge is scattered across 8 design docs,
   5 plans, 3 branch plans, and multiple generated reports
2. **LLM context bloat** - agents load massive files (4K-line experiment
   log) and waste context on stale history
3. **Stale/contradictory content** - old docs say things that are no
   longer true after 87 experiments
4. **Intimidating for onboarding** - 14K lines overwhelms new readers

The documentation is primarily consumed by agents, not humans. The main
use cases are: preventing re-running dead-end experiments, understanding
why current decisions were made, and knowing the current champion model.

## Requirements

- R1. Split the 4K-line experiment log into individual files by experiment
  family so agents can load only the relevant 50-100 lines
- R2. Provide a single index file with the experiment family map table
  for navigation
- R3. Consolidate the 7 remaining design docs into one concise "auto-BEQ
  technical reference" covering current architecture, champion model,
  and key decisions (~500 lines target)
- R4. Extract remaining golden learnings from design docs into
  `docs/solutions/` entries
- R5. Delete completed plans, generated reports, and stale branch plans
- R6. Update AGENTS.md, mkdocs.yml, and cross-references
- R7. Total documentation stays under 5,000 lines (from current 14K)

## Success Criteria

- An agent working on auto-BEQ improvements can find "did we try X" by
  scanning `docs/design/experiments/` filenames without loading all of
  them
- The single technical reference doc fits in one agent context load
- No generated reports are checked into the repo (they can be regenerated)
- No completed implementation plans remain in docs/plans/

## Scope

### In scope

- `docs/design/auto_beq_experiments.md` (4068 lines) - split into per-family files
- `docs/design/auto_beq.md` (409 lines) - consolidate into reference
- `docs/design/auto_beq_plan.md` (316 lines) - consolidate or archive
- `docs/design/auto_beq_how_it_works.md` (300 lines) - consolidate
- `docs/design/auto_beq_ml_experiments.md` (363 lines) - consolidate
- `docs/design/auto_beq_nn_future_experiments.md` (358 lines) - consolidate
- `docs/design/auto_beq_nn_paradigm_shifts.md` (401 lines) - consolidate
- `docs/design/auto_beq_library_sweep_plan.md` (469 lines) - consolidate
- `docs/plans/` completed plans - delete
- `branch-plans/` stale branch plans - delete
- Generated reports (`nn_comparison_report.md`, `acquisition_recommendations.md`,
  `f/g/h/i_experiment_results.md`, `wav_cache_bias.md`, `author_patterns.md`) - delete

### Out of scope

- `docs/architecture.md` - still current, keep as-is
- `docs/faq.md` - still current, keep as-is
- `docs/lfe_extractor.md` - still current, keep as-is
- `docs/ui/` and `docs/workflow/` - separate concern (GUI docs), untouched
- `docs/solutions/` - these are already well-structured, only additions
- `docs/install.md`, `docs/concepts.md`, `docs/index.md` - keep as-is

## Key Decisions

- **Experiment files named by experiment ID + short description** -
  e.g. `E01-E06-procedural-heuristics.md`, `E82-weighted-hybrid-champion.md`.
  Family groupings where experiments share a theme; individual files for
  major standalone experiments.

- **Archive stays in the repo, not just git history** - the experiment
  log exists only on this feature branch, so git history compaction
  could lose it. Breaking into files under `docs/design/experiments/`
  preserves it durably.

- **Generated reports deleted, not archived** - the code can regenerate
  them (`bin/beq-designer report *`). Stale snapshots cause more harm
  than value.

- **Completed plans deleted** - they served their purpose during
  implementation. The commit history preserves the record. The new
  review-fixes plan (003) stays since it's still active.

## Resolve Before Planning

None - all decisions made during brainstorm.
