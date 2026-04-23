# Quarterly Experiment Log

Use this file as the canonical running log for quarterly experiments.

## Current Canonical State

- Live modeling unit: one quarterly filing event
- Live target: `event_v2_21d_excess_threshold`
- Live validation: `5`-fold purged expanding-window CV with `2024-01-01` holdout start
- Baseline config: `configs/event_panel_v2_quarterly.yaml`
- Active benchmark anchor config: `configs/quarterly/quarterly_phase9_event_specific_sentiment_champion_v1.yaml`
- Active candidate config: none
- Promotion registry: `docs/quarterly/promoted_models.md`

## Required entry fields
- date
- config path
- panel path
- label definition
- validation policy
- feature families changed
- selected model
- mean CV AUC
- worst-fold AUC
- holdout AUC
- concentration / stability note
- promote, revise, or kill decision

## Open items
- Add the first dedicated improved-label quarterly experiment.
- Add a dedicated purged-validation quarterly diagnostics package.
- Add a promotion memo only after a stability-aware quarterly champion exists.

## Latest Decision

- date: `2026-04-23`
- config path: `configs/quarterly/quarterly_phase9_event_specific_sentiment_champion_v1.yaml`
- panel path: `data/interim/event_panel_v2_quarterly_feature_design.parquet`
- label definition: `21d_excess_thresholded`
- validation policy: unchanged `5`-fold purged expanding window with `2024-01-01` holdout
- feature families changed: promote Phase 9 event-specific sentiment only; keep broad filing sentiment and all Phase 8 market features excluded from the live benchmark stack
- selected model: `random_forest`
- mean CV AUC: `0.5210`
- worst-fold AUC: `0.4789`
- holdout AUC: `0.5367`
- concentration / stability note: event-specific sentiment-only improved both holdout AUC and worst-fold AUC versus the frozen core anchor, while the combined broad-plus-event block degraded performance
- promote, revise, or kill decision: promote Phase 9 event-specific sentiment-only as the current quarterly champion and benchmark to beat for Phase 10

## Prior Decision

- date: `2026-04-23`
- config path: `configs/quarterly/quarterly_core_no_market_anchor_v1.yaml`
- panel path: `data/interim/event_panel_v2_quarterly_feature_design.parquet`
- label definition: `21d_excess_thresholded`
- validation policy: unchanged `5`-fold purged expanding window with `2024-01-01` holdout
- feature families changed: none promoted; both generic Layer 2 and Phase 8 event-aware market features remain excluded from the active benchmark anchor
- selected model: `random_forest`
- mean CV AUC: `0.5126`
- worst-fold AUC: `0.4652`
- holdout AUC: `0.5291`
- concentration / stability note: Phase 8 market variants raised usable feature counts but degraded holdout performance versus `core_no_market`
- promote, revise, or kill decision: kill Phase 8 promotion path for now; keep `core_no_market` as the active benchmark anchor
