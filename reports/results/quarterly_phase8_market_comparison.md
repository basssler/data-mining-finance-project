# Quarterly Phase 8 Market Comparison

## Setup

- Frozen quarterly anchor family: `levels_plus_deltas_plus_cross_sectional`.
- Frozen label anchor: `21d_excess_thresholded`.
- Event timing, purged walk-forward validation, and 2024 holdout were kept unchanged from the frozen anchor config.

## Selected Models

| Setup | Selected Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Rows | Usable Features | Pre-Event Features | First-Tradable Features |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core_no_market | random_forest | 0.5126 | 0.0346 | 0.4652 | 0.5291 | 66 | 116 | 0 | 0 |
| generic_market_only | logistic_regression | 0.5653 | 0.0459 | 0.4788 | 0.4248 | 66 | 130 | 0 | 0 |
| event_aware_market_only | logistic_regression | 0.5780 | 0.0596 | 0.4788 | 0.4173 | 66 | 128 | 10 | 2 |
| generic_and_event_aware_market | logistic_regression | 0.5530 | 0.0473 | 0.4591 | 0.4060 | 66 | 142 | 10 | 2 |

## Direct Answers

- Does `event_aware_market_only` beat `generic_market_only`? `No`
- Does `generic_and_event_aware_market` beat `core_no_market`? `No`
- Are first-tradable-session features helping? `No`
- Should Phase 8 stay in the stack before Phase 9? `No`

## Winner

- Winning setup: `core_no_market` with `random_forest`.
- Winner metrics: CV AUC `0.5126`, CV AUC std `0.0346`, worst fold AUC `0.4652`, holdout AUC `0.5291`.

## Event-Aware Survivors In Winner

- No event-aware market features survived into the selected winner.
