# Phase 4 Decision Lock V1

## Locked Primary Setup

- Primary panel: `event_panel_v2`
- Primary label: `5-trading-day excess return sign`
- 21-day sign: not primary
- Quantile labels: dropped for now
- Model family set: `logistic_regression`, `random_forest`, `xgboost`
- Holdout policy: unchanged 2024 holdout
- CV policy: expanding purged date splits, `5` folds, `5` embargo days, `252` minimum pre-holdout train dates

## Locked Benchmark Anchor

Phase 4 reran the frozen benchmark matrix on the cleaned event-based setup and established the new anchor for the current 34-ticker universe.

- Selected primary model: `random_forest`
- Mean CV AUC: `0.5260`
- Mean CV log loss: `0.6989`
- 2024 holdout AUC: `0.5237`
- 2024 holdout log loss: `0.6945`

The full per-model benchmark is recorded in:

- `reports/results/event_panel_v2_primary_benchmark.csv`
- `reports/results/event_panel_v2_primary_benchmark.md`

## Feature Exclusion Rules

The following exclusions are now part of the locked primary setup:

- Explicit exclusions:
  - `gross_margin`
  - `current_filing_sentiment_available`
- Fold/holdout missingness rule:
  - drop any feature above `20%` missingness in the training slice
- Structural rule:
  - drop features that are constant inside the training slice

In the current benchmark run, the additional recurring train-slice exclusions were:

- `earnings_growth_yoy`
- `inventory_turnover`
- `receivables_turnover`
- `revenue_growth_yoy`

## GPU Decision

- XGBoost remains part of the anchor model set.
- In this environment, XGBoost ran on CPU.
- Reason: the local stack does not support clean CUDA prediction without the earlier device-mismatch warning.
- Policy going forward: use GPU only when the local stack supports clean CUDA prediction; otherwise fall back to CPU.

## What Is Frozen Or Dropped

Frozen for the next phase:

- observation unit = one row per ticker-event
- event types = current `10-Q` / `10-K` event panel design
- 5-day sign label
- anchor model family set only
- current benchmark and holdout rules

Dropped for now:

- 21-day sign as the main label
- quantile labels as the main label
- extra model families
- grouped SEC refinements
- analyst-title and other previously unproductive branches

## What Carries Forward To Universe Expansion

The following should carry forward unchanged into the next phase unless a hard blocker appears:

- `event_panel_v2` as the base research panel
- 5-day excess-return sign as the primary label
- `random_forest` as the current selected primary model
- logistic regression and XGBoost as benchmark comparators
- explicit dead-feature exclusions
- current CV and holdout policy

## What Will Not Be Rerun

The following are intentionally not part of the new anchor and should not be revived in this branch phase:

- old daily forward-filled panel benchmarking as the primary research path
- old `event_v1` branch logic as the primary path
- grouped 8-K feature branches
- analyst-title branches
- external-data experiments
- universe expansion before the locked anchor is accepted

## Interpretation

Relative to the old daily/event_v1 direction, the redesigned event-based setup is materially cleaner and modestly better on the benchmark metrics that matter most:

- old `event_v1_layer1` best CV AUC: `0.5056`
- new `event_panel_v2` primary CV AUC: `0.5260`
- old `event_v1_layer1` holdout AUC: `0.5180`
- new `event_panel_v2` primary holdout AUC: `0.5237`

That improvement is not large enough to justify method churn, but it is strong enough to lock the redesigned pipeline as the post-fix benchmark anchor before universe expansion.
