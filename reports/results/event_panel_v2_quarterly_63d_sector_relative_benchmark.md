# Event Panel V2 Quarterly 63D Sector-Relative Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_63d_sector_relative`
- Primary label: `63-trading-day sector-relative sign label`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Canonical final quarterly benchmark config using the 63-trading-day sector-relative sign label.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Rank IC | Backend | Dominant Feature | Concentration | Repro Holdout Std | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| logistic_regression | 0.5022 | 0.0571 | 0.4198 | 0.4992 | 0.7330 | 0.0019 | cpu | fund_snapshot_is_current_event | 0.8333 | 0.0000 | reference_only |
| random_forest | 0.5093 | 0.0711 | 0.3732 | 0.4487 | 0.7080 | -0.0969 | cpu | realized_vol_63d | 0.6667 | 0.0086 | reference_only |
| xgboost | 0.5127 | 0.0827 | 0.3522 | 0.5365 | 0.7448 | 0.0137 | cuda | fund_snapshot_is_current_event | 0.6667 | 0.0201 | candidate_only |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5127`
- CV AUC std: `0.0827`
- Worst fold AUC: `0.3522`
- 2024 holdout AUC: `0.5365`
- 2024 holdout log loss: `0.7448`
- Dominant feature concentration: `0.6667`
- Reproducibility holdout AUC std: `0.0201`
- Promotion status: `candidate_only`
- Promotion reason: `reproducibility_threshold_failed`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5127` and best holdout AUC from `0.5180` to `0.5365`.
- Use this as the active/default label contract for final project reporting. Historical shorter-horizon outputs are retained only as comparison artifacts.
