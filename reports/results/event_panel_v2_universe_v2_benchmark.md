# Event Panel V2 Primary Benchmark

## Locked Setup

- Primary panel: `event_panel_v2`
- Primary label: `5-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report is the new post-fix anchor to use before universe expansion.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4977 | 0.7022 | 0.4917 | 0.7013 | 0.4520 | 0.7048 | 0.0212 | cpu |  |
| random_forest | 0.5021 | 0.6979 | 0.5282 | 0.6932 | 0.4774 | 0.6035 | 0.0705 | cpu | yes |
| xgboost | 0.5006 | 0.7471 | 0.5242 | 0.7069 | 0.4803 | 0.5374 | 0.0570 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `random_forest`
- Mean CV AUC: `0.5021`
- Mean CV log loss: `0.6979`
- 2024 holdout AUC: `0.5282`
- 2024 holdout log loss: `0.6932`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5021` and best holdout AUC from `0.5180` to `0.5282`.
- The redesigned setup is directionally better than the old daily research path, but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
