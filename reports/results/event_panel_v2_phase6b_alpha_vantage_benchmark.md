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
| logistic_regression | 0.5164 | 0.7706 | 0.5141 | 0.7563 | 0.5132 | 0.5735 | 0.0414 | cpu |  |
| random_forest | 0.5253 | 0.6990 | 0.4876 | 0.6973 | 0.4648 | 0.4853 | -0.0277 | cpu |  |
| xgboost | 0.5262 | 0.7984 | 0.5580 | 0.7276 | 0.5714 | 0.5882 | 0.1223 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5262`
- Mean CV log loss: `0.7984`
- 2024 holdout AUC: `0.5580`
- 2024 holdout log loss: `0.7276`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5262` and best holdout AUC from `0.5180` to `0.5580`.
- The redesigned setup is directionally better than the old daily research path, but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
