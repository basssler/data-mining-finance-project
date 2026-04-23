# Event Panel V2 Quarterly Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly`
- Primary label: `21-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report is the new post-fix anchor to use before universe expansion.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4813 | 0.7844 | 0.4295 | 0.7674 | 0.4583 | 0.3188 | -0.0921 | cpu |  |
| random_forest | 0.5006 | 0.7109 | 0.4515 | 0.7121 | 0.4884 | 0.3043 | -0.1098 | cpu |  |
| xgboost | 0.5070 | 0.8708 | 0.4698 | 0.8278 | 0.4565 | 0.3043 | -0.1136 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5070`
- Mean CV log loss: `0.8708`
- 2024 holdout AUC: `0.4698`
- 2024 holdout log loss: `0.8278`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5070` and best holdout AUC from `0.5180` to `0.4698`.
- The redesigned setup is directionally better than the old daily research path, but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
