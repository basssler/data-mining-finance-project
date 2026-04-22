# Event Panel V2 Quarterly Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report is the new post-fix anchor to use before universe expansion.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5021 | 0.7834 | 0.4407 | 0.7666 | 0.5000 | 0.3448 | -0.1021 | cpu |  |
| random_forest | 0.5018 | 0.7016 | 0.4254 | 0.7106 | 0.5556 | 0.4310 | -0.1398 | cpu |  |
| xgboost | 0.5270 | 0.8109 | 0.4411 | 0.8103 | 0.5000 | 0.3966 | -0.1311 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5270`
- Mean CV log loss: `0.8109`
- 2024 holdout AUC: `0.4411`
- 2024 holdout log loss: `0.8103`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5270` and best holdout AUC from `0.5180` to `0.4411`.
- The redesigned setup is directionally better than the old daily research path, but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
