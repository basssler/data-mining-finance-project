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
| logistic_regression | 0.5254 | 0.7527 | 0.4919 | 0.7492 | 0.5072 | 0.5147 | 0.0538 | cpu |  |
| random_forest | 0.5228 | 0.6987 | 0.4960 | 0.6999 | 0.4545 | 0.4412 | -0.0455 | cpu |  |
| xgboost | 0.5415 | 0.7950 | 0.5388 | 0.7474 | 0.5616 | 0.6029 | 0.0902 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5415`
- Mean CV log loss: `0.7950`
- 2024 holdout AUC: `0.5388`
- 2024 holdout log loss: `0.7474`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5415` and best holdout AUC from `0.5180` to `0.5388`.
- The redesigned setup is directionally better than the old daily research path, but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
