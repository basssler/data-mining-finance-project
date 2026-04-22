# Event Panel V2 Quarterly Alpha Vantage Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_alpha_vantage`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report is a parallel quarterly experiment on the Alpha Vantage earnings-enriched panel, not a replacement for the 5-day anchor.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5069 | 0.8082 | 0.3841 | 0.8562 | 0.5349 | 0.3966 | -0.2168 | cpu |  |
| random_forest | 0.5066 | 0.6966 | 0.4190 | 0.7136 | 0.5238 | 0.3793 | -0.1425 | cpu |  |
| xgboost | 0.5251 | 0.7977 | 0.4683 | 0.8118 | 0.6000 | 0.4655 | -0.0908 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5251`
- Mean CV log loss: `0.7977`
- 2024 holdout AUC: `0.4683`
- 2024 holdout log loss: `0.8118`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5251` and best holdout AUC from `0.5180` to `0.4683`.
- Adding prior-quarter surprise and estimate-revision features improves the quarterly holdout versus the base 63-day lane, but the result still trails the historical 5-day anchor and should be treated as a research comparison rather than a promoted default.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
