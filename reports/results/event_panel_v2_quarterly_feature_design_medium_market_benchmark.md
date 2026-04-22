# Quarterly Feature Design Medium Market Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_feature_design_medium_market`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report adds a medium-horizon market context bucket onto the quarterly core family while keeping short-horizon trading features excluded.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4990 | 0.8194 | 0.4206 | 0.8422 | 0.4571 | 0.2759 | -0.1480 | cpu |  |
| random_forest | 0.5113 | 0.6968 | 0.3941 | 0.7241 | 0.4865 | 0.3103 | -0.1716 | cpu |  |
| xgboost | 0.5384 | 0.7987 | 0.4322 | 0.8472 | 0.5122 | 0.3621 | -0.1941 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, overnight_gap_1d, abs_return_shock_1d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5384`
- Mean CV log loss: `0.7987`
- 2024 holdout AUC: `0.4322`
- 2024 holdout log loss: `0.8472`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5384` and best holdout AUC from `0.5180` to `0.4322`.
- Promote this family only if medium-horizon market context improves the 63-day lane without reintroducing short-horizon trading dominance.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
