# Quarterly Stability Core Raw Components Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_core_raw_components`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run uses raw Alpha Vantage revision and surprise components without the concentrated quarterly interaction terms.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4997 | 0.7751 | 0.4286 | 0.8315 | 0.4722 | 0.2931 | -0.1379 | cpu |  |
| random_forest | 0.5086 | 0.6969 | 0.4611 | 0.7156 | 0.6000 | 0.3103 | -0.1023 | cpu | yes |
| xgboost | 0.4952 | 0.7998 | 0.4318 | 0.8385 | 0.5556 | 0.3448 | -0.1096 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `random_forest`
- Mean CV AUC: `0.5086`
- Mean CV log loss: `0.6969`
- 2024 holdout AUC: `0.4611`
- 2024 holdout log loss: `0.7156`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5086` and best holdout AUC from `0.5180` to `0.4611`.
- Use this run to test whether the quarterly lane can survive on raw revision, raw surprise, and event-history context before introducing stronger composites.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
