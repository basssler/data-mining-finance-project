# Quarterly Stability Sentiment Raw Components Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_sentiment_raw_components`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run combines raw quarterly revision and surprise components with filing-timed sentiment probabilities.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5065 | 0.7768 | 0.4302 | 0.8200 | 0.5349 | 0.3966 | -0.1345 | cpu | yes |
| random_forest | 0.4932 | 0.6991 | 0.4491 | 0.7145 | 0.5778 | 0.4483 | -0.1081 | cpu |  |
| xgboost | 0.5019 | 0.7898 | 0.4387 | 0.8326 | 0.5625 | 0.4655 | -0.0940 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Mean CV AUC: `0.5065`
- Mean CV log loss: `0.7768`
- 2024 holdout AUC: `0.4302`
- 2024 holdout log loss: `0.8200`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5065` and best holdout AUC from `0.5180` to `0.4302`.
- Use this run to test whether sentiment lifts the raw quarterly thesis without forcing the model back into concentrated interaction features.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
