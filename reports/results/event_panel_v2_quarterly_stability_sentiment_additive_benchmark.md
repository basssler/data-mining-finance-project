# Quarterly Stability Sentiment Additive Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_sentiment_additive`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run layers sentiment probabilities onto the additive quarterly composites and clipped volatility context.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5083 | 0.7764 | 0.4791 | 0.7696 | 0.5227 | 0.3966 | -0.0636 | cpu |  |
| random_forest | 0.5021 | 0.6959 | 0.4687 | 0.7057 | 0.5854 | 0.4138 | -0.0552 | cpu |  |
| xgboost | 0.5116 | 0.7777 | 0.4683 | 0.8127 | 0.5682 | 0.4310 | -0.0739 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5116`
- Mean CV log loss: `0.7777`
- 2024 holdout AUC: `0.4683`
- 2024 holdout log loss: `0.8127`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5116` and best holdout AUC from `0.5180` to `0.4683`.
- Use this run to test whether sentiment helps the simpler quarterly composites survive across folds without a single dominant interaction feature.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
