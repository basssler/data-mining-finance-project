# Quarterly Stability Sentiment Capped Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_sentiment_capped`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run adds filing-timed sentiment probabilities to the capped quarterly interaction style.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5100 | 0.7817 | 0.5044 | 0.7504 | 0.5227 | 0.3966 | -0.0302 | cpu | yes |
| random_forest | 0.5001 | 0.6964 | 0.4816 | 0.7033 | 0.5946 | 0.3793 | -0.0528 | cpu |  |
| xgboost | 0.5064 | 0.7927 | 0.4370 | 0.8183 | 0.5366 | 0.3793 | -0.1447 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Mean CV AUC: `0.5100`
- Mean CV log loss: `0.7817`
- 2024 holdout AUC: `0.5044`
- 2024 holdout log loss: `0.7504`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5100` and best holdout AUC from `0.5180` to `0.5044`.
- Use this run to test whether sentiment can preserve holdout performance after the quarterly interaction is capped and simplified.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
