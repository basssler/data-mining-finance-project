# Quarterly Stability Sentiment Bucketed Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_sentiment_bucketed`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run combines sentiment probabilities with the simplest quarterly regime buckets and flags.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5099 | 0.7751 | 0.4844 | 0.7642 | 0.5957 | 0.4828 | -0.0497 | cpu | yes |
| random_forest | 0.4949 | 0.6972 | 0.4723 | 0.7084 | 0.5278 | 0.3276 | -0.0831 | cpu |  |
| xgboost | 0.4864 | 0.7985 | 0.4515 | 0.8056 | 0.5581 | 0.4138 | -0.1122 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Mean CV AUC: `0.5099`
- Mean CV log loss: `0.7751`
- 2024 holdout AUC: `0.4844`
- 2024 holdout log loss: `0.7642`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5099` and best holdout AUC from `0.5180` to `0.4844`.
- Use this run to test whether the quarterly sentiment lane can keep signal while moving most of the dominant drivers into coarse, more stable regimes.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
