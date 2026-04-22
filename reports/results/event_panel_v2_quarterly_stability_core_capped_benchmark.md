# Quarterly Stability Core Capped Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_core_capped`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run caps the revision-surprise interaction and uses absolute deviation plus clipped trailing surprise volatility.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5043 | 0.7808 | 0.4804 | 0.7616 | 0.5250 | 0.3621 | -0.0671 | cpu |  |
| random_forest | 0.5071 | 0.6964 | 0.4218 | 0.7197 | 0.5312 | 0.2931 | -0.1225 | cpu |  |
| xgboost | 0.5140 | 0.7993 | 0.4615 | 0.8328 | 0.5278 | 0.3276 | -0.0660 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5140`
- Mean CV log loss: `0.7993`
- 2024 holdout AUC: `0.4615`
- 2024 holdout log loss: `0.8328`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5140` and best holdout AUC from `0.5180` to `0.4615`.
- Use this run to test whether the quarterly lane keeps most of the holdout signal while reducing reliance on extreme interaction values.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
