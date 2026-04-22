# Quarterly Stability Core Bucketed Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_core_bucketed`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run replaces the concentrated quarterly interactions with coarse buckets and regime flags.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4971 | 0.7740 | 0.4679 | 0.7772 | 0.5641 | 0.3793 | -0.0754 | cpu |  |
| random_forest | 0.4994 | 0.6988 | 0.4411 | 0.7173 | 0.5152 | 0.2931 | -0.1329 | cpu | yes |
| xgboost | 0.4882 | 0.8032 | 0.4631 | 0.8276 | 0.5116 | 0.3793 | -0.0954 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `random_forest`
- Mean CV AUC: `0.4994`
- Mean CV log loss: `0.6988`
- 2024 holdout AUC: `0.4411`
- 2024 holdout log loss: `0.7173`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.4994` and best holdout AUC from `0.5180` to `0.4411`.
- Use this run to test whether a simpler quarterly regime encoding survives more folds even if raw holdout performance softens.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
