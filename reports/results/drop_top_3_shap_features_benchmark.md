# Quarterly Feature Design Sentiment Benchmark drop_top_3_shap_features

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_feature_design_sentiment`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report adds filing-timed SEC sentiment probabilities onto the quarterly core family while keeping all market features excluded.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5123 | 0.7692 | 0.4727 | 0.7645 | 0.5556 | 0.4310 | -0.0834 | cpu | yes |
| random_forest | 0.5031 | 0.6970 | 0.4623 | 0.7094 | 0.5814 | 0.4310 | -0.0726 | cpu |  |
| xgboost | 0.5014 | 0.7871 | 0.4611 | 0.8188 | 0.5532 | 0.4483 | -0.1277 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count, qfd_av_revision_x_surprise, qfd_av_latest_surprise_vs_trailing_pct, av_trailing_4q_eps_surprise_pct_std`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Mean CV AUC: `0.5123`
- Mean CV log loss: `0.7692`
- 2024 holdout AUC: `0.4727`
- 2024 holdout log loss: `0.7645`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5123` and best holdout AUC from `0.5180` to `0.4727`.
- Promote this family only if the cleaner sentiment probability block adds 63-day signal without displacing the quarterly-thesis features as the main driver set.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
