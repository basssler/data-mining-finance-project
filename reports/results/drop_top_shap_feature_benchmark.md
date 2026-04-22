# Quarterly Feature Design Sentiment Benchmark drop_top_shap_feature

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_feature_design_sentiment`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report adds filing-timed SEC sentiment probabilities onto the quarterly core family while keeping all market features excluded.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5101 | 0.7759 | 0.4254 | 0.8262 | 0.5349 | 0.3966 | -0.1465 | cpu |  |
| random_forest | 0.5021 | 0.6968 | 0.4940 | 0.7078 | 0.6061 | 0.3448 | -0.0590 | cpu |  |
| xgboost | 0.5147 | 0.7844 | 0.4876 | 0.7894 | 0.5682 | 0.4310 | -0.0484 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count, qfd_av_revision_x_surprise`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5147`
- Mean CV log loss: `0.7844`
- 2024 holdout AUC: `0.4876`
- 2024 holdout log loss: `0.7894`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5147` and best holdout AUC from `0.5180` to `0.4876`.
- Promote this family only if the cleaner sentiment probability block adds 63-day signal without displacing the quarterly-thesis features as the main driver set.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
