# Quarterly Stability Core Additive Benchmark

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_stability_core_additive`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This run replaces the concentrated revision-surprise interaction with additive quarterly composites and clipped volatility context.

## Per-Model Results

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4998 | 0.7740 | 0.4623 | 0.7826 | 0.5385 | 0.3621 | -0.0845 | cpu |  |
| random_forest | 0.5058 | 0.6975 | 0.4399 | 0.7165 | 0.5161 | 0.2759 | -0.1235 | cpu |  |
| xgboost | 0.5262 | 0.7712 | 0.4880 | 0.8086 | 0.5641 | 0.3793 | -0.0101 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `none`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Mean CV AUC: `0.5262`
- Mean CV log loss: `0.7712`
- 2024 holdout AUC: `0.4880`
- 2024 holdout log loss: `0.8086`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5262` and best holdout AUC from `0.5180` to `0.4880`.
- Use this run to test whether additive quarterly composites hold up better across folds than the raw concentrated interaction.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
