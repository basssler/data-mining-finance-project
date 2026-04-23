# Quarterly Feature Design Core Benchmark levels_plus_deltas_plus_cross_sectional

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_feature_design_core`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report evaluates the quarterly core family only: fundamentals, profile scores, event context, Alpha Vantage earnings features, and quarterly derived interactions.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5044 | 0.0230 | 0.4695 | 0.4475 | 0.8575 | 0.5500 | 0.3793 | -0.1152 | cpu |  |
| random_forest | 0.5055 | 0.0533 | 0.4030 | 0.4631 | 0.7068 | 0.5366 | 0.3793 | -0.0710 | cpu |  |
| xgboost | 0.5200 | 0.0307 | 0.4921 | 0.5072 | 0.8018 | 0.5909 | 0.4483 | -0.0020 | cpu | yes |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `xgboost`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5200`
- CV AUC std: `0.0307`
- Worst fold AUC: `0.4921`
- 2024 holdout AUC: `0.5072`
- 2024 holdout log loss: `0.8018`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5200` and best holdout AUC from `0.5180` to `0.5072`.
- Promote this family only if it materially improves the 63-day holdout without reverting to short-horizon trading proxies as the dominant explanation.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
