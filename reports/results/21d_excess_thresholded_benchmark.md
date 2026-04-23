# Quarterly Feature Design Core Benchmark 21d_excess_thresholded

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_feature_design_core`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report evaluates the quarterly core family only: fundamentals, profile scores, event context, Alpha Vantage earnings features, and quarterly derived interactions.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5685 | 0.0864 | 0.4136 | 0.4521 | 1.3232 | 0.5312 | 0.4474 | -0.0340 | cpu |  |
| random_forest | 0.5126 | 0.0346 | 0.4652 | 0.5291 | 0.7018 | 0.6000 | 0.3947 | 0.0732 | cpu | yes |
| xgboost | 0.5123 | 0.0374 | 0.4495 | 0.4774 | 0.8380 | 0.5758 | 0.5000 | -0.0001 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `random_forest`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5126`
- CV AUC std: `0.0346`
- Worst fold AUC: `0.4652`
- 2024 holdout AUC: `0.5291`
- 2024 holdout log loss: `0.7018`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5126` and best holdout AUC from `0.5180` to `0.5291`.
- Promote this family only if it materially improves the 63-day holdout without reverting to short-horizon trading proxies as the dominant explanation.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
