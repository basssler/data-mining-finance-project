# Quarterly Phase 8 event_aware_market_only

## Locked Setup

- Primary panel: `quarterly_final_core_confirmation_v1`
- Primary label: `21-trading-day thresholded excess return (+/-1.5%)`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Final quarterly-core confirmation run using the rescued levels_plus_deltas_plus_cross_sectional stack and the provisional 21d thresholded excess label.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5780 | 0.0596 | 0.4788 | 0.4173 | 1.4409 | 0.5676 | 0.5526 | -0.0612 | cpu | yes |
| random_forest | 0.5093 | 0.0528 | 0.4400 | 0.4709 | 0.7097 | 0.5200 | 0.3421 | 0.0022 | cpu |  |
| xgboost | 0.4870 | 0.0614 | 0.3968 | 0.4784 | 0.8482 | 0.5588 | 0.5000 | 0.0284 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5780`
- CV AUC std: `0.0596`
- Worst fold AUC: `0.4788`
- 2024 holdout AUC: `0.4173`
- 2024 holdout log loss: `1.4409`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5780` and best holdout AUC from `0.5180` to `0.4173`.
- Promote this quarterly core only if the thresholded-label gain is meaningful relative to both the prior excess-sign winner and the older default-label baseline.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
