# Quarterly Phase 9 combined_sentiment_block

## Locked Setup

- Primary panel: `quarterly_phase9_combined_sentiment_block`
- Primary label: `21-trading-day thresholded excess return (+/-1.5%)`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Frozen quarterly core anchor plus both broad filing sentiment and Phase 9 event-specific sentiment.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5747 | 0.0774 | 0.4333 | 0.4352 | 1.3973 | 0.5676 | 0.5526 | -0.0938 | cpu |  |
| random_forest | 0.5015 | 0.0392 | 0.4545 | 0.5179 | 0.7035 | 0.6071 | 0.4474 | 0.0291 | cpu | yes |
| xgboost | 0.5165 | 0.0510 | 0.4253 | 0.3957 | 0.8881 | 0.4815 | 0.3421 | -0.1154 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, qfd_mkt_pre_event_return_5d, qfd_mkt_pre_event_return_21d, qfd_mkt_pre_event_excess_return_5d_sector, qfd_mkt_pre_event_excess_return_21d_sector, qfd_mkt_pre_event_excess_return_5d_market, qfd_mkt_pre_event_excess_return_21d_market, qfd_mkt_pre_event_volume_z_20d, qfd_mkt_pre_event_vol_5d, qfd_mkt_pre_event_vol_21d, qfd_mkt_pre_event_vol_ratio_5_21, qfd_mkt_first_tradable_gap, qfd_mkt_first_tradable_abnormal_volume`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `qfd_es_source_count`

## Selected Primary Model

- Selected model: `random_forest`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5015`
- CV AUC std: `0.0392`
- Worst fold AUC: `0.4545`
- 2024 holdout AUC: `0.5179`
- 2024 holdout log loss: `0.7035`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5015` and best holdout AUC from `0.5180` to `0.5179`.
- Use this as the active benchmark anchor until a later phase materially beats it on the same purged quarterly evaluation contract.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
