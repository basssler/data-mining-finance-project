# Quarterly Phase 8 generic_and_event_aware_market

## Locked Setup

- Primary panel: `quarterly_final_core_confirmation_v1`
- Primary label: `21-trading-day thresholded excess return (+/-1.5%)`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Final quarterly-core confirmation run using the rescued levels_plus_deltas_plus_cross_sectional stack and the provisional 21d thresholded excess label.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5530 | 0.0473 | 0.4591 | 0.4060 | 1.4894 | 0.5789 | 0.5789 | -0.0706 | cpu | yes |
| random_forest | 0.5053 | 0.0515 | 0.4305 | 0.4276 | 0.7156 | 0.4800 | 0.3158 | -0.0459 | cpu |  |
| xgboost | 0.5045 | 0.0609 | 0.4042 | 0.5000 | 0.7985 | 0.5517 | 0.4211 | 0.0086 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5530`
- CV AUC std: `0.0473`
- Worst fold AUC: `0.4591`
- 2024 holdout AUC: `0.4060`
- 2024 holdout log loss: `1.4894`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5530` and best holdout AUC from `0.5180` to `0.4060`.
- Promote this quarterly core only if the thresholded-label gain is meaningful relative to both the prior excess-sign winner and the older default-label baseline.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
