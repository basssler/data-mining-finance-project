# Quarterly Phase 9 broad_filing_sentiment_only

## Locked Setup

- Primary panel: `quarterly_phase9_broad_filing_sentiment_only`
- Primary label: `21-trading-day thresholded excess return (+/-1.5%)`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Frozen quarterly core anchor plus broad filing sentiment only.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.5605 | 0.0756 | 0.4189 | 0.4182 | 1.3851 | 0.5714 | 0.5263 | -0.1064 | cpu |  |
| random_forest | 0.5162 | 0.0390 | 0.4653 | 0.5263 | 0.6970 | 0.6207 | 0.4737 | 0.0566 | cpu | yes |
| xgboost | 0.5156 | 0.0325 | 0.4547 | 0.4427 | 0.8601 | 0.5625 | 0.4737 | -0.0645 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, qfd_mkt_pre_event_return_5d, qfd_mkt_pre_event_return_21d, qfd_mkt_pre_event_excess_return_5d_sector, qfd_mkt_pre_event_excess_return_21d_sector, qfd_mkt_pre_event_excess_return_5d_market, qfd_mkt_pre_event_excess_return_21d_market, qfd_mkt_pre_event_volume_z_20d, qfd_mkt_pre_event_vol_5d, qfd_mkt_pre_event_vol_21d, qfd_mkt_pre_event_vol_ratio_5_21, qfd_mkt_first_tradable_gap, qfd_mkt_first_tradable_abnormal_volume, qfd_es_filing_negative_prob, qfd_es_filing_negative_prob_delta_prev_q, qfd_es_filing_neutral_prob, qfd_es_filing_positive_prob, qfd_es_filing_positive_prob_delta_prev_q, qfd_es_filing_sentiment_abs, qfd_es_filing_sentiment_delta_prev_q, qfd_es_filing_sentiment_score, qfd_es_log_text_chunk_count, qfd_es_negative_surprise_vs_90d, qfd_es_negative_tone_jump, qfd_es_sentiment_entropy, qfd_es_sentiment_polarity_balance, qfd_es_sentiment_surprise_vs_90d, qfd_es_sentiment_uncertainty, qfd_es_source_count, qfd_es_text_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `random_forest`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5162`
- CV AUC std: `0.0390`
- Worst fold AUC: `0.4653`
- 2024 holdout AUC: `0.5263`
- 2024 holdout log loss: `0.6970`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5162` and best holdout AUC from `0.5180` to `0.5263`.
- Use this as the active benchmark anchor until a later phase materially beats it on the same purged quarterly evaluation contract.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
