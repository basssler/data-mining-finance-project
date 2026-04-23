# Quarterly Tuned Model Upgrade

## Locked Setup

- Primary panel: `quarterly_tuned_model_upgrade_v1`
- Primary label: `21-trading-day thresholded excess return (+/-1.5%)`
- Models: `catboost`, `hist_gradient_boosting`, `lightgbm`, `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Baseline keepers stay in the matrix while XGBoost, LightGBM, and CatBoost are tuned under the same purged quarterly evaluation policy.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Rank IC | Dominant Feature | Concentration | Repro Holdout Std | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| catboost | 0.5442 | 0.0760 | 0.4674 | 0.4868 | 0.7684 | -0.0576 | av_trailing_4q_eps_surprise_pct_mean | 0.5000 | 0.0206 | reference_only |
| hist_gradient_boosting | 0.5384 | 0.0609 | 0.4674 | 0.5244 | 0.9061 | 0.0458 | qfd_v2_roa_zsec | 0.5000 | 0.0000 | reference_only |
| lightgbm | 0.5107 | 0.0311 | 0.4673 | 0.5179 | 0.7917 | 0.0581 | qfd_av_surprise_pct_consistency | 0.3333 | 0.0132 | reference_only |
| logistic_regression | 0.5682 | 0.0776 | 0.4273 | 0.4286 | 1.3617 | -0.1051 | qfd_es_sentiment_entropy | 0.8333 | 0.0000 | reference_only |
| random_forest | 0.5210 | 0.0324 | 0.4789 | 0.5367 | 0.6976 | 0.0717 | av_trailing_4q_eps_surprise_pct_mean | 0.8333 | 0.0033 | candidate_only |
| xgboost | 0.5399 | 0.0610 | 0.4505 | 0.4765 | 0.7559 | -0.0186 | roa | 0.3333 | 0.0051 | reference_only |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, qfd_mkt_pre_event_return_5d, qfd_mkt_pre_event_return_21d, qfd_mkt_pre_event_excess_return_5d_sector, qfd_mkt_pre_event_excess_return_21d_sector, qfd_mkt_pre_event_excess_return_5d_market, qfd_mkt_pre_event_excess_return_21d_market, qfd_mkt_pre_event_volume_z_20d, qfd_mkt_pre_event_vol_5d, qfd_mkt_pre_event_vol_21d, qfd_mkt_pre_event_vol_ratio_5_21, qfd_mkt_first_tradable_gap, qfd_mkt_first_tradable_abnormal_volume, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `qfd_es_source_count`

## Selected Primary Model

- Selected model: `random_forest`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5210`
- CV AUC std: `0.0324`
- Worst fold AUC: `0.4789`
- 2024 holdout AUC: `0.5367`
- 2024 holdout log loss: `0.6976`
- Dominant feature concentration: `0.8333`
- Reproducibility holdout AUC std: `0.0033`
- Promotion status: `candidate_only`
- Promotion reason: `holdout_not_better_than_reference`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5210` and best holdout AUC from `0.5180` to `0.5367`.
- Promote only if the tuned winner improves on the current quarterly champion without sacrificing worst-fold behavior or reproducibility.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
- Reference benchmark for promotion was `random_forest` with holdout AUC `0.5367`.
