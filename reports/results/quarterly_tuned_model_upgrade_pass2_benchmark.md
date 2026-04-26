# Quarterly Tuned Model Upgrade Pass 2

## Locked Setup

- Primary panel: `quarterly_tuned_model_upgrade_pass2_v1`
- Primary label: `21-trading-day thresholded excess return (+/-1.5%)`
- Models: `catboost`, `hist_gradient_boosting`, `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- Second-pass retune focused on the only nonlinear candidates worth more search budget after the first benchmark run.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Rank IC | Backend | Dominant Feature | Concentration | Repro Holdout Std | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| catboost | 0.5436 | 0.0542 | 0.4439 | 0.4342 | 0.8110 | -0.0934 | cuda | earnings_growth_qoq | 0.3333 | 0.0369 | reference_only |
| hist_gradient_boosting | 0.5222 | 0.0751 | 0.3811 | 0.5019 | 0.8834 | 0.0184 | cpu | days_since_prior_event | 0.3333 | 0.0000 | reference_only |
| logistic_regression | 0.5726 | 0.0662 | 0.4879 | 0.4135 | 1.3732 | -0.1080 | cpu | qfd_es_sentiment_entropy | 0.6667 | 0.0000 | reference_only |
| random_forest | 0.5162 | 0.0373 | 0.4576 | 0.5367 | 0.7026 | 0.0571 | cpu | qfd_av_surprise_pct_consistency | 0.8333 | 0.0138 | candidate_only |
| xgboost | 0.5471 | 0.0654 | 0.4583 | 0.4934 | 0.7411 | 0.0177 | cuda | av_trailing_4q_eps_surprise_pct_mean | 0.3333 | 0.0104 | reference_only |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, qfd_mkt_pre_event_return_5d, qfd_mkt_pre_event_return_21d, qfd_mkt_pre_event_excess_return_5d_sector, qfd_mkt_pre_event_excess_return_21d_sector, qfd_mkt_pre_event_excess_return_5d_market, qfd_mkt_pre_event_excess_return_21d_market, qfd_mkt_pre_event_volume_z_20d, qfd_mkt_pre_event_vol_5d, qfd_mkt_pre_event_vol_21d, qfd_mkt_pre_event_vol_ratio_5_21, qfd_mkt_first_tradable_gap, qfd_mkt_first_tradable_abnormal_volume, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl, qfd_v2_gross_margin_d1, qfd_v3_gross_margin_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_capex_intensity_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_free_cash_flow_to_net_income_d1`
- Auto constant exclusions: `qfd_es_source_count`

## Selected Primary Model

- Selected model: `random_forest`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.5162`
- CV AUC std: `0.0373`
- Worst fold AUC: `0.4576`
- 2024 holdout AUC: `0.5367`
- 2024 holdout log loss: `0.7026`
- Dominant feature concentration: `0.8333`
- Reproducibility holdout AUC std: `0.0138`
- Promotion status: `candidate_only`
- Promotion reason: `holdout_not_better_than_reference,worst_fold_below_reference,reproducibility_threshold_failed`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.5162` and best holdout AUC from `0.5180` to `0.5367`.
- Promote only if the focused retune improves on the current quarterly champion and still beats the baseline keepers on holdout quality.
- Reference benchmark for promotion was `random_forest` with holdout AUC `0.5367`.
