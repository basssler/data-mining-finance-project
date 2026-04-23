# Quarterly Feature Design Core Benchmark levels_only

## Locked Setup

- Primary panel: `event_panel_v2_quarterly_feature_design_core`
- Primary label: `63-trading-day excess return sign`
- Models: `logistic_regression`, `random_forest`, `xgboost`
- 2024 holdout policy: unchanged
- This report evaluates the quarterly core family only: fundamentals, profile scores, event context, Alpha Vantage earnings features, and quarterly derived interactions.

## Per-Model Results

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| logistic_regression | 0.4880 | 0.0294 | 0.4472 | 0.4046 | 0.8391 | 0.4737 | 0.3103 | -0.1896 | cpu | yes |
| random_forest | 0.4982 | 0.0751 | 0.3622 | 0.4431 | 0.7155 | 0.5455 | 0.3103 | -0.0808 | cpu |  |
| xgboost | 0.4962 | 0.0529 | 0.4186 | 0.4302 | 0.8633 | 0.5517 | 0.2759 | -0.0616 | cpu |  |

## Feature Exclusions

- Explicit exclusions: `gross_margin, current_filing_sentiment_available, rel_return_5d, rel_return_10d, rel_return_21d, realized_vol_21d, realized_vol_63d, vol_ratio_21d_63d, beta_63d_to_sector, overnight_gap_1d, abs_return_shock_1d, drawdown_21d, return_zscore_21d, volume_ratio_20d, log_volume, abnormal_volume_flag, sec_sentiment_score, sec_positive_prob, sec_negative_prob, sec_neutral_prob, sec_sentiment_abs, sec_sentiment_change_prev, sec_positive_change_prev, sec_negative_change_prev, sec_chunk_count, sec_log_chunk_count, qfd_av_estimate_breadth_mean, qfd_av_latest_surprise_above_trailing_flag, qfd_av_latest_surprise_vs_trailing_pct, qfd_av_latest_surprise_vs_trailing_pct_abs, qfd_av_latest_surprise_vs_trailing_pct_bucket, qfd_av_revision_acceleration, qfd_av_revision_breadth_pressure, qfd_av_revision_direction_flag, qfd_av_revision_magnitude_bucket, qfd_av_revision_plus_surprise, qfd_av_revision_pressure, qfd_av_revision_surprise_same_sign_flag, qfd_av_revision_x_growth_quality, qfd_av_revision_x_surprise, qfd_av_revision_x_surprise_capped, qfd_av_surprise_consistency, qfd_av_surprise_consistency_bucket, qfd_av_surprise_pct_consistency, qfd_av_surprise_x_financial_health, qfd_av_trailing_surprise_pct_std_bucket, qfd_av_trailing_surprise_pct_std_clipped, qfd_delta_accruals_ratio, qfd_delta_asset_turnover, qfd_delta_earnings_growth_qoq, qfd_delta_growth_quality_profile_score, qfd_delta_liquidity_profile_score, qfd_delta_net_margin, qfd_delta_operating_margin, qfd_delta_overall_financial_health_score, qfd_delta_profitability_profile_score, qfd_delta_revenue_growth_qoq, qfd_delta_roa, qfd_delta_solvency_profile_score, qfd_event_gap_difference, qfd_event_gap_ratio, qfd_growth_delta_combo, qfd_log_days_since_last_earnings_release, qfd_log_days_since_prior_event, qfd_log_days_since_prior_same_event_type, qfd_margin_delta_combo, qfd_prior_event_gap_bucket, qfd_prior_same_type_gap_bucket, qfd_profile_delta_combo, qfd_short_cycle_flag, qfd_short_same_type_cycle_flag, qfd_v2_earnings_growth_yoy_accel, qfd_v2_net_margin_accel, qfd_v2_operating_margin_accel, qfd_v2_revenue_growth_yoy_accel, qfd_v2_accruals_ratio_zsec, qfd_v2_debt_to_assets_d1_ranksec, qfd_v2_debt_to_assets_zsec, qfd_v2_operating_margin_d1_ranksec, qfd_v2_operating_margin_d1_zsec, qfd_v2_profitability_profile_score_ranksec, qfd_v2_roa_zsec, qfd_v2_accruals_ratio_d1, qfd_v2_cfo_to_net_income_d1, qfd_v2_cfo_to_net_income_d4, qfd_v2_debt_to_assets_d1, qfd_v2_earnings_growth_yoy_d4, qfd_v2_gross_margin_d1, qfd_v2_net_margin_d1, qfd_v2_operating_margin_d1, qfd_v2_revenue_growth_yoy_d4, qfd_v2_roa_d1, qfd_v2_working_capital_to_total_assets_d1, qfd_v2_cfo_to_net_income_stb4q, qfd_v2_earnings_growth_yoy_stb4q, qfd_v2_net_margin_pos_count_stb4q, qfd_v2_net_margin_stb4q, qfd_v2_operating_margin_pos_count_stb4q, qfd_v2_operating_margin_stb4q, qfd_v2_revenue_growth_yoy_stb4q, qfd_v3_accruals_ratio_d1, qfd_v3_capex_intensity_d1, qfd_v3_cfo_to_net_income_d1, qfd_v3_free_cash_flow_margin_d1, qfd_v3_free_cash_flow_to_net_income_d1, qfd_v3_gross_margin_d1, qfd_v3_interest_coverage_d1, qfd_v3_leverage_change_qoq_d1, qfd_v3_net_margin_d1, qfd_v3_operating_margin_d1, qfd_v3_shareholder_payout_ratio_d1, qfd_v3_total_debt_to_assets_d1`
- Auto all-missing exclusions: `inventory_turnover, interest_coverage, capex_intensity, free_cash_flow, free_cash_flow_margin, free_cash_flow_to_net_income, shareholder_payout_ratio, qfd_v2_gross_margin_lvl`
- Auto constant exclusions: `none`

## Selected Primary Model

- Selected model: `logistic_regression`
- Promotion strategy: `stability_aware`
- Mean CV AUC: `0.4880`
- CV AUC std: `0.0294`
- Worst fold AUC: `0.4472`
- 2024 holdout AUC: `0.4046`
- 2024 holdout log loss: `0.8391`

## Interpretation

- Against the old daily/event_v1 direction (`event_v1_layer1` best model `hist_gradient_boosting`), the redesigned event setup improves best CV AUC from `0.5056` to `0.4880` and best holdout AUC from `0.5180` to `0.4046`.
- Promote this family only if it materially improves the 63-day holdout without reverting to short-horizon trading proxies as the dominant explanation.
- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning.
