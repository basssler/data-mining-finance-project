# Quarterly Promoted Models

This registry is the canonical status board for quarterly benchmark promotion.

## Promotion Rule

- beat the quarterly baseline on the agreed comparison metric set
- show acceptable worst-fold behavior and concentration stability
- have a reproducible config and checked-in benchmark artifact pair
- clear a review step recorded in `docs/quarterly/experiment_log.md`

## Registry

| Benchmark | Dataset Version | Label Version | Validation Version | Feature Family | Model Family | Status | Evidence |
|---|---|---|---|---|---|---|---|
| `event_panel_v2_quarterly` | `data/interim/event_panel_v2.parquet` | `event_v2_63d_sign` | `5`-fold purged expanding window, `2024` holdout | baseline quarterly event panel | `xgboost` selected | `baseline` | `reports/results/event_panel_v2_quarterly_benchmark.csv` |
| `quarterly_core_no_market_anchor_v1` | `data/interim/event_panel_v2_quarterly_feature_design.parquet` | `event_v2_21d_excess_threshold` | `5`-fold purged expanding window, `2024` holdout | quarterly core with both generic and Phase 8 market layers excluded | `random_forest` selected | `active_anchor` | `reports/results/quarterly_phase8_market_comparison.csv` |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `data/interim/event_panel_v2_quarterly_feature_design.parquet` | `event_v2_21d_excess_threshold` | `5`-fold purged expanding window, `2024` holdout | quarterly core plus event-specific sentiment only, with broad filing sentiment and Phase 8 market layers excluded | `random_forest` selected | `current_champion` | `reports/results/quarterly_phase9_event_specific_sentiment_champion_benchmark.csv` |
| `event_panel_v2_quarterly_feature_design_core` | `data/interim/event_panel_v2_quarterly_feature_design.parquet` | `event_v2_63d_sign` | `5`-fold purged expanding window, `2024` holdout | expanded accounting + Alpha Vantage additive quarter features | `xgboost` selected | `comparison_only` | `reports/results/event_panel_v2_quarterly_feature_design_core_benchmark.csv` |
| `event_panel_v2_quarterly_feature_design_medium_market` | `data/interim/event_panel_v2_quarterly_feature_design.parquet` | `event_v2_63d_sign` | `5`-fold purged expanding window, `2024` holdout | quarterly accounting plus medium-horizon market context | `xgboost` selected | `comparison_only` | `reports/results/event_panel_v2_quarterly_feature_design_medium_market_benchmark.csv` |
| `event_panel_v2_quarterly_feature_design_sentiment` | `data/interim/event_panel_v2_quarterly_feature_design.parquet` | `event_v2_63d_sign` | `5`-fold purged expanding window, `2024` holdout | quarterly accounting plus sentiment probabilities | `random_forest` selected | `comparison_only` | `reports/results/event_panel_v2_quarterly_feature_design_sentiment_benchmark.csv` |
| `event_panel_v2_quarterly_stability_core_additive` | `data/interim/event_panel_v2_quarterly_feature_design.parquet` | `event_v2_63d_sign` | `5`-fold purged expanding window, `2024` holdout | stability-oriented additive quarterly composites | `xgboost` selected | `candidate_under_review` | `reports/results/event_panel_v2_quarterly_stability_core_additive_benchmark.csv` |

## Current Read

- Active quarterly champion is `quarterly_phase9_event_specific_sentiment_champion_v1`.
- The prior frozen anchor `quarterly_core_no_market_anchor_v1` remains the last pre-Phase-9 comparison anchor.
- Phase 8 event-aware market features were reviewed and explicitly not promoted into the benchmark stack.
- Phase 9 event-specific sentiment passed the frozen thresholded comparison: `event_specific_sentiment_only` beat both `core_no_sentiment` and `broad_filing_sentiment_only`, while the combined sentiment block did not.
- Highest checked-in quarterly holdout AUC on the live 21-day thresholded contract is now `0.5367` from `quarterly_phase9_event_specific_sentiment_champion_v1`.
