# Quarterly Benchmark Ladder

Derived from the checked-in quarterly configs and benchmark artifacts in the current repo state.

| Step | Title | Status | Config | CSV | Markdown |
|---:|---|---|---|---|---|
| 1 | Old daily 5-day baseline | partial |  | `reports/results/event_v1_layer1_metrics.json` | `reports/results/event_v1_layer1.md` |
| 2 | Quarterly event baseline with old features | complete | `configs/event_panel_v2_quarterly.yaml` | `reports/results/event_panel_v2_quarterly_benchmark.csv` | `reports/results/event_panel_v2_quarterly_benchmark.md` |
| 3 | Quarterly event baseline with improved label | planned |  |  |  |
| 4 | Quarterly event baseline with purged validation | planned |  |  |  |
| 5 | Quarterly event plus expanded accounting | complete | `configs/event_panel_v2_quarterly_feature_design_core.yaml` | `reports/results/event_panel_v2_quarterly_feature_design_core_benchmark.csv` | `reports/results/event_panel_v2_quarterly_feature_design_core_benchmark.md` |
| 6 | Quarterly event plus market event layer | complete | `configs/event_panel_v2_quarterly_feature_design_medium_market.yaml` | `reports/results/event_panel_v2_quarterly_feature_design_medium_market_benchmark.csv` | `reports/results/event_panel_v2_quarterly_feature_design_medium_market_benchmark.md` |
| 7 | Quarterly event plus sentiment event layer | complete | `configs/event_panel_v2_quarterly_feature_design_sentiment.yaml` | `reports/results/event_panel_v2_quarterly_feature_design_sentiment_benchmark.csv` | `reports/results/event_panel_v2_quarterly_feature_design_sentiment_benchmark.md` |
| 8 | Tuned boosted-tree model | complete | `configs/event_panel_v2_quarterly_stability_core_additive.yaml` | `reports/results/event_panel_v2_quarterly_stability_core_additive_benchmark.csv` | `reports/results/event_panel_v2_quarterly_stability_core_additive_benchmark.md` |
| 9 | Frozen thresholded quarterly anchor | complete | `configs/quarterly/quarterly_core_no_market_anchor_v1.yaml` | `reports/results/quarterly_phase8_market_comparison.csv` | `reports/results/quarterly_phase8_summary.md` |
| 10 | Phase 9 event-specific sentiment champion | complete | `configs/quarterly/quarterly_phase9_event_specific_sentiment_champion_v1.yaml` | `reports/results/quarterly_phase9_event_specific_sentiment_champion_benchmark.csv` | `reports/results/quarterly_phase9_summary.md` |

## Operating Rule

Advance one major layer at a time: panel, labels, validation, features, model tuning, promotion.

## Current Gaps

- Step 1: `Old daily 5-day baseline` is `partial` (Historical baseline only. No quarterly config should point here.).
- Step 3: `Quarterly event baseline with improved label` is `planned` (Planned. The repo still needs a distinct improved-label quarterly config and result set.).
- Step 4: `Quarterly event baseline with purged validation` is `planned` (Current quarterly configs already use purged splits, but this rung is reserved for a dedicated validation package.).
- Step 9: `Frozen thresholded quarterly anchor` is `complete` (Phase 8 was reviewed and rejected for promotion; `core_no_market` was the last pre-Phase-9 anchor.).
- Step 10: `Phase 9 event-specific sentiment champion` is `complete` (Phase 9 passed the frozen comparison, so `quarterly_phase9_event_specific_sentiment_champion_v1` is now the benchmark to beat.).
