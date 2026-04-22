# Stock Direction Prediction: Master Results Summary

## Current Anchor

The canonical benchmark for this repo is the enriched `event_panel_v2` result in `reports/results/event_panel_v2_primary_benchmark.csv`.

- Universe: 34 Consumer Staples large caps
- Label: 5-trading-day excess return sign
- Selected model: `xgboost`
- CV AUC: `0.5415`
- Holdout AUC: `0.5388`

This is the benchmark all downstream phase summaries should be compared against unless a regenerated CSV clearly supersedes it.

## Phase Summary

### Phase 4: Enriched Event Panel Benchmark

| Model | CV AUC | Holdout AUC | Selected |
|---|---:|---:|---|
| Logistic Regression | 0.5254 | 0.4919 |  |
| Random Forest | 0.5228 | 0.4960 |  |
| XGBoost | **0.5415** | **0.5388** | yes |

Interpretation: this remains the main full-panel benchmark because it is the best current balance of cross-validation and holdout performance in the checked-in artifacts.

### Phase 5: Universe Expansion

- Compared the canonical 34-name benchmark against `event_panel_v2_universe_v2`.
- Expanded-universe selected model: `random_forest`
- Expanded-universe CV AUC: `0.5021`
- Expanded-universe holdout AUC: `0.5282`

Interpretation: the expanded universe underperforms the canonical anchor on both selected-model CV AUC and holdout AUC, so it is not promoted.

### Phase 6: SEC Sentiment Reproducibility

- `event_panel_v2_sec_sentiment_v1_benchmark.csv` currently matches the canonical `event_panel_v2` benchmark exactly on model-level CV AUC, holdout AUC, and selected model.
- Selected model: `xgboost`
- CV AUC: `0.5415`
- Holdout AUC: `0.5388`

Interpretation: this phase is a reproducibility confirmation for the SEC sentiment path, not a separate benchmark win.

### Phase 6B: Alpha Vantage Earnings Additive Test

- Manifest state: complete (`68` complete rows, no pending or failed rows)
- Selected model: `xgboost`
- CV AUC: `0.5262`
- Holdout AUC: `0.5580`
- Delta vs canonical anchor: CV AUC `-0.0154`, holdout AUC `+0.0192`

Interpretation: the additive result is real and no longer incomplete, but it is mixed rather than cleanly dominant. Because the selected-model CV AUC deteriorates while holdout AUC improves, the canonical enriched benchmark remains the default anchor and Phase 6B stays in `FREEZE`.

## Bottom Line

- The repo’s main benchmark is still the enriched `event_panel_v2` XGBoost result at holdout AUC `0.5388`.
- Universe expansion did not help.
- SEC sentiment reproducibility did not create a new benchmark distinct from the anchor.
- Alpha Vantage earnings is a completed additive test with mixed evidence, not an incomplete run.

## File References

- Canonical primary benchmark: `reports/results/event_panel_v2_primary_benchmark.csv`
- Universe comparison: `reports/results/event_panel_v2_universe_v2_benchmark.csv`
- SEC sentiment comparison: `reports/results/event_panel_v2_sec_sentiment_v1_benchmark.csv`
- Alpha Vantage comparison: `reports/results/event_panel_v2_phase6b_alpha_vantage_benchmark.csv`
- Runbook: `docs/benchmark_regeneration_runbook.md`
