# Event Panel V2 SEC Sentiment V1 Benchmark

## Scope

- Locked 34-ticker setup only.
- Same 5-trading-day excess return sign label.
- Same 2024 holdout and same three anchor models.
- No universe expansion and no new external datasets.

## Important Note

- The selected SEC filing sentiment artifact was already embedded in the locked `event_panel_v2` Phase 4 anchor.
- This Phase 6 run is therefore a reproducibility and explicit documentation pass for that existing dataset path, not a truly new incremental additive signal test.

## Panel Comparison

| Panel | Rows | Tickers | Feature Count | Selected Model |
|---|---:|---:|---:|---|
| Phase 4 anchor | 1,109 | 34 | 72 | random_forest |
| Phase 6 sec sentiment v1 | 1,109 | 34 | 72 | random_forest |

## Per-Model Comparison

| Model | Phase 4 CV AUC | Phase 6 CV AUC | Phase 4 CV Log Loss | Phase 6 CV Log Loss | Phase 4 Holdout AUC | Phase 6 Holdout AUC | Phase 4 Holdout Log Loss | Phase 6 Holdout Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5181 | 0.5181 | 0.7510 | 0.7510 | 0.5051 | 0.5051 | 0.7409 | 0.7409 |
| random_forest | 0.5260 | 0.5260 | 0.6989 | 0.6989 | 0.5237 | 0.5237 | 0.6945 | 0.6945 |
| xgboost | 0.5244 | 0.5244 | 0.8119 | 0.8119 | 0.5290 | 0.5290 | 0.7569 | 0.7569 |

