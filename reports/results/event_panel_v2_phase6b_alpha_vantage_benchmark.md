# Event Panel V2 Phase 6B Alpha Vantage Benchmark

## Scope

- Locked baseline preserved: `event_panel_v2`, 34 tickers, 5-trading-day excess return sign, 2024 holdout unchanged.
- One additive external dataset family only: Alpha Vantage earnings estimates/outcomes.
- Models unchanged: logistic regression, random forest, XGBoost.
- Manifest completion state: `{'complete': 68}`.

## Panel Summary

- Rows: `1,109`
- Tickers: `34`
- Total feature count: `95`
- New Alpha Vantage feature count: `20`

## Baseline vs Additive

| Model | Baseline CV AUC | Additive CV AUC | Baseline Holdout AUC | Additive Holdout AUC | Baseline CV Log Loss | Additive CV Log Loss | Baseline Holdout Log Loss | Additive Holdout Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5181 | 0.5096 | 0.5051 | 0.5341 | 0.7510 | 0.7703 | 0.7409 | 0.7448 |
| random_forest | 0.5260 | 0.5232 | 0.5237 | 0.5109 | 0.6989 | 0.6992 | 0.6945 | 0.6947 |
| xgboost | 0.5244 | 0.5233 | 0.5290 | 0.5774 | 0.8119 | 0.7990 | 0.7569 | 0.7132 |

## Decision

- Baseline selected model: `random_forest`
- Additive selected model: `xgboost`
- Current benchmark result is identical to baseline because every new Alpha Vantage feature was dropped by the existing 20% train-fold missingness rule under the partial backfill coverage.
- This should be treated as a partial-cache diagnostic run, not the final official Phase 6B verdict, until the remaining manifest rows are fetched with refreshed or replacement API keys.
- This benchmark should be read as the apples-to-apples Phase 6B test against the locked Phase 4 anchor.
