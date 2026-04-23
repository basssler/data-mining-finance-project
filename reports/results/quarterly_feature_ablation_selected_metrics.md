# Quarterly Feature Ablation Selected Models

## Metrics

| Regime | Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Usable Features |
|---|---|---:|---:|---:|---:|---:|
| levels_only | logistic_regression | 0.5060 | 0.0155 | 0.4847 | 0.4006 | 50 |
| levels_plus_deltas | logistic_regression | 0.5053 | 0.0453 | 0.4522 | 0.4619 | 93 |
| levels_plus_deltas_plus_cross_sectional | logistic_regression | 0.5047 | 0.0443 | 0.4520 | 0.4647 | 96 |

## Usable Feature Counts By Family

| Regime | Model | Family | Usable Feature Count |
|---|---|---|---:|
| levels_only | logistic_regression | base_or_other | 43 |
| levels_only | logistic_regression | level | 7 |
| levels_plus_deltas | logistic_regression | base_or_other | 43 |
| levels_plus_deltas | logistic_regression | delta | 5 |
| levels_plus_deltas | logistic_regression | legacy_delta | 38 |
| levels_plus_deltas | logistic_regression | level | 7 |
| levels_plus_deltas_plus_cross_sectional | logistic_regression | base_or_other | 43 |
| levels_plus_deltas_plus_cross_sectional | logistic_regression | cross_sectional | 3 |
| levels_plus_deltas_plus_cross_sectional | logistic_regression | delta | 5 |
| levels_plus_deltas_plus_cross_sectional | logistic_regression | legacy_delta | 38 |
| levels_plus_deltas_plus_cross_sectional | logistic_regression | level | 7 |
