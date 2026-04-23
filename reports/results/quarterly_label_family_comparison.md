# Quarterly Label Family Comparison

## Selected Models

| Label Family | Selected Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Rows | Class 1 Rate | Dropped Ambiguous |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 21d_excess_quantile | logistic_regression | 0.6048 | 0.1653 | 0.4074 | 0.5500 | 9 | 0.5500 | 939 |
| 21d_excess_thresholded | random_forest | 0.5126 | 0.0346 | 0.4652 | 0.5291 | 66 | 0.5176 | 303 |
| 21d_excess_sign | xgboost | 0.5002 | 0.0240 | 0.4757 | 0.5084 | 82 | 0.5150 | 0 |
| 10d_excess_thresholded | logistic_regression | 0.5229 | 0.0609 | 0.4667 | 0.4958 | 62 | 0.4916 | 455 |
| 10d_excess_sign | logistic_regression | 0.5609 | 0.0435 | 0.4816 | 0.4681 | 86 | 0.4873 | 0 |

## Readout

- Best label family by holdout/CV ordering: `21d_excess_quantile` with selected model `logistic_regression` and holdout AUC `0.5500`.
