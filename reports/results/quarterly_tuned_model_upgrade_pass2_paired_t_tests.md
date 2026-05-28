# Paired t-tests: quarterly tuned model upgrade pass 2

Input: `outputs\quarterly\validation\quarterly_tuned_model_upgrade_pass2_v1\fold_summary.csv`

Tests compare matched validation-fold scores across the same five purged time-series folds. Positive `mean_diff_a_minus_b` means `model_a` scored higher than `model_b`; for `log_loss`, lower is better despite the sign convention.

Because n=5 folds, treat p-values as directional evidence, not a strong standalone promotion rule. No multiple-comparison correction is applied.

## auc

| model_a | model_b | n | mean_a | mean_b | diff | t | p | 95% CI diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | random_forest | 5 | 0.5436 | 0.5162 | 0.0274 | 2.054 | 0.1092 | [-0.0096, 0.0643] |
| logistic_regression | random_forest | 5 | 0.5726 | 0.5162 | 0.0563 | 1.800 | 0.1463 | [-0.0306, 0.1433] |
| random_forest | xgboost | 5 | 0.5162 | 0.5471 | -0.0309 | -1.659 | 0.1724 | [-0.0827, 0.0208] |
| catboost | hist_gradient_boosting | 5 | 0.5436 | 0.5222 | 0.0214 | 1.551 | 0.1959 | [-0.0169, 0.0597] |
| hist_gradient_boosting | logistic_regression | 5 | 0.5222 | 0.5726 | -0.0504 | -1.387 | 0.2377 | [-0.1512, 0.0505] |
| catboost | logistic_regression | 5 | 0.5436 | 0.5726 | -0.0290 | -1.160 | 0.3104 | [-0.0983, 0.0404] |
| hist_gradient_boosting | xgboost | 5 | 0.5222 | 0.5471 | -0.0249 | -0.933 | 0.4036 | [-0.0992, 0.0493] |
| logistic_regression | xgboost | 5 | 0.5726 | 0.5471 | 0.0254 | 0.842 | 0.4470 | [-0.0584, 0.1092] |
| hist_gradient_boosting | random_forest | 5 | 0.5222 | 0.5162 | 0.0060 | 0.276 | 0.7965 | [-0.0543, 0.0662] |
| catboost | xgboost | 5 | 0.5436 | 0.5471 | -0.0036 | -0.174 | 0.8704 | [-0.0604, 0.0533] |

## f1

| model_a | model_b | n | mean_a | mean_b | diff | t | p | 95% CI diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | random_forest | 5 | 0.5555 | 0.5135 | 0.0420 | 3.037 | 0.0385 | [0.0036, 0.0804] |
| random_forest | xgboost | 5 | 0.5135 | 0.5586 | -0.0451 | -2.783 | 0.0497 | [-0.0901, -0.0001] |
| hist_gradient_boosting | random_forest | 5 | 0.5476 | 0.5135 | 0.0341 | 2.232 | 0.0894 | [-0.0083, 0.0764] |
| catboost | random_forest | 5 | 0.5722 | 0.5135 | 0.0587 | 2.005 | 0.1154 | [-0.0226, 0.1399] |
| catboost | hist_gradient_boosting | 5 | 0.5722 | 0.5476 | 0.0246 | 1.715 | 0.1614 | [-0.0152, 0.0644] |
| hist_gradient_boosting | xgboost | 5 | 0.5476 | 0.5586 | -0.0111 | -0.882 | 0.4278 | [-0.0459, 0.0238] |
| catboost | xgboost | 5 | 0.5722 | 0.5586 | 0.0135 | 0.689 | 0.5290 | [-0.0410, 0.0681] |
| catboost | logistic_regression | 5 | 0.5722 | 0.5555 | 0.0167 | 0.584 | 0.5906 | [-0.0625, 0.0958] |
| hist_gradient_boosting | logistic_regression | 5 | 0.5476 | 0.5555 | -0.0079 | -0.434 | 0.6868 | [-0.0588, 0.0429] |
| logistic_regression | xgboost | 5 | 0.5555 | 0.5586 | -0.0031 | -0.223 | 0.8345 | [-0.0421, 0.0358] |

## log_loss

| model_a | model_b | n | mean_a | mean_b | diff | t | p | 95% CI diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | hist_gradient_boosting | 5 | 0.8242 | 0.9788 | -0.1546 | -6.323 | 0.0032 | [-0.2224, -0.0867] |
| hist_gradient_boosting | random_forest | 5 | 0.9788 | 0.7008 | 0.2779 | 4.571 | 0.0103 | [0.1091, 0.4467] |
| hist_gradient_boosting | xgboost | 5 | 0.9788 | 0.7543 | 0.2244 | 4.431 | 0.0114 | [0.0838, 0.3650] |
| logistic_regression | random_forest | 5 | 1.1044 | 0.7008 | 0.4035 | 3.891 | 0.0177 | [0.1156, 0.6914] |
| logistic_regression | xgboost | 5 | 1.1044 | 0.7543 | 0.3500 | 3.828 | 0.0187 | [0.0961, 0.6039] |
| catboost | logistic_regression | 5 | 0.8242 | 1.1044 | -0.2802 | -3.726 | 0.0204 | [-0.4889, -0.0714] |
| random_forest | xgboost | 5 | 0.7008 | 0.7543 | -0.0535 | -3.015 | 0.0394 | [-0.1028, -0.0042] |
| catboost | random_forest | 5 | 0.8242 | 0.7008 | 0.1234 | 2.612 | 0.0593 | [-0.0078, 0.2545] |
| catboost | xgboost | 5 | 0.8242 | 0.7543 | 0.0699 | 2.043 | 0.1106 | [-0.0251, 0.1648] |
| hist_gradient_boosting | logistic_regression | 5 | 0.9788 | 1.1044 | -0.1256 | -1.340 | 0.2513 | [-0.3858, 0.1346] |

## Bottom line

Uncorrected alpha=0.05 significant comparisons:
- log_loss: catboost vs hist_gradient_boosting, diff=-0.1546, p=0.0032
- log_loss: hist_gradient_boosting vs random_forest, diff=0.2779, p=0.0103
- log_loss: hist_gradient_boosting vs xgboost, diff=0.2244, p=0.0114
- log_loss: logistic_regression vs random_forest, diff=0.4035, p=0.0177
- log_loss: logistic_regression vs xgboost, diff=0.3500, p=0.0187
- log_loss: catboost vs logistic_regression, diff=-0.2802, p=0.0204
- f1: logistic_regression vs random_forest, diff=0.0420, p=0.0385
- log_loss: random_forest vs xgboost, diff=-0.0535, p=0.0394
- f1: random_forest vs xgboost, diff=-0.0451, p=0.0497
