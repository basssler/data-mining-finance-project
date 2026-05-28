# Same-model paired t-tests across cumulative layers

This evaluates the same model across the same five validation folds for three cumulative layer comparisons:

- `L1_vs_L1_L2`: Layer 1 vs Layer 1+2
- `L1_L2_vs_L1_L2_L3`: Layer 1+2 vs Layer 1+2+3
- `L1_vs_L1_L2_L3`: Layer 1 vs Layer 1+2+3

Layer mapping used:

| Layer | Artifact | Meaning |
|---|---|---|
| L1 | `outputs/quarterly/validation/capitaliq_ladder_core/fold_summary.csv` | Layer 1: fundamentals/core |
| L1_L2 | `outputs/quarterly/validation/capitaliq_ladder_core_plus_market/fold_summary.csv` | Layer 1+2: core + market |
| L1_L2_L3 | `outputs/quarterly/validation/capitaliq_ladder_sector_adjusted_sentiment/fold_summary.csv` | Layer 1+2+3: core + market + sector-adjusted sentiment |

Positive `mean_diff_right_minus_left` is better for AUC and F1; negative is better for log loss. P-values are two-sided and uncorrected. Each test uses `n=5` paired folds.

## Layer means by model

| Model | Layer | CV AUC | CV F1 | CV log loss |
|---|---|---:|---:|---:|
| logistic_regression | L1 | 0.4998 | 0.4738 | 0.7245 |
| logistic_regression | L1_L2 | 0.4941 | 0.4851 | 0.7759 |
| logistic_regression | L1_L2_L3 | 0.4989 | 0.4960 | 0.8438 |
| random_forest | L1 | 0.4866 | 0.4541 | 0.7032 |
| random_forest | L1_L2 | 0.5271 | 0.4938 | 0.6979 |
| random_forest | L1_L2_L3 | 0.5052 | 0.4967 | 0.7009 |
| xgboost | L1 | 0.4878 | 0.4724 | 0.8179 |
| xgboost | L1_L2 | 0.5304 | 0.5244 | 0.8119 |
| xgboost | L1_L2_L3 | 0.5018 | 0.4922 | 0.8361 |

## Paired tests

### logistic_regression

| Comparison | Metric | left mean | right mean | diff | t | p | 95% CI diff |
|---|---|---:|---:|---:|---:|---:|---:|
| L1_vs_L1_L2 | auc | 0.4998 | 0.4941 | -0.0057 | -0.396 | 0.7120 | [-0.0454, 0.0340] |
| L1_vs_L1_L2 | f1 | 0.4738 | 0.4851 | 0.0113 | 0.354 | 0.7412 | [-0.0773, 0.1000] |
| L1_vs_L1_L2 | log_loss | 0.7245 | 0.7759 | 0.0513 | 1.532 | 0.2002 | [-0.0417, 0.1443] |
| L1_L2_vs_L1_L2_L3 | auc | 0.4941 | 0.4989 | 0.0047 | 0.219 | 0.8371 | [-0.0551, 0.0645] |
| L1_L2_vs_L1_L2_L3 | f1 | 0.4851 | 0.4960 | 0.0109 | 0.641 | 0.5561 | [-0.0364, 0.0582] |
| L1_L2_vs_L1_L2_L3 | log_loss | 0.7759 | 0.8438 | 0.0679 | 2.066 | 0.1077 | [-0.0233, 0.1592] |
| L1_vs_L1_L2_L3 | auc | 0.4998 | 0.4989 | -0.0009 | -0.033 | 0.9752 | [-0.0802, 0.0783] |
| L1_vs_L1_L2_L3 | f1 | 0.4738 | 0.4960 | 0.0222 | 0.645 | 0.5540 | [-0.0735, 0.1179] |
| L1_vs_L1_L2_L3 | log_loss | 0.7245 | 0.8438 | 0.1192 | 2.384 | 0.0757 | [-0.0196, 0.2581] |

### random_forest

| Comparison | Metric | left mean | right mean | diff | t | p | 95% CI diff |
|---|---|---:|---:|---:|---:|---:|---:|
| L1_vs_L1_L2 | auc | 0.4866 | 0.5271 | 0.0405 | 2.431 | 0.0719 | [-0.0058, 0.0868] |
| L1_vs_L1_L2 | f1 | 0.4541 | 0.4938 | 0.0397 | 2.780 | 0.0498 | [0.0000, 0.0794] |
| L1_vs_L1_L2 | log_loss | 0.7032 | 0.6979 | -0.0053 | -1.346 | 0.2494 | [-0.0161, 0.0056] |
| L1_L2_vs_L1_L2_L3 | auc | 0.5271 | 0.5052 | -0.0219 | -1.908 | 0.1291 | [-0.0538, 0.0100] |
| L1_L2_vs_L1_L2_L3 | f1 | 0.4938 | 0.4967 | 0.0029 | 0.185 | 0.8622 | [-0.0411, 0.0469] |
| L1_L2_vs_L1_L2_L3 | log_loss | 0.6979 | 0.7009 | 0.0030 | 1.168 | 0.3078 | [-0.0041, 0.0100] |
| L1_vs_L1_L2_L3 | auc | 0.4866 | 0.5052 | 0.0186 | 0.768 | 0.4852 | [-0.0487, 0.0860] |
| L1_vs_L1_L2_L3 | f1 | 0.4541 | 0.4967 | 0.0427 | 2.178 | 0.0950 | [-0.0117, 0.0971] |
| L1_vs_L1_L2_L3 | log_loss | 0.7032 | 0.7009 | -0.0023 | -0.516 | 0.6332 | [-0.0147, 0.0101] |

### xgboost

| Comparison | Metric | left mean | right mean | diff | t | p | 95% CI diff |
|---|---|---:|---:|---:|---:|---:|---:|
| L1_vs_L1_L2 | auc | 0.4878 | 0.5304 | 0.0426 | 1.662 | 0.1718 | [-0.0285, 0.1137] |
| L1_vs_L1_L2 | f1 | 0.4724 | 0.5244 | 0.0520 | 1.430 | 0.2260 | [-0.0490, 0.1531] |
| L1_vs_L1_L2 | log_loss | 0.8179 | 0.8119 | -0.0060 | -0.341 | 0.7505 | [-0.0545, 0.0426] |
| L1_L2_vs_L1_L2_L3 | auc | 0.5304 | 0.5018 | -0.0286 | -1.689 | 0.1665 | [-0.0755, 0.0184] |
| L1_L2_vs_L1_L2_L3 | f1 | 0.5244 | 0.4922 | -0.0322 | -2.411 | 0.0735 | [-0.0692, 0.0049] |
| L1_L2_vs_L1_L2_L3 | log_loss | 0.8119 | 0.8361 | 0.0242 | 1.741 | 0.1567 | [-0.0144, 0.0627] |
| L1_vs_L1_L2_L3 | auc | 0.4878 | 0.5018 | 0.0140 | 0.369 | 0.7310 | [-0.0914, 0.1194] |
| L1_vs_L1_L2_L3 | f1 | 0.4724 | 0.4922 | 0.0199 | 0.445 | 0.6791 | [-0.1040, 0.1437] |
| L1_vs_L1_L2_L3 | log_loss | 0.8179 | 0.8361 | 0.0182 | 0.680 | 0.5336 | [-0.0561, 0.0925] |

## Bottom line

Uncorrected alpha=0.05 significant same-model layer comparisons:
- `random_forest` `L1_vs_L1_L2` on `f1`: right layer higher by 0.0397 (better), p=0.0498.

Fold-level paired values are in `reports\results\capitaliq_layer_same_model_fold_pairs.csv`.
