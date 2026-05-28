# Paired t-tests: 21d layers by model

Target label: `event_v2_21d_excess_threshold`.

Each row compares the same model across two layers on matched validation folds. For example, `catboost` in Layer 11 vs `catboost` in Layer 12 is a model-preserving layer comparison.

Positive `mean_diff_b_minus_a` means Layer B is higher than Layer A. Higher is better for AUC/F1; lower is better for log loss. P-values are two-sided and uncorrected.

## Layer coverage

| Layer | Panel name | Models present |
|---|---|---|
| L1: Final core confirmation | `quarterly_final_core_confirmation_v1` | logistic_regression, random_forest, xgboost |
| L2: Phase 8 core, no market | `quarterly_phase8_core_no_market` | logistic_regression, random_forest, xgboost |
| L3: Phase 8 generic market only | `quarterly_phase8_generic_market_only` | logistic_regression, random_forest, xgboost |
| L4: Phase 8 event-aware market only | `quarterly_phase8_event_aware_market_only` | logistic_regression, random_forest, xgboost |
| L5: Phase 8 generic + event-aware market | `quarterly_phase8_generic_and_event_aware_market` | logistic_regression, random_forest, xgboost |
| L6: Phase 9 core, no sentiment | `quarterly_phase9_core_no_sentiment` | logistic_regression, random_forest, xgboost |
| L7: Phase 9 broad filing sentiment only | `quarterly_phase9_broad_filing_sentiment_only` | logistic_regression, random_forest, xgboost |
| L8: Phase 9 event-specific sentiment only | `quarterly_phase9_event_specific_sentiment_only` | logistic_regression, random_forest, xgboost |
| L9: Phase 9 combined sentiment block | `quarterly_phase9_combined_sentiment_block` | logistic_regression, random_forest, xgboost |
| L10: Phase 9 event-specific sentiment champion | `quarterly_phase9_event_specific_sentiment_champion_v1` | logistic_regression, random_forest, xgboost |
| L11: Tuned model upgrade v1 | `quarterly_tuned_model_upgrade_v1` | catboost, hist_gradient_boosting, lightgbm, logistic_regression, random_forest, xgboost |
| L12: Tuned model upgrade pass 2 | `quarterly_tuned_model_upgrade_pass2_v1` | catboost, hist_gradient_boosting, logistic_regression, random_forest, xgboost |

## Adjacent Layer Comparisons

| Panel A | Panel B | Model | Metric | Layer A | Layer B | n pairs | mean A | mean B | mean diff B-A | t | p | 95% CI diff | Note |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | logistic_regression | auc | L1 | L2 | 5 | 0.5685 | 0.5685 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | logistic_regression | f1 | L1 | L2 | 5 | 0.5373 | 0.5373 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | logistic_regression | log_loss | L1 | L2 | 5 | 1.0274 | 1.0274 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | random_forest | auc | L1 | L2 | 5 | 0.5126 | 0.5126 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | random_forest | f1 | L1 | L2 | 5 | 0.5066 | 0.5066 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | random_forest | log_loss | L1 | L2 | 5 | 0.7009 | 0.7009 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | xgboost | auc | L1 | L2 | 5 | 0.5123 | 0.5123 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | xgboost | f1 | L1 | L2 | 5 | 0.5333 | 0.5333 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_final_core_confirmation_v1` | `quarterly_phase8_core_no_market` | xgboost | log_loss | L1 | L2 | 5 | 0.8477 | 0.8477 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | logistic_regression | auc | L2 | L3 | 5 | 0.5685 | 0.5653 | -0.0032 | -0.137 | 0.8978 | [-0.0682, 0.0618] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | logistic_regression | f1 | L2 | L3 | 5 | 0.5373 | 0.5485 | 0.0113 | 0.438 | 0.6841 | [-0.0601, 0.0827] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | logistic_regression | log_loss | L2 | L3 | 5 | 1.0274 | 1.1751 | 0.1478 | 3.177 | 0.0336 | [0.0186, 0.2769] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | random_forest | auc | L2 | L3 | 5 | 0.5126 | 0.5124 | -0.0003 | -0.018 | 0.9862 | [-0.0434, 0.0428] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | random_forest | f1 | L2 | L3 | 5 | 0.5066 | 0.5251 | 0.0186 | 0.682 | 0.5329 | [-0.0570, 0.0942] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | random_forest | log_loss | L2 | L3 | 5 | 0.7009 | 0.6999 | -0.0010 | -0.545 | 0.6147 | [-0.0061, 0.0041] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | xgboost | auc | L2 | L3 | 5 | 0.5123 | 0.5202 | 0.0079 | 0.573 | 0.5970 | [-0.0303, 0.0461] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | xgboost | f1 | L2 | L3 | 5 | 0.5333 | 0.5494 | 0.0161 | 1.460 | 0.2182 | [-0.0145, 0.0468] |  |
| `quarterly_phase8_core_no_market` | `quarterly_phase8_generic_market_only` | xgboost | log_loss | L2 | L3 | 5 | 0.8477 | 0.8782 | 0.0304 | 1.173 | 0.3058 | [-0.0416, 0.1025] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | logistic_regression | auc | L3 | L4 | 5 | 0.5653 | 0.5780 | 0.0128 | 0.944 | 0.3987 | [-0.0248, 0.0503] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | logistic_regression | f1 | L3 | L4 | 5 | 0.5485 | 0.5739 | 0.0254 | 1.011 | 0.3694 | [-0.0443, 0.0951] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | logistic_regression | log_loss | L3 | L4 | 5 | 1.1751 | 1.1220 | -0.0531 | -1.212 | 0.2922 | [-0.1749, 0.0686] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | random_forest | auc | L3 | L4 | 5 | 0.5124 | 0.5093 | -0.0031 | -0.303 | 0.7771 | [-0.0313, 0.0251] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | random_forest | f1 | L3 | L4 | 5 | 0.5251 | 0.5023 | -0.0228 | -0.716 | 0.5136 | [-0.1112, 0.0656] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | random_forest | log_loss | L3 | L4 | 5 | 0.6999 | 0.7013 | 0.0014 | 1.157 | 0.3115 | [-0.0019, 0.0046] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | xgboost | auc | L3 | L4 | 5 | 0.5202 | 0.4870 | -0.0332 | -2.191 | 0.0935 | [-0.0752, 0.0089] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | xgboost | f1 | L3 | L4 | 5 | 0.5494 | 0.5242 | -0.0252 | -0.667 | 0.5414 | [-0.1302, 0.0798] |  |
| `quarterly_phase8_generic_market_only` | `quarterly_phase8_event_aware_market_only` | xgboost | log_loss | L3 | L4 | 5 | 0.8782 | 0.9364 | 0.0582 | 3.043 | 0.0383 | [0.0051, 0.1114] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | logistic_regression | auc | L4 | L5 | 5 | 0.5780 | 0.5530 | -0.0251 | -1.551 | 0.1959 | [-0.0699, 0.0198] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | logistic_regression | f1 | L4 | L5 | 5 | 0.5739 | 0.5425 | -0.0315 | -2.467 | 0.0692 | [-0.0668, 0.0039] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | logistic_regression | log_loss | L4 | L5 | 5 | 1.1220 | 1.2593 | 0.1373 | 5.858 | 0.0042 | [0.0722, 0.2024] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | random_forest | auc | L4 | L5 | 5 | 0.5093 | 0.5053 | -0.0040 | -0.896 | 0.4207 | [-0.0163, 0.0083] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | random_forest | f1 | L4 | L5 | 5 | 0.5023 | 0.4970 | -0.0053 | -0.159 | 0.8812 | [-0.0980, 0.0874] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | random_forest | log_loss | L4 | L5 | 5 | 0.7013 | 0.6994 | -0.0019 | -1.397 | 0.2349 | [-0.0057, 0.0019] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | xgboost | auc | L4 | L5 | 5 | 0.4870 | 0.5045 | 0.0175 | 2.500 | 0.0668 | [-0.0019, 0.0369] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | xgboost | f1 | L4 | L5 | 5 | 0.5242 | 0.5332 | 0.0090 | 0.284 | 0.7908 | [-0.0792, 0.0972] |  |
| `quarterly_phase8_event_aware_market_only` | `quarterly_phase8_generic_and_event_aware_market` | xgboost | log_loss | L4 | L5 | 5 | 0.9364 | 0.9103 | -0.0261 | -1.596 | 0.1858 | [-0.0716, 0.0193] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | logistic_regression | auc | L5 | L6 | 5 | 0.5530 | 0.5685 | 0.0155 | 0.646 | 0.5534 | [-0.0511, 0.0821] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | logistic_regression | f1 | L5 | L6 | 5 | 0.5425 | 0.5373 | -0.0052 | -0.372 | 0.7285 | [-0.0438, 0.0335] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | logistic_regression | log_loss | L5 | L6 | 5 | 1.2593 | 1.0274 | -0.2319 | -8.557 | 0.0010 | [-0.3072, -0.1567] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | random_forest | auc | L5 | L6 | 5 | 0.5053 | 0.5126 | 0.0073 | 0.377 | 0.7250 | [-0.0466, 0.0613] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | random_forest | f1 | L5 | L6 | 5 | 0.4970 | 0.5066 | 0.0096 | 0.198 | 0.8529 | [-0.1247, 0.1438] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | random_forest | log_loss | L5 | L6 | 5 | 0.6994 | 0.7009 | 0.0015 | 0.585 | 0.5898 | [-0.0058, 0.0089] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | xgboost | auc | L5 | L6 | 5 | 0.5045 | 0.5123 | 0.0078 | 0.522 | 0.6291 | [-0.0336, 0.0492] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | xgboost | f1 | L5 | L6 | 5 | 0.5332 | 0.5333 | 0.0001 | 0.004 | 0.9973 | [-0.0758, 0.0760] |  |
| `quarterly_phase8_generic_and_event_aware_market` | `quarterly_phase9_core_no_sentiment` | xgboost | log_loss | L5 | L6 | 5 | 0.9103 | 0.8477 | -0.0625 | -2.110 | 0.1025 | [-0.1448, 0.0198] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | logistic_regression | auc | L6 | L7 | 5 | 0.5685 | 0.5605 | -0.0080 | -1.106 | 0.3307 | [-0.0280, 0.0120] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | logistic_regression | f1 | L6 | L7 | 5 | 0.5373 | 0.5356 | -0.0017 | -0.116 | 0.9132 | [-0.0415, 0.0382] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | logistic_regression | log_loss | L6 | L7 | 5 | 1.0274 | 1.0880 | 0.0606 | 2.823 | 0.0477 | [0.0010, 0.1202] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | random_forest | auc | L6 | L7 | 5 | 0.5126 | 0.5162 | 0.0036 | 0.309 | 0.7724 | [-0.0287, 0.0359] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | random_forest | f1 | L6 | L7 | 5 | 0.5066 | 0.5094 | 0.0029 | 0.426 | 0.6920 | [-0.0159, 0.0216] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | random_forest | log_loss | L6 | L7 | 5 | 0.7009 | 0.7000 | -0.0009 | -0.311 | 0.7710 | [-0.0089, 0.0071] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | xgboost | auc | L6 | L7 | 5 | 0.5123 | 0.5156 | 0.0033 | 0.364 | 0.7340 | [-0.0220, 0.0287] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | xgboost | f1 | L6 | L7 | 5 | 0.5333 | 0.5412 | 0.0079 | 0.340 | 0.7512 | [-0.0565, 0.0722] |  |
| `quarterly_phase9_core_no_sentiment` | `quarterly_phase9_broad_filing_sentiment_only` | xgboost | log_loss | L6 | L7 | 5 | 0.8477 | 0.8529 | 0.0051 | 0.219 | 0.8372 | [-0.0598, 0.0700] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | logistic_regression | auc | L7 | L8 | 5 | 0.5605 | 0.5682 | 0.0078 | 1.256 | 0.2773 | [-0.0094, 0.0249] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | logistic_regression | f1 | L7 | L8 | 5 | 0.5356 | 0.5729 | 0.0372 | 1.336 | 0.2524 | [-0.0401, 0.1146] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | logistic_regression | log_loss | L7 | L8 | 5 | 1.0880 | 1.0653 | -0.0227 | -1.598 | 0.1854 | [-0.0622, 0.0168] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | random_forest | auc | L7 | L8 | 5 | 0.5162 | 0.5210 | 0.0048 | 0.994 | 0.3765 | [-0.0086, 0.0181] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | random_forest | f1 | L7 | L8 | 5 | 0.5094 | 0.5002 | -0.0093 | -0.652 | 0.5501 | [-0.0487, 0.0302] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | random_forest | log_loss | L7 | L8 | 5 | 0.7000 | 0.7001 | 0.0000 | 0.067 | 0.9501 | [-0.0012, 0.0012] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | xgboost | auc | L7 | L8 | 5 | 0.5156 | 0.5186 | 0.0030 | 0.331 | 0.7570 | [-0.0218, 0.0277] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | xgboost | f1 | L7 | L8 | 5 | 0.5412 | 0.5367 | -0.0045 | -0.699 | 0.5233 | [-0.0224, 0.0134] |  |
| `quarterly_phase9_broad_filing_sentiment_only` | `quarterly_phase9_event_specific_sentiment_only` | xgboost | log_loss | L7 | L8 | 5 | 0.8529 | 0.8571 | 0.0042 | 0.214 | 0.8413 | [-0.0502, 0.0586] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | logistic_regression | auc | L8 | L9 | 5 | 0.5682 | 0.5747 | 0.0065 | 2.640 | 0.0576 | [-0.0003, 0.0133] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | logistic_regression | f1 | L8 | L9 | 5 | 0.5729 | 0.5778 | 0.0050 | 0.753 | 0.4933 | [-0.0133, 0.0232] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | logistic_regression | log_loss | L8 | L9 | 5 | 1.0653 | 1.0624 | -0.0029 | -0.631 | 0.5620 | [-0.0156, 0.0098] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | random_forest | auc | L8 | L9 | 5 | 0.5210 | 0.5015 | -0.0195 | -2.924 | 0.0431 | [-0.0380, -0.0010] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | random_forest | f1 | L8 | L9 | 5 | 0.5002 | 0.4911 | -0.0090 | -0.593 | 0.5851 | [-0.0513, 0.0333] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | random_forest | log_loss | L8 | L9 | 5 | 0.7001 | 0.7043 | 0.0043 | 2.016 | 0.1140 | [-0.0016, 0.0102] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | xgboost | auc | L8 | L9 | 5 | 0.5186 | 0.5165 | -0.0021 | -0.108 | 0.9194 | [-0.0553, 0.0511] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | xgboost | f1 | L8 | L9 | 5 | 0.5367 | 0.5677 | 0.0310 | 1.510 | 0.2055 | [-0.0260, 0.0881] |  |
| `quarterly_phase9_event_specific_sentiment_only` | `quarterly_phase9_combined_sentiment_block` | xgboost | log_loss | L8 | L9 | 5 | 0.8571 | 0.8672 | 0.0101 | 0.464 | 0.6666 | [-0.0505, 0.0708] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | logistic_regression | auc | L9 | L10 | 5 | 0.5747 | 0.5682 | -0.0065 | -2.640 | 0.0576 | [-0.0133, 0.0003] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | logistic_regression | f1 | L9 | L10 | 5 | 0.5778 | 0.5729 | -0.0050 | -0.753 | 0.4933 | [-0.0232, 0.0133] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | logistic_regression | log_loss | L9 | L10 | 5 | 1.0624 | 1.0653 | 0.0029 | 0.631 | 0.5620 | [-0.0098, 0.0156] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | random_forest | auc | L9 | L10 | 5 | 0.5015 | 0.5210 | 0.0195 | 2.924 | 0.0431 | [0.0010, 0.0380] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | random_forest | f1 | L9 | L10 | 5 | 0.4911 | 0.5002 | 0.0090 | 0.593 | 0.5851 | [-0.0333, 0.0513] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | random_forest | log_loss | L9 | L10 | 5 | 0.7043 | 0.7001 | -0.0043 | -2.016 | 0.1140 | [-0.0102, 0.0016] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | xgboost | auc | L9 | L10 | 5 | 0.5165 | 0.5186 | 0.0021 | 0.108 | 0.9194 | [-0.0511, 0.0553] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | xgboost | f1 | L9 | L10 | 5 | 0.5677 | 0.5367 | -0.0310 | -1.510 | 0.2055 | [-0.0881, 0.0260] |  |
| `quarterly_phase9_combined_sentiment_block` | `quarterly_phase9_event_specific_sentiment_champion_v1` | xgboost | log_loss | L9 | L10 | 5 | 0.8672 | 0.8571 | -0.0101 | -0.464 | 0.6666 | [-0.0708, 0.0505] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | logistic_regression | auc | L10 | L11 | 5 | 0.5682 | 0.5726 | 0.0043 | 0.225 | 0.8332 | [-0.0490, 0.0576] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | logistic_regression | f1 | L10 | L11 | 5 | 0.5729 | 0.5555 | -0.0173 | -0.894 | 0.4221 | [-0.0712, 0.0365] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | logistic_regression | log_loss | L10 | L11 | 5 | 1.0653 | 1.1044 | 0.0391 | 0.724 | 0.5092 | [-0.1109, 0.1891] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | random_forest | auc | L10 | L11 | 5 | 0.5210 | 0.5162 | -0.0048 | -0.799 | 0.4690 | [-0.0215, 0.0119] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | random_forest | f1 | L10 | L11 | 5 | 0.5002 | 0.5135 | 0.0133 | 1.413 | 0.2305 | [-0.0129, 0.0396] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | random_forest | log_loss | L10 | L11 | 5 | 0.7001 | 0.7008 | 0.0008 | 0.882 | 0.4278 | [-0.0017, 0.0032] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | xgboost | auc | L10 | L11 | 5 | 0.5186 | 0.5426 | 0.0240 | 1.187 | 0.3011 | [-0.0321, 0.0801] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | xgboost | f1 | L10 | L11 | 5 | 0.5367 | 0.5504 | 0.0137 | 0.352 | 0.7428 | [-0.0943, 0.1216] |  |
| `quarterly_phase9_event_specific_sentiment_champion_v1` | `quarterly_tuned_model_upgrade_v1` | xgboost | log_loss | L10 | L11 | 5 | 0.8571 | 0.9760 | 0.1189 | 5.119 | 0.0069 | [0.0544, 0.1834] |  |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | catboost | auc | L11 | L12 | 5 | 0.5598 | 0.5436 | -0.0163 | -0.950 | 0.3960 | [-0.0638, 0.0313] |  |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | catboost | f1 | L11 | L12 | 5 | 0.5430 | 0.5722 | 0.0292 | 2.234 | 0.0892 | [-0.0071, 0.0655] |  |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | catboost | log_loss | L11 | L12 | 5 | 0.9278 | 0.8242 | -0.1036 | -2.438 | 0.0713 | [-0.2216, 0.0144] |  |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | hist_gradient_boosting | auc | L11 | L12 | 5 | 0.5222 | 0.5222 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | hist_gradient_boosting | f1 | L11 | L12 | 5 | 0.5476 | 0.5476 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | hist_gradient_boosting | log_loss | L11 | L12 | 5 | 0.9788 | 0.9788 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | logistic_regression | auc | L11 | L12 | 5 | 0.5726 | 0.5726 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | logistic_regression | f1 | L11 | L12 | 5 | 0.5555 | 0.5555 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | logistic_regression | log_loss | L11 | L12 | 5 | 1.1044 | 1.1044 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | random_forest | auc | L11 | L12 | 5 | 0.5162 | 0.5162 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | random_forest | f1 | L11 | L12 | 5 | 0.5135 | 0.5135 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | random_forest | log_loss | L11 | L12 | 5 | 0.7008 | 0.7008 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | xgboost | auc | L11 | L12 | 5 | 0.5426 | 0.5471 | 0.0046 | 0.300 | 0.7788 | [-0.0377, 0.0468] |  |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | xgboost | f1 | L11 | L12 | 5 | 0.5504 | 0.5586 | 0.0083 | 0.283 | 0.7911 | [-0.0730, 0.0896] |  |
| `quarterly_tuned_model_upgrade_v1` | `quarterly_tuned_model_upgrade_pass2_v1` | xgboost | log_loss | L11 | L12 | 5 | 0.9760 | 0.7543 | -0.2216 | -7.009 | 0.0022 | [-0.3094, -0.1338] |  |

## Bottom line

Uncorrected alpha=0.05 significant adjacent same-model layer comparisons:
- logistic_regression log_loss L5+L6 (Phase 8 generic + event-aware market -> Phase 9 core, no sentiment): diff=-0.2319, p=0.0010.
- xgboost log_loss L11+L12 (Tuned model upgrade v1 -> Tuned model upgrade pass 2): diff=-0.2216, p=0.0022.
- logistic_regression log_loss L4+L5 (Phase 8 event-aware market only -> Phase 8 generic + event-aware market): diff=0.1373, p=0.0042.
- xgboost log_loss L10+L11 (Phase 9 event-specific sentiment champion -> Tuned model upgrade v1): diff=0.1189, p=0.0069.
- logistic_regression log_loss L2+L3 (Phase 8 core, no market -> Phase 8 generic market only): diff=0.1478, p=0.0336.
- xgboost log_loss L3+L4 (Phase 8 generic market only -> Phase 8 event-aware market only): diff=0.0582, p=0.0383.
- random_forest auc L9+L10 (Phase 9 combined sentiment block -> Phase 9 event-specific sentiment champion): diff=0.0195, p=0.0431.
- random_forest auc L8+L9 (Phase 9 event-specific sentiment only -> Phase 9 combined sentiment block): diff=-0.0195, p=0.0431.
- logistic_regression log_loss L6+L7 (Phase 9 core, no sentiment -> Phase 9 broad filing sentiment only): diff=0.0606, p=0.0477.

Full all-pairs output: `reports\results\quarterly_21d_layer_by_model_paired_t_tests.csv`
Model coverage output: `reports\results\quarterly_21d_layer_by_model_coverage.csv`
