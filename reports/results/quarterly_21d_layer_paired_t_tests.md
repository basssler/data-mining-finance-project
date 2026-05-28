# Paired t-tests: 21d quarterly layer evaluation

Target label: `event_v2_21d_excess_threshold`.

Tests compare each layer's selected primary model on the same five validation folds. `mean_diff_right_minus_left` is the later/right layer minus the earlier/left layer. Positive is better for AUC and F1; negative is better for log loss.

With n=5 folds, these are directional diagnostics. P-values are two-sided and uncorrected for multiple comparisons.

## Included layers

| Order | Layer | Selected model | CV AUC | CV F1 | CV log loss |
|---:|---|---|---:|---:|---:|
| 1 | Final core confirmation | random_forest | 0.5126 | 0.5066 | 0.7009 |
| 2 | Phase 8 core, no market | random_forest | 0.5126 | 0.5066 | 0.7009 |
| 3 | Phase 8 generic market only | logistic_regression | 0.5653 | 0.5485 | 1.1751 |
| 4 | Phase 8 event-aware market only | logistic_regression | 0.5780 | 0.5739 | 1.1220 |
| 5 | Phase 8 generic + event-aware market | logistic_regression | 0.5530 | 0.5425 | 1.2593 |
| 6 | Phase 9 core, no sentiment | random_forest | 0.5126 | 0.5066 | 0.7009 |
| 7 | Phase 9 broad filing sentiment only | random_forest | 0.5162 | 0.5094 | 0.7000 |
| 8 | Phase 9 event-specific sentiment only | random_forest | 0.5210 | 0.5002 | 0.7001 |
| 9 | Phase 9 combined sentiment block | random_forest | 0.5015 | 0.4911 | 0.7043 |
| 10 | Phase 9 event-specific sentiment champion | random_forest | 0.5210 | 0.5002 | 0.7001 |

## Adjacent layer tests

| Comparison | Metric | left mean | right mean | diff | t | p | 95% CI diff | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Final core confirmation -> Phase 8 core, no market | auc | 0.5126 | 0.5126 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| Final core confirmation -> Phase 8 core, no market | f1 | 0.5066 | 0.5066 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| Final core confirmation -> Phase 8 core, no market | log_loss | 0.7009 | 0.7009 | 0.0000 | NA | NA | [NA, NA] | identical fold scores |
| Phase 8 core, no market -> Phase 8 generic market only | auc | 0.5126 | 0.5653 | 0.0526 | 2.853 | 0.0463 | [0.0014, 0.1038] |  |
| Phase 8 core, no market -> Phase 8 generic market only | f1 | 0.5066 | 0.5485 | 0.0420 | 1.993 | 0.1171 | [-0.0165, 0.1005] |  |
| Phase 8 core, no market -> Phase 8 generic market only | log_loss | 0.7009 | 1.1751 | 0.4742 | 7.037 | 0.0021 | [0.2871, 0.6613] |  |
| Phase 8 generic market only -> Phase 8 event-aware market only | auc | 0.5653 | 0.5780 | 0.0128 | 0.944 | 0.3987 | [-0.0248, 0.0503] |  |
| Phase 8 generic market only -> Phase 8 event-aware market only | f1 | 0.5485 | 0.5739 | 0.0254 | 1.011 | 0.3694 | [-0.0443, 0.0951] |  |
| Phase 8 generic market only -> Phase 8 event-aware market only | log_loss | 1.1751 | 1.1220 | -0.0531 | -1.212 | 0.2922 | [-0.1749, 0.0686] |  |
| Phase 8 event-aware market only -> Phase 8 generic + event-aware market | auc | 0.5780 | 0.5530 | -0.0251 | -1.551 | 0.1959 | [-0.0699, 0.0198] |  |
| Phase 8 event-aware market only -> Phase 8 generic + event-aware market | f1 | 0.5739 | 0.5425 | -0.0315 | -2.467 | 0.0692 | [-0.0668, 0.0039] |  |
| Phase 8 event-aware market only -> Phase 8 generic + event-aware market | log_loss | 1.1220 | 1.2593 | 0.1373 | 5.858 | 0.0042 | [0.0722, 0.2024] |  |
| Phase 8 generic + event-aware market -> Phase 9 core, no sentiment | auc | 0.5530 | 0.5126 | -0.0403 | -2.120 | 0.1013 | [-0.0931, 0.0125] |  |
| Phase 8 generic + event-aware market -> Phase 9 core, no sentiment | f1 | 0.5425 | 0.5066 | -0.0359 | -3.447 | 0.0261 | [-0.0648, -0.0070] |  |
| Phase 8 generic + event-aware market -> Phase 9 core, no sentiment | log_loss | 1.2593 | 0.7009 | -0.5584 | -8.687 | 0.0010 | [-0.7368, -0.3799] |  |
| Phase 9 core, no sentiment -> Phase 9 broad filing sentiment only | auc | 0.5126 | 0.5162 | 0.0036 | 0.309 | 0.7724 | [-0.0287, 0.0359] |  |
| Phase 9 core, no sentiment -> Phase 9 broad filing sentiment only | f1 | 0.5066 | 0.5094 | 0.0029 | 0.426 | 0.6920 | [-0.0159, 0.0216] |  |
| Phase 9 core, no sentiment -> Phase 9 broad filing sentiment only | log_loss | 0.7009 | 0.7000 | -0.0009 | -0.311 | 0.7710 | [-0.0089, 0.0071] |  |
| Phase 9 broad filing sentiment only -> Phase 9 event-specific sentiment only | auc | 0.5162 | 0.5210 | 0.0048 | 0.994 | 0.3765 | [-0.0086, 0.0181] |  |
| Phase 9 broad filing sentiment only -> Phase 9 event-specific sentiment only | f1 | 0.5094 | 0.5002 | -0.0093 | -0.652 | 0.5501 | [-0.0487, 0.0302] |  |
| Phase 9 broad filing sentiment only -> Phase 9 event-specific sentiment only | log_loss | 0.7000 | 0.7001 | 0.0000 | 0.067 | 0.9501 | [-0.0012, 0.0012] |  |
| Phase 9 event-specific sentiment only -> Phase 9 combined sentiment block | auc | 0.5210 | 0.5015 | -0.0195 | -2.924 | 0.0431 | [-0.0380, -0.0010] |  |
| Phase 9 event-specific sentiment only -> Phase 9 combined sentiment block | f1 | 0.5002 | 0.4911 | -0.0090 | -0.593 | 0.5851 | [-0.0513, 0.0333] |  |
| Phase 9 event-specific sentiment only -> Phase 9 combined sentiment block | log_loss | 0.7001 | 0.7043 | 0.0043 | 2.016 | 0.1140 | [-0.0016, 0.0102] |  |
| Phase 9 combined sentiment block -> Phase 9 event-specific sentiment champion | auc | 0.5015 | 0.5210 | 0.0195 | 2.924 | 0.0431 | [0.0010, 0.0380] |  |
| Phase 9 combined sentiment block -> Phase 9 event-specific sentiment champion | f1 | 0.4911 | 0.5002 | 0.0090 | 0.593 | 0.5851 | [-0.0333, 0.0513] |  |
| Phase 9 combined sentiment block -> Phase 9 event-specific sentiment champion | log_loss | 0.7043 | 0.7001 | -0.0043 | -2.016 | 0.1140 | [-0.0102, 0.0016] |  |

## Bottom line

Uncorrected alpha=0.05 significant adjacent layer changes:
- Phase 8 generic + event-aware market -> Phase 9 core, no sentiment on log_loss: right layer lower by -0.5584, p=0.0010.
- Phase 8 core, no market -> Phase 8 generic market only on log_loss: right layer higher by 0.4742, p=0.0021.
- Phase 8 event-aware market only -> Phase 8 generic + event-aware market on log_loss: right layer higher by 0.1373, p=0.0042.
- Phase 8 generic + event-aware market -> Phase 9 core, no sentiment on f1: right layer lower by -0.0359, p=0.0261.
- Phase 9 combined sentiment block -> Phase 9 event-specific sentiment champion on auc: right layer higher by 0.0195, p=0.0431.
- Phase 9 event-specific sentiment only -> Phase 9 combined sentiment block on auc: right layer lower by -0.0195, p=0.0431.
- Phase 8 core, no market -> Phase 8 generic market only on auc: right layer higher by 0.0526, p=0.0463.

Full all-pair layer results are in `reports\results\quarterly_21d_layer_paired_t_tests.csv`.
