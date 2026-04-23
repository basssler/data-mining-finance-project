# Quarterly Pruning Checkpoint

| Regime | Model | Feature Count | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Delta vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| unpruned_logistic | logistic_regression | 103 | 0.5044 | 0.0230 | 0.4695 | 0.4447 | 0.0000 |
| unpruned_elastic_net | elastic_net_logistic | 103 | 0.5125 | 0.0163 | 0.4926 | 0.4399 | -0.0048 |
| correlation_pruned_logistic | logistic_regression | 74 | 0.4957 | 0.0283 | 0.4477 | 0.4346 | -0.0100 |
| correlation_pruned_elastic_net | elastic_net_logistic | 74 | 0.5057 | 0.0160 | 0.4791 | 0.4254 | -0.0192 |

## Readout

- Best pruning regime: `unpruned_logistic` with holdout AUC `0.4447`.
