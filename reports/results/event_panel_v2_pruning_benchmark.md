# Event Panel V2 Pruning Benchmark

## Summary

- Locked baseline reference from primary benchmark: CV AUC `0.5415`, holdout AUC `0.5388`.
- Reproduced full_49 baseline in pruning lane: CV AUC `0.5415`, holdout AUC `0.5388`.
- Conservative rule: a regime is promotable only if it is non-worse on CV AUC, non-worse on CV log loss, and at least as good on holdout AUC or holdout log loss.

## Regime Comparison

| Regime | Group | Features | CV AUC | CV AUC Std | CV Log Loss | CV Log Loss Std | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | Promotion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| drop_beta_63d_to_sector | stress_test | 48 | 0.5234 | 0.0452 | 0.8141 | 0.0463 | 0.4891 | 0.7607 | 0.4706 | 0.4706 | -0.0462 | not_promotable |
| drop_cash_ratio | stress_test | 48 | 0.5328 | 0.0484 | 0.8038 | 0.0385 | 0.5139 | 0.7681 | 0.5432 | 0.6471 | 0.0383 | not_promotable |
| drop_log_volume | stress_test | 48 | 0.5348 | 0.0541 | 0.8112 | 0.0452 | 0.5023 | 0.7561 | 0.4714 | 0.4853 | 0.0262 | not_promotable |
| drop_sec_positive_prob | stress_test | 48 | 0.5236 | 0.0469 | 0.8151 | 0.0418 | 0.5134 | 0.7599 | 0.5263 | 0.5882 | 0.0032 | not_promotable |
| full_49_baseline | block_mix | 49 | 0.5415 | 0.0602 | 0.7950 | 0.0487 | 0.5388 | 0.7474 | 0.5616 | 0.6029 | 0.0902 | baseline_reference |
| no_availability_flags | hygiene | 47 | 0.5274 | 0.0467 | 0.8143 | 0.0418 | 0.5399 | 0.7370 | 0.5476 | 0.6765 | 0.0890 | interesting_not_promotable |
| price_volume_only | block_mix | 14 | 0.5414 | 0.0352 | 0.7983 | 0.0583 | 0.5367 | 0.7305 | 0.5250 | 0.6176 | 0.0482 | interesting_not_promotable |
| price_volume_plus_fundamentals | block_mix | 35 | 0.5396 | 0.0682 | 0.8083 | 0.0636 | 0.5058 | 0.7561 | 0.5072 | 0.5147 | 0.0182 | not_promotable |
| price_volume_plus_sentiment | block_mix | 24 | 0.5320 | 0.0233 | 0.8077 | 0.0327 | 0.5194 | 0.7433 | 0.5250 | 0.6176 | 0.0535 | interesting_not_promotable |
| reduced_vol_cluster | hygiene | 40 | 0.5414 | 0.0677 | 0.8053 | 0.0587 | 0.5124 | 0.7714 | 0.5375 | 0.6324 | 0.0137 | not_promotable |
| top_10_shap_only | compact | 10 | 0.5511 | 0.0242 | 0.8009 | 0.0491 | 0.5789 | 0.7286 | 0.5806 | 0.5294 | 0.1493 | interesting_not_promotable |
| top_15_shap_only | compact | 15 | 0.5524 | 0.0545 | 0.7848 | 0.0326 | 0.5904 | 0.7258 | 0.5676 | 0.6176 | 0.1911 | candidate_winner |
| top_20_shap_only | compact | 20 | 0.5696 | 0.0655 | 0.7619 | 0.0392 | 0.5249 | 0.7761 | 0.5395 | 0.6029 | 0.0420 | not_promotable |

## Readout

- Candidate winner regimes under the conservative rule: `top_15_shap_only`.
- Best compact regime by holdout/CV ordering was `top_15_shap_only` with `15` features, CV AUC `0.5524`, and holdout AUC `0.5904`.
- Compact regime fold-driver stability snapshot: `{"days_since_prior_same_event_type": 5, "fund_snapshot_is_current_event": 4, "cash_ratio": 1, "log_volume": 1, "realized_vol_63d": 1};` holdout top drivers `["days_since_prior_same_event_type", "sec_positive_prob", "realized_vol_21d", "rel_return_10d", "realized_vol_63d"]`.
- Availability-flag removal moved holdout AUC to `0.5399` and holdout log loss to `0.7370`.
- Reduced volatility/volume cluster regime reached CV AUC `0.5414` and holdout AUC `0.5124`.
- Dropping `beta_63d_to_sector` changed holdout AUC from `0.5388` to `0.4891`, confirming whether the feature remains fragile-important after retraining.

## WRDS Checkpoint

- Treat the current full_49 baseline as locked until the same matrix is rerun with WRDS-added feature sets.
- If a compact regime is promising now, keep it as a lean benchmark candidate rather than replacing the default model immediately.
