# Quarterly Feature Ablation

## Summary

- Reference regime: CV AUC `0.5111`, holdout AUC `0.4210`.
- Best ablation regime by holdout/CV ordering: `drop_top_shap_feature` with holdout AUC `0.4876`.

## Regime Comparison

| Regime | Added Exclusions | Model | CV AUC | CV Log Loss | Holdout AUC | Holdout Log Loss | Delta Holdout AUC | Top SHAP Feature |
|---|---|---|---:|---:|---:|---:|---:|---|
| drop_top_shap_feature | ["qfd_av_revision_x_surprise"] | xgboost | 0.5147 | 0.7844 | 0.4876 | 0.7894 | 0.0666 | qfd_av_latest_surprise_vs_trailing_pct |
| drop_top_3_shap_features | ["qfd_av_revision_x_surprise", "qfd_av_latest_surprise_vs_trailing_pct", "av_trailing_4q_eps_surprise_pct_std"] | logistic_regression | 0.5123 | 0.7692 | 0.4727 | 0.7645 | 0.0517 | n/a |
| full_reference | [] | xgboost | 0.5111 | 0.7850 | 0.4210 | 0.8218 | 0.0000 | qfd_av_revision_x_surprise |

## Readout

- Use this report to demote short-horizon proxy features only when retraining shows they do not support the 63-day holdout.
