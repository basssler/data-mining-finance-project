# Label Comparison V1

## Scope

- Primary panel: `event_panel_v2` only
- Holdout policy: unchanged 2024 holdout
- CV policy: expanding purged date splits reused from event_v1, with `min_train_dates=252` because the event panel has fewer unique dates than the daily panel
- Model families: logistic regression, random forest, XGBoost

## Feature Exclusions

- Candidate feature count before exclusions: `50`
- Global all-missing exclusions: `gross_margin`
- Global constant exclusions: `current_filing_sentiment_available`
- Additional fold-level exclusions were applied when train-fold missingness exceeded 20% or a feature became constant inside a training fold.

## Best Sign-Horizon Comparison

| Horizon | Best Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5-day | random_forest | 0.5260 | 0.6989 | 0.5237 | 0.6945 | 0.5143 | 0.5294 | 0.0201 |
| 21-day | xgboost | 0.5194 | 0.8573 | 0.5022 | 0.7737 | 0.5532 | 0.3768 | -0.0722 |

## Full Model Table

| Variant | Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | Best For Variant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| event_v2_21d_quantile_20 | logistic_regression | 0.4194 | 0.9431 | 0.4194 | 0.8507 | 0.4444 | 0.3750 | -0.0763 |  |
| event_v2_21d_quantile_20 | random_forest | 0.4936 | 0.7075 | 0.4113 | 0.7216 | 0.4643 | 0.4062 | -0.0829 |  |
| event_v2_21d_quantile_20 | xgboost | 0.5113 | 0.8611 | 0.4476 | 0.8530 | 0.4828 | 0.4375 | -0.0727 | yes |
| event_v2_21d_sign | logistic_regression | 0.4824 | 0.7787 | 0.4290 | 0.7589 | 0.4667 | 0.3043 | -0.1006 |  |
| event_v2_21d_sign | random_forest | 0.5046 | 0.7100 | 0.4493 | 0.7116 | 0.4762 | 0.2899 | -0.1100 |  |
| event_v2_21d_sign | xgboost | 0.5194 | 0.8573 | 0.5022 | 0.7737 | 0.5532 | 0.3768 | -0.0722 | yes |
| event_v2_5d_quantile_20 | logistic_regression | 0.5128 | 0.9032 | 0.5057 | 0.7615 | 0.6000 | 0.6923 | 0.0050 |  |
| event_v2_5d_quantile_20 | random_forest | 0.4813 | 0.7038 | 0.4023 | 0.7124 | 0.4865 | 0.4615 | -0.0654 |  |
| event_v2_5d_quantile_20 | xgboost | 0.5419 | 0.8248 | 0.4430 | 0.8682 | 0.5312 | 0.4359 | -0.1263 | yes |
| event_v2_5d_sign | logistic_regression | 0.5181 | 0.7510 | 0.5051 | 0.7409 | 0.5000 | 0.5441 | 0.0778 |  |
| event_v2_5d_sign | random_forest | 0.5260 | 0.6989 | 0.5237 | 0.6945 | 0.5143 | 0.5294 | 0.0201 | yes |
| event_v2_5d_sign | xgboost | 0.5236 | 0.8163 | 0.5271 | 0.7481 | 0.5467 | 0.6029 | 0.0368 |  |

## Recommendation

- Direct recommendation: **keep 5-day as primary**
- Recommendation rule: prefer the horizon whose best model improves CV AUC and CV log loss without reversing the direction on the 2024 holdout.
- If neither horizon wins cleanly across both CV and holdout, treat the horizon choice as unresolved rather than forcing a switch.

