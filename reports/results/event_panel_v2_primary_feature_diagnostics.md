# Event Panel V2 Primary Feature Diagnostics

## Scope

- Selected model under test: `xgboost`
- Baseline holdout AUC: `0.5388`
- Baseline holdout log loss: `0.7474`
- Baseline holdout rank IC: `0.0902`
- SHAP anchor features used for diagnostics: `beta_63d_to_sector, sec_positive_prob, rel_return_10d, cash_ratio, log_volume, vol_ratio_21d_63d, realized_vol_21d, rel_return_5d, volume_ratio_20d, abs_return_shock_1d`

## Feature Ablation

| Scenario | Feature Count | Holdout AUC | Holdout Log Loss | Precision | Recall | Rank IC |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 49 | 0.5388 | 0.7474 | 0.5616 | 0.6029 | 0.0902 |
| drop_beta_63d_to_sector | 48 | 0.4891 | 0.7607 | 0.4706 | 0.4706 | -0.0462 |
| drop_cash_ratio | 48 | 0.5139 | 0.7681 | 0.5432 | 0.6471 | 0.0383 |
| drop_log_volume | 48 | 0.5023 | 0.7561 | 0.4714 | 0.4853 | 0.0262 |
| drop_rel_return_10d | 48 | 0.5505 | 0.7429 | 0.5385 | 0.6176 | 0.1001 |
| drop_sec_positive_prob | 48 | 0.5134 | 0.7599 | 0.5263 | 0.5882 | 0.0032 |
| drop_top_3_shap | 46 | 0.4597 | 0.7928 | 0.4638 | 0.4706 | -0.1053 |
| drop_top_5_shap | 44 | 0.4606 | 0.7873 | 0.4861 | 0.5147 | -0.0920 |
| no_availability_flags | 47 | 0.5399 | 0.7370 | 0.5476 | 0.6765 | 0.0890 |
| price_volume_only | 14 | 0.5367 | 0.7305 | 0.5250 | 0.6176 | 0.0482 |
| sentiment_only | 10 | 0.4098 | 0.8209 | 0.4337 | 0.5294 | -0.1170 |
| top_10_shap_only | 10 | 0.5789 | 0.7286 | 0.5806 | 0.5294 | 0.1493 |

## Key Read

- Strongest direct removal signal came from `drop_top_3_shap`, moving holdout AUC from `0.5388` to `0.4597`.
- Most efficient reduced feature set in this run was `top_10_shap_only`, reaching holdout AUC `0.5789` with `10` features.

## Top-Feature Deciles

| Feature | Lowest Decile Hit Rate | Highest Decile Hit Rate | Lowest Decile Mean Pred Prob | Highest Decile Mean Pred Prob |
|---|---:|---:|---:|---:|
| beta_63d_to_sector | 0.5000 | 0.2143 | 0.6698 | 0.5226 |
| sec_positive_prob | 0.5714 | 0.6429 | 0.5216 | 0.6122 |
| rel_return_10d | 0.7143 | 0.5714 | 0.5958 | 0.5588 |
| cash_ratio | 0.3846 | 0.6923 | 0.5579 | 0.5546 |
| log_volume | 0.4286 | 0.5714 | 0.5359 | 0.4888 |
| vol_ratio_21d_63d | 0.2857 | 0.4286 | 0.4075 | 0.5211 |
| realized_vol_21d | 0.5714 | 0.5000 | 0.5250 | 0.4987 |
| rel_return_5d | 0.5000 | 0.5714 | 0.5150 | 0.4716 |
| volume_ratio_20d | 0.4286 | 0.5000 | 0.5102 | 0.5100 |
| abs_return_shock_1d | 0.6429 | 0.5000 | 0.5375 | 0.5050 |
