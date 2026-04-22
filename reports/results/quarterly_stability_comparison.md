# Quarterly Stability Comparison

## Summary

- Plain quarterly checkpoint: holdout AUC `0.4411`.
- Quarterly Alpha Vantage checkpoint: holdout AUC `0.4683`.
- Current best redesigned quarterly checkpoint: holdout AUC `0.4956`.
- Best overall stable recommendation: family `quarterly_plus_sentiment`, style `capped`, model `logistic_regression`, holdout AUC `0.5044`, worst-fold AUC `0.4213`, CV AUC std `0.0480`, and dominant top-3 feature `qfd_delta_growth_quality_profile_score` appearing in `2` folds.
- Best raw holdout configuration remains family `quarterly_plus_sentiment`, style `capped`, model `logistic_regression`, with holdout AUC `0.5044`.

## Family Views

| Family | View | Style | Model | Holdout AUC | CV Mean AUC | CV AUC Std | Worst Fold AUC | Dominant Top-3 Feature | Dominant Top-3 Folds | Holdout Rows | Beat AV | Beat Current Best |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|
| quarterly_core | Trainer Selected | additive | xgboost | 0.4880 | 0.5262 | 0.0501 | 0.4292 | av_days_since_last_earnings_release | 3 | 101 | yes |  |
| quarterly_core | Best Holdout | additive | xgboost | 0.4880 | 0.5262 | 0.0501 | 0.4292 | av_days_since_last_earnings_release | 3 | 101 | yes |  |
| quarterly_core | Best Stable | additive | xgboost | 0.4880 | 0.5262 | 0.0501 | 0.4292 | av_days_since_last_earnings_release | 3 | 101 | yes |  |
| quarterly_plus_sentiment | Trainer Selected | capped | logistic_regression | 0.5044 | 0.5100 | 0.0480 | 0.4213 | qfd_delta_growth_quality_profile_score | 2 | 101 | yes | yes |
| quarterly_plus_sentiment | Best Holdout | capped | logistic_regression | 0.5044 | 0.5100 | 0.0480 | 0.4213 | qfd_delta_growth_quality_profile_score | 2 | 101 | yes | yes |
| quarterly_plus_sentiment | Best Stable | capped | logistic_regression | 0.5044 | 0.5100 | 0.0480 | 0.4213 | qfd_delta_growth_quality_profile_score | 2 | 101 | yes | yes |

## Readout

- A stabilized quarterly candidate now matches or exceeds the current best redesigned quarterly holdout checkpoint.
- `trainer_selected` is kept for audit only. `best_stable` is the quarterly recommendation because it balances holdout performance with fold survival and driver concentration.
